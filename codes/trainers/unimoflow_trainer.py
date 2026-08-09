import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict, OrderedDict
import os
from os.path import join as pjoin
import time
import numpy as np
import random
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    torch.serialization.add_safe_globals([
        np._core.multiarray.scalar,
        np.core.multiarray.scalar,
        np.dtype,
        np.float32,
        np.float64,
        np.int64,
        np.int32,
    ])
except AttributeError:
    pass


def def_value():
    return 0.0


class UniMoFlowTrainer:
    """
    UniMoFlow trainer for edit and text-to-motion generation.

    Supported schedules:
    - probabilistic: each optimizer step randomly chooses edit or gen by gen_prob.
    - all_data_joint: each optimizer step uses one edit batch and one gen batch, sums losses,
      and cycles the shorter loader.
    Validation and evaluation still run edit/gen independently.
    """

    def __init__(self, cfg, model, vae, device, local_rank=None, world_size=None,
                 gen_prob=0.5, geo_loss=False, mixing_strategy=None):
        self.cfg = cfg
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if local_rank is None else local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1)) if world_size is None else world_size
        self.is_main_process = (self.local_rank == 0)
        self.gen_prob = gen_prob
        self.mixing_strategy = mixing_strategy or cfg.training.get('mixing_strategy', 'probabilistic')
        self.mixing_strategy = self._normalize_mixing_strategy(self.mixing_strategy)
        self.geo_loss = geo_loss
        self.device = device

        # VAE setup（用于解码评估，不训练）
        self.vae = vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # 模型
        self.model = model.to(self.device)
        self.model.train()

        # DDP包装
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

        # Logger
        if cfg.exp.is_train and self.is_main_process:
            self.logger = SummaryWriter(cfg.exp.log_dir)
        else:
            self.logger = None

        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        # 双模式指标追踪器
        from utils.unimoflow_eval import UniMoFlowMetricsTracker
        self.tracker = UniMoFlowMetricsTracker()

    @staticmethod
    def _normalize_mixing_strategy(strategy):
        strategy = str(strategy).lower().strip()
        aliases = {
            'prob': 'probabilistic',
            'random': 'probabilistic',
            'probabilistic': 'probabilistic',
            'all_data': 'all_data_joint',
            'all_data_joint': 'all_data_joint',
            'joint': 'all_data_joint',
            'simultaneous': 'all_data_joint',
        }
        if strategy not in aliases:
            raise ValueError(
                f"Unknown mixing_strategy={strategy!r}. Expected 'probabilistic' or 'all_data_joint'."
            )
        return aliases[strategy]

    def prepare_edit_batch(self, batch_data):
        """准备编辑任务的batch数据"""
        source = batch_data["source"].detach().float().to(self.device, non_blocking=True)
        target = batch_data["target"].detach().float().to(self.device, non_blocking=True)
        edit_text = batch_data["edit_text"]
        m_lens = batch_data["length"].detach().long().to(self.device, non_blocking=True)

        x = {
            "source": source,
            "target": target,
            "edit_text": edit_text,
            "length": m_lens,
            "mode": "edit",
        }
        return x

    def prepare_gen_batch(self, batch_data):
        """准备生成任务的batch数据（来自 LatentTextMotionDataset）"""
        conds, motion, m_lens = batch_data

        motion = motion.detach().float().to(self.device, non_blocking=True)
        m_lens = m_lens.detach().long().to(self.device, non_blocking=True)

        x = {
            "target": motion,
            "text": conds,
            "length": m_lens,
            "mode": "gen",
        }
        return x

    def forward(self, batch_data, mode="edit"):
        if mode == "edit":
            x = self.prepare_edit_batch(batch_data)
        else:
            x = self.prepare_gen_batch(batch_data)

        loss_dict = self.model(x)
        loss = loss_dict['total']
        mse = loss_dict['mse']
        return loss, mse

    def update(self, batch_data, mode, step_scheduler=True):
        self.opt_model.zero_grad()

        if mode == "edit":
            x = self.prepare_edit_batch(batch_data)
        else:
            x = self.prepare_gen_batch(batch_data)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.amp.autocast("cuda", dtype=dtype):
            loss_dict = self.model(x)
            loss = loss_dict['total']
            mse = loss_dict['mse']

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt_model)

        model_params = self.model.module.parameters() if isinstance(self.model, DDP) else self.model.parameters()
        torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)

        self.scaler.step(self.opt_model)
        self.scaler.update()

        if step_scheduler:
            self.scheduler.step()

        return loss.item(), mse.item(), mode

    def update_joint(self, edit_batch_data, gen_batch_data, step_scheduler=True):
        """Run edit and gen losses in the same optimizer step."""
        self.opt_model.zero_grad()
        edit_x = self.prepare_edit_batch(edit_batch_data)
        gen_x = self.prepare_gen_batch(gen_batch_data)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.amp.autocast("cuda", dtype=dtype):
            edit_loss_dict = self.model(edit_x)
            gen_loss_dict = self.model(gen_x)
            edit_loss = edit_loss_dict['total']
            gen_loss = gen_loss_dict['total']
            loss = edit_loss + gen_loss
            mse = 0.5 * (edit_loss_dict['mse'] + gen_loss_dict['mse'])

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt_model)

        model_params = self.model.module.parameters() if isinstance(self.model, DDP) else self.model.parameters()
        torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)

        self.scaler.step(self.opt_model)
        self.scaler.update()

        if step_scheduler:
            self.scheduler.step()

        return {
            'loss': loss.item(),
            'mse': mse.item(),
            'edit_loss': edit_loss.item(),
            'edit_mse': edit_loss_dict['mse'].item(),
            'gen_loss': gen_loss.item(),
            'gen_mse': gen_loss_dict['mse'].item(),
        }

    def save(self, file_name, ep, total_it):
        if not self.is_main_process:
            return

        state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        state = {
            'model': state_dict,
            'optimizer': self.opt_model.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
            'tracker': self.tracker.state_dict(),
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Checkpoint not found: {model_dir}")

        try:
            checkpoint = torch.load(model_dir, map_location=self.device)
        except Exception as e:
            if "Weights only load failed" in str(e) or "UnpicklingError" in str(e):
                print(f"[Rank {self.local_rank}] Safe load failed, retrying with weights_only=False...")
                checkpoint = torch.load(model_dir, map_location=self.device, weights_only=False)
            else:
                raise e

        model_state = checkpoint['model']
        current_model = self.model.module if isinstance(self.model, DDP) else self.model
        missing, unexpected = current_model.load_state_dict(model_state, strict=False)

        if self.is_main_process:
            if len(missing) > 0:
                print(f"Warning: Missing keys in checkpoint: {missing}")
            if len(unexpected) > 0:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected}")

        try:
            if 'optimizer' in checkpoint:
                self.opt_model.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint and hasattr(self, 'scheduler'):
                self.scheduler.load_state_dict(checkpoint['scheduler'])
        except Exception as e:
            if self.is_main_process:
                print(f'Resume warning: Could not load optimizer/scheduler state: {e}')

        start_ep = checkpoint.get('ep', 0)
        start_it = checkpoint.get('total_it', 0)

        if 'tracker' in checkpoint:
            self.tracker.load_state_dict(checkpoint['tracker'])

        if self.is_main_process:
            print(f"[Resume] Loaded checkpoint from epoch {start_ep}, iteration {start_it}")
            print(f"[Resume] Edit best overall: {self.tracker.edit_best_overall:.4f}, "
                  f"Gen best FID: {self.tracker.gen_best_fid:.4f}")

        return start_ep, start_it

    def _safe_barrier(self, desc=""):
        if self.world_size <= 1:
            return
        try:
            dist.barrier()
        except Exception as e:
            print(f"[Rank {self.local_rank}] Barrier failed ({desc}): {e}")
            raise

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_model.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def train(self, edit_loader, gen_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval,
              gen_val_loader=None, gen_eval_val_loader=None, gen_plot_eval=None):
        """
        混合训练主循环

        Args:
            edit_loader: 编辑数据集的 DataLoader（训练）
            gen_loader: 生成数据集的 DataLoader（训练）
            val_loader: 编辑验证集的 DataLoader（计算val loss）
            eval_val_loader: 编辑评估集的 DataLoader（evaluate_edit_model）
            eval_wrapper: 评估包装器
            plot_eval: 编辑模式可视化函数
            gen_val_loader: 生成验证集的 DataLoader（计算gen val loss，可选）
            gen_eval_val_loader: 生成评估集的 DataLoader（evaluation_diffusion_model，可选）
            gen_plot_eval: 生成模式可视化函数（可选）
        """
        # 优化器
        self.opt_model = optim.AdamW(
            self.model.parameters(),
            betas=(0.9, 0.99),
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.get('weight_decay', 1e-5))
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.opt_model,
            milestones=self.cfg.training.milestones,
            gamma=self.cfg.training.gamma
        )

        epoch = 0
        it = 0

        if self.cfg.exp.is_continue:
            model_dir = pjoin(self.cfg.exp.model_dir, 'latest.tar')
            if os.path.exists(model_dir):
                epoch, it = self.resume(model_dir)
                if self.is_main_process:
                    print(f"Load model epoch:{epoch} iterations:{it}")

        start_time = time.time()
        if self.mixing_strategy == 'all_data_joint':
            steps_per_epoch = max(len(edit_loader), len(gen_loader))
            schedule_desc = (
                f"{steps_per_epoch} joint steps, each step uses one edit batch and one gen batch; "
                f"shorter loader cycles"
            )
        else:
            steps_per_epoch = len(edit_loader)
            schedule_desc = f"{steps_per_epoch} edit-based steps, switch to gen with p={self.gen_prob:.2f}"
        max_iters = self.cfg.training.max_epoch * steps_per_epoch
        logs = defaultdict(def_value, OrderedDict())

        if self.is_main_process:
            print(f'Total Epochs: {self.cfg.training.max_epoch}, Max Iters: {max_iters}')
            print(f'Edit Loader Size: {len(edit_loader):04d}, Gen Loader Size: {len(gen_loader):04d}')
            print(f'Mixing Strategy: {self.mixing_strategy}')
            print(f'Epoch schedule: {schedule_desc}')
            print(f'Gen Eval Loader: {"provided" if gen_eval_val_loader is not None else "NOT provided"}')

        if self.world_size > 1:
            dist.barrier()

        while epoch < self.cfg.training.max_epoch:
            # 设置 epoch
            if self.world_size > 1:
                if isinstance(edit_loader.sampler, DistributedSampler):
                    edit_loader.sampler.set_epoch(epoch)
                if isinstance(gen_loader.sampler, DistributedSampler):
                    gen_loader.sampler.set_epoch(epoch)

            self.model.train()
            self.vae.eval()

            edit_iter = iter(edit_loader)
            gen_iter = iter(gen_loader)
            iterator = range(steps_per_epoch)

            if self.is_main_process:
                iterator = tqdm(iterator, total=steps_per_epoch, desc=f"Epoch {epoch}")

            for i in iterator:
                it += 1
                if it < self.cfg.training.warm_up_iter:
                    self.update_lr_warm_up(it, self.cfg.training.warm_up_iter, self.cfg.training.lr)

                if self.mixing_strategy == 'all_data_joint':
                    try:
                        edit_batch = next(edit_iter)
                    except StopIteration:
                        edit_iter = iter(edit_loader)
                        edit_batch = next(edit_iter)
                    try:
                        gen_batch = next(gen_iter)
                    except StopIteration:
                        gen_iter = iter(gen_loader)
                        gen_batch = next(gen_iter)

                    step_stats = self.update_joint(
                        edit_batch, gen_batch, it >= self.cfg.training.warm_up_iter
                    )
                    logs['loss'] += step_stats['loss']
                    logs['mse'] += step_stats['mse']
                    logs['edit_loss'] += step_stats['edit_loss']
                    logs['edit_mse'] += step_stats['edit_mse']
                    logs['gen_loss'] += step_stats['gen_loss']
                    logs['gen_mse'] += step_stats['gen_mse']
                    logs['lr'] += self.opt_model.param_groups[0]['lr']
                    logs['edit_count'] += 1
                    logs['gen_count'] += 1
                else:
                    mode = "gen" if random.random() < self.gen_prob else "edit"
                    if mode == "gen":
                        try:
                            batch = next(gen_iter)
                        except StopIteration:
                            gen_iter = iter(gen_loader)
                            batch = next(gen_iter)
                    else:
                        try:
                            batch = next(edit_iter)
                        except StopIteration:
                            edit_iter = iter(edit_loader)
                            batch = next(edit_iter)

                    loss, mse, actual_mode = self.update(batch, mode, it >= self.cfg.training.warm_up_iter)
                    logs['loss'] += loss
                    logs['mse'] += mse
                    logs['lr'] += self.opt_model.param_groups[0]['lr']
                    logs[f'{actual_mode}_count'] += 1

                if it % self.cfg.training.log_every == 0 and self.is_main_process:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        if tag.endswith('_count'):
                            continue
                        self.logger.add_scalar(f'Train/{tag}', value / self.cfg.training.log_every, it)
                        mean_loss[tag] = value / self.cfg.training.log_every
                    logs = defaultdict(def_value, OrderedDict())

                    current_lr = self.opt_model.param_groups[0]['lr']
                    msg = (
                        f"[Epoch {epoch} Iter {it}/{max_iters}] Loss: {mean_loss['loss']:.4f}, "
                        f"MSE: {mean_loss['mse']:.4f}, LR: {current_lr:.6f}"
                    )
                    if self.mixing_strategy == 'all_data_joint':
                        msg += (
                            f", EditLoss: {mean_loss.get('edit_loss', 0.0):.4f}, "
                            f"GenLoss: {mean_loss.get('gen_loss', 0.0):.4f}"
                        )
                    print(msg)

            # 保存 latest
            self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            # ========== 验证阶段 ==========
            if self.is_main_process:
                print('Validation time:')

            self._safe_barrier("epoch_end")

            self.model.eval()

            # --- 编辑模式验证 ---
            val_loss = []
            val_mse = []

            with torch.no_grad():
                for batch_data in val_loader:
                    loss, mse = self.forward(batch_data, mode="edit")
                    val_loss.append(loss.item())
                    val_mse.append(mse.item())

            mean_val_loss = np.mean(val_loss)
            mean_val_mse = np.mean(val_mse)

            # --- 生成模式验证（如果提供 gen_val_loader）---
            gen_val_loss = []
            gen_val_mse = []

            if gen_val_loader is not None:
                with torch.no_grad():
                    for batch_data in gen_val_loader:
                        loss, mse = self.forward(batch_data, mode="gen")
                        gen_val_loss.append(loss.item())
                        gen_val_mse.append(mse.item())

                mean_gen_val_loss = np.mean(gen_val_loss) if gen_val_loss else 0.0
                mean_gen_val_mse = np.mean(gen_val_mse) if gen_val_mse else 0.0
            else:
                mean_gen_val_loss = 0.0
                mean_gen_val_mse = 0.0

            # 多卡聚合
            if self.world_size > 1:
                metrics_tensor = torch.tensor(
                    [mean_val_loss, mean_val_mse, mean_gen_val_loss, mean_gen_val_mse],
                    device=self.device
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
                mean_val_loss, mean_val_mse = metrics_tensor[0].item(), metrics_tensor[1].item()
                mean_gen_val_loss, mean_gen_val_mse = metrics_tensor[2].item(), metrics_tensor[3].item()

            if self.is_main_process:
                print(f"Val Edit Loss: {mean_val_loss:.3f}, MSE: {mean_val_mse:.3f}")
                self.logger.add_scalar('Val/edit_loss', mean_val_loss, epoch)
                self.logger.add_scalar('Val/edit_mse', mean_val_mse, epoch)
                if gen_val_loader is not None:
                    print(f"Val Gen  Loss: {mean_gen_val_loss:.3f}, MSE: {mean_gen_val_mse:.3f}")
                    self.logger.add_scalar('Val/gen_loss', mean_gen_val_loss, epoch)
                    self.logger.add_scalar('Val/gen_mse', mean_gen_val_mse, epoch)

            # ========== 评估阶段（每2个epoch）==========
            self._safe_barrier("epoch_end")

            if epoch % 2 == 0:
                from utils.unimoflow_eval import evaluate_edit, evaluate_gen

                # --- 编辑模式评估 ---
                if self.is_main_process:
                    print(f'\n{"=" * 60}')
                    print(f'Edit Evaluation (Epoch {epoch})')
                    print(f'{"=" * 60}')

                fid_edit, match_score, struct_score, overall_score, top1_edit, top2_edit, top3_edit, positive_improve = \
                    evaluate_edit(
                        out_dir=self.cfg.exp.model_dir,
                        val_loader=eval_val_loader,
                        unimoflow_model=self.model,
                        vae_model=self.vae,
                        eval_wrapper=eval_wrapper,
                        writer=self.logger,
                        ep=epoch,
                        tracker=self.tracker,
                        device=self.device,
                        plot_func=plot_eval if self.is_main_process else None,
                        num_denoise_steps=self.cfg.model.get('eval_noise_steps', 10),
                        cfg_scale=self.cfg.model.cfg_scale,
                        save_ckpt=self.is_main_process,
                        save_anim=(epoch % self.cfg.training.get('save_anim_every', 10) == 0),
                        distributed=self.world_size > 1,
                        world_size=self.world_size,
                        local_rank=self.local_rank,
                        structure_weight=self.cfg.training.get('structure_weight', 0.5),
                        matching_weight=self.cfg.training.get('matching_weight', 0.5),
                    )

                # 更新编辑模式最佳指标
                if self.is_main_process:
                    model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
                    self.tracker.update_edit_best(
                        fid_edit, match_score, struct_score, overall_score, top1_edit, positive_improve,
                        self.cfg.exp.model_dir, model_state, epoch
                    )

                self._safe_barrier("eval_edit_done")

                # --- 生成模式评估（如果提供 gen_eval_val_loader）---
                if gen_eval_val_loader is not None:
                    if self.is_main_process:
                        print(f'\n{"=" * 60}')
                        print(f'Generation Evaluation (Epoch {epoch})')
                        print(f'{"=" * 60}')

                    fid_gen, diversity, top1_gen, top2_gen, top3_gen, matching_gen = \
                        evaluate_gen(
                            out_dir=self.cfg.exp.model_dir,
                            val_loader=gen_eval_val_loader,
                            unimoflow_model=self.model,
                            vae_model=self.vae,
                            eval_wrapper=eval_wrapper,
                            writer=self.logger,
                            ep=epoch,
                            tracker=self.tracker,
                            device=self.device,
                            plot_func=gen_plot_eval if self.is_main_process else None,
                            num_denoise_steps=self.cfg.model.get('eval_noise_steps', 10),
                            cfg_scale=self.cfg.model.cfg_scale,
                            save_ckpt=self.is_main_process,
                            save_anim=(epoch % self.cfg.training.get('save_anim_every', 10) == 0),
                            distributed=self.world_size > 1,
                            world_size=self.world_size,
                            local_rank=self.local_rank,
                        )

                    # 更新生成模式最佳指标
                    if self.is_main_process:
                        model_state = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
                        self.tracker.update_gen_best(
                            fid_gen, diversity, top1_gen, top2_gen, top3_gen, matching_gen,
                            self.cfg.exp.model_dir, model_state, epoch
                        )

                self._safe_barrier("eval_gen_done")
