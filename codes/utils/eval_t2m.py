import os

# import clip
import numpy as np
import torch
# from scipy import linalg
from utils.metrics import *
import torch.nn.functional as F
from tqdm import tqdm


def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor: # type: ignore
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


@torch.no_grad()
def evaluation_evaluator(out_dir, eval_val_loader, writer, ep, best_top1, best_top2, best_top3, 
                         best_matching, eval_model, device, save_ckpt=True, draw=True):
    # eval_model.eval()

    def save(file_path, ep):
        state = {
            "latent_enc": eval_model.latent_enc.state_dict(),
            "text_enc": eval_model.text_enc.state_dict(),
            "ep": ep,
        }

        if "motion_enc" in eval_model.state_dict():
            state["motion_enc"] = eval_model.motion_enc.state_dict()
        
        # if "text_enc" in eval_model.state_dict():
        #     state["text_enc"] = eval_model.text_enc.state_dict(),


        torch.save(state, file_path)

    motion_annotation_list = []

    R_precision_real = 0

    nb_sample = 0
    matching_score_real = 0
    for batch in eval_val_loader:
        # print(len(batch))
        texts, motions, m_lengths = batch

        motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_model.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_model.encode_motion(motions, m_lengths, sample_mean=True)

        bs, _ = motions.shape[0], motions.shape[1]


        motion_annotation_list.append(fid_em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample

    matching_score_real = matching_score_real / nb_sample

    msg = "--> \t Eva. Ep %d:, Diversity Real. %.4f, R_precision_real. (%.4f, %.4f, %.4f), matching_score_real. %.4f"%\
          (ep, diversity_real, R_precision_real[0],R_precision_real[1], R_precision_real[2], matching_score_real ) # type: ignore
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('Eval/Diversity', diversity_real, ep)
        writer.add_scalar('Eval/top1', R_precision_real[0], ep) # type: ignore
        writer.add_scalar('Eval/top2', R_precision_real[1], ep)
        writer.add_scalar('Eval/top3', R_precision_real[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_real, ep)


    # msg = "--> --> \t Diversity %.5f !!!"%(diversity_real)
    # print(msg)
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision_real[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision_real[0])
        if draw: print(msg)
        best_top1 = R_precision_real[0]
        if save_ckpt:
            save(os.path.join(out_dir, 'net_best_top1.tar'), ep)
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision_real[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision_real[1])
        if draw: print(msg)
        best_top2 = R_precision_real[1]

    if R_precision_real[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision_real[2])
        if draw: print(msg)
        best_top3 = R_precision_real[2]

    if matching_score_real > best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_real)
        if draw: print(msg)
        best_matching = matching_score_real
        if save_ckpt:
            # save(os.path.join(out_dir, 'net_best_mm.tar'),
            #      ep
            #      )
            save(os.path.join(out_dir, 'net_best_mm.tar'), ep)
    # eval_model.train()

    return diversity_real, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, best_mpjpe, nfeats,
                     eval_wrapper, device, fk_func, save_ckpt=True, draw=True):
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0

    mpjpe_error_sum = 0
    frame_count_sum = 0

    net.eval()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                desc=f"Epoch {ep}", ncols=120)
    for i, batch in pbar:
        texts, motions, m_lengths = batch

        # motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs, _ = motions.shape[0], motions.shape[1]


        if 'vq' in out_dir:
            _, all_codes = net.encode(motions[...,:nfeats], m_lengths.clone())
        else:
            all_codes = net.encode(motions[..., :nfeats])
        rec_motions = net.decode(all_codes, m_lengths.clone())
        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(rec_motions[..., :148], m_lengths, sample_mean=True)
        orig_len = motions.shape[1]
        recon_len = rec_motions.shape[1]

        if orig_len != recon_len:
            min_len = min(orig_len, recon_len)
            # 截取到相同长度
            motions_aligned = motions[:, :min_len, :]
            rec_motions_aligned = rec_motions[:, :min_len, :]
            # 更新mask长度
            m_lengths_mpjpe = torch.clamp(m_lengths, max=min_len)
        else:
            motions_aligned = motions
            rec_motions_aligned = rec_motions
            m_lengths_mpjpe = m_lengths

        batch_mpjpe_error, batch_frame_count = calculate_mpjpe(
            fk_func(rec_motions_aligned),
            fk_func(motions_aligned),
            mask=length_to_mask(m_lengths_mpjpe, motions_aligned.shape[1]),
            only_local=False
            )
        
        mpjpe_error_sum += batch_mpjpe_error
        frame_count_sum += batch_frame_count

        motion_pred_list.append(fid_em_pred)
        motion_annotation_list.append(fid_em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    pbar.close()
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe_error = mpjpe_error_sum / frame_count_sum

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f, mpjpe. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe_error )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('Eval/FID', fid, ep)
        writer.add_scalar('Eval/Diversity', diversity, ep)
        writer.add_scalar('Eval/top1', R_precision[0], ep)
        writer.add_scalar('Eval/top2', R_precision[1], ep)
        writer.add_scalar('Eval/top3', R_precision[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_pred, ep)
        writer.add_scalar('Eval/mpjpe', mpjpe_error, ep)

    draw = True
    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save_ckpt:
            torch.save({'model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save_ckpt:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred > best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save_ckpt:
            torch.save({'model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    if mpjpe_error < best_mpjpe:
        msg = f"--> --> \t mpjpe Improved from %.5f to %.5f !!!" % (best_mpjpe, mpjpe_error)
        if draw: print(msg)
        best_mpjpe = mpjpe_error
        if save_ckpt:
            torch.save({'model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mpjpe.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    # net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe


@torch.no_grad()
def evaluation_vqvae_hml(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                         best_top2, best_top3, best_matching, eval_wrapper, device='cuda', save=True, draw=True):
    net.eval()
    net.to(device)  # 确保模型在正确设备

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0

    # 使用 tqdm 显示进度
    pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                desc=f"Epoch {ep}", ncols=120)

    for i, batch in pbar:
        # 解包 batch - 注意顺序要和 dataloader 一致
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        # 统一使用传入的 device
        motion = motion.to(device).float().detach()
        m_length = m_length.to(device).long().detach()
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        # pred_pose_eval, loss_commit, perplexity = net(motion)
        if 'vq' in out_dir:
            _, all_codes = net.encode(motion, m_length.clone())
        else:
            all_codes = net.encode(motion, m_length.clone())
        # _, all_codes = net.encode(motion, m_length.clone())
        pred_pose_eval = net.decode(all_codes, m_length.clone())
        motion = eval_wrapper.inv_transform(motion)
        pred_pose_eval = eval_wrapper.inv_transform(pred_pose_eval)
        orig_len = motion.shape[1]
        recon_len = pred_pose_eval.shape[1]

        if orig_len != recon_len:
            min_len = min(orig_len, recon_len)
            # 截取到相同长度
            motion = motion[:, :min_len, :]
            pred_pose_eval = pred_pose_eval[:, :min_len, :]
            # 更新mask长度
            m_length = torch.clamp(m_length, max=min_len)
        else:
            motion = motion
            pred_pose_eval = pred_pose_eval
            m_length = m_length

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        # 🔴 关键：立即检查并清理 embedding 中的 NaN/Inf
        def sanitize_tensor(t, name):
            if torch.isnan(t).any() or torch.isinf(t).any():
                nan_count = torch.isnan(t).sum().item()
                inf_count = torch.isinf(t).sum().item()
                print(f"[WARNING] {name} has {nan_count} NaN, {inf_count} Inf in batch {i}")
                # 替换为0（或者可以用均值填充）
                t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            return t

        em = sanitize_tensor(em, "em (gt)")
        em_pred = sanitize_tensor(em_pred, "em_pred")
        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    pbar.close()
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    # # 检查并修复异常值
    # def sanitize_array(arr, name):
    #     if not np.isfinite(arr).all():
    #         print(f"[WARNING] {name} has {(~np.isfinite(arr)).sum()} non-finite values")
    #         arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
    #     return arr

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    # motion_annotation_np = sanitize_array(motion_annotation_np, "annotation")
    # motion_pred_np = sanitize_array(motion_pred_np, "prediction")

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    print('1')
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    print('2')
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    print('3')
    fid = calculate_frechet_distance_torch(gt_mu, gt_cov, mu, cov, device=device)
    print('4')
    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)
    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))
    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, device, plot_func, time_steps=20,
                           cond_scale = 4, save_ckpt=False, save_anim=False, draw=True):


    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0


    nb_sample = 0
    # for i in range(1):
    for batch in tqdm(val_loader):
        texts, motions, m_lengths = batch
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs, _ = motions.shape[0], motions.shape[1]

        # mids, _ = vq_model.encode(motions)
        # mids = mids[..., 0:1]
        # motion_codes = motion_codes.permute(0, 2, 1)
        mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, temperature=1)
        pred_motions = vq_model.forward_decoder(mids, m_lengths.clone())


        # mids, _ = vq_model.encode(motions)
        # mids = mids['top']

        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)

        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    
    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    if draw: print(msg)

    if draw:
        writer.add_scalar('Eval/FID', fid, ep)
        writer.add_scalar('Eval/Diversity', diversity, ep)
        writer.add_scalar('Eval/top1', R_precision[0], ep)
        writer.add_scalar('Eval/top2', R_precision[1], ep)
        writer.add_scalar('Eval/top3', R_precision[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_pred, ep)


    draw = True
    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        if draw:print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            torch.save({"t2m_transformer":trans.state_dict(), "ep":ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if matching_score_pred > best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        if draw:print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        if draw:print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        if draw:print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        if draw:print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        if draw:print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [texts[k] for k in rand_idx]
        lengths = m_lengths[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_mask_transformer_hml(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper,device, plot_func, time_steps = 10,
                           cond_scale=4, save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        # m_length = m_length.cuda()
        # motions = motions.to(device).float().detach()
        m_length = m_length.to(device).long().detach()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
        pred_motions = vq_model.forward_decoder(mids, m_length.clone())

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.to(device).float().detach()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar('Eval/FID', fid, ep)
    writer.add_scalar('Eval/Diversity', diversity, ep)
    writer.add_scalar('Eval/top1', R_precision[0], ep)
    writer.add_scalar('Eval/top2', R_precision[1], ep)
    writer.add_scalar('Eval/top3', R_precision[2], ep)
    writer.add_scalar('Eval/matching_score', matching_score_pred, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir,  'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching




@torch.no_grad()
def evaluation_momask(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper, 
                      time_steps, cond_scale, temperature, topkr, gsample=True, 
                      force_mask=False, cal_mm=True, res_cond_scale=5):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    device = res_model.device

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 1

    for i, batch in enumerate(tqdm(val_loader)):
        texts, motions, m_lengths = batch

        # motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs, _ = motions.shape[0], motions.shape[1]

        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):

                mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, 
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                pred_ids = res_model.generate(mids, texts, m_lengths//4, temperature=1, cond_scale=res_cond_scale)
                # pred_ids = mids.unsqueeze(-1)

                pred_motions = vq_model.forward_decoder(pred_ids)

                fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(fid_em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, 
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

            pred_ids = res_model.generate(mids, texts, m_lengths//4, temperature=1, cond_scale=res_cond_scale)
            # pred_ids = mids.unsqueeze(-1)
            
            # pred_ids, _ = vq_model.encode(motions)
            pred_motions = vq_model.forward_decoder(pred_ids)

            # pred_motions[..., 1] = 0
            # motions[..., 90:100] = 0
            # pred_motions += torch.randn_like(pred_motions) 

            fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)

        # pose = pose.cuda().float()
        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_momask_plus(val_loader, vq_model, trans, repeat_id, eval_wrapper, 
                      time_steps, cond_scale, temperature, topkr, gsample=True, cal_mm=True):
    trans.eval()
    vq_model.eval()

    device = trans.device

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 1
    else:
        num_mm_batch = 0

    for i, batch in enumerate(tqdm(val_loader)):
        texts, motions, m_lengths = batch

        # motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs, _ = motions.shape[0], motions.shape[1]

        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):

                mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, 
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample)

                pred_motions = vq_model.forward_decoder(mids, m_lengths.clone())

                fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(fid_em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(texts, m_lengths//4, time_steps, cond_scale, 
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample)

            pred_motions = vq_model.forward_decoder(mids, m_lengths.clone())

            fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)

        # fid_em_pred, em_pred = fid_em, em
        # pose = pose.cuda().float()
        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_momask_plus_hml(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if  cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
                pred_motions = vq_model.forward_decoder(mids, m_length.clone())

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
            pred_motions = vq_model.forward_decoder(mids, m_length.clone())

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_diffusion_model(out_dir, val_loader, diffusion_model, vae_model, writer, ep,
                               best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
                               eval_wrapper, device, plot_func, num_denoise_steps=20, cfg_scale=4.0,
                               save_ckpt=True, save_anim=False,
                               distributed=False, world_size=1, local_rank=0):
    """
    支持多卡分布式评估的evaluation函数
    """
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    diffusion_model.eval()
    if vae_model is not None:
        vae_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    # 修复1: 初始化为numpy数组或列表，而不是标量0
    R_precision_real = np.array([0., 0., 0.])
    R_precision = np.array([0., 0., 0.])
    matching_score_real = 0.0
    matching_score_pred = 0.0

    nb_sample = 0

    if local_rank == 0:
        print(f"--> \t Evaluating Diffusion Model...")

    # 如果是分布式评估，设置sampler的epoch
    if distributed and hasattr(val_loader.sampler, 'set_epoch'):
        val_loader.sampler.set_epoch(ep)
    viz_batch_data = None
    for batch in tqdm(val_loader, disable=(local_rank != 0)):
        texts, motions, m_lengths = batch
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        # 1. 获取真实数据的特征
        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs = motions.shape[0]

        # 2. Diffusion 生成
        if vae_model is not None:
            x_in = {
                "text": texts,
                "feature_length": m_lengths // getattr(vae_model, 'downsample_factor', 4)
            }
        else:
            x_in = {
                "text": texts,
                "feature_length": m_lengths
            }

        old_cfg = diffusion_model.cfg_scale if hasattr(diffusion_model, 'cfg_scale') else cfg_scale
        if hasattr(diffusion_model, 'cfg_scale'):
            diffusion_model.cfg_scale = cfg_scale
        gen_output = diffusion_model.generate(x_in, num_denoise_steps=num_denoise_steps)
        pred_latents = gen_output["generated"] if isinstance(gen_output, dict) else gen_output
        if hasattr(diffusion_model, 'cfg_scale'):
            diffusion_model.cfg_scale = old_cfg

        # 3. VAE 解码
        if vae_model is not None:
            if hasattr(vae_model, 'forward_decoder'):
                pred_motions = vae_model.forward_decoder(pred_latents, m_lengths.clone())
            else:
                pred_motions = vae_model.decode(pred_latents, m_lengths.clone())
        else:
            pred_motions = pred_latents

        # 4. 获取生成数据的特征
        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)

        # 收集特征（用于FID和Diversity）
        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred)

        # 5. 计算 Text-Motion Retrieval 相似度
        # 修复2: 直接累加数组，不要转为标量
        # Real
        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True, is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()

        # 确保temp_R是数组并直接累加
        R_precision_real += temp_R  # temp_R 是 [top1, top2, top3]
        matching_score_real += float(temp_match)

        # Generated
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,
                                       is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()

        R_precision += temp_R  # temp_R 是 [top1, top2, top3]
        matching_score_pred += float(temp_match)

        nb_sample += bs
        if local_rank == 0 and viz_batch_data is None:
            viz_batch_data = {
                "pred_motions": pred_motions.detach().cpu(), # 移至 CPU 节省显存
                "texts": texts,
                "m_lengths": m_lengths.detach().cpu()
            }

    # 6. 多卡结果聚合
    if distributed and dist.is_initialized():
        # 收集motion特征用于全局FID计算
        world_size = dist.get_world_size() if world_size == 1 else world_size

        # 收集motion_annotation_list
        local_annotation = torch.cat(motion_annotation_list, dim=0) if motion_annotation_list else torch.empty(0, 768,
                                                                                                               device=device)
        local_pred = torch.cat(motion_pred_list, dim=0) if motion_pred_list else torch.empty(0, 768, device=device)

        # 使用all_gather收集所有特征
        gathered_annotations = [torch.zeros_like(local_annotation) for _ in range(world_size)]
        gathered_preds = [torch.zeros_like(local_pred) for _ in range(world_size)]

        dist.all_gather(gathered_annotations, local_annotation)
        dist.all_gather(gathered_preds, local_pred)

        # 只在rank 0合并
        if local_rank == 0:
            motion_annotation_list = gathered_annotations
            motion_pred_list = gathered_preds
        else:
            motion_annotation_list = []
            motion_pred_list = []

        # 修复3: 聚合标量指标 - R_precision 现在是数组 [3]，需要正确处理
        # 将 R_precision_real 和 R_precision 转为 tensor 进行聚合
        rp_real_tensor = torch.from_numpy(R_precision_real).to(device)
        rp_tensor = torch.from_numpy(R_precision).to(device)

        metrics_tensor = torch.tensor([
            float(matching_score_real),
            float(matching_score_pred),
            float(nb_sample)
        ], device=device, dtype=torch.float64)

        # All-reduce R-precision arrays
        dist.all_reduce(rp_real_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(rp_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        R_precision_real = rp_real_tensor.cpu().numpy()
        R_precision = rp_tensor.cpu().numpy()

        total_samples = int(metrics_tensor[2].item())
        matching_score_real = metrics_tensor[0].item()
        matching_score_pred = metrics_tensor[1].item()
    else:
        total_samples = nb_sample

    # 7. 计算整体指标（只在rank 0执行）
    if not distributed or local_rank == 0:
        if motion_annotation_list:
            motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
            motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
        else:
            # 空数据保护
            motion_annotation_np = np.zeros((1, 768))
            motion_pred_np = np.zeros((1, 768))

        # FID
        gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
        mu, cov = calculate_activation_statistics(motion_pred_np)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        # Diversity
        diversity_real = calculate_diversity(motion_annotation_np, min(300, len(motion_annotation_np)))
        diversity = calculate_diversity(motion_pred_np, min(300, len(motion_pred_np)))

        # 修复4: 平均化检索指标 - R_precision 已经是 [top1, top2, top3] 数组
        R_precision_real_avg = R_precision_real / total_samples if total_samples > 0 else np.array([0., 0., 0.])
        R_precision_avg = R_precision / total_samples if total_samples > 0 else np.array([0., 0., 0.])
        matching_score_real_avg = matching_score_real / total_samples if total_samples > 0 else 0
        matching_score_pred_avg = matching_score_pred / total_samples if total_samples > 0 else 0

        # R_precision_values 直接就是平均后的数组
        R_precision_values = list(R_precision_avg)

        # 8. 日志输出
        msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
              f"R_precision_real. {R_precision_real_avg}, R_precision. {R_precision_values}, " \
              f"matching_score_real. {matching_score_real_avg:.4f}, matching_score_pred. {matching_score_pred_avg:.4f}"
        print(msg)

        if writer is not None:
            writer.add_scalar('Eval/FID', fid, ep)
            writer.add_scalar('Eval/Diversity', diversity, ep)
            writer.add_scalar('Eval/top1', R_precision_values[0], ep)
            writer.add_scalar('Eval/top2', R_precision_values[1], ep)
            writer.add_scalar('Eval/top3', R_precision_values[2], ep)
            writer.add_scalar('Eval/matching_score', matching_score_pred_avg, ep)

        # 9. 最佳模型保存逻辑
        if fid < best_fid:
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            print(msg)
            best_fid = fid
            if save_ckpt:
                torch.save({"model": diffusion_model.state_dict(), "ep": ep},
                           os.path.join(out_dir, 'net_best_fid.tar'))

        if matching_score_pred_avg > best_matching:
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred_avg:.5f} !!!"
            print(msg)
            best_matching = matching_score_pred_avg

        if abs(diversity_real - diversity) < abs(diversity_real - best_div):
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            print(msg)
            best_div = diversity

        if R_precision_values[0] > best_top1:
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision_values[0]:.4f} !!!"
            print(msg)
            best_top1 = R_precision_values[0]
            if save_ckpt:
                torch.save({"model": diffusion_model.state_dict(), "ep": ep},
                           os.path.join(out_dir, 'net_best_top1.tar'))

        if R_precision_values[1] > best_top2:
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision_values[1]:.4f} !!!"
            print(msg)
            best_top2 = R_precision_values[1]

        if R_precision_values[2] > best_top3:
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision_values[2]:.4f} !!!"
            print(msg)
            best_top3 = R_precision_values[2]

        # 10. 可视化保存
        if save_anim and plot_func is not None and 'pred_motions' in locals():
            try:
                print(f"--> \t Generating animation for Epoch {ep}...")

                # 从缓存中恢复数据
                viz_motions = viz_batch_data["pred_motions"].to(device)  # 转回 device 如果 plot_func 需要，或者根据 plot_func 实现决定
                viz_texts = viz_batch_data["texts"]
                viz_lengths = viz_batch_data["m_lengths"]

                bs = viz_motions.shape[0]
                num_samples = min(bs, 4)  # 限制保存数量，例如 4 个

                # 随机或固定选择索引
                rand_idx = torch.arange(num_samples)  # 使用前 N 个，或者用 torch.randint

                data = viz_motions[rand_idx].cpu().detach().numpy()  # 假设 plot_func 需要 numpy
                captions = [viz_texts[k] for k in rand_idx]
                lengths = viz_lengths[rand_idx].numpy()

                save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
                os.makedirs(save_dir, exist_ok=True)

                # 调用绘图函数
                plot_func(data, save_dir, captions, lengths)
                print(f"--> \t Visualizations saved to {save_dir}")

            except Exception as e:
                print(f"Error saving animation: {e}")
                import traceback
                traceback.print_exc()
        elif save_anim and viz_batch_data is None:
            print("Warning: save_anim is True but no batch data was captured (Loader empty?).")
    else:
        # 非rank 0返回当前值（会被广播覆盖）
        fid = best_fid
        diversity = best_div
        R_precision_values = [best_top1, best_top2, best_top3]
        matching_score_pred_avg = best_matching

    # 同步最佳指标给所有进程
    if distributed and dist.is_initialized():
        best_metrics = torch.tensor([
            float(best_fid),
            float(best_div),
            float(best_top1),
            float(best_top2),
            float(best_top3),
            float(best_matching)
        ], device=device)
        dist.broadcast(best_metrics, src=0)
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = best_metrics.cpu().numpy()

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching

@torch.no_grad()
def evaluation_diff_withMM(val_loader, diffusion_model, vae_model, repeat_id, eval_wrapper,
                           num_denoise_steps, cfg_scale, device,
                           temperature=1.0, cal_mm=True, distributed=False, world_size=1, local_rank=0):
    """
    支持 Multimodality 计算的 Diffusion 模型评估函数
    参考 evaluation_momask_plus 的形式，使用 evaluation_diffusion_model 的生成逻辑

    Args:
        val_loader: 验证数据加载器
        diffusion_model: Diffusion 模型（OmniMoEditDiT）
        vae_model: VAE 解码器（HRVAE），可为 None
        repeat_id: 重复实验 ID
        eval_wrapper: 评估器包装器
        num_denoise_steps: 去噪步数
        cfg_scale: CFG 引导强度
        device: 计算设备
        temperature: 采样温度（Diffusion 通常不使用，保留接口兼容性）
        cal_mm: 是否计算 Multimodality
        distributed: 是否分布式评估
        world_size: 总进程数
        local_rank: 当前进程 rank
    """
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    diffusion_model.eval()
    if vae_model is not None:
        vae_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []  # 存储 MM 计算的 features

    # 修复: 初始化为 numpy 数组
    R_precision_real = np.array([0., 0., 0.])
    R_precision = np.array([0., 0., 0.])
    matching_score_real = 0.0
    matching_score_pred = 0.0

    nb_sample = 0

    # 确定计算 MM 的 batch 数量（参考 momask_plus：通常只在前 3 个 batch 计算 MM）
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    # 设置 sampler epoch（分布式时确保数据一致性）
    if distributed and hasattr(val_loader.sampler, 'set_epoch'):
        val_loader.sampler.set_epoch(repeat_id)

    for i, batch in enumerate(tqdm(val_loader, disable=(local_rank != 0))):
        texts, motions, m_lengths = batch

        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        bs = motions.shape[0]

        # 1. 编码真实数据
        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)

        # 2. 准备 Diffusion 输入
        if vae_model is not None:
            downsample_factor = getattr(vae_model, 'downsample_factor', 4)
            if hasattr(vae_model, 'module'):
                downsample_factor = getattr(vae_model.module, 'downsample_factor', 4)
            x_in = {
                "text": texts,
                "feature_length": m_lengths // downsample_factor
            }
        else:
            x_in = {
                "text": texts,
                "feature_length": m_lengths
            }

        # 3. Multimodality 计算：对前 num_mm_batch 个 batch 生成 30 次
        if i < num_mm_batch:
            motion_multimodality_batch = []

            for _ in range(30):  # 生成 30 个不同样本
                # 临时设置不同的随机种子以确保多样性（可选）
                # torch.manual_seed(torch.randint(0, 10000, (1,)).item())

                # Diffusion 生成
                if hasattr(diffusion_model, 'module'):
                    old_cfg = diffusion_model.module.cfg_scale
                    diffusion_model.module.cfg_scale = cfg_scale
                    gen_output = diffusion_model.module.generate(x_in, num_denoise_steps=num_denoise_steps)
                    diffusion_model.module.cfg_scale = old_cfg
                else:
                    old_cfg = diffusion_model.cfg_scale
                    diffusion_model.cfg_scale = cfg_scale
                    gen_output = diffusion_model.generate(x_in, num_denoise_steps=num_denoise_steps)
                    diffusion_model.cfg_scale = old_cfg

                pred_latents = gen_output["generated"] if isinstance(gen_output, dict) else gen_output

                # VAE 解码
                if vae_model is not None:
                    if hasattr(vae_model, 'module'):
                        pred_motions = vae_model.module.forward_decoder(pred_latents, m_lengths.clone())
                    else:
                        pred_motions = vae_model.forward_decoder(pred_latents, m_lengths.clone())
                else:
                    pred_motions = pred_latents

                # 编码预测动作
                fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths,
                                                                     sample_mean=True)
                motion_multimodality_batch.append(fid_em_pred.unsqueeze(1))  # [bs, 1, dim]

            # 合并 30 个样本: [bs, 30, dim]
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1)
            motion_multimodality.append(motion_multimodality_batch)

            # 使用最后一次生成的结果作为该 batch 的主预测结果（用于 FID/R-Precision）
            fid_em_pred_final = fid_em_pred
            em_pred_final = em_pred

        else:
            # 普通生成（单次）
            if hasattr(diffusion_model, 'module'):
                old_cfg = diffusion_model.module.cfg_scale
                diffusion_model.module.cfg_scale = cfg_scale
                gen_output = diffusion_model.module.generate(x_in, num_denoise_steps=num_denoise_steps)
                diffusion_model.module.cfg_scale = old_cfg
            else:
                old_cfg = diffusion_model.cfg_scale
                diffusion_model.cfg_scale = cfg_scale
                gen_output = diffusion_model.generate(x_in, num_denoise_steps=num_denoise_steps)
                diffusion_model.cfg_scale = old_cfg

            pred_latents = gen_output["generated"] if isinstance(gen_output, dict) else gen_output

            # VAE 解码
            if vae_model is not None:
                if hasattr(vae_model, 'module'):
                    pred_motions = vae_model.module.forward_decoder(pred_latents, m_lengths.clone())
                else:
                    pred_motions = vae_model.forward_decoder(pred_latents, m_lengths.clone())
            else:
                pred_motions = pred_latents

            # 编码预测动作
            fid_em_pred_final, em_pred_final, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths,
                                                                             sample_mean=True)

        # 4. 收集特征用于 FID 和 Diversity
        motion_annotation_list.append(fid_em)
        motion_pred_list.append(fid_em_pred_final)

        # 5. 计算 R-Precision 和 Matching Score
        # Real
        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True, is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += float(temp_match)

        # Generated
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred_final.cpu().numpy(), top_k=3, sum_all=True,
                                       is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred_final.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += float(temp_match)

        nb_sample += bs

    # 6. 分布式结果聚合
    if distributed and dist.is_initialized():
        # 收集所有特征
        local_annotation = torch.cat(motion_annotation_list, dim=0) if motion_annotation_list else torch.empty(0, 768,
                                                                                                               device=device)
        local_pred = torch.cat(motion_pred_list, dim=0) if motion_pred_list else torch.empty(0, 768, device=device)

        gathered_annotations = [torch.zeros_like(local_annotation) for _ in range(world_size)]
        gathered_preds = [torch.zeros_like(local_pred) for _ in range(world_size)]

        dist.all_gather(gathered_annotations, local_annotation)
        dist.all_gather(gathered_preds, local_pred)

        if local_rank == 0:
            motion_annotation_list = gathered_annotations
            motion_pred_list = gathered_preds
        else:
            motion_annotation_list = []
            motion_pred_list = []

        # 聚合 MM 特征（如果计算了 MM）
        if cal_mm and motion_multimodality:
            # MM 特征形状: list of [bs, 30, dim]，需要小心处理
            local_mm = torch.cat(motion_multimodality, dim=0)  # [total_mm_samples, 30, dim]
            gathered_mm = [torch.zeros_like(local_mm) for _ in range(world_size)]
            dist.all_gather(gathered_mm, local_mm)
            if local_rank == 0:
                motion_multimodality = gathered_mm
            else:
                motion_multimodality = []

        # 聚合标量指标
        rp_real_tensor = torch.from_numpy(R_precision_real).to(device)
        rp_tensor = torch.from_numpy(R_precision).to(device)

        dist.all_reduce(rp_real_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(rp_tensor, op=dist.ReduceOp.SUM)

        metrics_tensor = torch.tensor([
            float(matching_score_real),
            float(matching_score_pred),
            float(nb_sample)
        ], device=device, dtype=torch.float64)

        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        R_precision_real = rp_real_tensor.cpu().numpy()
        R_precision = rp_tensor.cpu().numpy()
        matching_score_real = metrics_tensor[0].item()
        matching_score_pred = metrics_tensor[1].item()
        total_samples = int(metrics_tensor[2].item())
    else:
        total_samples = nb_sample

    # 7. 计算最终指标（只在主进程或单卡时计算）
    if not distributed or local_rank == 0:
        # 合并特征
        if motion_annotation_list:
            motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
            motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
        else:
            motion_annotation_np = np.zeros((1, 768))
            motion_pred_np = np.zeros((1, 768))

        # 计算 Multimodality
        if cal_mm and motion_multimodality:
            # 合并所有 MM 样本: [total_samples, 30, dim]
            mm_tensor = torch.cat(motion_multimodality, dim=0)
            motion_multimodality_np = mm_tensor.cpu().numpy()
            multimodality = calculate_multimodality(motion_multimodality_np, 10)  # 每 10 个样本计算一次平均
        else:
            multimodality = 0.0

        # FID
        gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
        mu, cov = calculate_activation_statistics(motion_pred_np)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        # Diversity
        diversity_real = calculate_diversity(motion_annotation_np, 300 if total_samples > 300 else 100)
        diversity = calculate_diversity(motion_pred_np, 300 if total_samples > 300 else 100)

        # R-Precision 和 Matching（平均）
        R_precision_real_avg = R_precision_real / total_samples if total_samples > 0 else np.array([0., 0., 0.])
        R_precision_avg = R_precision / total_samples if total_samples > 0 else np.array([0., 0., 0.])
        matching_score_real_avg = matching_score_real / total_samples if total_samples > 0 else 0
        matching_score_pred_avg = matching_score_pred / total_samples if total_samples > 0 else 0

        # 8. 打印结果
        msg = (
            f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, "
            f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
            f"R_precision_real. [{R_precision_real_avg[0]:.4f}, {R_precision_real_avg[1]:.4f}, {R_precision_real_avg[2]:.4f}], "
            f"R_precision. [{R_precision_avg[0]:.4f}, {R_precision_avg[1]:.4f}, {R_precision_avg[2]:.4f}], "
            f"matching_score_real. {matching_score_real_avg:.4f}, matching_score_pred. {matching_score_pred_avg:.4f}, "
            f"multimodality. {multimodality:.4f}"
        )
        print(msg)

        return (
            fid,
            diversity,
            R_precision_avg,  # 返回数组 [top1, top2, top3]
            matching_score_pred_avg,
            multimodality
        )
    else:
        # 非主进程返回占位符（会被忽略）
        return 0.0, 0.0, np.array([0., 0., 0.]), 0.0, 0.0


@torch.no_grad()
def evaluation_edit_selfcond(out_dir, val_loader, edit_model, vae_model, writer, ep,
                             best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
                             eval_wrapper, device, plot_func=None, num_denoise_steps=10,
                             save_ckpt=True, save_anim=False,
                             distributed=False, world_size=1, local_rank=0):
    """
    自条件重建评估函数：
    - 输入：masked source (从target构造)
    - 输出：reconstructed target
    - 评估：与原始target的FID、Diversity等指标

    与evaluation_diffusion_model类似，但适配双流编辑模型的自条件模式
    """
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    edit_model.eval()
    if vae_model is not None:
        vae_model.eval()

    motion_annotation_list = []  # 原始动作特征
    motion_recon_list = []  # 重建动作特征

    R_precision_real = np.array([0., 0., 0.])
    R_precision_recon = np.array([0., 0., 0.])
    matching_score_real = 0.0
    matching_score_recon = 0.0

    nb_sample = 0

    if local_rank == 0:
        print(f"--> \t Evaluating Self-Condition Reconstruction...")

    if distributed and hasattr(val_loader.sampler, 'set_epoch'):
        val_loader.sampler.set_epoch(ep)

    for batch in tqdm(val_loader, disable=(local_rank != 0)):
        texts, motions, m_lengths = batch

        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        # 1. 编码原始动作特征（用于对比）
        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs = motions.shape[0]

        # 2. 准备自条件输入（模拟训练时的mask逻辑）
        if vae_model is not None:
            with torch.no_grad():
                # 编码到latent空间
                latents = vae_model.encode(motions, m_lengths)
                latent_lens = m_lengths // getattr(vae_model, 'downsample_factor', 4)

                # 构造masked source（与训练时一致）
                if hasattr(edit_model, 'random_mask_source'):
                    source_masked, _ = edit_model.random_mask_source(latents, latent_lens)
                else:
                    # 备用：简单的random mask
                    source_masked = latents.clone()
                    for b in range(bs):
                        valid_len = latent_lens[b].item()
                        mask_len = int(valid_len * 0.5)  # 默认mask 50%
                        if valid_len > mask_len:
                            start = torch.randint(0, valid_len - mask_len, (1,)).item()
                            source_masked[b, start:start + mask_len] = 0.0
        else:
            # 如果没有VAE，直接在动作空间mask（较少见）
            source_masked = motions.clone()
            latent_lens = m_lengths
            for b in range(bs):
                valid_len = m_lengths[b].item()
                mask_len = int(valid_len * 0.5)
                if valid_len > mask_len:
                    start = torch.randint(0, valid_len - mask_len, (1,)).item()
                    source_masked[b, start:start + mask_len] = 0.0
            latents = motions

        # 3. 自条件重建生成
        x_in = {
            "source": source_masked,
            "length": latent_lens,
            "feature_length": latent_lens,
        }

        # 使用generate方法进行重建（mode="self_condition"）
        gen_output = edit_model.generate(
            x_in,
            num_denoise_steps=num_denoise_steps,
            cfg_scale=1.0,  # 自条件通常不需要CFG或CFG=1
            mode="self_condition"
        )

        recon_latents = gen_output["generated"] if isinstance(gen_output, dict) else gen_output

        # 4. VAE解码回动作空间
        if vae_model is not None:
            if hasattr(vae_model, 'forward_decoder'):
                recon_motions = vae_model.forward_decoder(recon_latents, m_lengths.clone())
            else:
                recon_motions = vae_model.decode(recon_latents, m_lengths.clone())
        else:
            recon_motions = recon_latents

        # 5. 编码重建动作
        fid_em_recon, em_recon, _ = eval_wrapper.encode_motion(recon_motions[..., :148], m_lengths, sample_mean=True)

        # 收集特征
        motion_annotation_list.append(fid_em)
        motion_recon_list.append(fid_em_recon)

        # 6. 计算R-precision和Matching Score（评估重建质量）
        # 原始 vs 文本
        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True, is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += float(temp_match)

        # 重建 vs 文本（看重建是否保留语义）
        temp_R = calculate_R_precision(et.cpu().numpy(), em_recon.cpu().numpy(), top_k=3, sum_all=True,
                                       is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_recon.cpu().numpy()).trace()
        R_precision_recon += temp_R
        matching_score_recon += float(temp_match)

        nb_sample += bs

    # 7. 分布式聚合
    if distributed and dist.is_initialized():
        # 收集特征
        local_annotation = torch.cat(motion_annotation_list, dim=0) if motion_annotation_list else torch.empty(0, 768,
                                                                                                               device=device)
        local_recon = torch.cat(motion_recon_list, dim=0) if motion_recon_list else torch.empty(0, 768, device=device)

        gathered_annotations = [torch.zeros_like(local_annotation) for _ in range(world_size)]
        gathered_recons = [torch.zeros_like(local_recon) for _ in range(world_size)]

        dist.all_gather(gathered_annotations, local_annotation)
        dist.all_gather(gathered_recons, local_recon)

        if local_rank == 0:
            motion_annotation_list = gathered_annotations
            motion_recon_list = gathered_recons
        else:
            motion_annotation_list = []
            motion_recon_list = []

        # 聚合指标
        rp_real_tensor = torch.from_numpy(R_precision_real).to(device)
        rp_recon_tensor = torch.from_numpy(R_precision_recon).to(device)

        dist.all_reduce(rp_real_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(rp_recon_tensor, op=dist.ReduceOp.SUM)

        metrics_tensor = torch.tensor([
            float(matching_score_real),
            float(matching_score_recon),
            float(nb_sample)
        ], device=device, dtype=torch.float64)

        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        R_precision_real = rp_real_tensor.cpu().numpy()
        R_precision_recon = rp_recon_tensor.cpu().numpy()
        matching_score_real = metrics_tensor[0].item()
        matching_score_recon = metrics_tensor[1].item()
        total_samples = int(metrics_tensor[2].item())
    else:
        total_samples = nb_sample

    # 8. 计算最终指标（只在主进程）
    if not distributed or local_rank == 0:
        if motion_annotation_list:
            motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
            motion_recon_np = torch.cat(motion_recon_list, dim=0).cpu().numpy()
        else:
            motion_annotation_np = np.zeros((1, 768))
            motion_recon_np = np.zeros((1, 768))

        # FID（原始 vs 重建，衡量重建质量，越低越好）
        gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
        recon_mu, recon_cov = calculate_activation_statistics(motion_recon_np)
        fid = calculate_frechet_distance(gt_mu, gt_cov, recon_mu, recon_cov)

        # Diversity
        diversity_real = calculate_diversity(motion_annotation_np, min(300, len(motion_annotation_np)))
        diversity_recon = calculate_diversity(motion_recon_np, min(300, len(motion_recon_np)))

        # R-Precision和Matching（平均）
        R_precision_real_avg = R_precision_real / total_samples if total_samples > 0 else np.array([0., 0., 0.])
        R_precision_recon_avg = R_precision_recon / total_samples if total_samples > 0 else np.array([0., 0., 0.])
        matching_score_real_avg = matching_score_real / total_samples if total_samples > 0 else 0
        matching_score_recon_avg = matching_score_recon / total_samples if total_samples > 0 else 0

        R_precision_values = list(R_precision_recon_avg)

        # 输出日志（与原框架风格一致）
        msg = (f"--> \t Eva. Ep {ep} (SelfCond) :, FID. {fid:.4f}, "
               f"Diversity Real. {diversity_real:.4f}, Diversity Recon. {diversity_recon:.4f}, "
               f"R_precision_real. [{R_precision_real_avg[0]:.4f}, {R_precision_real_avg[1]:.4f}, {R_precision_real_avg[2]:.4f}], "
               f"R_precision_recon. [{R_precision_recon_avg[0]:.4f}, {R_precision_recon_avg[1]:.4f}, {R_precision_recon_avg[2]:.4f}], "
               f"matching_score_real. {matching_score_real_avg:.4f}, matching_score_recon. {matching_score_recon_avg:.4f}")
        print(msg)

        if writer is not None:
            writer.add_scalar('Eval_SelfCond/FID', fid, ep)
            writer.add_scalar('Eval_SelfCond/Diversity_Real', diversity_real, ep)
            writer.add_scalar('Eval_SelfCond/Diversity_Recon', diversity_recon, ep)
            writer.add_scalar('Eval_SelfCond/top1', R_precision_values[0], ep)
            writer.add_scalar('Eval_SelfCond/top2', R_precision_values[1], ep)
            writer.add_scalar('Eval_SelfCond/top3', R_precision_values[2], ep)
            writer.add_scalar('Eval_SelfCond/matching_score', matching_score_recon_avg, ep)

        # 保存最佳模型（基于FID）
        if fid < best_fid:
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            print(msg)
            best_fid = fid
            if save_ckpt:
                torch.save({"model": edit_model.state_dict(), "ep": ep},
                           os.path.join(out_dir, 'net_best_fid_selfcond.tar'))

        # 其他指标更新（参照原逻辑）
        if abs(diversity_real - diversity_recon) < abs(diversity_real - best_div):
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity_recon:.5f} !!!"
            print(msg)
            best_div = diversity_recon

        if R_precision_values[0] > best_top1:
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision_values[0]:.4f} !!!"
            print(msg)
            best_top1 = R_precision_values[0]

        if R_precision_values[1] > best_top2:
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision_values[1]:.4f} !!!"
            print(msg)
            best_top2 = R_precision_values[1]

        if R_precision_values[2] > best_top3:
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision_values[2]:.4f} !!!"
            print(msg)
            best_top3 = R_precision_values[2]

        if matching_score_recon_avg > best_matching:
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_recon_avg:.5f} !!!"
            print(msg)
            best_matching = matching_score_recon_avg

        # 可视化（可选）
        if save_anim and plot_func is not None:
            try:
                save_dir = os.path.join(out_dir, 'animation_selfcond', 'E%04d' % ep)
                os.makedirs(save_dir, exist_ok=True)
                # 这里可以添加特定的可视化逻辑
            except Exception as e:
                print(f"Error saving selfcond animation: {e}")

    else:
        fid = best_fid
        diversity_recon = best_div
        R_precision_values = [best_top1, best_top2, best_top3]
        matching_score_recon_avg = best_matching

    # 同步最佳指标
    if distributed and dist.is_initialized():
        best_metrics = torch.tensor([
            float(best_fid), float(best_div), float(best_top1),
            float(best_top2), float(best_top3), float(best_matching)
        ], device=device)
        dist.broadcast(best_metrics, src=0)
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching = best_metrics.cpu().numpy()

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching
