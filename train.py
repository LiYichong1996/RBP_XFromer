import numpy as np
import wandb
import time
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
import os
import psutil
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts

from utils.data import LengthAwareSampler, RBPDataSet, custom_collate_fn, get_pos_weight
from utils.loss import MultiLabelCustomLoss, combined_loss, AdaptiveAuxiliaryLoss
from utils.loss import multilabel_mcc
from enformer_pytorch import EnformerConfig, str_to_one_hot
from model.rbp_xformer import RBPXFormer


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 
device_id = [2,3] #[2,3] #

data_root = './data/'
PAD_IDX = 0
batch_size = 32
pg_root = os.getcwd()




def compute_label_union(labels):
    """
    在n维度上对标签进行或运算（并集操作）
    
    参数:
        labels: torch.Tensor, 形状为(b,n,166)的标签张量，值为0或1
        
    返回:
        torch.Tensor, 形状为(b,166)的张量，表示在n维度上的并集结果
    """
    # # 确保输入是二值张量
    # assert torch.max(labels, dim=1), "输入标签必须是0或1"
    
    # 方法1: 使用max操作
    # 在n维度上取最大值，相当于或运算
    union_labels = torch.max(labels, dim=1)[0]
    
    
    return union_labels


def calculate_rbp_frequency(y, sample_total_counts):
    """
    计算每个RBP的真实频率
    
    Parameters:
    y: shape (n_samples, n_rbps) 的结合位点数组
    sample_total_counts: shape (n_samples,) 每个样本自己的总计数
    
    Returns:
    freqs: 每个RBP的频率
    """
    # 对于每个RBP：所有样本的结合位点总和 / 所有样本的总计数之和
    rbp_freqs = y.sum(axis=0) / sample_total_counts.sum()
    return rbp_freqs




def main(args):
    batch_size = args.batch_size
    device_id = [args.device_id]
    data_dir = args.data_dir
    rbp_dir = args.rbp_dir
    p_dict_dir = args.p_dict_dir
    save_root = args.save_root
    
    config = EnformerConfig.from_json_file("./utils/config_rbp.json")
    # save_root = data_root + 'rbp_data_train_2/'
    # data_dir = data_root + '/train_test_dataset_1w_85_POSTAR20.pkl'
    # rbp_dir = data_root + '/rbp_list.pkl'
    # p_dict_dir = data_root + '/p_{}_dict_320.pkl'.format(config.esm_data)
    
    
    
    with open(data_dir, 'rb') as f:
        seq_list_train, label_list_train, seq_list_test, label_list_test = pickle.load(f)

    feature_list_train = []
    for seq in seq_list_train:
        feature = str_to_one_hot(seq)
        feature_list_train.append(feature)

    feature_list_test = []
    for seq in seq_list_test:
        feature = str_to_one_hot(seq)
        feature_list_test.append(feature)

    aim_rbp_indices = [i for i in range(config.data_version)]


    config.output_heads['rbp'] = config.data_version
    config.num_label = len(aim_rbp_indices)
    config.aim_rbp_indices = aim_rbp_indices

    dataset_t = RBPDataSet(feature_list_train, label_list_train, rbp_dir, p_dict_dir, aim_rbp_indices)
    train_indices = list(range(len(seq_list_train)))
    train_sampler = LengthAwareSampler(dataset_t, train_indices, batch_size)
    t_loader = DataLoader(dataset_t, batch_size=batch_size, sampler=train_sampler, collate_fn=custom_collate_fn)

    dataset_s = RBPDataSet(feature_list_test, label_list_test, rbp_dir, p_dict_dir, aim_rbp_indices)
    test_indices = list(range(len(seq_list_test)))
    test_sampler = LengthAwareSampler(dataset_s, test_indices, batch_size, shuffle=False)
    s_loader = DataLoader(dataset_s, batch_size=batch_size, sampler=test_sampler, collate_fn=custom_collate_fn)
 
    local_time = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    

    device = torch.device("cuda:{}".format(device_id[0]) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = RBPXFormer(config, enformer_param_dir=None)
    model = model.to(device)
    # model = torch.nn.parallel.DataParallel(model, output_device=device, device_ids=device_id)

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    save_root += (local_time + '/')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # 给RBP也加上全局权重
    label_list_train_p = [
        torch.tensor(np.max(label, axis=0)).view(1,-1) for label in label_list_train
    ]
    labels = torch.cat(label_list_train_p, dim=0)
    label_cnt = torch.sum(labels, dim=0)
    l = labels.shape[0]
    label_frequencies = label_cnt / l
    avg_freq = torch.mean(label_frequencies)
    label_frequencies = avg_freq / label_frequencies
    pos_weight_p = torch.clamp(label_frequencies, min=0.1, max=100)
    pos_weight = get_pos_weight(data_loader=t_loader, threshold=100, lower_bound=0.1).to(device)

    logits2probs = torch.nn.Sigmoid()

    criterion_p = MultiLabelCustomLoss(pos_weights=pos_weight_p, gamma=1.5, alpha=0.9, smooth=1e-6)
    criterion = MultiLabelCustomLoss(pos_weights=pos_weight, gamma=1.5, alpha=0.9, smooth=1e-6)

    evaluate_criterion = BCEWithLogitsLoss()
    

    # 获取enformer模块和其他模块的参数
    enformer_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'enformer' in name or 'cell_line_emb_layer' in name: #or 'cross_attn' in name:
            enformer_params.append(param)
        else:
            other_params.append(param)
    
    # 为不同模块设置不同的学习率
    optim = torch.optim.AdamW([
        {'params': enformer_params, 'lr': 3e-6, 'weight_decay': 0.01},  # enformer模块使用小学习率
        {'params': other_params, 'lr': 5e-6, 'weight_decay': 0.01}      # 其他模块使用大学习率
    ])

    warmup_scheduler = LinearLR(
        optim, 
        start_factor=0.01,  # 初始学习率为设定值的1%
        end_factor=1.0, 
        total_iters=500
    )

    main_scheduler = CosineAnnealingWarmRestarts(
        optim,
        T_0=20,           # 重启周期（epoch数）
        T_mult=1,         # 周期倍增因子
        eta_min=1e-6      # 最小学习率
    )

    from torch.optim.lr_scheduler import SequentialLR
    scheduler = SequentialLR(
        optim,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[500]  # 500步后切换至余弦退火
    )
    

    max_epoch = 999

    adaptive_loss = AdaptiveAuxiliaryLoss(initial_weight=0.01, patience=5).to(device)
    
    for epo in range(1, max_epoch):
        loss_train = 0
        bce_loss_sum = 0
        bce_loss_sum_p = 0
        model.train()
        with tqdm(total=len(t_loader)//10, desc='training epoch {}...'.format(epo), unit='it') as pbar:
            cnt = 1
            for features, labels, p_cls_reps, cell_line_indices_tensor, name_list in t_loader:
                # labels = compute_label_union(labels)
                features = features.to(device)
                labels = labels.to(torch.float32).to(device)
                mask = labels != -1

                labels_p = compute_label_union(labels)
                labels_p = labels_p.to(torch.float32).to(device)
                mask_p = labels_p != -1

                p_cls_reps = p_cls_reps.to(device)
                cell_line_indices_tensor = cell_line_indices_tensor.to(device) 
                
                logits, logits_p = model(
                    features,
                    esm2_reps=p_cls_reps, 
                    cell_line_tensor=cell_line_indices_tensor,
                )
                
                loss, bce_loss, _ = combined_loss(logits, labels, mask, criterion)
                loss_p, bce_loss_p, _ = combined_loss(logits_p, labels_p, mask_p, criterion_p)
                                               #, focal=focal, undersample=undersample, dice=dice)
                
                # loss = loss + loss_p
                if config.loss_type == 'rna':
                    loss = loss
                elif config.loss_type == 'add':
                    loss = loss + loss_p
                else:
                    loss, current_weight = adaptive_loss(loss, loss_p)

                loss.backward()

                optim.step()
                scheduler.step()
                optim.zero_grad()

                loss_train += loss.detach().item() * features.shape[0] / len(t_loader) 
                bce_loss_sum += bce_loss * features.shape[0] / len(t_loader)
                bce_loss_sum_p += bce_loss_p * features.shape[0] / len(t_loader)
                # break
                cnt += 1

                if cnt % 10 == 0:
                    memo_result = psutil.virtual_memory()
                    pbar.set_postfix({'used':memo_result.used, 'total': memo_result.total, 'ratio': memo_result.used/memo_result.total})
                    pbar.update(1)
   
        y_true_list = []
        y_pred_list = []
        loss_eval = 0

        y_true_list_p = []
        y_pred_list_p = []
        loss_eval_p = 0
        
        model.eval()
        with tqdm(total=len(s_loader)//10, desc='validating...', unit='it') as pbar:
            cnt = 1
            for features, labels, p_cls_reps, cell_line_indices_tensor, name_list in s_loader:
                with torch.no_grad():
                    labels_p = compute_label_union(labels)
                    labels_p = labels_p.to(torch.float32).to(device)
                    mask_p = labels_p != -1

                    p_cls_reps = p_cls_reps.to(device)
                    cell_line_indices_tensor = cell_line_indices_tensor.to(device) 
                    # labels = compute_label_union(labels)
                    
                    features = features.to(device)
                    labels = labels.to(torch.float32).to(device)  
                    logits, logits_p = model(
                        features, 
                        # x_mask=x_mask, 
                        esm2_reps=p_cls_reps, 
                        cell_line_tensor=cell_line_indices_tensor,
                    )          
                    preds = logits2probs(logits)
                    preds_p = logits2probs(logits_p)
                y_true = labels.cpu()
                y_pred = preds.cpu()

                logits = logits.cpu()

                n = y_true.shape[-1]
                mask = labels != -1
                mask_f = mask.view(-1, n)
                indices = torch.where(torch.any(mask_f, dim=1))[0].cpu()

                y_true_f = y_true.view(-1, n)
                y_pred_f = y_pred.view(-1, n)
                logits_f = logits.view(-1, n)
                y_true_masked = y_true_f[indices, :]
                y_pred_masked = y_pred_f[indices, :]
                logits_masked = logits_f[indices, :]
                loss = evaluate_criterion(logits_masked, y_true_masked)
                loss_eval += loss.detach().item() * features.shape[0]
                y_true_list.append(y_true_masked)
                y_pred_list.append(y_pred_masked)

                y_true_p = labels_p.cpu()
                y_pred_p = preds_p.cpu()

                logits_p = logits_p.cpu()

                n = y_true_p.shape[-1]
                mask_p = mask_p.view(-1, n)
                indices_p = torch.where(torch.any(mask_p, dim=1))[0].cpu()

                y_true_f_p = y_true_p.view(-1, n)
                y_pred_f_p = y_pred_p.view(-1, n)
                logits_f_p = logits_p.view(-1, n)
                y_true_masked_p = y_true_f_p[indices_p, :]
                y_pred_masked_p = y_pred_f_p[indices_p, :]
                logits_masked_p = logits_f_p[indices_p, :]
                loss_p = evaluate_criterion(logits_masked_p, y_true_masked_p)
                loss_eval_p += loss_p.detach().item() * features.shape[0]
                y_true_list_p.append(y_true_masked_p)
                y_pred_list_p.append(y_pred_masked_p)

                cnt += 1
                if cnt % 10 == 0:
                    pbar.update(1)

        y_true_f = torch.cat(y_true_list,dim=0)
        y_pred_f = torch.cat(y_pred_list,dim=0)
        best_threshold = 0.5 #, best_f1 = find_best_threshold(y_true_f, y_pred_f)
        y_pl_f = (y_pred_f >= best_threshold).float()
        auc_label = roc_auc_score(y_true_f, y_pred_f, multi_class='ovo', average='macro')
        pr_label = average_precision_score(y_true_f, y_pred_f, average='macro')
        f1_label = f1_score(y_true_f, y_pl_f, pos_label=1, average='macro')
        mcc_label = multilabel_mcc(y_true_f, y_pl_f, average='macro')

        y_true_f_p = torch.cat(y_true_list_p,dim=0)
        y_pred_f_p = torch.cat(y_pred_list_p,dim=0)
        y_pl_f_p = (y_pred_f_p >= best_threshold).float()
        auc_label_p = roc_auc_score(y_true_f_p, y_pred_f_p, multi_class='ovo', average='macro')
        pr_label_p = average_precision_score(y_true_f_p, y_pred_f_p, average='macro')
        f1_label_p = f1_score(y_true_f_p, y_pl_f_p, pos_label=1, average='macro')
        mcc_label_p = multilabel_mcc(y_true_f_p, y_pl_f_p, average='macro')

        current_weight = adaptive_loss.update_weight(1-pr_label, 1-pr_label_p)


        log_dict = {
            'focal_loss_train': loss_train,
            'bce_loss_train': bce_loss_sum,
            'bce_loss_train_p': bce_loss_sum_p,
            # 'auc_sample': auc_sample,
            'auc_label': auc_label,
            # 'pr_sample': pr_sample,
            'pr_label': pr_label,
            # 'f1_sample': f1_sample,
            'f1_label': f1_label,
            # 'mcc_sample': mcc_sample,
            'mcc_label': mcc_label,
            'eval loss': loss_eval,
            'auc_label_p': auc_label_p,
            # 'pr_sample': pr_sample,
            'pr_label_p': pr_label_p,
            # 'f1_sample': f1_sample,
            'f1_label_p': f1_label_p,
            # 'mcc_sample': mcc_sample,
            'mcc_label_p': mcc_label_p,
            'eval loss_p': loss_eval_p,
        }

        save_dir = save_root + '{}.pt'.format(epo)

        torch.save(model.state_dict(), save_dir)

        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default='./data/train_test_dataset_1w_85_POSTAR20.pkl')
    parser.add_argument("--rbp_dir", type=str, default='./data/rbp_list.pkl')
    parser.add_argument("--p_dict_dir", type=str, default='./data/p_cls_dict_320.pkl')
    parser.add_argument("--save_root", type=str, default='./data/logs/')

    args = parser.parse_args()
    main(args)