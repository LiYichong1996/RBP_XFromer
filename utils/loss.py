import numpy as np
from sklearn.metrics import precision_recall_curve, matthews_corrcoef
import torch
from torch import nn


def find_best_threshold(y_true, y_scores):
    """
    通过PR-AUC找到多标签分类中一个统一的最佳阈值。

    参数:
    y_true (ndarray): 真实标签的数组，形状为 (num_samples, num_labels)。
    y_scores (ndarray): 模型预测的分数数组，形状为 (num_samples, num_labels)。

    返回:
    best_threshold (float): 最佳的统一阈值。
    best_f1 (float): 在最佳阈值下的平均F1 Score。
    """
    num_labels = y_true.shape[1]
    best_thresholds = []
    best_f1_scores = []

    for i in range(num_labels):
        y_true_label = y_true[:, i]
        y_scores_label = y_scores[:, i]

        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true_label, y_scores_label)

        # 检查并处理 NaN 和 Inf 值
        if np.any(np.isnan(precision)) or np.any(np.isnan(recall)):
            print(f"NaN values found in precision or recall for label {i}")
        
        if np.any(np.isinf(precision)) or np.any(np.isinf(recall)):
            print(f"Inf values found in precision or recall for label {i}")

        # 处理 NaN 和 Inf 值
        precision = np.nan_to_num(precision, nan=0.0, posinf=0.0, neginf=0.0)
        recall = np.nan_to_num(recall, nan=0.0, posinf=0.0, neginf=0.0)

        # 过滤掉 precision 和 recall 都为零的情况
        valid_indices = (precision + recall) > 0
        valid_precision = precision[valid_indices]
        valid_recall = recall[valid_indices]
        valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds 比 precision 和 recall 少一个元素

        # 计算F1 Score
        f1_scores = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall)

        # 找到F1 Score最大的阈值
        best_index = np.argmax(f1_scores)
        best_thresholds.append(valid_thresholds[best_index])
        best_f1_scores.append(f1_scores[best_index])

    # 选择全局最佳阈值（例如使用中位数）
    best_threshold = np.median(best_thresholds)
    best_f1 = np.mean(best_f1_scores)

    return best_threshold, best_f1


def f1_loss(y_pred, y_true):
    epsilon = 1e-7 

    y_pred = torch.sigmoid(y_pred)

    tp = (y_true * y_pred).sum(dim=0)
    predicted_positives = y_pred.sum(dim=0)
    actual_positives = y_true.sum(dim=0)

    precision = tp / (predicted_positives + epsilon)
    recall = tp / (actual_positives + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    f1_loss = 1 - f1.mean()  

    return f1_loss, 0


class MultiLabelCustomLoss(nn.Module):
    def __init__(self, pos_weights, gamma=2.0, alpha=0.25, smooth=1e-6):
        super(MultiLabelCustomLoss, self).__init__()
        self.pos_weights = pos_weights
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, targets, focal=True):

        probas = torch.sigmoid(logits)
        if self.pos_weights is not None:
            pos_weights = self.pos_weights.to(targets.device)
            bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)(logits, targets)
        else:
            pos_weights = self.pos_weights
            bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        bce_loss_out = bce_loss.mean().item()
        
        if focal:
            pt = torch.where(targets == 1, probas, 1 - probas)
            focal_weight = self.alpha * (1 - pt) ** self.gamma
            
            focal_loss = focal_weight * bce_loss

            loss = focal_loss.mean()

            loss = bce_loss.mean()
        
        return loss, bce_loss_out, 0


def mask_reshape(mask):
    N = mask.shape[-1]
    B = mask.shape[0]
    L = mask.shape[1]

    mask_f = mask.reshape(-1, N)
    select_indices = torch.where(torch.any(mask_f, dim=1))[0]

    return select_indices 


def balance_index(y_true, n_p_ratio=1):
    pos_index = torch.where(torch.any(y_true == 1, dim=1))[0]
    neg_index = torch.where(torch.all(y_true == 0, dim=1))[0]
    n_pos = pos_index.shape[0]
    n_neg = neg_index.shape[0]
    # print(n_pos, n_neg)
    n_neg_remain = int(n_p_ratio * n_pos)
    indices = torch.randperm(n_neg)[:n_neg_remain]
    neg_index = neg_index[indices]

    b_index = torch.cat([pos_index, neg_index], dim=0)
    return b_index

def combined_loss(y_pred, y_true, mask, criterion, undersample=False):
    n = mask.shape[-1]
    y_true_f = y_true.view(-1, n)
    y_pred_f = y_pred.reshape(-1, n)
    select_indices = mask_reshape(mask)
    
    y_true_masked = y_true_f[select_indices, :]
    y_pred_masked = y_pred_f[select_indices, :]

    if undersample:
        b_index = balance_index(y_true_masked, n_p_ratio=1)
        y_true_masked = y_true_masked[b_index]
        y_pred_masked = y_pred_masked[b_index]
    # loss_f1 = f1_loss(y_true, y_pred)
    # loss_poisson = poisson_nll_loss(y_true_masked, y_pred_masked)
    # loss_multinomial = cross_entropy(y_true_masked, y_pred_masked)
    # loss_func = BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss_func = BCEWithLogitsLoss()
    # loss = criterion(y_true_masked_f, y_pred_masked_f)
    # loss, bce_loss = criterion(y_pred_masked, y_true_masked, sample_balance=sample_balance, label_balance=label_balance)
    loss, bce_loss, pr_loss = criterion(y_pred_masked, y_true_masked)#, focal=focal, dice=dice)
    # alpha, beta, gamma = 0.3, 0.3, 0.4

    # 结合损失
    # total_loss = alpha * loss_f1 + beta * loss_poisson + gamma * loss_multinomial
    return loss, bce_loss, pr_loss #total_loss


class AdaptiveAuxiliaryLoss(nn.Module):
    def __init__(self, initial_weight=0.1, patience=5, min_weight=0.01, max_weight=0.5):
        super().__init__()
        self.aux_weight = nn.Parameter(torch.tensor(initial_weight))
        self.patience = patience
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # 跟踪指标
        self.best_main_metric = float('inf')
        self.patience_counter = 0
        self.main_metric_history = []
        self.prev_main_metric = None
        self.moving_average = 0
        self.beta = 0.9  # 用于计算移动平均
        
    def update_weight(self, main_metric, aux_metric=None):
        """
        在epoch结束时更新权重
        main_metric: 主任务的验证指标
        aux_metric: 辅助任务的验证指标（可选）
        """
        # 更新移动平均
        self.moving_average = self.beta * self.moving_average + (1 - self.beta) * main_metric
        
        # 保存历史记录
        self.main_metric_history.append(main_metric)
        if len(self.main_metric_history) > 5:  # 保持最近5个epoch的历史
            self.main_metric_history.pop(0)
        
        # 计算相对变化（使用移动平均减少噪声）
        if self.prev_main_metric is not None:
            relative_change = (self.moving_average - self.prev_main_metric) / self.prev_main_metric
            
            # 主任务性能显著变差
            if relative_change > 0.02:  # 2%的阈值
                self.aux_weight.data *= 0.85
                print(f"Main task degraded by {relative_change:.2%}, reducing aux weight to {self.aux_weight.item():.4f}")
            
            # 主任务性能改善
            elif relative_change < -0.01:  # 1%的阈值
                if self.aux_weight.item() < self.max_weight:
                    self.aux_weight.data *= 1.05
                    print(f"Main task improved by {-relative_change:.2%}, increasing aux weight to {self.aux_weight.item():.4f}")
        
        # 更新最佳指标和耐心计数器
        if main_metric < self.best_main_metric:
            self.best_main_metric = main_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.aux_weight.data *= 0.9
                self.patience_counter = 0
                print(f"No improvement for {self.patience} epochs, reducing aux weight to {self.aux_weight.item():.4f}")
        
        # 确保权重在合理范围内
        self.aux_weight.data = torch.clamp(self.aux_weight.data, self.min_weight, self.max_weight)
        self.prev_main_metric = self.moving_average
        
        return self.aux_weight.item()

    def forward(self, main_loss, aux_loss):
        """
        在训练过程中使用当前权重计算加权损失
        """
        return main_loss + self.aux_weight * aux_loss, self.aux_weight.item()
    

def multilabel_mcc(y_true, y_pred, average='macro'):
    """
    计算多标签分类的MCC。

    参数:
    y_true -- 真实标签，二维numpy数组，形状为 (n_samples, n_labels)
    y_pred -- 预测标签，二维numpy数组，形状为 (n_samples, n_labels)

    返回:
    mcc -- 多标签分类的MCC
    """
    
    mcc_sum = 0

    if average == 'samples':
        n_samples = y_true.shape[0]
        for i in range(n_samples):
            mcc_sum += matthews_corrcoef(y_true[i, :], y_pred[i, :])
        mcc = mcc_sum / n_samples
    else:
        n_labels = y_true.shape[1]
        for i in range(n_labels):
            mcc_sum += matthews_corrcoef(y_true[:, i], y_pred[:, i])
        mcc = mcc_sum / n_labels
    return mcc