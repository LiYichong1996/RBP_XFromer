import torch
from torch.nn.parallel import DataParallel
import pandas as pd 
import gc
import math
import torch.utils
from tqdm import tqdm
from enformer_pytorch import EnformerConfig, str_to_one_hot
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import BCEWithLogitsLoss, MultiLabelSoftMarginLoss
import wandb
import time
from torch.nn.utils.rnn import pad_sequence
import os
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import psutil
import math
from sklearn import metrics
import random
import psutil
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchvision.ops import sigmoid_focal_loss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from torch import nn



class RBPDataSet(Dataset):
    def __init__(self, feature_list, label_list, rbp_dir, p_dict_dir, aim_rbp_indices=None, name_list=None) -> None:
        if aim_rbp_indices is None:
            aim_rbp_indices = [i for i in range(label_list[0].shape[1])]
        self.feature_list = feature_list
        self.label_list = [label[:,aim_rbp_indices] for label in label_list]
        self.name_list = name_list
        if name_list is None:
            self.name_list = [f'{i}' for i in range(len(feature_list))]
        with open(rbp_dir, 'rb') as f:
            rbp_list = pickle.load(f)

        with open(p_dict_dir, 'rb') as f:
            p_cls_dict = pickle.load(f)

        p_cls_list = []
        for p_c in rbp_list:
            p_name = p_c.split('_')[0]
            p_cls_list.append(p_cls_dict[p_name])

        self.p_cls_reps = torch.stack(p_cls_list, dim=0)[aim_rbp_indices,:]

        c_list = [rbp.split('_')[-1] for rbp in rbp_list]
        c_list = sorted(list(set(c_list)))

        cell_line_indices = []
        for p_c in rbp_list:
            cl = p_c.split('_')[1]
            cell_line_indices.append(c_list.index(cl))
        self.cell_line_indices_tensor = torch.tensor(cell_line_indices)[aim_rbp_indices]

        self.p_cls_reps_list = [self.p_cls_reps for _ in range(len(self.feature_list))]
        self.cell_line_indices_tensor_list = [self.cell_line_indices_tensor for _ in range(len(self.feature_list))]
    
    def __len__(self):
        return len(self.feature_list)
    
    def __getitem__(self, index):
       label = self.label_list[index]
       feature = self.feature_list[index]
       p_cls_reps = self.p_cls_reps_list[index]
       cell_line_indices_tensor = self.cell_line_indices_tensor_list[index]
       name = self.name_list[index]
       return feature, label, p_cls_reps, cell_line_indices_tensor, name
    

def get_pos_weight(data_loader=None, label_list=None,threshold=10, lower_bound=1, ):

    if data_loader is not None:
        label_list = []
        for i in data_loader.batch_sampler.sampler.indices:
            label_list.append(
                torch.tensor(data_loader.dataset.label_list[i])
            )
    labels = torch.cat(label_list, dim=0)
    label_cnt = torch.sum(labels, dim=0)

    l = labels.shape[0]
    label_frequencies = label_cnt / l

    avg_freq = torch.mean(label_frequencies)

    label_frequencies = avg_freq / label_frequencies

    pos_weight = torch.clamp(label_frequencies, min=lower_bound, max=threshold)

    return pos_weight


class LengthAwareSampler(Sampler):
    def __init__(self, data_source, indices, batch_size, shuffle=True):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 仅统计传入indices对应的数据长度
        self.lengths = [len(data_source[idx][0]) for idx in indices]
        
        # 对传入的indices进行排序（根据长度）
        self.sorted_indices = [x for _, x in sorted(zip(self.lengths, indices), key=lambda pair: pair[0])]
    
    def __iter__(self):
        # 按照排序后的（或打乱后的）索引生成每个batch
        batches = [
            self.sorted_indices[i:i + self.batch_size] 
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        
        # 随机打乱这些batch的顺序（可选）
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            # 只打乱批次的顺序，保持批次内部的长度相近
            random.shuffle(batches)
        
        # 将打乱后的batch展平成一个索引序列
        return iter([idx for batch in batches for idx in batch])
    
    def __len__(self):
        return len(self.indices)
    

def custom_collate_fn(batch):
    features, labels, p_cls_reps_list, cell_line_indices_tensor_list, name_list = zip(*batch)  # 解压特征和标签
    p_cls_reps = torch.stack(p_cls_reps_list, dim=0)
    cell_line_indices_tensor = torch.stack(cell_line_indices_tensor_list, dim=0)
    # 对特征进行填充
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels = [torch.tensor(l) for l in labels]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    return features_padded, labels_padded, p_cls_reps, cell_line_indices_tensor, name_list