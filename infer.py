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
from Bio import SeqIO
import math

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

def main(args):
    batch_size = args.batch_size
    device_id = [args.device_id]
    out_root = args.out_root
    model_dir = args.model_dir
    rna_dir = args.rna_dir
    
    config = EnformerConfig.from_json_file("./utils/config_rbp.json")
    # save_root = data_root + 'rbp_data_train_2/'
    # data_dir = data_root + '/train_test_dataset_1w_85_POSTAR20.pkl'
    rbp_dir = data_root + '/rbp_list.pkl'
    p_dict_dir = data_root + '/p_{}_dict_320.pkl'.format(config.esm_data)
    
    
    seq_list = []
    name_list = []
    label_list = []
    for record in SeqIO.parse(rna_dir, "fasta"):
        seq_list.append(str(record.seq))
        name_list.append(record.id)
        label_list.append(np.zeros([math.ceil(len(record.seq)/128), 166]))

    feature_list = []
    for seq in seq_list:
        feature = str_to_one_hot(seq)
        feature_list.append(feature)

    aim_rbp_indices = [i for i in range(config.data_version)]


    config.output_heads['rbp'] = config.data_version
    config.num_label = len(aim_rbp_indices)
    config.aim_rbp_indices = aim_rbp_indices

    dataset_t = RBPDataSet(feature_list, label_list, rbp_dir, p_dict_dir, aim_rbp_indices)
    t_loader = DataLoader(dataset_t, batch_size=batch_size, collate_fn=custom_collate_fn)
 
    local_time = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    

    device = torch.device("cuda:{}".format(device_id[0]) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = RBPXFormer(config, enformer_param_dir=None)
 
    checkpoint = torch.load(model_dir, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    checkpoint_keys = set(state_dict.keys())

    # 找出匹配的键
    matched_keys = model_keys & checkpoint_keys
    missing_keys = model_keys - checkpoint_keys

    # 加载参数

    new_state_dict = {k: state_dict[k] for k in matched_keys}
    model.load_state_dict(new_state_dict, strict=False)
    
    model = model.to(device)
    # model = torch.nn.parallel.DataParallel(model, output_device=device, device_ids=device_id)

    save_root = out_root + (local_time + '/')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    logits2probs = torch.nn.Sigmoid()
    model.eval()
    with tqdm(total=len(t_loader), desc='validating...', unit='it') as pbar:
        for features, labels, p_cls_reps, cell_line_indices_tensor, name_list in t_loader:
            with torch.no_grad():
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
                
            y_pred = preds.cpu()

            for i in range(len(name_list)):
                save_dir = save_root + '/' + name_list[i] + '.pkl'
                with open(save_dir, 'wb') as f:
                    pickle.dump(y_pred[i,:math.ceil(len(seq_list[i])/128)], f)
            pbar.update(1)

    

        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default='./params/rbp_xformer.pt')
    parser.add_argument("--out_root", type=str, default='./results/')
    parser.add_argument("--rna_dir", type=str, default='./example/rna_seq.fasta')

    args = parser.parse_args()
    main(args)