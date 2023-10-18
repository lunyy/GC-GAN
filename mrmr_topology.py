import os
import torch
import numpy as np
import pandas as pd
from config import config
from load_data import load_data_list, load_graph_data
from mrmr import mrmr_classif

args = config()

def flatten_fc(data):
    # print("subject :", data.shape[1])  # 3810
    if isinstance(data, torch.Tensor):
        data = data.unsqueeze(0) if int(data.ndim) == 2 else data
    else:
        data = np.expand_dims(data, axis=0) if int(data.ndim) == 2 else data
    x, y = np.triu_indices(data.shape[1], k=1)
    # print(x, y)
    FC_flatten = data[:, x, y]
    # print(FC_flatten.shape)
    return FC_flatten

def flatten2dense(flatten, ROI):
    if isinstance(flatten, torch.Tensor):
        flatten = flatten.unsqueeze(0) if int(flatten.ndim) == 1 else flatten
    else:
        flatten = np.expand_dims(flatten, axis=0) if int(flatten.ndim) == 1 else flatten
    B, F = flatten.shape
    x, y = np.triu_indices(ROI, k=1)
    if isinstance(flatten, torch.Tensor):
        sym = torch.zeros([B, ROI, ROI], device=flatten.device)
        sym[:, x, y] = flatten
        sym = sym + torch.transpose(sym, -1, -2)
    else:
        sym = np.zeros((B, ROI, ROI))
        sym[:, x, y] = flatten
        sym = sym + np.transpose(sym, (0, 2, 1))
    return sym


def mrmr(train_data, train_label, K=620, fold = None, timestamp = None, pseudo_epoch =1):
    train_data = flatten_fc(train_data)
    X = pd.DataFrame(train_data)
    y = pd.Series(train_label)
    selected_features_index = mrmr_classif(X=X, y=y, K=K)
    mask = np.zeros((6216))
    mask[selected_features_index] = 1

    topology = np.ones((112, 112))
    r, c = np.triu_indices(112, 1)
    topology[r, c] = mask
    topology[c, r] = mask

    print("fold {} | Edge count:".format(fold), topology.sum())
    if args.ex_type == "Main" :
        if not (os.path.isdir('GAN/Edge_topology/model{}/'.format(timestamp))):
            os.makedirs(os.path.join('GAN/Edge_topology/model{}/'.format(timestamp)))
        np.save("GAN/Edge_topology/model{}/mrmr_MDD_Harvard_FC_map_fold_{}_{}.npy".format(timestamp, fold, pseudo_epoch), topology)
    elif args.ex_type == "Single":
        if not (os.path.isdir('GAN/Edge_topology/model{}/'.format(timestamp))):
            os.makedirs(os.path.join('GAN/Edge_topology/model{}/'.format(timestamp)))
        np.save("GAN/Edge_topology/model{}/mrmr_S20_MDD_Harvard_FC_map_fold_{}_{}.npy".format(timestamp, fold, pseudo_epoch), topology)
