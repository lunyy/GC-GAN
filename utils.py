import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if type(m) == nn.Conv3d or type(m) == nn.Conv1d or type(m) == nn.Linear:
        try:
            torch.nn.init.xavier_uniform_(m.weight)
        except:
            pass

def weights_init_normal(m):
    if type(m) == nn.Linear:
        try:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass
    elif type(m) == nn.BatchNorm1d :
        try:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant(m.bias.dasta, 0)
        except:
            pass

def accuracy(out, label):
    out = np.array(out)
    label = np.array(label)
    total = out.shape[0]
    correct = (out == label).sum().item() / total
    return correct

def sensitivity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label == 1.)
    sens = np.sum(out[mask]) / np.sum(mask)
    return sens

def specificity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label <= 1e-5)
    total = np.sum(mask)
    spec = (total - np.sum(out[mask])) / total
    return spec

def correlation_distance(x):
    corr_mat = torch.zeros(112,112)
    for row in range(len(x[0])):
        for col in range(len(x[0])):
            u = x[row,:]
            u_bar = torch.mean(u)
            v = x[col,:]
            v_bar = torch.mean(v)
            corr_mat[row][col] =  1 - (torch.dot((u-u_bar),(v-v_bar)) / (torch.norm(u-u_bar)*torch.norm(v-v_bar)))
    return corr_mat
