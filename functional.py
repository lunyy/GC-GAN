import math
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
EPS = 1e-15
MAX_LOGSTD = 10

from torch.nn.parameter import Parameter


# https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/functional.py
'''
Script provides functional interface for Mish activation function.
'''
# import pytorch
@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))

'''
Applies the mish function element-wise:
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
'''
# import pytorch
from torch import nn
class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        m = Mish()
        input = torch.randn(2)
        output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

def reset_normal_param(L, stdv, weight_scale = 1.):
    assert type(L) == torch.nn.Linear
    torch.nn.init.normal(L.weight, std=weight_scale / math.sqrt(L.weight.size()[0]))


def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
    r"""Given latent variables :obj:`z`, computes the binary cross
    entropy loss for positive edges :obj:`pos_edge_index` and negative
    sampled edges.

    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (LongTensor): The positive edges to train against.
        neg_edge_index (LongTensor, optional): The negative edges to train
            against. If not given, uses negative sampling to calculate
            negative edges. (default: :obj:`None`)
    """

    pos_loss = -torch.log(
        self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    # Do not include self-loops in negative samples
    pos_edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index, _ = add_self_loops(pos_edge_index)
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

    return pos_loss + neg_loss