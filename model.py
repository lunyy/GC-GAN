'''
# ## Code resources
1. https://github.com/rusty1s/pytorch_geometric/blob/master/examples/autoencoder.py
2. https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html
3. https://github.com/anonymous1025/Deep-Graph-Translation-
'''

import torch
import torch.nn as nn
import math
from graph_utils import laplacian_norm_adj, add_self_loop_cheb
from config import config
from seed import set_seed
from functional import mish

args = config()
set_seed()
# ChebyGCN written by younghanSon

class myChebConv(torch.nn.Module):
    """
    simple GCN layer
    Z = f(X, A) = softmax(A` * ReLU(A` * X * W0)* W1)
    A` = D'^(-0.5) * A * D'^(-0.5)
    """
    def __init__(self, in_channels, out_channels, K=2, bias=True):
        # input
        super(myChebConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        # BN operations
        stdv = 1. / math.sqrt(self.weight.shape[1])
        # for weight in self.weight:
        # nn.init.kaiming_uniform_(weight)
        # nn.init.xavier_uniform_(weight)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            # self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x, adj_orig):
        # A` * X * W
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = laplacian_norm_adj(adj_orig)
        adj = add_self_loop_cheb(-adj)
        Tx_0 = x  # T0(x) = 1, 여기서는 k=1시작
        Tx_1 = x  # Dummy.
        out = torch.matmul(Tx_0, self.weight[0])
        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.shape[0] > 1:
            Tx_1 = torch.matmul(adj, x)
            out = out + torch.matmul(Tx_1, self.weight[1])
        for k in range(2, self.weight.shape[0]):
            Tx_2 = torch.matmul(adj, Tx_1)
            Tx_2 = 2. * Tx_2 - Tx_0  # Chebyshef polynomial
            out = out + torch.matmul(Tx_1, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        if self.bias is not None:
            out += self.bias
        return out
    def __repr__(self):
        # print layer's structure
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

class myGraphAE(torch.nn.Module): # pretraining GAE
    def __init__(self):
        super(myGraphAE, self).__init__()
        self.econv1 = myChebConv(in_channels=args.e_channels[0]+1, out_channels=args.e_channels[1], K=2)
        self.econv2 = myChebConv(in_channels=args.e_channels[1], out_channels=args.e_channels[2], K=2)
        self.gconv1 = myChebConv(in_channels=args.g_channels[0]+1, out_channels=args.g_channels[1], K=2)
        self.gconv2 = myChebConv(in_channels=args.g_channels[1], out_channels=args.g_channels[2], K=2)
        self.linear = nn.Linear(in_features=112 * args.g_channels[2], out_features=112 * 112)
        # self.fc1 = nn.Linear(in_features=112 * args.e_channels[2], out_features= 112 * args.e_channels[2])
        # self.fc2 = nn.Linear(in_features=112 * args.e_channels[2], out_features=112 * args.e_channels[2])
        self.act = mish
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(2, 112)  # label embedding
        self.diag_mask = torch.eye(112, 112, dtype=torch.bool)  # set diagonal to one

    # def reparametrize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + std * eps


    def forward(self, x, edge_index, label = None):
        label = self.embedding(label)
        label = label.reshape((-1, 112, 1))
        x = torch.cat([x, label], dim=2)
        x = self.act(self.econv1(x, edge_index))
        x = self.act(self.econv2(x, edge_index))
        # x = x.reshape((-1, 112 * args.e_channels[2]))
        # mu = self.act(self.fc1(x))
        # logvar = self.act(self.fc2(x))
        # z = self.reparametrize(mu, logvar)
        # x = z.reshape((-1,112, args.e_channels[2]))
        x = torch.cat([x, label], dim=2)
        x = self.act(self.gconv1(x, edge_index))
        x = self.act(self.gconv2(x, edge_index))
        x = x.reshape((-1, 112 * args.g_channels[2]))
        x = self.tanh(self.linear(x))
        x = x.reshape((-1, 112, 112))

        x = (x + torch.transpose(x, 1, 2)) / 2.0  # make symmetric matrix
        x = torch.arctanh(0.999 * x)

        for s in range(x.size(0)):  # set diagonal to one
            x[s][self.diag_mask] = 1.0

        x = x.reshape((-1, 112, 112))  # reshape [len(train_files)*112, 112] -> fake FC matrix 생성

        return x


class Generator(torch.nn.Module): # triple gan generator
    def __init__(self):
        super(Generator, self).__init__()
        self.gconv1 = myChebConv(in_channels=args.g_channels[0]+1, out_channels=args.g_channels[1], K=2)
        self.gconv2 = myChebConv(in_channels=args.g_channels[1], out_channels=args.g_channels[2], K=2)
        # self.pre = nn.Linear(in_features=112 * (args.g_channels[0]), out_features=112 * (args.g_channels[0]))
        self.li1 = nn.Linear(in_features=112 * args.g_channels[2], out_features=112 * 112)
        self.act = mish
        self.tanh = nn.Tanh()
        self.diag_mask = torch.eye(112,112, dtype=torch.bool) # set diagonal to one
        self.embedding = nn.Embedding(2,112) #label embedding

    def forward(self, x, edge_index, label):
        # x = x.reshape(-1, 112 * (args.g_channels[0]))
        # x = self.act(self.pre(x))
        # x = x.reshape(-1, 112, (args.g_channels[0]))
        label = self.embedding(label)
        label = label.reshape((-1, 112, 1))
        x = torch.cat([x,label],dim=2)
        x = self.act(self.gconv1(x, edge_index))
        x = self.act(self.gconv2(x, edge_index))

        x = x.reshape((-1, 112 * args.g_channels[2]))
        x = self.tanh(self.li1(x))
        x = x.reshape((-1, 112, 112))

        x = (x + torch.transpose(x, 1, 2)) / 2.0  # make symmetric matrix
        x = torch.arctanh(0.999 * x)

        for s in range(x.size(0)):  # set diagonal to one
            x[s][self.diag_mask] = 1.0

        x = x.reshape((-1, 112, 112))  # reshape [len(train_files)*112, 112] -> fake FC matrix 생성

        return x

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cconv1 = myChebConv(in_channels=args.d_channels[0], out_channels=args.d_channels[1], K=2)
        self.cconv2 = myChebConv(in_channels=args.d_channels[1], out_channels=args.d_channels[2], K=2)
        self.li1 = nn.Linear(in_features=112 * (args.d_channels[2]), out_features=112 * 64)
        self.li2 = nn.Linear(in_features=112 * 64, out_features=112 * 32)
        self.cla = nn.Linear(in_features=112 * 32, out_features=1)
        self.act = mish
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding(2, 112 * 224)  # label embedding

    def forward(self, x, edge_index):
        # label = self.embedding(label)
        x = self.act(self.cconv1(x, edge_index))
        x = self.act(self.cconv2(x, edge_index))
        x = x.reshape((-1, 112 * args.d_channels[2]))
        # x = torch.mul(x,label)
        x = self.act(self.li1(x))
        x = self.act(self.li2(x))
        features = x
        predict = self.sigmoid(self.cla(x))

        return predict, features

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.cconv1 = myChebConv(in_channels=args.c_channels[0], out_channels=args.c_channels[1],K=2)
        self.cconv2 = myChebConv(in_channels=args.c_channels[1], out_channels=args.c_channels[2],K=2)
        self.lin1 = nn.Linear(in_features=112 * args.c_channels[2], out_features=112 * 64)
        self.lin2 = nn.Linear(in_features=112 * 64 , out_features=112 * 32)
        self.bn1 = nn.BatchNorm1d(num_features=112 * 64)
        self.bn2 = nn.BatchNorm1d(num_features=112 * 32)
        self.final = nn.Linear(in_features=112 * 32, out_features=2)
        self.act = mish

    def forward(self, x, edge_index):

        x = self.act(self.cconv1(x, edge_index))
        x = self.act(self.cconv2(x, edge_index))
        x = x.reshape((-1, 112 * args.c_channels[2]))
        x = self.act(self.bn1(self.lin1(x)))
        features = x
        x = self.act(self.bn2(self.lin2(x)))
        logits = self.final(x)

        return logits, features

class ACDiscriminator(torch.nn.Module):
    def __init__(self):
        super(ACDiscriminator,self).__init__()
        self.cconv1 = myChebConv(in_channels=args.d_channels[0], out_channels=args.d_channels[1],K=2)
        self.cconv2 = myChebConv(in_channels=args.d_channels[1], out_channels=args.d_channels[2],K=2)
        self.li1 = nn.Linear(in_features=112 * (args.d_channels[2]), out_features= 112 * 64)
        self.li2 = nn.Linear(in_features=112 * 64, out_features=112 * 32)
        self.cla = nn.Linear(in_features=112 * 32, out_features=4)
        self.act = mish

    def forward(self, x, edge_index):

        x = self.act(self.cconv1(x, edge_index))
        x = self.act(self.cconv2(x, edge_index))
        x = x.reshape((-1, 112 * args.d_channels[2]))
        x = self.act(self.li1(x))
        x = self.act(self.li2(x))
        features = x
        logits = self.cla(x)

        return  logits,  features
