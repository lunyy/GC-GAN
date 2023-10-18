
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from torch_geometric.utils import dense_to_sparse
from copy import copy


def make_node_features_ones(data, data_type="MDD_Data"):
    if data_type == "MDD_Data":
        node_list = np.ones((len(data), 112, 25))  # Harvard
    elif data_type == "ADNI":
        node_list = np.ones(len(data), 116, 116)  # AAL

    return node_list


def edge_connection_from_pvalue(data, num, p_value, data_type="", GCN_type="ChebGCN"):
    """
    :param data: graph data(FC map)
    :param num: fold num
    :param p_value: p_value
    :param data_type: ADNI or MDD_Data
    :param GCN_type: ChebGCN or GAT(binary)
    :return:
    """
    # if label != 0 and label != 1:
    #     raise ValueError

    # if data_type == "ADNI":
    #     t = np.load("C:/Users/MAILAB/PycharmProjects/ADNI&MDD_ttest_feature_selection/ttest_pvalue_FCmap/t-test_FC_map_fold_{num}.npy".format(num=num))

    if data_type == "MDD_Data":
        t = np.load("MDD_Data/MDD_topology/t-test_MDD_FC_map_fold_{num}_20210525175638.npy".format(num=num))

    # voxels = np.array(np.where(t >= (1 - p_value))).T
    pixels = np.array(np.where(t <= p_value)).T  # 왜 transpose를 취하지? (8672, 2) -> 어떻게 matrix 형태로 바꿀까..

    for i in range(len(pixels)):
        # print("t:",t[voxels[i][0], voxels[i][1], voxels[i][2]])
        if t[pixels[i][0], pixels[i][1]] > p_value:
            print(t[pixels[i][0], pixels[i][1]])  # ex) [6 29 24]
        elif t[pixels[i][0], pixels[i][1]] < p_value:
            pass

    # data : [381, 112, 112]
    subj_edge_list = []
    for subj in range(len(data)):
        edge_init = np.zeros_like(t)
        for i in range(len(pixels)):
            if GCN_type == "ChebGCN":
                # edge_init[pixels[i][0], pixels[i][1]] = data[subj][pixels[i][0], pixels[i][1]]  # FCmap 값으로 edge weight 추가
                edge_init[pixels[i][0], pixels[i][1]] = 1.  # no weight
            elif GCN_type == "GAT":
                edge_init[pixels[i][0], pixels[i][1]] = 1.   # edge weight없음  20210526012533 -> p value 0.5
        subj_edge_list.append(edge_init)

    return np.array(subj_edge_list)


def make_zero_normalization_node_feature(data, data_type):
    new_input = copy(data)  #  얕은 복사인가..? 같이 바뀌는거 같음
    if data_type == "dynamic":
        for t in range(len(data)):  # 10
            for i in range(len(data[0])):  # 112 roi
                signal_mean = sum(data[t][i]) / len(data[t][i])
                single_roi = data[t][i]
                for j in range(len(single_roi)):
                    new_input[t][i][j] = single_roi[j] - signal_mean

                # print("signal_zero_mean: ", sum(new_input[t][i]))  # 0, -2.220446049250313e-15

    elif data_type == "static":
        for i in range(len(data)):
            signal_mean = sum(data[i]) / len(data[i])
            # print("signal_mean: ", signal_mean)
            single_roi = data[i]
            for j in range(len(single_roi)):
                new_input[i][j] = single_roi[j] - signal_mean

            # print("signal_zero_mean: ", sum(new_input[i]))  # 0

    elif data_type == "time_series":
        pass

    return new_input


def edge_normalzation_add_loop_dynamic(edge_matrix, add_loop=True):
    # [subj, time, row, col]
    subj_block = []
    # subject
    for subj in range(len(edge_matrix)):
        time_block = []
        # time
        for t in range(len(edge_matrix[1])):
            # print(edge_matrix[subj][t].size())
            # node
            N, _ = edge_matrix[subj][t].size()
            if add_loop:
                adj = edge_matrix[subj][t].clone()
                idx = torch.arange(N, dtype=torch.long, device=edge_matrix[subj][t].device)
                adj[idx, idx] = 1

            deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
            adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)  # D^-(1/2)*A*D^-(1/2)

            time_block.append(adj.unsqueeze(0))
        time_block = torch.cat(time_block, dim=0)

        subj_block.append(time_block.unsqueeze(0))
    subj_block = torch.cat(subj_block, dim=0)  # type : torch.Tensor

    return subj_block

def edge_normalzation_add_loop_static(edge_matrix, add_loop=True):
    # [subj, time, row, col]
    subj_block = []
    # subject
    for subj in range(len(edge_matrix)):
        # print(edge_matrix[subj][t].size())
        # node
        N, _ = edge_matrix[subj].size()
        if add_loop:
            adj = edge_matrix[subj].clone()
            idx = torch.arange(N, dtype=torch.long, device=edge_matrix[subj].device)
            adj[idx, idx] = 1

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)  # deg_inv_sqrt은 diagonal matrix가 아닌거 같음 확인 필요 2021.5.18
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)  # D^-(1/2)*A*D^-(1/2)

        subj_block.append(adj.unsqueeze(0))
    subj_block = torch.cat(subj_block, dim=0)  # type : torch.Tensor

    return subj_block

def add_self_loop(adj, fill_value: float = 1.):
    """
    add self loop to matrix A
    :param adj: (N,V,V) matrix
    :param fill_value: value of diagonal
    :return: self loop added matrix A
    """
    num_node = adj.shape[1]
    x = torch.arange(0, num_node)
    adj[:, x, x] = fill_value
    return adj

def make_edge_index_attr(tensor):
    assert tensor.dim() == 3   # [subj, nodes, node_features]
    subj_edge_index = []
    subj_edge_attr = []

    for subj in range(len(tensor)):
        edge_index, edge_attr = dense_to_sparse(tensor[subj])  # index -> edge_index,  value -> edge_attr

        subj_edge_index.append(edge_index.unsqueeze(0))
        subj_edge_attr.append(edge_attr.unsqueeze(0))

    EdgeIndex = torch.cat(subj_edge_index, dim=0)
    EdgeAttr = torch.cat(subj_edge_attr, dim=0)

    return EdgeIndex, EdgeAttr

def make_dynamic_signal(mat, roi="Harvard_Oxford"):
    if roi == "AAL":
        mat = mat[:, :116]
    elif roi == "Harvard_Oxford":
        mat = mat[:, 116:228]

    sliding_window = 50
    stride = 20
    time_block = []
    end_point = len(mat) // stride - 1  # 230 // 20 - 1 = 10
    # print(end_point) # 10
    mat_transpose = np.array(mat).transpose()
    # print(mat_transpose)
    # print(type(mat_transpose))  # <class 'list'>
    for i in range(end_point): # i : 0~9
        time_block_per_roi = []

        for j in range(len(mat_transpose)):  # 1~112
            time_block_per_roi.append(mat_transpose[j, i*stride:sliding_window+(i*stride)])
            # print(type(time_block_per_roi)) # <class 'list'>
        time_block.append(time_block_per_roi)
    # print(np.array(time_block).shape)  # (10, 112, 50)
    # print(type(time_block)) # <class 'list'>
    return time_block

def make_static_signal(mat, roi="Harvard_Oxford"):

    if roi == "AAL":
        mat = mat[:, :116]
    elif roi == "Harvard_Oxford":
        mat = mat[:, 116:228]
    mat_transpose = np.array(mat).transpose()

    return mat_transpose

# 2021.5.13 gru data
def make_time_series(mat, roi="Harvard_Oxford"):
    # 추가 확장 필요 roi
    if roi == "AAL":
        mat = mat[:, :116]
    elif roi == "Harvard_Oxford":
        mat = mat[:, 116:228]

    return mat


# [subject, row, col]
def flatten_fc(data):
    print("subject :", data.shape[1])  # 3810
    x, y = np.triu_indices(data.shape[1], k=1)
    # print(x, y)
    FC_flatten = data[:, x, y]
    # print(FC_flatten.shape)
    return FC_flatten

def make_cosine_edge_static(data):
    subj_edge_matrix = []
    for subj in range(len(data)):
        cos_adj = cosine_similarity(data[subj], data[subj])
        subj_edge_matrix.append(cos_adj)

    return subj_edge_matrix

def make_cosine_edge_dynamic(data):
    timeblock = []
    for subj in range(len(data)):
        subj_edge_matrix = []
        for t in range(10):
            cos_adj = cosine_similarity(data[subj][t], data[subj][t])
            subj_edge_matrix.append(cos_adj)
        timeblock.append(subj_edge_matrix)

    return timeblock

# statc FC edge, 2021.5.12 보류
def make_cosine_edge_static_FC(data):
    # similarity에 similarity?
    pass


def make_rbf_edge(feature, gender, protocol=None):
    rbf_adj = rbf_kernel(feature, feature)  # similarity 를 어떤 방식으로 사용할 것인지
    np.fill_diagonal(rbf_adj, 0)
    # 약 829개의 gender list 에서 829 개 각자 성별, 디바이스가 같은 지를 판별하기 위해 np.repeat 로 axis=1,0 829x829 로 만들어서 비교
    sex_same_1 = np.repeat(gender.reshape(-1, 1), gender.shape[0], axis=1)
    sex_same_2 = np.repeat(gender.reshape(1, -1), gender.shape[0], axis=0)
    # protocol_same_1 = np.repeat(protocol.reshape(-1, 1), protocol.shape[0], axis=1)
    # protocol_same_2 = np.repeat(protocol.reshape(1, -1), protocol.shape[0], axis=0)

    sex_same = np.array(sex_same_1, copy=True)  # same 을 담을 그릇
    # protocol_same = np.array(protocol_same_1, copy=True)

    sex_same[sex_same_2 == sex_same_1] = 2
    sex_same[sex_same_2 != sex_same_1] = 0
    # protocol_same[protocol_same_2 == protocol_same_1] = 1
    # protocol_same[protocol_same_2 != protocol_same_1] = 0
    # weight_edge = sex_same.astype(np.float) + protocol_same.astype(np.float)
    weight_edge = sex_same.astype(np.float)

    # edge = rbf simil * (sex_same + device_same)

    edge_adj = rbf_adj * weight_edge

    return edge_adj

def laplacian_norm_adj(A):
    """
    return norm adj matrix
    A` = D'^(-0.5) * A * D'^(-0.5)
    :param A: (N, V, V)
    :return: norm matrix
    """
    A = remove_self_loop(A)
    adj = graph_norm(A)
    # adj = torch.where(torch.isnan(adj), torch.zeros_like(adj), adj)
    return adj


def remove_self_loop(adj):
    """
    remove self loop
    :param adj: (N, V, V)
    :return: (N, V, V)
    """
    num_node = adj.shape[1]
    if isinstance(adj, torch.Tensor):
        x = torch.arange(0, num_node)
    else:
        x = np.arange(0, num_node)
    adj = adj.unsqueeze(0) if int(adj.ndim) == 2 else adj
    adj[:, x, x] = 0
    return adj


def graph_norm(A):
    D = A.sum(dim=-1).clamp(min=1).pow(-0.5)
    adj = D.unsqueeze(-1) * A * D.unsqueeze(-2)  # 여기서 broadcasting으로 대각행렬처럼 곱해지게 됨
    return adj


def add_self_loop_cheb(adj, fill_value: float = 1.):
    """
    add self loop to matrix A
    :param adj: (N,V,V) matrix
    :param fill_value: value of diagonal
    :return: self loop added matrix A
    """
    num_node = adj.shape[1]
    if isinstance(adj, torch.Tensor):
        x = torch.arange(0, num_node)
    else:
        x = np.arange(0, num_node)
    adj[:, x, x] = fill_value
    return adj

def make_kNN_graph_edge(adj, k=20):
    """
    :param adj: adjacency matrix (N, V, V)
    :param k: k-NN per roi
    :return:
    """
    normal_adj = []
    for subj in range(len(adj)):
        new_a = torch.zeros(len(adj[0]), len(adj[0]))
        for row in range(len(adj[0])):
            a = adj[subj][row]
            topid = sorted(range(len(a)), key=lambda i: a[i])[-k:]  # 상위 20개 index

            for index in topid:
                if new_a[index][row] == 0:  # symmetric, 중복 방지
                    new_a[row][index] = 1.

        normal_adj.append(new_a.unsqueeze(0))

    norm_adj = torch.cat(normal_adj, dim=0)

    return norm_adj
