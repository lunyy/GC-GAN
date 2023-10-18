import os
import numpy as np
from config import config

args = config()

print(os.path.abspath((__file__)))

# harvard, aal zero value
nan_fc_subject_list = ['ROISignals_S20-1-0251.mat', 'ROISignals_S20-2-0095.mat']

def load_data_list(fold, data_type):

    nan_idx = []

    if args.ex_type == "Main":
        print("Main!")
        if data_type == "train":
            data_source = "Data_txt_list/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_trainvalid_data_list_fold_{}.txt".format(fold)
        elif data_type == "test":
            data_source = "Data_txt_list/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_test_data_list_fold_{}.txt".format(fold)
        with open(data_source, 'r') as f:
            data_filenames = np.array([line.strip() for line in f.readlines()])

    elif args.ex_type == "Single":
        print("Single!")
        if data_type == "train":
            data_source = "Data_txt_list/S20_MSGAN_MDD_Harvard_list.txt"
        elif data_type == "test":
            if args.site == "S1":
                data_source = "Data_txt_list/S1_MSGAN_MDD_Harvard_list.txt"
            elif args.site == "S21":
                data_source = "Data_txt_list/S21_MSGAN_MDD_Harvard_list.txt"
            else:
                data_source = "Data_txt_list/MDD_Harvard_site20/20210331151631_MDD_site20_group/MDD_test_data_list_fold_{}.txt".format(fold)
        with open(data_source, 'r') as f:
            data_filenames = np.array([line.strip() for line in f.readlines()])


    for i, nan_sub in enumerate(data_filenames):
        if nan_sub in nan_fc_subject_list: nan_idx.append(i)

    data_filenames = np.delete(data_filenames, nan_idx)

    print(data_filenames.shape[0])
    return data_filenames


# load DNN data
def load_dnn_real_data(fold, data_filenames, timestamp):
    data_path = 'MDD_Data/MS_MDD_Harvard_FC_fisher'
    fc_data = []
    fc_label = []

    fc = np.load(os.path.join(data_path, 'ROISignals_S21-1-0002.npy'))
    fc[fc == fc[0]] = 0  # change inf to zero for upper half
    mask = (fc == 0)  # get a mask

    for subject in data_filenames:  # len(data_filenames) = 381
        fc_matrix = np.load(os.path.join(data_path, subject))
        fc_matrix = np.triu(fc_matrix).flatten()
        fc_matrix = np.delete(fc_matrix, mask)

        if subject.split('-')[1] == "1":
            label = 1  # real mdd
        else:
            label = 0 # real nc

        fc_data.append(fc_matrix)
        fc_label.append(label)

    [fc_data,  fc_label] = [np.array(fc_data), np.array(fc_label)]

    return [fc_data,  fc_label]

def load_dnn_fake_data(fold, data_filenames,fixed_noise, timestamp):
    fc_data = []
    fc_label = []

    for z,subject in enumerate(data_filenames):
        fc_matrix = fixed_noise[z]
        if subject.split('-')[1] == "1":
            label = 3 # fake mdd
        else:
            label = 2 # fake nc

        fc_data.append(fc_matrix)
        fc_label.append(label)

    [fc_data,  fc_label] = [np.array(fc_data), np.array(fc_label)]

    return [fc_data,  fc_label]

def load_cnn_real_data(data_filenames):
    data_path = 'MDD_Data/MS_MDD_Harvard_FC_fisher'

    fc_data = []
    fc_label = []

    for subject in data_filenames:
        node_feature = np.load(os.path.join(data_path, subject))
        if subject.split('-')[1] == "1" :
            label = 1 # real mdd
        else:
            label = 0 # real nc

        fc_data.append(node_feature)
        fc_label.append(label)

    [fc_data, fc_label] = [np.array(fc_data),np.array(fc_label)]

    fc_data = np.expand_dims(fc_data, axis=1)  # 112x112 -> 1x112x112

    return [fc_data,fc_label]

def load_cnn_fake_data(data_filenames, fixed_noise):
    fc_data = []
    fc_label = []

    for z, subject in enumerate(data_filenames):
        node_feature = fixed_noise[z]
        if subject.split('-')[1] == "1":
            label = 1  # real mdd
        else:
            label = 0  # real nc
        fc_data.append(node_feature)
        fc_label.append(label)

        [fc_data, fc_label] = [np.array(fc_data), np.array(fc_label)]

    return [fc_data,fc_label]

def load_graph_data(fold, data_filenames, timestamp, pseudo_epoch = None, data_type="train"):
    data_path = 'MDD_Data/MS_MDD_Harvard_FC_fisher'

    graph_data = []
    graph_edge = []
    graph_label = []

    if args.ex_type == "Main":
        if pseudo_epoch == 0 or pseudo_epoch == None:
            edge_topology = np.load('GAN/Edge_topology/mrmr_seed_{}_GAT/mrmr_MS_MDD_Harvard_FC_map_{}_Multisite_1.npy'.format(args.seed,fold))
        else:
            edge_topology = np.load("GAN/Edge_topology/model{}/mrmr_MDD_Harvard_FC_map_fold_{}_{}.npy".format(timestamp, fold, pseudo_epoch))

    elif args.ex_type == "Single":
        if pseudo_epoch == 0 or pseudo_epoch == None:
            edge_topology = np.load('MDD_topology/mrmr_S20_MS_MDD_Harvard_FC_map.npy')
        else:
            edge_topology = np.load("GAN/Edge_topology/model{}/mrmr_S20_MDD_Harvard_FC_map_fold_{}_{}.npy".format(timestamp, fold, pseudo_epoch))

    for subject in data_filenames:
        node_feature = np.load(os.path.join(data_path,subject.replace('.mat', '.npy')))
        edge_index = edge_topology
        if subject.split('-')[1] == "1" :
            label = 1 # real mdd
        else:
            label = 0 # real nc

        graph_data.append(node_feature)
        graph_edge.append(edge_index)
        graph_label.append(label)

    [graph_data, graph_edge, graph_label] = [np.array(graph_data),np.array(graph_edge),np.array(graph_label)]

    return [graph_data, graph_edge, graph_label]


def load_real_data(fold, data_filenames, timestamp, pseudo_epoch=None,data_type="train"):
    data_path = 'MDD_Data/MS_MDD_Harvard_FC_fisher'

    graph_data = []
    graph_edge = []
    graph_label = []

    if args.ex_type == "Main":
        if pseudo_epoch == 0 or pseudo_epoch == None:
            edge_topology = np.load('MDD_topology/mrmr_MDD_Harvard_FC_map_fold_{}.npy'.format(fold))
        else:
            edge_topology = np.load("GAN/Edge_topology/model{}/mrmr_MDD_Harvard_FC_map_fold_{}_{}.npy".format(timestamp, fold, pseudo_epoch))

    elif args.ex_type == "Single":
        if pseudo_epoch == 0 or pseudo_epoch == None:
            edge_topology = np.load('MDD_topology/mrmr_S20_MS_MDD_Harvard_FC_map.npy')
        else:
            edge_topology = np.load("GAN/Edge_topology/model{}/mrmr_S20_MDD_Harvard_FC_map_fold_{}_{}.npy".format(timestamp, fold, pseudo_epoch))

    for subject in data_filenames:
        node_feature = np.load(os.path.join(data_path,subject.replace('.mat', '.npy')))
        edge_index = edge_topology
        if subject.split('-')[1] == "1":
            label = 1  # fake mdd
        else:
            label = 0 # fake nc

        graph_data.append(node_feature)
        graph_edge.append(edge_index)
        graph_label.append(label)

    [graph_data, graph_edge, graph_label] = [np.array(graph_data), np.array(graph_edge), np.array(graph_label)]

    return [graph_data, graph_edge, graph_label]

    return [graph_data, graph_edge, graph_label]

def load_fake_data(fold, data_filenames, fixed_noise, timestamp, pseudo_epoch = None): # noise shape [real_data_cnt, 112, 32]
    fake_data = []
    fake_edge = []
    fake_label = []

    if args.ex_type == "Main":
        if pseudo_epoch == 0 or pseudo_epoch == None:
            edge_topology = np.load('MDD_topology/mrmr_MDD_Harvard_FC_map_fold_{}.npy'.format(fold))
        else:
            edge_topology = np.load("GAN/Edge_topology/model{}/mrmr_MDD_Harvard_FC_map_fold_{}_{}.npy".format(timestamp, fold, pseudo_epoch))

    elif args.ex_type == "Single":
        if pseudo_epoch == 0 or pseudo_epoch == None:
            edge_topology = np.load('MDD_topology/mrmr_S20_MS_MDD_Harvard_FC_map.npy')
        else:
            edge_topology = np.load("GAN/Edge_topology/model{}/mrmr_S20_MDD_Harvard_FC_map_fold_{}_{}.npy".format(timestamp, fold, pseudo_epoch))

    for z,subject in enumerate(data_filenames):
        node_feature = fixed_noise[z]
        edge_index = edge_topology
        if subject.split('-')[1] == "1":
            label = 3  # fake mdd
        else:
            label = 2 # fake nc

        fake_data.append(node_feature)
        fake_edge.append(edge_index)
        fake_label.append(label)

    [fake_data, fake_edge, fake_label] = [np.array(fake_data), np.array(fake_edge), np.array(fake_label)]

    return [fake_data, fake_edge, fake_label]

if __name__ == "__main__":
    for fold in range(1,6):
        data_filenames = load_data_list(fold, data_type = "test")
        load_graph_data(fold,data_filenames,args.timestamp,data_type = "test")
