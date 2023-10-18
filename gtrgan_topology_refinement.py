# label = 0 : real nc, label = 1 : real mdd, label = 2 : fake nc, label = 3 : fake mdd
import os
import time
import torch
import torch.nn as nn
import numpy as np
import sklearn
import glob
import torch.nn.functional as F
import pandas as pd
from config import config
from load_data import load_data_list, load_real_data,load_fake_data, load_graph_data
from mrmr_topology import mrmr
from model import ACDiscriminator, Generator, Classifier
from utils import accuracy, sensitivity, specificity
from gat_baseline import Single_GAT
from graphsage_baseline import Single_GraphSAGE

# load hyper parameter
args = config()

# make TSNE directory
os.mkdir('GAN/TSNE/tsne{}'.format(args.timestamp))

def generate_pseudo_data(fold,timestamp,gen_no):

    # model declaration
    generator = Generator()
    discriminator = ACDiscriminator()
    classifier = Classifier()

    if torch.cuda.is_available():
        generator.to(args.device)
        discriminator.to(args.device)
        classifier.to(args.device)

    if args.ex_type == "Main":
        generator.load_state_dict(torch.load('GAN/Model/mrmr_seed_{}_fix/G_model_{}_final.pth'.format(args.seed, fold), map_location=args.device))
        discriminator.load_state_dict(torch.load('GAN/Model/mrmr_seed_{}_fix/D_model_{}_final.pth'.format(args.seed, fold), map_location=args.device))
    elif args.ex_type == "Single":
        generator.load_state_dict(torch.load('GAN/Model/mrmr_seed_{}_S20/G_model_{}_final.pth'.format(args.seed, fold), map_location=args.device))
        discriminator.load_state_dict(torch.load('GAN/Model/mrmr_seed_{}_S20/D_model_{}_final.pth'.format(args.seed, fold), map_location=args.device))


    generator.eval()
    discriminator.eval()
    classifier.eval()

    start_log = "======== Pseudo data generation start! ======== \t timestamp: {} \t fold : {} \t".format(args.timestamp, fold)
    print(start_log)

    # load real data
    real_filenames = load_data_list(fold=fold, data_type="train")
    [train_x, train_edge_index, train_label] = load_real_data(fold=fold, data_filenames=real_filenames, timestamp=args.timestamp, pseudo_epoch=gen_no-1)
    real_x = torch.FloatTensor(train_x).to(args.device)
    real_edge_index = torch.LongTensor(train_edge_index).to(args.device)
    real_label = torch.LongTensor(train_label).to(args.device)

    pseudo_size = args.pseudo_size
    mdd_no = round(len(real_filenames) * pseudo_size * 0.48)
    nc_no = round(len(real_filenames) * pseudo_size * 0.52)
    mdd_cnt = 0
    nc_cnt = 0
    flag = True
    seed_value = 0

    while flag:
        np.random.seed(seed_value)
        noise = np.random.normal(0, 1, (len(real_filenames), 112, args.z_dim))

        [gen_data, gen_edge, gen_label] = load_fake_data(fold=fold, data_filenames=real_filenames, fixed_noise=noise, timestamp=args.timestamp, pseudo_epoch = gen_no-1)
        fake_x = torch.FloatTensor(gen_data).to(args.device)
        fake_edge_index = torch.LongTensor(gen_edge).to(args.device)
        fake_x = generator(fake_x, fake_edge_index, real_label)  # real distribution

        D_real_class, D_real_features = discriminator(real_x.detach(), real_edge_index.detach())
        D_fake_class, D_fake_features = discriminator(fake_x.detach(), real_edge_index.detach())

        d_out = (np.argmax(D_fake_class.cpu().detach().tolist(), axis=1))
        d_mask = (d_out == real_label.cpu().detach().numpy())

        mask = np.invert(d_mask)
        generated_fake_x = np.delete(fake_x.cpu().detach().numpy(), mask, axis=0)  # pseudo data filtering
        generated_fake_label = np.delete(real_label.cpu().detach().numpy(), mask, axis=0)  # pseudo label filtering

        for sub in range(generated_fake_x.shape[0]):
            if generated_fake_label[sub] == 1:
                if mdd_cnt == 0:
                    generated_fake_x_mdd = [generated_fake_x[sub]]
                    generated_fake_label_mdd = [generated_fake_label[sub]]
                else:
                    generated_fake_x_mdd = np.concatenate((generated_fake_x_mdd, [generated_fake_x[sub]]), axis=0)
                    generated_fake_label_mdd = np.concatenate((generated_fake_label_mdd, [generated_fake_label[sub]]), axis=0)
                mdd_cnt = mdd_cnt + 1
            else:
                if nc_cnt == 0:
                    generated_fake_x_nc = [generated_fake_x[sub]]
                    generated_fake_label_nc = [generated_fake_label[sub]]
                else:
                    generated_fake_x_nc = np.concatenate((generated_fake_x_nc, [generated_fake_x[sub]]), axis=0)
                    generated_fake_label_nc = np.concatenate((generated_fake_label_nc, [generated_fake_label[sub]]), axis=0)
                nc_cnt = nc_cnt + 1

        if mdd_cnt > mdd_no and nc_cnt > nc_no:
            flag = False
        seed_value = seed_value + 1

    # concat pseudo data & save as numpy
    generated_data = np.concatenate((generated_fake_x_mdd[:mdd_no], generated_fake_x_nc[:nc_no]), axis=0)
    generated_label = np.concatenate((generated_fake_label_mdd[:mdd_no], generated_fake_label_nc[:nc_no]), axis=0)

    if not (os.path.isdir('GAN/Pseudo_data/model{}/'.format(args.timestamp))):
        os.makedirs(os.path.join('GAN/Pseudo_data/model{}/'.format(args.timestamp)))
    np.save('GAN/Pseudo_data/model{}/pseudo_data_{}_{}.npy'.format(args.timestamp, fold, gen_no), generated_data)
    np.save('GAN/Pseudo_data/model{}/pseudo_label_{}_{}.npy'.format(args.timestamp, fold, gen_no), generated_label)

    generated_data = torch.FloatTensor(generated_data).to(args.device)
    generated_label = torch.LongTensor(generated_label).to(args.device)

    if args.load_data == 1:
        real_filenames = load_data_list(fold=fold, data_type="train")
        if args.gan_type == "semi":
            generated_data = np.load('SemiGAN/Pseudo_data/gnn_semi_{}(sota)/pseudo_data_{}.npy'.format(args.seed, fold))
            generated_label = np.load('SemiGAN/Pseudo_data/gnn_semi_{}(sota)/pseudo_label_{}.npy'.format(args.seed, fold))

            generated_data = torch.FloatTensor(generated_data).to(args.device)
            generated_label = torch.LongTensor(generated_label).to(args.device)

        elif args.gan_type == "brainnetcnn":
            generated_data = np.load('BrainnetCNN/Pseudo_data/BrainNet_seed_{}(sota)/pseudo_data_{}.npy'.format(args.seed, fold))
            generated_label = np.load('BrainnetCNN/Pseudo_data/BrainNet_seed_{}(sota)/pseudo_label_{}.npy'.format(args.seed, fold))

            generated_data = torch.FloatTensor(generated_data).to(args.device).squeeze()
            generated_label = torch.LongTensor(generated_label).to(args.device)

        elif args.gan_type == "dnn":
            generated_fc = np.load('DNN/Pseudo_data/dnn_seed_{}(sota)/pseudo_data_{}.npy'.format(args.seed, fold))
            generated = np.ones((112, 112))
            r, c = np.triu_indices(112, 1)
            for n in range(generated_fc.shape[0]):
                generated[r, c] = generated_fc[n]
                generated[c, r] = generated_fc[n]
                if n == 0:
                    generated_data = [generated]
                else:
                    generated_data = np.concatenate((generated_data, [generated]), axis=0)

            generated_label = np.load('DNN/Pseudo_data/dnn_seed_{}(sota)/pseudo_label_{}.npy'.format(args.seed, fold))

            generated_data = torch.FloatTensor(generated_data).to(args.device)
            generated_label = torch.LongTensor(generated_label).to(args.device)

        elif args.gan_type == "wgan":
            generated_data = np.load('WGAN-GP/Pseudo_data/wgan_seed_{}(sota)/pseudo_data_{}.npy'.format(args.seed, fold))
            generated_label = np.load('WGAN-GP/Pseudo_data/wgan_seed_{}(sota)/pseudo_label_{}.npy'.format(args.seed, fold))

            generated_data = torch.FloatTensor(generated_data).to(args.device)
            generated_label = torch.LongTensor(generated_label).to(args.device)

        elif args.gan_type == "dnngnn":
            generated_data = np.load('DNNGNN/Pseudo_data/dnngnn_seed_{}_2/pseudo_data_{}.npy'.format(args.seed, fold))
            generated_label = np.load('DNNGNN/Pseudo_data/dnngnn_seed_{}_2/pseudo_label_{}.npy'.format(args.seed, fold))

            generated_data = torch.FloatTensor(generated_data).to(args.device)
            generated_label = torch.LongTensor(generated_label).to(args.device)

        elif args.gan_type == "acgan":
            generated_data = np.load('ACGAN/Pseudo_data/acgan_seed_{}(sota)/pseudo_data_{}.npy'.format(args.seed, fold))
            generated_label = np.load('ACGAN/Pseudo_data/acgan_seed_{}(sota)/pseudo_label_{}.npy'.format(args.seed, fold))

            generated_data = torch.FloatTensor(generated_data).to(args.device)
            generated_label = torch.LongTensor(generated_label).to(args.device)

    [batch_real_x, batch_real_edge_index, batch_real_label] = load_real_data(fold=fold, data_filenames=real_filenames, timestamp=args.timestamp)
    pseudo_data = torch.FloatTensor(np.concatenate([batch_real_x, generated_data.cpu().detach().numpy()], axis=0)).to(args.device)
    pseudo_label = torch.LongTensor(np.concatenate([batch_real_label, generated_label.cpu().detach().numpy()], axis=0)).to(args.device)

    if args.edge_type == "mrmr":
        mrmr(train_data=pseudo_data.cpu().detach().numpy(), train_label=pseudo_label.cpu().detach().numpy(), K=121, fold=fold, timestamp=args.timestamp, pseudo_epoch=gen_no)

    [_, real_edge, _] = load_graph_data(fold=fold, data_filenames=real_filenames, timestamp=args.timestamp, pseudo_epoch=gen_no)
    pseudo_edge_index = torch.FloatTensor(real_edge).to(args.device)
    for e in range(1, 4):
        pseudo_edge_index = torch.cat([pseudo_edge_index, pseudo_edge_index[:generated_data.shape[0]]])
        real_edge_index = torch.cat([real_edge_index, real_edge_index[:generated_data.shape[0]]])

    # return fold, pseudo_data, pseudo_edge_index, pseudo_label
    return fold, real_x, pseudo_edge_index, real_label


def train_classifier(fold,train_x, train_edge_index, train_label,gen_no):

    print("============Classifier Training Start ! ============")

    if args.gnn_type == "GAT":
        classifier = Single_GAT().to(args.device)
    elif args.gnn_type == "GraphSAGE":
        classifier = Single_GraphSAGE().to(args.device)
    else:
        classifier = Classifier().to(args.device)

    c_weight_decay = args.c_weight_decay
    batch_size = args.batch_size

    if args.gnn_type == "GAT":
        classifier.load_state_dict(torch.load('GAT/Model/model_seed_{}/GAT_model_{}_init.pth'.format(args.seed, fold), map_location=args.device), strict=True)  # weight initialize
    elif args.gnn_type == "GraphSAGE":
        classifier.load_state_dict(torch.load('GraphSAGE/Model/model_seed_{}/GraphSAGE_model_{}_init.pth'.format(args.seed, fold), map_location=args.device), strict=True)  # weight initialize
    else:
        if args.ex_type == "Main":
            classifier.load_state_dict(torch.load('GCN/Model/model_seed_{}/GCN_model_{}_init.pth'.format(args.seed,fold),map_location=args.device), strict=True) # weight initialize
        elif args.ex_type == "Single":
            classifier.load_state_dict(torch.load('GCN/Model/model_seed_{}_S20/GCN_model_{}_init.pth'.format(args.seed, fold), map_location=args.device), strict=True)  # weight initialize


    optimizer_c = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=c_weight_decay, betas=(0.5, 0.9))
    scheduler_c = torch.optim.lr_scheduler.ExponentialLR(optimizer_c, gamma=0.998)
    mdd_cnt = train_label.count_nonzero()
    nc_cnt = train_x.shape[0] - mdd_cnt
    info_gain_weight = torch.tensor((mdd_cnt/(mdd_cnt+nc_cnt), nc_cnt/(mdd_cnt+nc_cnt)), dtype=torch.float).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=info_gain_weight)

    # save log
    if not (os.path.isdir('GAN/Results_dgan/{}/model{}'.format(args.edge_type,args.timestamp))):
        os.makedirs(os.path.join('GAN/Results_dgan/{}/model{}'.format(args.edge_type,args.timestamp)))
    path_save_info = 'GAN/Results_dgan/{}/model{}'.format(args.edge_type,args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, fold)

    # save parameter log
    with open(path_save_info.replace(".csv", "c_info.txt"), "w") as f:
        f.write('Classifier: {}\n\n Parameters : {}\n\n Optimizer C : {}\n\n '.format(str(classifier), args.Return_args(), optimizer_c))

    step_size = (train_x.shape[0]//batch_size)

    for epoch in range(1,2):
        out_stack = []
        label_stack = []
        c_loss = []
        epoch_time = time.time()

        indices = torch.randperm(train_x.size(0)).to(args.device)
        train_x = torch.index_select(train_x, dim=0, index=indices)
        train_label = torch.index_select(train_label, dim=0, index=indices)

        for step in range(step_size+1):
            if step == step_size:
                final_batch = train_x.shape[0] - (step_size) * args.batch_size  # real data batch
                batch_mask = range(step * batch_size, (step * args.batch_size) + final_batch)
                if final_batch == 0 or final_batch == 1:
                    continue
            else:
                batch_mask = range(step * batch_size, (step + 1) * args.batch_size)

            optimizer_c.zero_grad()
            C_real_logits, C_real_features = classifier(train_x[batch_mask], train_edge_index[batch_mask])
            c_step_loss = torch.mean(criterion(C_real_logits, train_label[batch_mask]))
            c_loss.append(c_step_loss)
            c_step_loss.backward()
            optimizer_c.step()

            out_stack.extend(np.argmax(C_real_logits.cpu().detach().tolist(), axis=1))  # real prediction stack
            label_stack.extend(train_label[batch_mask].cpu().detach().tolist())  # real prediction stack

        scheduler_c.step()
        C_acc = accuracy(out_stack, label_stack)
        C_sen = sensitivity(out_stack, label_stack)
        C_spec = specificity(out_stack, label_stack)

        print("[Train] [Epoch:{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}]".format(epoch,time.time() - epoch_time, sum(c_loss)/step_size, C_acc, C_sen, C_spec))

        with open(path_save_info.replace(".csv", "_c_train_{}.csv".format(gen_no)), "a") as f:
            f.write('{},{},{},{},{}\n'.format(epoch,sum(c_loss)/step_size, C_acc, C_sen, C_spec))

    model_save_path = 'GAN/Model/model{}/'.format(args.timestamp)
    if not (os.path.isdir(model_save_path)):
        os.mkdir(model_save_path)
    model_name_C = model_save_path + 'C_model_{}_final_{}.pth'.format(fold, gen_no)
    torch.save(classifier.state_dict(), model_name_C)

    test(fold, gen_no)
    files = glob.glob(model_save_path + '*.pth')
    for f in files:
        if 'final' not in f:
            os.remove(f)  # except final model
    print("[!!] Training finished\n\n")


def test(fold, gen_no):
    # model declaration
    if args.gnn_type == "GAT":
        classifier = Single_GAT()
    elif args.gnn_type == "GraphSAGE":
        classifier = Single_GraphSAGE()
    else :
        classifier = Classifier()

    if torch.cuda.is_available():
        classifier.to(args.device)

    classifier.load_state_dict(torch.load('GAN/Model/model{}/C_model_{}_final_{}.pth'.format(args.timestamp, fold, gen_no)))
    classifier.eval()

    # sample loss function
    criterion = nn.CrossEntropyLoss()

    # load test data
    test_filenames = load_data_list(fold=fold, data_type="test")
    [test_data, test_edge, test_label] = load_graph_data(fold=fold, data_filenames=test_filenames, timestamp=args.timestamp, pseudo_epoch=gen_no)
    test_x = torch.FloatTensor(test_data).to(args.device)
    test_edge_index = torch.FloatTensor(test_edge).to(args.device)
    test_label = torch.LongTensor(test_label).to(args.device)

    epoch_time = time.time()

    total_info = {}
    eval_list = ['loss', 'acc', 'sen', 'spec', 'f1']
    label_stack = []
    out_stack = []

    C_logits, C_features = classifier(test_x, test_edge_index)
    test_loss_d = criterion(C_logits, test_label)

    label_stack.extend(test_label.cpu().detach().tolist())
    out_stack.extend(np.argmax(C_logits.cpu().detach().tolist(), axis=1))

    total_info['loss'] = test_loss_d.item()

    acc = accuracy(out_stack, label_stack)
    total_info['acc'] = acc

    sen = sensitivity(out_stack, label_stack)
    total_info['sen'] = sen

    spec = specificity(out_stack, label_stack)
    total_info['spec'] = spec

    f1 = sklearn.metrics.f1_score(label_stack, out_stack)
    total_info['f1'] = f1

    log = "[Test] time [{:.1f}s]".format(time.time() - epoch_time) + \
          " | loss [{:.3}] acc [{:.3}] sen [{:.3f}] spec [{:.3f}] f1 [{:.3f}]".format(total_info['loss'], total_info['acc'], total_info['sen'], total_info['spec'], total_info['f1'])
    print(log)

    out = {}
    for key in eval_list:
        out[key] = total_info[key]

    path_save_info = 'GAN/Results_dgan/{}/model{}'.format(args.edge_type, args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, fold)
    with open(path_save_info.replace(".csv", "_test_{}.csv".format(gen_no)), "a") as f:
        log = ",".join([str(out[key]) for key in out.keys()]) + "\n"
        f.write(log)


if __name__ == "__main__":
    total_result = [[0 for j in range(5)] for i in range(4)]

    if not (os.path.isdir('GAN/Results_dgan/{}/model{}'.format(args.edge_type,args.timestamp))):
        os.makedirs(os.path.join('GAN/Results_dgan/{}/model{}'.format(args.edge_type,args.timestamp)))

    # single classifier_results
    for fold in range(1, 3):
            # save log
        path_save_info = 'GAN/Results_dgan/{}/model{}'.format(args.edge_type,args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, fold)
        with open(path_save_info.replace(".csv", "_test_{}.csv".format(1)), "w") as f:
            f.write("C_loss,C_acc,C_sen,C_spec,C_F1\n")
        f1, pseudo_data, pseudo_edge_index, pseudo_label = generate_pseudo_data(fold,args.timestamp,1)
        train_classifier(fold, pseudo_data, pseudo_edge_index, pseudo_label,1)


    with open('GAN/Results_dgan/{}/model{}/cross_val_result{}_{}.csv'.format(args.edge_type,args.timestamp, args.timestamp,1), 'w') as f:
        f.write('fold,acc,sen,spec,f1\n')

    for fold in range(1, 6):
            result_csv = pd.read_csv(
                'GAN/Results_dgan/{}/model{}/train_info{}_{}_test_{}.csv'.format(args.edge_type,args.timestamp, args.timestamp, fold, 1))
            test_result = result_csv.iloc[-1]

            with open('GAN/Results_dgan/{}/model{}/cross_val_result{}_{}.csv'.format(args.edge_type,args.timestamp, args.timestamp,1), 'a') as f:
                f.write('{},{},{},{},{}\n'.format(fold, round(test_result[1], 3) * 100, round(test_result[2], 3) * 100,
                                                      round(test_result[3], 3) * 100, round(test_result[4], 3) * 100))
                total_result[0][fold - 1] = round(test_result[1], 3)
                total_result[1][fold - 1] = round(test_result[2], 3)
                total_result[2][fold - 1] = round(test_result[3], 3)
                total_result[3][fold - 1] = round(test_result[4], 3)


    with open('GAN/Results_dgan/{}/model{}/cross_val_result{}_{}.csv'.format(args.edge_type,args.timestamp, args.timestamp, 1), 'a') as f:
            f.write('avg,{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(np.mean(total_result[0]) * 100, np.mean(total_result[1]) * 100, np.mean(total_result[2]) * 100, np.mean(total_result[3]) * 100))
            f.write('std,{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(total_result[0], ddof=1) * 100, np.nanstd(total_result[1], ddof=1) * 100, np.nanstd(total_result[2], ddof=1) * 100, np.nanstd(total_result[3], ddof=1) * 100))





