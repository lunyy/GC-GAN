import os
import time
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn
import random
import sklearn.metrics as metrics
from config import config
from load_data import load_data_list, load_graph_data
from model import Classifier
from utils import weights_init, accuracy, sensitivity, specificity

# set hyper parameter
args = config()

def train(fold):
    # fix seed
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    batch_size = args.batch_size

    # model declaration
    classifier = Classifier()
    if torch.cuda.is_available():
        classifier.to(args.device)

    # classifier.apply(weights_init)
    if args.ex_type == "Main":
        classifier.load_state_dict(torch.load('GCN/Model/model_seed_{}/GCN_model_{}_init.pth'.format(args.seed, fold), map_location=args.device), strict=True)
    elif args.ex_type == "Single":
        classifier.load_state_dict(torch.load('GCN/Model/model_seed_{}_S20/GCN_model_{}_init.pth'.format(args.seed, fold), map_location=args.device), strict=True)  # unseen

    # save log
    if not (os.path.isdir('GCN/Results/model{}'.format(args.timestamp))):
        os.makedirs(os.path.join('GCN/Results/model{}'.format(args.timestamp)))
    path_save_info = 'GCN/Results/model{}'.format(args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, fold)
    with open(path_save_info, "w") as f:
        f.write("C_loss,C_acc,C_sen,C_spec\n")
    with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("C_loss,C_acc,C_sen,C_spec,C_f1\n")

    # optimizer
    if args.optim == "ADAM":
        print("current optimizer : ADAM")
        optimizer_d = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.c_weight_decay, betas=(0.5, 0.9))

    # scheduler
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=args.gamma)

    # save parameter log
    with open(path_save_info.replace(".csv", ".txt"), "w") as f:
        f.write('Classifier: {}\n\n Parameters :{} \n\n Optimizer C : {}\n\n Scheduler C {} : {}\n\n'.format \
                    (str(classifier), args.Return_args(), args.batch_size, optimizer_d, scheduler_d, scheduler_d.state_dict()))

    train_log = "##### Training start! ###### \t timestamp: {} \t fold : {} \t lr : {} \t wd : {}".format(args.timestamp, fold, ray_config["lr"][fold-1],ray_config["wd"][fold-1])
    print(train_log)

    # load train data
    train_filenames = load_data_list(fold=fold, data_type="train")
    [train_data, train_edge, train_label] = load_graph_data(fold=fold, data_filenames=train_filenames, timestamp=args.timestamp, pseudo_epoch=None)
    real_label = torch.LongTensor(train_label).to(args.device)

    mdd_cnt = (real_label == 1).sum()

    # sample loss function
    info_gain_weight = torch.tensor((mdd_cnt/real_label.shape[0],(real_label.shape[0]-mdd_cnt)/real_label.shape[0]), dtype=torch.float).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=info_gain_weight)

    step_size = (len(train_filenames) // batch_size)


    for epoch in range(1,2001):

        epoch_time = time.time()
        step_loss_d = []  # loss
        label_stack = []; out_stack = []

        classifier.train()

        np.random.shuffle(train_filenames) # shuffle train files

        for step in range(step_size + 1):
            if step == (step_size):
                final_batch = len(train_filenames) - (step_size) * batch_size
                batch_mask = range(step * batch_size, (step * batch_size) + final_batch)
            else:
                batch_mask = range(step * batch_size, (step + 1) * batch_size)

            [train_data, train_edge, train_label] = load_graph_data(fold=fold, data_filenames=train_filenames[batch_mask],timestamp=args.timestamp,pseudo_epoch=None)
            real_x = torch.FloatTensor(train_data).to(args.device)
            real_edge_index = torch.LongTensor(train_edge).to(args.device)
            real_label = torch.LongTensor(train_label).to(args.device)

            optimizer_d.zero_grad()

            logits_real,_= classifier(real_x,real_edge_index)
            logz_label = criterion(logits_real,real_label)
            loss_d = logz_label

            loss_d.backward()

            optimizer_d.step()

            step_loss_d.append(loss_d.item())

            label_stack.extend(real_label.cpu().detach().tolist())
            out_stack.extend(np.argmax(logits_real.cpu().detach().tolist(), axis=1))

        scheduler_d.step()

        C_loss = sum(step_loss_d) / step_size
        C_acc = accuracy(out_stack, label_stack)
        C_sen = sensitivity(out_stack, label_stack)
        C_spec = specificity(out_stack, label_stack)


        print("[Train] [Epoch:{}/{}] [Time:{:.1f}] [Loss C:{:.4f}] [ACC C:{:.4f}] [SEN C:{:.4f}] [SPEC C:{:.4f}]".format(epoch,
                                                                                args.iter_size, time.time()-epoch_time,C_loss,C_acc,C_sen,C_spec))

        with open(path_save_info, "a") as f:
            f.write('{},{},{},{}\n'.format(C_loss, C_acc, C_sen, C_spec))

        model_save_path = 'GCN/Model/model{}/'.format(args.timestamp)
        if not (os.path.isdir(model_save_path)):
            os.mkdir(model_save_path)
        if epoch == 1:
            torch.save(classifier.state_dict(), model_save_path + 'GCN_model_{}_init.pth'.format(fold))

    torch.save(classifier.state_dict(), model_save_path + 'GCN_model_{}_final.pth'.format(fold))

    test(fold)

    print("[!!] Training finished\n\n")


def test(fold):

    if not (os.path.isdir('GCN/Results/model{}'.format(args.timestamp))):
        os.makedirs(os.path.join('GCN/Results/model{}'.format(args.timestamp)))
    path_save_info = 'GCN/Results/model{}'.format(args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, fold)
    with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("C_loss,C_acc,C_sen,C_spec,C_f1,C_auc\n")

    # model declaration
    classifier = Classifier()
    if torch.cuda.is_available():
        classifier.to(args.device)

    # load test model
    classifier.load_state_dict(torch.load('GCN/Model/model{}/GCN_model_{}_final.pth'.format(args.timestamp, fold)))
    classifier.eval()

    # sample loss function
    criterion = nn.CrossEntropyLoss()

    # load test data
    test_filenames = load_data_list(fold=fold, data_type="test")
    [test_data, test_edge, test_label] = load_graph_data(fold=fold, data_filenames=test_filenames,timestamp=args.timestamp)
    test_x = torch.FloatTensor(test_data).to(args.device)
    test_edge_index = torch.LongTensor(test_edge).to(args.device)
    test_label = torch.LongTensor(test_label).to(args.device)


    epoch_time = time.time()
    total_info = {}
    eval_list = ['loss', 'acc', 'sen', 'spec', 'f1','auc']
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

    fpr, tpr, thresholds = metrics.roc_curve(label_stack, out_stack)
    total_info['auc'] = metrics.auc(fpr, tpr)

    log = "[Test] time [{:.1f}s]".format(time.time() - epoch_time) + \
          " | loss [{:.3}] acc [{:.3}] sen [{:.3f}] spec [{:.3f}] f1 [{:.3f}] auc [{:.3f}]".format(total_info['loss'], total_info['acc'], total_info['sen'], total_info['spec'], total_info['f1'], total_info['auc'])
    print(log)

    out = {}
    for key in eval_list:
        out[key] = total_info[key]

    path_save_info = 'GCN/Results/model{}'.format(args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, fold)
    with open(path_save_info.replace(".csv", "_test.csv"), "a") as f:
        log = ",".join([str(out[key]) for key in out.keys()]) + "\n"
        f.write(log)



if __name__ == "__main__":

    total_result = [[0 for j in range(5)] for i in range(5)]
    for fold in range(1,6):
        train(fold)

    with open('GCN/Results/model{}/cross_val_result{}.csv'.format(args.timestamp, args.timestamp), 'w') as f:
        f.write('fold,acc,sen,spec,f1,auc\n')


    for fold in range(1, 6):
        result_csv = pd.read_csv(
            'GCN/Results/model{}/train_info{}_{}_test.csv'.format(args.timestamp, args.timestamp, fold))
        test_result = result_csv.iloc[-1]
        with open('GCN/Results/model{}/cross_val_result{}.csv'.format(args.timestamp, args.timestamp), 'a') as f:
            f.write('{},{},{},{},{},{}\n'.format(fold, round(test_result[1], 3) * 100, round(test_result[2], 3) * 100,
                                              round(test_result[3], 3) * 100, round(test_result[4], 3) * 100, round(test_result[5], 3) * 100))
        total_result[0][fold - 1] = round(test_result[1], 3)
        total_result[1][fold - 1] = round(test_result[2], 3)
        total_result[2][fold - 1] = round(test_result[3], 3)
        total_result[3][fold - 1] = round(test_result[4], 3)
        total_result[4][fold - 1] = round(test_result[5], 3)


    with open('GCN/Results/model{}/cross_val_result{}.csv'.format(args.timestamp, args.timestamp), 'a') as f:
        f.write('avg,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(np.mean(total_result[0]) * 100, np.mean(total_result[1]) * 100,
                                                    np.mean(total_result[2]) * 100, np.mean(total_result[3]) * 100, np.mean(total_result[4]) * 100))
        f.write('std,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.nanstd(total_result[0],ddof=1) * 100, np.nanstd(total_result[1],ddof=1) * 100,
                                                  np.nanstd(total_result[2],ddof=1) * 100, np.nanstd(total_result[3],ddof=1) * 100, np.nanstd(total_result[4]) * 100))
