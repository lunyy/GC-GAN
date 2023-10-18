import os
import time
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import config
from seed import set_seed
from load_data import load_data_list, load_graph_data
from model import myGraphAE
from utils import weights_init

# set hyper parameter
args = config()
timestamp = args.timestamp
set_seed()

def train_ae(fold, gen_no):

    genlr = args.genlr
    g_weight_decay = args.g_weight_decay
    batch_size = args.gan_batch_size

    # model declaration
    generator = myGraphAE()
    if torch.cuda.is_available():
        generator.to(args.device)

    # weight initialization - default xavier uniform
    generator.apply(weights_init)

    # save log
    if not (os.path.isdir('GraphAE/Results/model{}'.format(args.timestamp))):
        os.makedirs(os.path.join('GraphAE/Results/model{}'.format(args.timestamp)))
    path_save_info = 'GraphAE/Results/model{}'.format(args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, fold)
    with open(path_save_info, "w") as f:
        f.write("G_loss\n")
    with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
        f.write("G_loss\n")

    # optimizer
    if args.optim == "ADAM":
        print("current optimizer : ADAM")
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=genlr, weight_decay=g_weight_decay, betas=args.betas)

    scheduler_g= torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.gamma)

    # sample loss function
    mse_loss = nn.MSELoss()

    # save parameter log
    with open(path_save_info.replace(".csv", ".txt"), "w") as f:
            f.write('Generator: {}\n\n Parameters : {}\n\n Optimizer G : {}\n\n Scheduler G {} : {}\n\n'.format \
                    (str(myGraphAE),args.Return_args(),optimizer_g,scheduler_g,scheduler_g.state_dict()))

    train_log = "##### Training start! ###### \t timestamp: {} \t fold : {} \t".format(args.timestamp, fold)
    print(train_log)

    train_filenames = load_data_list(fold=fold, data_type="train")
    step_size = (len(train_filenames) // batch_size)

    for epoch in range(1000):
        epoch_time = time.time()
        step_loss_g = [];

        generator.train()

        np.random.shuffle(train_filenames)  # shuffle train files

        for step in range(step_size + 1):
            if step == (step_size):
                final_batch = len(train_filenames) - (step_size) * batch_size
                batch_mask = range(step * batch_size, (step * batch_size) + final_batch)
            else:
                batch_mask = range(step * batch_size, (step + 1) * batch_size)
            # load train data

            [train_data, train_edge, train_label] = load_graph_data(fold=fold, data_filenames=train_filenames[batch_mask],timestamp=timestamp, pseudo_epoch=gen_no, data_type="train")
            real_x = torch.FloatTensor(train_data).to(args.device)
            real_edge_index = torch.LongTensor(train_edge).to(args.device)
            real_label = torch.LongTensor(train_label).to(args.device)

            # training generator
            optimizer_g.zero_grad()
            generated = generator(real_x, real_edge_index, real_label)
            loss_recon = mse_loss(generated,real_x)
            loss_g = 10 * loss_recon
            loss_g.backward()

            optimizer_g.step()
            scheduler_g.step()

            step_loss_g.append(loss_g.item())

        G_loss = sum(step_loss_g) / step_size  # generator loss

        print("[Train] [Epoch:{}/{}] [Time:{:.1f}] [Loss G:{:.4f}]".format(epoch + 1,1000, time.time()-epoch_time,G_loss))

        with open(path_save_info, "a") as f:
            f.write('{}\n'.format(G_loss))

        model_save_path = 'GraphAE/Model/model{}/'.format(args.timestamp)
        if args.model_checkpoint != 0 and (epoch + 1) % args.model_checkpoint == 0:
            print("checkpoint saving..")
            if not (os.path.isdir(model_save_path)):
                os.mkdir(model_save_path)
            model_name_G = model_save_path + 'GAE_model_{}_{:04}.pth'.format(fold, epoch + 1)
            torch.save(generator.state_dict(), model_name_G)

            test_result = test(fold, epoch)
            with open(path_save_info.replace(".csv", "_test.csv"), "a") as f:
                log = ",".join([str(test_result[key]) for key in test_result.keys()]) + "\n"
                f.write(log)

    torch.save(generator.state_dict(), model_save_path + 'GAE_model_{}_final.pth'.format(fold))

    files = glob.glob(model_save_path + '*.pth')
    for f in files:
        if 'final' not in f:
            os.remove(f)  # except final model

    print("[!!] Training finished\n\n")

    return args.timestamp

def test(fold, epoch):

    # model declaration
    generator = myGraphAE()
    if torch.cuda.is_available():
        generator.to(args.device)

    # load test model
    generator.load_state_dict(torch.load('GraphAE/Model/model{}/GAE_model_{}_{:04}.pth'.format(args.timestamp, fold, epoch + 1)))
    generator.eval()

    # sample loss function
    mse_loss = nn.MSELoss()

    # load test data
    test_filenames = load_data_list(fold = fold, data_type = "test")
    [test_data, test_edge, test_label] = load_graph_data(fold = fold, data_filenames = test_filenames,timestamp=timestamp, pseudo_epoch=None, data_type="test")
    test_x = torch.FloatTensor(test_data).to(args.device)
    test_edge_index = torch.LongTensor(test_edge).to(args.device)
    test_label = torch.LongTensor(test_label).to(args.device)

    epoch_time = time.time()

    total_info = {}
    eval_list = ['loss']

    recon_img = generator(test_x,test_edge_index, test_label)
    test_loss_g = mse_loss(recon_img,test_x)

    recon_img = recon_img.cpu().detach().numpy()

    for n in range(len(test_filenames)):
        if n < 10:
            plt.figure(figsize=(6, 5))
            sns.heatmap(data=recon_img[n], annot=False,
                        fmt='.2f', linewidths=0, cmap='jet', cbar=True, xticklabels=False, yticklabels=False)
            plt.title('Generated FC matrix fold{}'.format(fold))
            plt.tight_layout()
            plt.savefig(
                'GraphAE/Results/model{}/generated_fold{}_no{}.png'.format(args.timestamp, fold, n + 1),
                format='png')
            plt.close()

        if n == 0:
            mean_generated = recon_img[n]
        else:
            mean_generated = np.add(mean_generated, recon_img[n])
    mean_generated = mean_generated / len(test_filenames)

    plt.figure(figsize=(6, 5))
    sns.heatmap(data=mean_generated, annot=False,
                fmt='.2f', linewidths=0, cmap='jet', cbar=True, xticklabels=False, yticklabels=False, vmin=-0.1,
                vmax=1.5)
    plt.title('Mean generated FC matrix fold{}'.format(fold))
    plt.tight_layout()
    plt.savefig('GraphAE/Results/model{}/mean_generated_fold{}.png'.format(args.timestamp, fold),
                format='png')
    plt.close()


    total_info['loss'] = test_loss_g.item()


    log = "[Test] time [{:.1f}s]".format(time.time() - epoch_time) + \
              " | loss [{:.3}]".format(total_info['loss'])
    print(log)

    out = {}
    for key in eval_list:
        out[key] = total_info[key]

    return out


if __name__ == "__main__":
    for fold in range(1,6):
        train_ae(fold,0)

