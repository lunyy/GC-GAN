import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from config import config
import glob
from load_data import load_data_list, load_real_data,load_fake_data
from model import ACDiscriminator, Generator
from utils import accuracy,weights_init

# load hyper parameter
args = config()

def train_gan(fold,gen_no):
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

    # hyperparameter setting
    batch_size = args.gan_batch_size
    iter_size = args.iter_size
    genlr = args.genlr
    dislr = args.dislr

    g_weight_decay = args.g_weight_decay
    d_weight_decay = args.d_weight_decay
    scale_rate = 1

    # model declaration
    generator = Generator()
    discriminator = ACDiscriminator()

    if torch.cuda.is_available():
        generator.to(args.device)
        discriminator.to(args.device)

    # # load GAN's initial weight
    # generator.load_state_dict(torch.load('GraphAE/Model/mrmr_seed_{}_S20/GAE_model_{}_final.pth'.format(args.seed, fold), map_location=args.device), strict=False)
    # discriminator.load_state_dict(torch.load('GCN/Model/mrmr_seed_{}_S20/GCN_model_{}_final.pth'.format(args.seed, fold), map_location=args.device), strict=False)

    generator.load_state_dict(torch.load('GAN/Model/{}_seed_{}_fix/G_model_{}_final.pth'.format(args.edge_type,args.seed, fold), map_location=args.device), strict=False)
    discriminator.load_state_dict(torch.load('GAN/Model/{}_seed_{}_{}_fix/D_model_{}_final.pth'.format(args.edge_type,args.seed, fold), map_location=args.device), strict=False)

    # generator.apply(weights_init)
    # discriminator.apply(weights_init)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=genlr, weight_decay=g_weight_decay, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=dislr, weight_decay=d_weight_decay, betas=(0.5, 0.9))

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.998)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.998)

    # save log
    if not (os.path.isdir('GAN/Results_gan/model{}'.format(args.timestamp))):
        os.makedirs(os.path.join('GAN/Results_gan/model{}'.format(args.timestamp)))
    path_save_info = 'GAN/Results_gan/model{}'.format(args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, fold)
    with open(path_save_info.replace(".csv","_gan_train_{}.csv".format(gen_no)), "w") as f:
        f.write("G_loss,D_loss,Target_acc,Gen_acc\n")
    with open(path_save_info.replace(".csv","_c_train_{}.csv".format(gen_no)), "w") as f:
        f.write("C_loss,C_real_acc,C_real_sen,C_real_spec,C_F1\n")

    # save parameter log
    with open(path_save_info.replace(".csv", "_gan.txt"), "w") as f:
        f.write('Generator: {}\n\n Discriminator: {}\n\n Parameters : {}\n\n Optimizer G : {}\n\n Optimizer D : {}\n\n'.format(str(generator), str(discriminator), args.Return_args(), optimizer_g, optimizer_d))

    start_log = "======== GAN Training start! ======== \t timestamp: {} \t fold : {} \t".format(args.timestamp, fold)
    print(start_log)

    # load train data
    real_filenames = load_data_list(fold=fold, data_type="train")
    [_,_, train_label] = load_real_data(fold=fold, data_filenames=real_filenames, timestamp=args.timestamp)

    # CE loss weight
    mdd_cnt = np.count_nonzero(train_label)
    nc_cnt = train_label.shape[0] - mdd_cnt
    info_gain_weight = torch.tensor((mdd_cnt / 2*(mdd_cnt + nc_cnt), nc_cnt / 2*(mdd_cnt + nc_cnt), mdd_cnt / 2*(mdd_cnt + nc_cnt), nc_cnt / 2*(mdd_cnt + nc_cnt)), dtype=torch.float).to(args.device)

    # loss function
    criterion = nn.CrossEntropyLoss(weight=info_gain_weight)
    mse_loss = nn.MSELoss()

    step_size = (len(real_filenames) // args.gan_batch_size)

    for epoch in range(1,2):

        # label stack
        c_class = []

        # discriminator feature stack
        D_real_feature_stack = []
        D_fake_feature_stack = []
        c_real_stack = []
        c_fake_stack = []

        # loss list
        g_loss = []
        D_loss = []

        epoch_time = time.time()

        # shuffle train files
        np.random.shuffle(real_filenames) # for gan

        for step in range(step_size + 1):
            # set batch size
            if step == (step_size):
                final_batch = len(real_filenames) - (step_size) * batch_size  # real data batch
                batch_mask = range(step * batch_size, (step * batch_size) + final_batch)
            else:
                batch_mask = range(step * batch_size, (step + 1) * batch_size)

            # load real data for D
            [train_data, train_edge, train_label] = load_real_data(fold=fold, data_filenames=real_filenames[batch_mask], timestamp=args.timestamp, pseudo_epoch = gen_no-1,data_type="train")
            real_x = torch.FloatTensor(train_data).to(args.device)
            real_edge_index = torch.LongTensor(train_edge).to(args.device)
            real_label = torch.LongTensor(train_label).to(args.device)


            # fake data generation
            noise = np.random.normal(0, 1, (len(batch_mask), 112, args.z_dim))  # noise shape [381, 112, 64]
            [gen_data, gen_edge, gen_label] = load_fake_data(fold=fold, data_filenames=real_filenames[batch_mask],fixed_noise=noise, timestamp=args.timestamp, pseudo_epoch = gen_no-1)
            fake_x = torch.FloatTensor(gen_data).to(args.device)
            fake_edge_index = torch.LongTensor(gen_edge).to(args.device)
            fake_label = torch.LongTensor(gen_label).to(args.device)
            fake_x = generator(fake_x, fake_edge_index, real_label)  # real distribution 이랑 똑같게 생성(개수도)

            # D  loss 
            optimizer_d.zero_grad()
            D_real_class, D_real_features = discriminator(real_x, real_edge_index)
            D_fake_class, D_fake_features = discriminator(fake_x.detach(), real_edge_index.detach())
            c_loss_real = criterion(D_real_class,real_label) # 0 1
            c_loss_fake = criterion(D_fake_class,fake_label) # 2 3
            c_class_loss = (c_loss_real + c_loss_fake)
            D_step_loss = c_class_loss
            D_loss.append(D_step_loss)
            D_step_loss.backward()
            optimizer_d.step()

            # G loss
            optimizer_g.zero_grad()
            D_fake_class, D_fake_features = discriminator(fake_x, real_edge_index)
            gc_loss_pos= criterion(D_fake_class, real_label) # adversarial loss
            gc_loss_scale = mse_loss(fake_x,real_x)
            g_step_loss =  gc_loss_pos + scale_rate * gc_loss_scale
            g_loss.append(g_step_loss)
            g_step_loss.backward()
            optimizer_g.step()

            # D stack
            D_real_feature_stack.extend(D_real_features.cpu().detach().tolist())  # D real feature
            D_fake_feature_stack.extend(D_fake_features.cpu().detach().tolist())  # D fake feature
            c_real_stack.extend(np.argmax(D_real_class.cpu().detach().tolist(), axis=1))  # c real class
            c_fake_stack.extend(np.argmax(D_fake_class.cpu().detach().tolist(), axis=1))  # c fake class
            c_class.extend(real_label.cpu().detach().tolist())  # real data label

        scale_rate = scale_rate * args.mse
        scheduler_g.step()
        scheduler_d.step()

        target_prob = accuracy(c_real_stack, c_class)  # D real cla prob
        gen_prob = accuracy(c_fake_stack, c_class)  # D fake cla prob

        if args.model_checkpoint != 0:
            print("[Train] [Epoch:{}/{}] [Time:{:.1f}] [Loss G:{:.4f}] [Loss D:{:.4f}] [TARGET ACC:{:.4f}] [GEN ACC:{:.4f}]  ".format(epoch, args.iter_size, time.time() - epoch_time, sum(g_loss) / step_size,  sum(D_loss) / step_size, target_prob,gen_prob))

            with open(path_save_info.replace(".csv", "_gan_train_{}.csv".format(gen_no)), "a") as f:
                f.write('{},{},{},{}\n'.format(sum(g_loss) / step_size, sum(D_loss) / step_size, target_prob, gen_prob))

        if args.model_checkpoint != 0 and epoch % args.model_checkpoint == 0:
            model_save_path = 'GAN/Model/model{}/'.format(args.timestamp)
            print("checkpoint saving..")
            if not (os.path.isdir(model_save_path)):
                os.mkdir(model_save_path)
            model_name_G = model_save_path + 'G_model_{}_{:04}.pth'.format(fold, epoch)
            model_name_D = model_save_path + 'D_model_{}_{:04}.pth'.format(fold, epoch)
            torch.save(generator.state_dict(), model_name_G)
            torch.save(discriminator.state_dict(), model_name_D)
            check_image(fake_x, fold) # generated image check

    # save model
    torch.save(generator.state_dict(), model_save_path + 'G_model_{}_final.pth'.format(fold))
    torch.save(discriminator.state_dict(), model_save_path + 'D_model_{}_final.pth'.format(fold))

    # remove all models except final model
    files = glob.glob(model_save_path + '*.pth')
    for f in files:
        if 'final' not in f:
            os.remove(f)

    finish_log = "======== GAN Training finished! ======== \t timestamp: {} \t fold : {} \t".format(args.timestamp, fold)
    print(finish_log)

def check_image(fake_data,fold):
    fake_data = fake_data.cpu().detach().numpy()
    print(fake_data.shape[0])
    if fake_data.shape[0] > 10 : sample = 10
    else: sample = fake_data.shape[0]

    for n in range(sample):
        plt.figure(figsize=(6, 5))
        sns.heatmap(data=fake_data[n], annot=False,
                    fmt='.2f', linewidths=0, cmap='jet', cbar=True, xticklabels=False, yticklabels=False)
        plt.title('Generated FC matrix fold{}'.format(fold))
        plt.tight_layout()
        plt.savefig('GAN/Results_gan/model{}/generated_fold{}_no_{}.png'.format(args.timestamp, fold, n + 1),
                            format='png')
        plt.close()


if __name__ == "__main__":
    total_result = [[0 for j in range(5)] for i in range(4)]
    if not (os.path.isdir('GAN/Results_gan/model{}'.format(args.timestamp))):
        os.makedirs(os.path.join('GAN/Results_gan/model{}'.format(args.timestamp)))

    # single classifier_results
    for i in range(1, args.gen_no + 1):
        for f1 in range(1, 6):
            # save log
            path_save_info = 'GAN/Results_gan/model{}'.format(args.alpha,args.timestamp) + os.path.sep + "train_info{}_{}.csv".format(args.timestamp, f1)
            train_gan(f1, i)
