import torch
import argparse
from datetime import datetime

class config():
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description='Argparse GCN')
        data_keys = ["data", "label"]

        timestamp = datetime.today().strftime("_%Y%m%d_%H%M%S")

        # pytorch base
        parser.add_argument("--cuda_num", default=2, type=str)
        parser.add_argument("--device")
        parser.add_argument("--data_keys", default=data_keys, type=list, help="")

        # data
        parser.add_argument("--data_type", default ="MDD", type=str)

        # training hyperparams.
        parser.add_argument('--seed', type=int, default=500, help='random seed')
        parser.add_argument('--dislr', default=9e-05, type=float, help='discriminator learning rate')
        parser.add_argument('--genlr', default=1e-04, type=float,help='generator learning rate')
        parser.add_argument('--lr', default=5e-06, type=float, help='classifier learning rate')
        parser.add_argument("--batch_size", default=200, type=int, help="batch size")
        parser.add_argument("--gan_batch_size", default=100, type=int, help="gan_batch_size")
        parser.add_argument('--iter_size', default=2000, type=int, help='num_epoch')
        parser.add_argument('--g_weight_decay', default=5e-03,type=float, help='weight decay')
        parser.add_argument('--d_weight_decay', default=5e-03,type=float, help='weight decay')
        parser.add_argument('--c_weight_decay', default=1e-04, type=float, help='weight decay')
        parser.add_argument('--optim', default="ADAM", type=str, help='optimizer')
        parser.add_argument('--betas', default=(0.5, 0.9), type=tuple, help='adam betas')
        parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum - SGD, MADGRAD")
        parser.add_argument("--gamma", default=0.998, type=float, help="gamma for lr learning")
        parser.add_argument("--mse", default=0.995, type=float, help="gamma for lr learning")
        parser.add_argument("--info_gain_weight", default=(0.52,0.48), type=list, help="Results gain weight")  # 0 : NC   1: MDD_Data, MDD가 약간 더 많음, 안써도 될거 같음, 2021.5.6 살짝 수정 필요 383->381

        # gan
        parser.add_argument("--gen_no", default=1, type=int, help="generation iteration")
        parser.add_argument("--z_dim", default=56, type=int, help="Gaussian noise dim")
        parser.add_argument('--pseudo_size', default=1, type=float, help='pseudo_data_size')
        parser.add_argument("--alpha", default=0.9, type=float, help="confidence score")
        parser.add_argument("--load_data", default=0, type=float, help="load saved generated data")
        parser.add_argument("--p_value", default=0.1, type=float, help="t-test  pvalue")
        parser.add_argument("--edge_type", default="mrmr", type=str, help="edge type")
        parser.add_argument("--gnn_type", default="GCN", type=str, help="data type")
        parser.add_argument("--gan_type", default="semi", type=str, help="data type")
        parser.add_argument("--ex_type", default = "Main", type=str, help="experiment type")
        parser.add_argument("--site", default="S20", type=str, help="MDD site")

        # save & load
        parser.add_argument("--model_save_overwrite", default=False, type=bool, help="")
        parser.add_argument("--timestamp", default=timestamp, type=str, help="")
        parser.add_argument("--path_model", default="Model/Model{}".format(timestamp), type=str, help="path to save and to load GraphVAE")
        parser.add_argument("--model_checkpoint", default=100, type=float, help="check point epoch size")
        parser.add_argument("--train_load_model", default=False, type=bool, help="true or false to load GraphVAE during training")  # finetuing
        parser.add_argument("--train_load_model_name", default="Model/.pth", type=str, help="")
        parser.add_argument("--test_model", default="Model.pth".format(timestamp), type=str, help="")
        parser.add_argument("--path_save_info", default="Results", type=str, help="")


        self.args = parser.parse_args()

        self.cuda_num = self.args.cuda_num
        self.device = torch.device("cuda:{}".format(self.cuda_num) if torch.cuda.is_available() else "cpu")
        self.data_keys = self.args.data_keys

        self.data_type = self.args.data_type
        self.gen_no = self.args.gen_no

        self.seed = self.args.seed
        self.dislr = self.args.dislr
        self.genlr = self.args.genlr
        self.lr = self.args.lr
        self.batch_size = self.args.batch_size
        self.gan_batch_size = self.args.gan_batch_size
        self.iter_size = self.args.iter_size
        self.pseudo_size = self.args.pseudo_size
        self.g_weight_decay = self.args.g_weight_decay
        self.d_weight_decay = self.args.d_weight_decay
        self.c_weight_decay = self.args.c_weight_decay
        self.optim = self.args.optim
        self.betas = self.args.betas
        self.momentum = self.args.momentum
        self.gamma = self.args.gamma
        self.mse = self.args.mse
        self.info_gain_weight = self.args.info_gain_weight

        # GAN
        self.z_dim = self.args.z_dim
        self.alpha = self.args.alpha
        self.load_data = self.args.load_data
        self.p_value = self.args.p_value
        self.edge_type = self.args.edge_type
        self.gnn_type = self.args.gnn_type
        self.gan_type = self.args.gan_type
        self.site = self.args.site
        self.ex_type = self.args.ex_type
        self.e_channels = [112,112,56]
        self.g_channels = [56,112,112]
        self.d_channels = [112,112,224]
        self.c_channels = [112,112,224]

        self.model_save_overwrite = self.args.model_save_overwrite
        self.timestamp = self.args.timestamp
        self.path_model = self.args.path_model
        self.model_checkpoint = self.args.model_checkpoint
        self.train_load_model = self.args.train_load_model
        self.train_load_model_name = self.args.train_load_model_name
        self.test_model = self.args.test_model
        self.path_save_info = self.args.path_save_info

    def Return_args(self):
        return self.args

print(config().Return_args())
