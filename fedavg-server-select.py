import argparse
import sys
from torch import nn

sys.path.append("/")
from fedlab.utils.logger import Logger
import src.fedlab.models as fmodels
from fedlab.models.mlp import MLP
import torch
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.network import DistNetwork
from fedlab.utils.logger import Logger

from fedlab.models.mlp import MLP

from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.network import DistNetwork
from fedavg_serverhandler import FedavgServerHandler
import src.model.net as net
import src.model.data_loader as data_loader
import src.model.resnet as resnet
import torch.optim as optim
parser = argparse.ArgumentParser(description='FL server example')

parser.add_argument('--ip', type=str)
parser.add_argument('--port', type=str)
parser.add_argument('--world_size', type=int)
parser.add_argument('--ethernet', type=str, default=None)

parser.add_argument('--round', type=int, default=3)
parser.add_argument('--sample', type=float, default=1)


args = parser.parse_args()
class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextCNN'
        self.save_path =  self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 45                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 15                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        out=F.softmax(out)
        return out


import torch.nn.functional as F
from sklearn import metrics
import time


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
config = Config()
model = Model(config)
init_network(model)
LOGGER = Logger(log_name="server")
from fedavg_handler_select import FedavgServerHandler
handler = FedavgServerHandler(model,
                                     global_round=args.round,
                                     logger=LOGGER,
                                     sample_ratio=args.sample)


network = DistNetwork(address=(args.ip, args.port),
                      world_size=args.world_size,
                      rank=0,
                      ethernet=args.ethernet)

manager_ = SynchronousServerManager(handler=handler,
                                    network=network,
                                    mode="GLOBAL",
                                    logger=LOGGER)
manager_.run()
