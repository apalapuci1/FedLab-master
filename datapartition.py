import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from torch.utils.data import DataLoader, TensorDataset
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import partition_report

num_clients = 4
num_classes = 45
seed = 2023
hist_color = '#4169E1'
trainset = torch.load("./datasets/new/train_db.pth")
dataloader=DataLoader(trainset,batch_size=128)
targets=[]
for i, (trains, labels) in enumerate(dataloader):
    for label in labels:
        targets.append(label.item())
# 1
# hetero_dir_part = CIFAR10Partitioner(trainset.targets,
#                                      num_clients,
#                                      balance=None,
#                                      partition="dirichlet",
#                                      dir_alpha=0.3,
#                                      seed=seed)
# 2
# num_shards = 200
# part = CIFAR10Partitioner(targets,
#                                  num_clients,
#                                  balance=None,
#                                  partition="shards",
#                                  num_shards=num_shards,
#                                  seed=seed)
#3
part = CIFAR10Partitioner(targets,
                                      num_clients,
                                      balance=True,
                                      partition="dirichlet",
                                      dir_alpha=0.3,
                                      seed=seed)
# 4
# unbalance_dir_part = CIFAR10Partitioner(trainset.targets,
#                                         num_clients,
#                                         balance=False,
#                                         partition="dirichlet",
#                                         unbalance_sgm=0.3,
#                                         dir_alpha=0.3,
#                                         seed=seed)

# hetero_dir_part.client_dict= { 0: indices of dataset,
#                                1: indices of dataset,
#                                ...
#                                100: indices of dataset }
csv_file = "cifar10_hetero_dir_0.3_100clients.csv"
partition_report(targets, part.client_dict,
                 class_num=num_classes,
                 verbose=False, file=csv_file)
hetero_dir_part_df = pd.read_csv(csv_file,header=1)
hetero_dir_part_df = hetero_dir_part_df.set_index('client')
col_names = [f"class{i}" for i in range(num_classes)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)
hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
plt.tight_layout()
plt.xlabel('sample num')
plt.savefig(f"cifar10_hetero_dir_0.5_100clients.png", dpi=400)
clt_sample_num_df = part.client_sample_count
sns.histplot(data=clt_sample_num_df,
             x="num_samples",
             edgecolor='none',
             alpha=0.7,
             shrink=0.95,
             color=hist_color)
plt.savefig(f"cifar10_hetero_dir_0.5_100clients_dist.png", dpi=400, bbox_inches = 'tight')
# num_shards = 200
# shards_part = CIFAR10Partitioner(trainset.targets,
#                                  num_clients,
#                                  balance=None,
#                                  partition="shards",
#                                  num_shards=num_shards,
#                                  seed=seed)

trainset.indices=part.client_dict[0].tolist()
torch.save(trainset,"./datasets/new/9client0")
trainset.indices=part.client_dict[1].tolist()
torch.save(trainset,"./datasets/new/9client1")
trainset.indices=part.client_dict[2].tolist()
torch.save(trainset,"./datasets/new/9client2")
trainset.indices=part.client_dict[3].tolist()
torch.save(trainset,"./datasets/new/9client3")