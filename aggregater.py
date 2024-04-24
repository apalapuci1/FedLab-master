# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)
import csv
import heapq

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset
import src.fedlab.utils.serialization
import src.model.net as net
import src.model.data_loader as data_loader
import src.model.resnet as resnet
import src.model.wrn as wrn
import src.model.densenet as densenet
import src.model.resnext as resnext
import src.model.preresnet as preresnet
import src.fedlab.models as fmodels


class Aggregators(object):
    """Define the algorithm of parameters aggregation"""

    def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
        T = temperature
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (
                    alpha * T * T) + F.cross_entropy(outputs, labels) * (1. - alpha)
        return KD_loss

    @staticmethod
    def fedavg_aggregate(serialized_params_list, weights=None):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        gradient_list = []
        var_list = []
        for enum in serialized_params_list:
            gradient_list.append(enum[-1])
            enum.pop()
            sum=0
            for e in enum:
                sum+=e.var()
            var_list.append(sum)
        #     top-k
        k = 2
        largest_k = heapq.nlargest(k, var_list)
        indices = {value: index for index, value in enumerate(var_list) if value in largest_k}
        indices = [indices[value] for value in largest_k]
        serialized_params_list=[]
        for i in indices:
            serialized_params_list.append(gradient_list[i])
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * weights, dim=-1)
        return serialized_parameters

    @staticmethod
    def fedasync_aggregate(server_param, new_param, alpha):
        """FedAsync aggregator

        Paper: https://arxiv.org/abs/1903.03934
        """
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, new_param)
        return serialized_parameters

    @staticmethod
    # labels=[0,1,2,3,4,5,6,7,8,9]
    def kd_aggregate(serialized_params_list, dataloader, model, labelss) -> torch.nn.Module:

        # set model to training mode
        model.train()
        # compute avgloss for all clients loss
        avg_loss = {}
        gradient_list=[]
        var_list=[]
        for enum in serialized_params_list:
            gradient_list.append(enum[-1])
            enum.pop()
            sum = 0
            for e in enum:
                sum += e.var()
            var_list.append(sum)
        #     top-k
        k=2
        largest_k = heapq.nlargest(k,var_list)
        with open('gongxianliang.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(var_list)
        indices = {value: index for index, value in enumerate(var_list) if value in largest_k}
        indices = [indices[value] for value in largest_k]
        for label in labelss:
            sum = torch.zeros(len(labelss))
            i=0
            for enum in serialized_params_list:
                if i in indices:
                    sum += enum[label]
                    i+=1
            avg_loss[label] = sum / len(serialized_params_list)


        # summary for current training loop and a running average object for loss
        # summ = []
        # loss_avg = utils.utils_kd.RunningAverage()

        # Use tqdm for progress bar
        def train(config, model, train_iter):
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

            # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            total_batch = 0  # 记录进行到多少batch
            dev_best_loss = float('inf')
            last_improve = 0  # 记录上次验证集loss下降的batch数
            flag = False  # 记录是否很久没有效果提升
            for epoch in range(config.num_epochs):
                print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
                # scheduler.step() # 学习率衰减
                for i, (trains, labels) in enumerate(train_iter):
                    outputs = model(trains)
                    model.zero_grad()
                    teacher_output = []
                    for label in labels:
                        teacher_output.append(avg_loss[label.item()])
                    teacher_output = torch.stack(teacher_output)
                    # loss = F.cross_entropy(outputs, labels)
                    loss = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_output, dim=1)) * (
                        0.5) + F.cross_entropy(outputs, labels) * (1. - 0.5)
                    loss.backward()
                    optimizer.step()
                    if total_batch % 100 == 0:
                        # 每多少轮输出在训练集和验证集上的效果
                        true = labels.data.cpu()
                        predic = torch.max(outputs.data, 1)[1].cpu()
                        train_acc = metrics.accuracy_score(true, predic)
                        print(total_batch, "              ", train_acc)
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                        model.train()
                    total_batch += 1
                    if total_batch - last_improve > config.require_improvement:
                        # 验证集loss超过1000batch没下降，结束训练
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break
                if flag:
                    break

        def test(config, model, test_iter):
            # test
            model.eval()

            test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
            msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
            print(msg.format(test_loss, test_acc))
            print("Precision, Recall and F1-Score...")
            print(test_report)
            print("Confusion Matrix...")
            print(test_confusion)
            with open('result_acc.csv','a')as file:
                writer = csv.writer(file)
                writer.writerow([test_acc,test_loss])

        def evaluate(config, model, data_iter, test=False):
            model.eval()
            loss_total = 0
            predict_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)
            with torch.no_grad():
                for texts, labels in data_iter:
                    outputs = model(texts)
                    loss = F.cross_entropy(outputs, labels)
                    loss_total += loss
                    labels = labels.data.cpu().numpy()
                    predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                    labels_all = np.append(labels_all, labels)
                    predict_all = np.append(predict_all, predic)

            acc = metrics.accuracy_score(labels_all, predict_all)
            if test:
                report = metrics.classification_report(labels_all, predict_all, digits=4)
                confusion = metrics.confusion_matrix(labels_all, predict_all)
                return acc, loss_total / len(data_iter), report, confusion
            return acc, loss_total / len(data_iter)

        class Config(object):

            """配置参数"""

            def __init__(self):
                self.model_name = 'TextCNN'
                self.save_path = self.model_name + '.ckpt'  # 模型训练结果
                self.log_path = self.model_name
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

                self.dropout = 0.5  # 随机失活
                self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
                self.num_classes = 45  # 类别数
                self.n_vocab = 0  # 词表大小，在运行时赋值
                self.num_epochs = 5  # epoch数
                self.batch_size = 128  # mini-batch大小
                self.pad_size = 15  # 每句话处理成的长度(短填长切)
                self.learning_rate = 1e-3  # 学习率
                self.embed = 300  # 字向量维度
                self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
                self.num_filters = 256  # 卷积核数量(channels数)

        config = Config()
        train(config, model, dataloader)
        test_data = torch.load('/src/data/test/test_db.pth')
        dataloader = DataLoader(test_data, batch_size=128)
        test(config, model, dataloader)
        return model




