import sys
sys.path.append("../")

from copy import deepcopy
import torch
from fedlab.core.client.trainer import ClientTrainer, SerialClientTrainer
from fedlab.utils import Logger, SerializationTool
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset
class SGDClientTrainer(ClientTrainer):
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool, optional): use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): :object of :class:`Logger`.
    """
    def __init__(self,
                 model:torch.nn.Module,
                 cuda:bool=False,
                 device:str=None,
                 logger:Logger=None):
        super(SGDClientTrainer, self).__init__(model, cuda, device)
        self.batch_size=128
        self.logits = None
        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """

        return self.logits


    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size.
            lr (float): Learning rate.
        """
        pass

    def local_process(self, payload, id):
        model_parameters = payload[0]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)

    def train(self, model_parameters, train_loader) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        labels = [i for i in range(45)]
        # self.logits=[]
        # for ep in range(self.epochs):
        #     self._model.train()
        #     for data, target in train_loader:
        #         if self.cuda:
        #             data, target = data.cuda(self.device), target.cuda(self.device)
        #
        #         outputs = self._model(data)
        #         loss = self.criterion(outputs, target)
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()
        # #   跑遍整个该节点的生成数据集来收集logits信息，传输logits及对应的编号给中心节点，由于生成数据集已经发给中心节点
        # #         因此中心节点只用知道编号就能知道所用哪个数据。

        # produced_datasets=
        # # 返回logits及对应的编号
        # logits,no=produce_logits()
        # self.logits.append(logits)
        self.logits = {}
        def train(config, model, train_iter):
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

            # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            total_batch = 0  # 记录进行到多少batch
            dev_best_loss = float('inf')
            last_improve = 0  # 记录上次验证集loss下降的batch数
            flag = False  # 记录是否很久没有效果提升
            data_size=0
            for epoch in range(config.num_epochs):
                print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
                # scheduler.step() # 学习率衰减
                for i, (trains, labels) in enumerate(train_iter):
                    data_size+=len(labels)
                    outputs = model(trains)
                    model.zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    # loss = nn.KLDivLoss()(F.log_softmax(outputs, dim=1), F.softmax(teacher_output, dim=1)) * (
                    #     0.5) + F.cross_entropy(outputs, labels) * (1. - 0.5)
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
            return data_size,model

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

        def evaluate(config, model, data_iter, test=False):
            model.eval()
            loss_total = 0
            predict_all = np.array([], dtype=int)
            labels_all = np.array([], dtype=int)
            with torch.no_grad():
                for texts, target in data_iter:
                    outputs = model(texts)
                    loss = F.cross_entropy(outputs, target)
                    loss_total += loss
                    target = target.data.cpu().numpy()
                    predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                    labels_all = np.append(labels_all, target)
                    predict_all = np.append(predict_all, predic)
                    for i in range(min(self.batch_size, target.shape[0])):
                        try:
                            if target[i].item() in self.logits:
                                self.logits[target[i].item()] = (outputs[i] + self.logits[target[i].item()]) / 2
                            else:
                                self.logits[target[i].item()] = outputs[i,]
                        except RuntimeError:
                            print("error")
                self.logits = [self.logits[i] if i in self.logits else torch.tensor(0) for i in labels]
                self.logits.append(self.model_parameters)

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
        train(config, self._model, train_loader)
        test_data = torch.load('/src/data/test/test_db.pth')
        dataloader = DataLoader(test_data, batch_size=128)
        test(config, self._model, dataloader)

        # for ep in range(self.epochs):
        #     self._model.train()
        #     for data, target in train_loader:
        #         if self.cuda:
        #             data, target = data.cuda(self.device), target.cuda(self.device)
        #
        #         outputs = self._model(data)
        #         loss = self.criterion(outputs, target)
        #
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()
        #
        #         for i in range(min(self.batch_size,target.shape[0])):
        #             try:
        #                 if target[i].item() in self.logits:
        #                     self.logits[target[i].item()]=(outputs[i]+self.logits[target[i].item()])/2
        #                 else:
        #                     self.logits[target[i].item()]=outputs[i,]
        #             except RuntimeError:
        #                 print("error")
        # self.logits=[self.logits[i] if i in self.logits else torch.tensor(0) for i in labels ]
        self._LOGGER.info("Local train procedure is finished")


class SGDSerialClientTrainer(SerialClientTrainer):
    """
    Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        num (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): Object of :class:`Logger`.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """
    def __init__(self, model, num, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.chache = []

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size.
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    @property
    def uplink_package(self):
        package = deepcopy(self.chache)
        self.chache = []
        return package

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader)
            self.chache.append(pack)

    def train(self, model_parameters, train_loader):
        """

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]