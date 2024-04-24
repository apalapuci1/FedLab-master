import torch
import random
from copy import deepcopy
import src
from typing import List
from fedlab.utils import Logger, Aggregators, SerializationTool
from fedlab.core.server.handler import ServerHandler
from src.fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset

class FednovaServerHandler(ServerHandler):
    """Synchronous Parameter Server Handler.

    Backend of synchronous parameter server: this class is responsible for backend computing in synchronous server.

    Synchronous parameter server will wait for every client to finish local training process before
    the next FL round.

    Details in paper: http://proceedings.mlr.press/v54/mcmahan17a.html

    Args:
        model (torch.nn.Module): Model used in this federation.
        global_round (int): stop condition. Shut down FL system when global round is reached.
        sample_ratio (float): The result of ``sample_ratio * client_num`` is the number of clients for every FL round.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None. If device is None and cuda is True, FedLab will set the gpu with the largest memory as default.
        logger (Logger, optional): object of :class:`Logger`.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 global_round: int,
                 sample_ratio: float,
                 cuda: bool = False,
                 device:str=None,
                 logger: Logger = None):
        super(FednovaServerHandler, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger
        assert sample_ratio >= 0.0 and sample_ratio <= 1.0

        # basic setting
        self.client_num = 2
        # self.client_num = 4
        self.sample_ratio = sample_ratio

        # client buffer
        self.client_buffer_cache = []
        option = "weighted_scale"
        self.option = option  # weighted_scale, uniform, weighted_com
        # stop condition
        self.global_round = global_round
        self.round = 0

    @property
    def downlink_package(self) -> List[torch.Tensor]:
        """Property for manager layer. Server manager will call this property when activates clients."""
        return [self.model_parameters]

    @property
    def if_stop(self):
        """:class:`NetworkManager` keeps monitoring this attribute, and it will stop all related processes and threads when ``True`` returned."""
        return self.round >= self.global_round

    @property
    def client_num_per_round(self):
        return max(1, int(self.sample_ratio * self.client_num))

    def sample_clients(self):
        """Return a list of client rank indices selected randomly. The client ID is from ``0`` to
        ``self.client_num -1``."""
        self.client_num=self.client_num_per_round
        selection = random.sample(range(self.client_num),
                                  self.client_num_per_round)
        return sorted(selection)

    def global_update(self, buffer):
        models = [elem[0] for elem in buffer]
        taus = [elem[1] for elem in buffer]

        deltas = [(model - self.model_parameters) / tau for model, tau in zip(models, taus)]

        # p is the FedAvg weight, we simply set it 1/m here.
        p = [
            1.0 / self.num_clients_per_round
            for _ in range(self.num_clients_per_round)
        ]

        if self.option == 'weighted_scale':
            K = len(deltas)
            N = self.num_clients
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = sum([dk * pk
                         for dk, pk in zip(deltas, p)]) * N / K

        elif self.option == 'uniform':
            tau_eff = 1.0 * sum(taus) / len(deltas)
            delta = Aggregators.fedavg_aggregate(deltas)

        elif self.option == 'weighted_com':
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = sum([dk * pk for dk, pk in zip(deltas, p)])

        else:
            sump = sum(p)
            p = [pk / sump for pk in p]
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = sum([dk * pk for dk, pk in zip(deltas, p)])

        self.set_model(self.model_parameters + tau_eff * delta)

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
        test_data = torch.load('/src/data/test/test_db.pth')
        dataloader = DataLoader(test_data, batch_size=128)
        test(config, self._model, dataloader)


    def load(self, payload: List[torch.Tensor]) -> bool:
        """Update global model with collected parameters from clients.

        Note:
            Server handler will call this method when its ``client_buffer_cache`` is full. User can
            overwrite the strategy of aggregation to apply on :attr:`model_parameters_list`, and
            use :meth:`SerializationTool.deserialize_model` to load serialized parameters after
            aggregation into :attr:`self._model`.

        Args:
            payload (list[torch.Tensor]): A list of tensors passed by manager layer.
        """
        assert len(payload) > 0
        self.client_buffer_cache.append(deepcopy(payload))

        assert len(self.client_buffer_cache) <= self.client_num_per_round

        if len(self.client_buffer_cache) == self.client_num_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            # reset cache
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False
    # def getgenerateddataset(self):
    #     dataset = torch.load('/home/admin/github/src/data/train_db.pth')
    #     dataloader = DataLoader(dataset, batch_size=128)
    #     return dataloader