import torch
import random
from copy import deepcopy
import src
from typing import List
from fedlab.utils import Logger, Aggregators, SerializationTool
from fedlab.core.server.handler import ServerHandler
from src.fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from torch.utils.data import DataLoader

class SyncServerHandler(ServerHandler):
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
        super(SyncServerHandler, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger
        assert sample_ratio >= 0.0 and sample_ratio <= 1.0

        # basic setting
        self.client_num = 2
        # self.client_num = 4
        self.sample_ratio = sample_ratio

        # client buffer
        self.client_buffer_cache = []

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
        selection = random.sample(range(self.client_num),
                                  self.client_num_per_round)
        return sorted(selection)

    def global_update(self, buffer):
        dataloader=self.getgenerateddataset()
        labels=[i for i in range(45)]
        self._model= Aggregators.kd_aggregate(buffer,dataloader,self._model,labels)


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
    def getgenerateddataset(self):
        dataset = torch.load('/src/data/train_db.pth')
        dataloader = DataLoader(dataset, batch_size=128)
        return dataloader