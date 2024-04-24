

from src.fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from torch.utils.data import DataLoader
def getcommondataset():
    dataset = PathologicalMNIST(root='../../datasets/mnist/',
                                path="../datasets/mnist/",
                                num_clients=100)
    dataloader = DataLoader(dataset.get_com_datasets(),batch_size=128)
    return dataloader