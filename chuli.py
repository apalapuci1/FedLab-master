import torch
from torch.utils.data import DataLoader, TensorDataset
x=torch.load('./datasets/train_db.pth')
y=torch.load('./datasets/test_db.pth')
#
# y=torch.tensor(y,dtype=int)
# y[19696]=torch.tensor(0)
# y[19697]=torch.tensor(0)
#
# dataset=TensorDataset(x,y)
# train_db, test_db = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
# torch.save(train_db,"./datasets/train_db.pth")
# torch.save(test_db,"./datasets/test_db.pth")