import torch
from torch.utils.data import DataLoader, TensorDataset
dataset=torch.load("./datasets/new/3client0")
# data1, data2 = torch.utils.data.random_split(dataset, [int(0.5*len(dataset)), len(dataset)-int(0.5*len(dataset))])
# torch.save(data1,'./datasets/new/data0.pkl')
# torch.save(data2,'./datasets/new/data1.pkl')
dataloader=DataLoader(dataset,batch_size=128)
label_index=[i for i in range(45)]
dic={}
for i, (trains, labels) in enumerate(dataloader):
    for j in range(min(128,labels.shape[0])):
        if labels[j].item() in dic:
            dic[labels[j].item()].append(trains[j])
        else:
            dic[labels[j].item()]=[]
            dic[labels[j].item()].append(trains[j])
for label in label_index:
    temp=torch.stack(dic[label])
    torch.save(temp,"./datasets/new/class{}".format(label))