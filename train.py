import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from src.fedlab.contrib.dataset.pathological_mnist import PathologicalMNIST
from torch.utils.data import DataLoader

import src.fedlab.models as fmodels
learning_rate=0.01
model = fmodels.cnn.CNN_MNIST().cuda() if torch.cuda.is_available() else fmodels.cnn.CNN_MNIST()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
dataset = PathologicalMNIST(root='../../datasets/mnist/',
                                    path="../datasets/mnist/",
                                    num_clients=100)
dataloader = DataLoader(dataset.get_com_datasets(), batch_size=128)
# if model_name == "resnet18":
#     teacher_model = resnet.ResNet18()
# teacher_checkpoint = 'experiments/base_resnet18/best.pth.tar'
# teacher_model = teacher_model.cuda() if params.cuda else teacher_model

# elif model_name == "wrn":
#     teacher_model = wrn.WideResNet(depth=28, num_classes=10, widen_factor=10,
#                                    dropRate=0.3)
#     # teacher_checkpoint = 'experiments/base_wrn/best.pth.tar'
#     # teacher_model = nn.DataParallel(teacher_model).cuda()
#
# elif model_name == "densenet":
#     teacher_model = densenet.DenseNet(depth=100, growthRate=12)
#     # teacher_checkpoint = 'experiments/base_densenet/best.pth.tar'
#     # teacher_model = nn.DataParallel(teacher_model).cuda()
#
# elif model_name == "resnext29":
#     teacher_model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=10)
#     # teacher_checkpoint = 'experiments/base_resnext29/best.pth.tar'
#     # teacher_model = nn.DataParallel(teacher_model).cuda()
#
# elif model_name == "preresnet110":
#     teacher_model = preresnet.PreResNet(depth=110, num_classes=10)
#     # teacher_checkpoint = 'experiments/base_preresnet110/best.pth.tar'
#     # teacher_model = nn.DataParallel(teacher_model).cuda()
# elif model_name == "CNNCifar":
#     teacher_model = CNNCifar(args=args)

# set model to training mode
model.train()
# compute avgloss for all clients loss

# summary for current training loop and a running average object for loss
# summ = []
# loss_avg = utils.utils_kd.RunningAverage()

# Use tqdm for progress bar
for epoch in range(30):
    for i, (train_batch, labels_batch) in enumerate(dataloader):
        # move to GPU if available
        if torch.cuda.is_available():
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
        # # convert to torch Variables
        # train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        # compute model output, fetch teacher output, and compute KD loss
        output_batch = model(train_batch)

        # get one batch output from teacher_outputs list
        loss = F.cross_entropy(output_batch, labels_batch)
        if i % 100 == 0:
            print(loss)
        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()
        # performs updates using calculated gradients
        optimizer.step()
torch.save(model.state_dict(), 'aggregated.pth')
print('model saved')