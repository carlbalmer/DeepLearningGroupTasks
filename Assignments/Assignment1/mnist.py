from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # input is 28x28
        # padding=0 for valid conv
        self.conv1 = nn.Conv2d(1, 64, 3, padding=0,stride=2)
        # feature map size is 13*13 after conv
        # padding=1 for same padding(if kernel is 2*2)
        #meaning 3 shd be 2?
        self.conv2 = nn.Conv2d(64, 32, 2, padding=1,stride=1)
        # feature map size is 12*12 by pooling -> (13-2+2*1) = 13*13 which is the effect of same
        #meaning fc1 shoud be built on 13*13?
        self.fc1 = nn.Linear(32 * 13 * 13, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x= F.max_pool2d(x,2,stride=1)
        x = F.dropout(x,p=0.35)
        x = x.view(-1,5408)  # reshape Variable

        x = F.dropout(self.fc1(x),p=0.5)
        x = self.fc2(x)
        return F.log_softmax(x)


model = MnistModel() if not torch.cuda.is_available() else MnistModel().cuda() # use cuda if available


batch_size = 50
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=1000)


for p in model.parameters():
    print(p.size())


# optimizers = ["op1","op2","op3"]
# foreach op in ops (so is all in one place)
# optimizer = optim.Adadelta(model.parameters(), lr=0.01)
optimizers = { 'SGD': optim.SGD(model.parameters(), lr=0.01)}

for name, optimizer in optimizers.items():
    model.train()
    train_loss = []
    train_accu = []
    i = 0
    for epoch in range(10):
        for data, target in train_loader:
            data, target = (Variable(data), Variable(target)) if not torch.cuda.is_available() else (Variable(data).cuda(), Variable(target).cuda()) # use cuda if available
            optimizer.zero_grad()
            output = model(data)

            loss = F.cross_entropy(output, target)
            loss.backward()    # calc gradients
            train_loss.append(loss.data[0])
            optimizer.step()   # update gradients
            prediction = output.data.max(1)[1]   # first column has actual prob.

            accuracy = (float(prediction.eq(target.data).sum())/batch_size)
            train_accu.append(accuracy)
            if i % 1000 == 0:
                print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}\tOptimizer: {}'.format(i, loss.data[0], accuracy, name))
            i += 1

    plt.plot(np.arange(len(train_loss)), train_loss, label='loss')
    plt.plot(np.arange(len(train_accu)), train_accu, label='accuracy')
    plt.legend()
    plt.title(name)
    plt.ylabel('loss/accuracy')
    plt.xlabel('epoch')
    plt.savefig(name+str(randint(0,1000))+'.png', bbox_inches='tight')
    plt.close()

# end of foreach
