import torch.nn as nn
import Maxout as Maxout
import torch.nn.functional as F



class DiabeticModel(nn.Module):
    def __init__(self):
        super(DiabeticModel, self).__init__()

        self.convblock1 = nn.Sequential()
        self.convblock1.add_module("conv1",nn.Conv2d(1, 16, 5, padding=1,stride=2))
        self.convblock1.add_module("relu_1", nn.LeakyReLU)
        self.convblock1.add_module("conv2",nn.Conv2d(16,16,3,padding=1,stride=1))
        self.convblock1.add_module("relu_2", nn.LeakyReLU)
        self.convblock1.add_module("max-pool1",nn.MaxPool2d(kernel_size=2))
        self.convblock1.add_module("relu_3", nn.LeakyReLU)
        self.convblock1.add_module("dropout1",nn.Dropout(p=0.1))

        self.convblock2 = nn.Sequential()
        self.convblock2.add_module("conv3",nn.Conv2d(16,32,3,padding=1,stride=2))
        self.convblock2.add_module("relu_4", nn.LeakyReLU)
        self.convblock2.add_module("conv4",nn.Conv2d(32,32,3,padding=1,stride=1))
        self.convblock2.add_module("relu_5", nn.LeakyReLU)
        self.convblock2.add_module("max-pool2",nn.MaxPool2d(kernel_size=2))
        self.convblock2.add_module("relu_6", nn.LeakyReLU)
        self.convblock2.add_module("dropout2",nn.Dropout(p=0.2))

        self.convblock3 = nn.Sequential()
        self.convblock3.add_module("conv5",nn.Conv2d(32,48,3,padding=1,stride=1))
        self.convblock3.add_module("relu_8", nn.LeakyReLU)
        self.convblock3.add_module("conv6", nn.Conv2d(48, 48, 3, padding=1, stride=1))
        self.convblock3.add_module("relu_9", nn.LeakyReLU)
        self.convblock3.add_module("conv7", nn.Conv2d(48, 48, 3, padding=1, stride=1))
        self.convblock3.add_module("relu_9", nn.LeakyReLU)
        self.convblock3.add_module("max-pool3",nn.MaxPool2d(kernel_size=2))
        self.convblock3.add_module("relu_10", nn.LeakyReLU)
        self.convblock3.add_module(("dropout3"),nn.Dropout(p=0.3))

        self.convblock4 = nn.Sequential()
        self.convblock4.add_module("conv8",nn.Conv2d(48,64,3,padding=1,stride=1))
        self.convblock4.add_module("relu_11", nn.LeakyReLU)
        self.convblock4.add_module("conv9", nn.Conv2d(64, 64, 3, padding=1, stride=1))
        self.convblock4.add_module("relu_12", nn.LeakyReLU)
        self.convblock4.add_module("conv10", nn.Conv2d(64, 64, 3, padding=1, stride=1))
        self.convblock4.add_module("relu_13", nn.LeakyReLU)
        self.convblock4.add_module("max-pool4",nn.MaxPool2d(kernel_size=2))
        self.convblock4.add_module("relu_14", nn.LeakyReLU)
        self.convblock4.add_module("dropout4",nn.Dropout(p=0.4))

        self.convblock5 = nn.Sequential()
        self.convblock5.add_module(("conv11",nn.Conv2d(64,128,3,padding=1,stride=1)))
        self.convblock5.add_module("relu_15", nn.LeakyReLU)
        self.convblock5.add_module(("conv12", nn.Conv2d(128, 128, 3, padding=1, stride=1)))
        self.convblock5.add_module("relu_16", nn.LeakyReLU)
        self.convblock5.add_module("max-pool5",nn.MaxPool2d(kernel_size=2))
        self.convblock5.add_module("relu_17", nn.LeakyReLU)
        self.convblock5.add_module("dropout5",nn.Dropout(p=0.5))

        self.fc = nn.Sequential()
        self.fc.add_module("fc1",nn.Linear(2048,400))
        self.fc.add_module("maxout",Maxout(400,200,2))
        self.fc.add_module("dropout6",nn.Dropout(p=0.5))
        self.fc.add_module("fc2",nn.Linear(400,400))
        self.fc.add_module("maxout", Maxout(400, 200, 2))
        self.fc.add_module("fc3", nn.Linear(400, 5))



    def forward(self, x):
         x = self.convblock1.forward(x)
         x = self.convblock2.forward(x)
         x = self.convblock3.forward(x)
         x = self.convblock4.forward(x)
         x = self.convblock5.forward(x)
         x = x.view(-1,2048)
         x = self.fc.forward(x)
         return F.log_softmax(x)



