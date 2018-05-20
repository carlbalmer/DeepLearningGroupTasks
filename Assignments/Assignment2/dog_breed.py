import pandas as pd
from torch import np  # Torch wrapper for Numpy

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import numpy as np
import torchvision
from torchvision import datasets, models

IMG_PATH = 'dogs/train/'
IMG_EXT = '.jpg'
TRAIN_LABELS = 'dogs/labels.csv'


class DogsDataset(Dataset):
    """Dataset
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path, img_ext, transform=None):
        tmp_df = pd.read_csv(csv_path)
        # print(tmp_df)
        # assert tmp_df['breed'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
        # "Some images referenced in the CSV file were not found"

        self.mlb = LabelEncoder()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['id']
        self.y_train = self.mlb.fit_transform(tmp_df['breed'])  # having problem shaping it in the right size

        # print(self.y_train[0])

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = (self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train.index)


transformations = transforms.Compose([
    transforms.RandomSizedCrop(224),

    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dset = DogsDataset(TRAIN_LABELS, IMG_PATH, IMG_EXT, transformations)


def make_stratified_splits(D_in: DogsDataset):
    X = D_in.X_train
    y = D_in.y_train
    test_straf = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=4456)

    train_straf = StratifiedShuffleSplit(n_splits=1, test_size=0.125, train_size=0.875, random_state=58778)
    rest_index, test_index = next(test_straf.split(X, y))
    train_index, val_index = next(train_straf.split(X[rest_index], y[rest_index]))
    return (train_index, val_index, test_index)


train_index, val_index, test_index = make_stratified_splits(dset)
print(len(train_index), '\n', len(val_index))
# define dataloaders
train_loader = DataLoader(dset, batch_size=50, sampler=SubsetRandomSampler(train_index))
val_loader = DataLoader(dset, batch_size=10, sampler=SubsetRandomSampler(val_index))
test_loader = DataLoader(dset, batch_size=50, sampler=SubsetRandomSampler(test_index))

# make the net
model = torchvision.models.resnet18(pretrained=False)
# for param in model.parameters():
#   param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)
model = model.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
# optimizer = optim.SGD(model.fc.parameters(), lr= 1e-3, momentum=0.9)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)


def train(epoch):
    best_accu = 0
    validate = False
    model.train()
    best_loss = 1.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data.cuda()), Variable(target.long().cuda())

        optimizer.zero_grad()
        output = model(data)
        # print( output.size(), target.data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_index),
                       100. * batch_idx / len(train_loader), loss.data[0]))
        if loss < best_loss:
            best_loss = loss
            validate = True
            # best_model_wts = model.state_dict()
    if (validate):
        val_loss = 0.0
        val_accu = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
            data, target = Variable(data.cuda()), Variable(target.long().cuda())

            output = model(data)
            _, preds = torch.max(output.data, 1)
            val_loss = criterion(output, target)
            val_accu += torch.sum(preds == target.data)

            # get accuraccy
            if batch_idx % 10 == 0:
                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},\tAccu: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(val_index),
                           100. * batch_idx / len(val_loader), loss.data[0], val_accu / len(val_index)))
            if val_accu > best_accu:
                best_accu = val_accu

                best_model_wts = model.state_dict()


for epoch in range(1, 25):
    train(epoch)