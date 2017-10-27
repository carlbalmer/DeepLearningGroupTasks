import os

import pandas as pd
import torch.optim as optim
import torchvision
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

IMG_PATH = 'dogs/train/'
IMG_EXT = '.jpg'
TRAIN_DATA = 'dogs/labels.csv'


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
        assert tmp_df['breed'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
            "Some images referenced in the CSV file were not found"

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

dset = DogsDataset(TRAIN_DATA, IMG_PATH, IMG_EXT, transformations)


def make_stratified_splits(D_in: DogsDataset):
    X = D_in.X_train
    y = D_in.y_train
    test_straf = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=4456)

    train_straf = StratifiedShuffleSplit(n_splits=1, test_size=0.125, train_size=0.875, random_state=58778)
    rest_index, test_index = next(test_straf.split(X, y))
    train_index, val_index = next(train_straf.split(X[rest_index], y[rest_index]))
    return train_index, val_index, test_index


train_index, val_index, test_index = make_stratified_splits(dset)
# define dataloaders
train_loader = DataLoader(dset, batch_size=50, sampler=SubsetRandomSampler(train_index))
validation_loader = DataLoader(dset, batch_size=50, sampler=SubsetRandomSampler(val_index))
test_loader = DataLoader(dset, batch_size=50, sampler=SubsetRandomSampler(test_index))

# make the net
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)
model = model.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer = optim.SGD(model.fc.parameters(), lr=1e-3, momentum=0.9)


def train(epoch):
    model.train()
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
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


for epoch in range(1, 25):
    train(epoch)
