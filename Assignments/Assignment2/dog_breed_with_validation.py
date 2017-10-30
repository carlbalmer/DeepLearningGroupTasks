import pandas as pd
import numpy as np
import time
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models


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


IMG_PATH = 'train/'
IMG_EXT = '.jpg'
TRAIN_DATA = 'labels.csv'
epochs = 50
do_transfer_learning = False
print("Transfer learning:" + str(do_transfer_learning))

transformations = transforms.Compose([transforms.RandomSizedCrop(224), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

dog_dataset = DogsDataset(TRAIN_DATA, IMG_PATH, IMG_EXT, transformations)

# get stratified indixes
'''_, test_index = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=92748301).split(np.zeros(len(dog_dataset.y_train)), dog_dataset.y_train))
train_index, validation_index = next(StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=78547820).split(np.zeros(len(_)), dog_dataset.y_train[_]))
'''

def make_stratified_splits(dataset):
    X = dataset.X_train
    y = dataset.y_train
    test_straf = StratifiedShuffleSplit(n_splits=1, test_size= 0.2, train_size=0.8, random_state=4456)
    
    train_straf = StratifiedShuffleSplit(n_splits=1, test_size= 0.125,train_size=0.875,  random_state=58778)
    rest_index, test_index = next(test_straf.split(X, y))
    #print("rest:", X[rest_index], "\nTEST:", X[test_index])
  
    train_index, val_index =next( train_straf.split(X[rest_index], y[rest_index]))
    #print("train:", X[train_index], "\nval:", X[val_index])
    
    # we can equiv also retrn these indexes for the random sampler to do its job 
    #print(test_index,train_index,val_index)
    return (train_index,val_index,test_index)

train_index, validation_index, test_index = make_stratified_splits(dog_dataset)
# define dataloaders
train_loader = DataLoader(dog_dataset,batch_size=50,
                          sampler=SubsetRandomSampler(train_index),
                          num_workers=1, pin_memory=True)
validation_loader = DataLoader(dog_dataset,batch_size=50, sampler=SubsetRandomSampler(validation_index), num_workers=1, pin_memory=True)
test_loader = DataLoader(dog_dataset,batch_size=50, sampler=SubsetRandomSampler(test_index), num_workers=1, pin_memory=True)

# create models and change fc layer
#alexnet_pretrained = models.alexnet(pretrained=True)
#num_ftrs = alexnet_pretrained.classifier._modules['6'].in_features
#alexnet_pretrained.classifier._modules['6'] = nn.Linear(num_ftrs, 120)

#alexnet = models.alexnet(num_classes=120)

#model = alexnet

#if do_transfer_learning:
#    model = alexnet_pretrained

# make the net
model = torchvision.models.resnet18(pretrained=do_transfer_learning)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)

model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)



model = model.cuda()

# define loss function, etc.

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001)

best_acc = 0
for epoch in range(epochs):
    model.train(True)
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data.cuda()), Variable(target.long().cuda())

        optimizer.zero_grad()
        output = model(data)
        # print( output.size(), target.data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(output.data, 1)
        running_corrects += torch.sum(preds == target.data)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

    epoch_acc = running_corrects / len(train_index)
    print('{} Acc: {:.4f}'.format(
        'train',  epoch_acc))

    model.train(False)
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(validation_loader):
        data, target = Variable(data.cuda()), Variable(target.long().cuda())
        output = model(data)

        _, preds = torch.max(output.data, 1)
        running_corrects += torch.sum(preds == target.data)

    epoch_acc = running_corrects / len(validation_index)
    print('{} Acc: {:.4f}'.format(
        'valid', epoch_acc))
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = model.state_dict()

print('Best val Acc: {:4f}'.format(best_acc))
model.load_state_dict(best_model_wts)

model.train(False)
running_corrects = 0
for batch_idx, (data, target) in enumerate(test_loader):
    data, target = Variable(data.cuda()), Variable(target.long().cuda())
    output = model(data)

    _, preds = torch.max(output.data, 1)
    running_corrects += torch.sum(preds == target.data)

test_acc = running_corrects / len(test_index)

print('Test Acc: {:4f}'.format(test_acc))
