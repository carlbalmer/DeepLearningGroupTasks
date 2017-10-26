import csv

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset


class DogDataset(Dataset):
    def __init__(self, breeds, filenames, root_dir, transform=None):
        self.breeds = breeds
        self.filenames = filenames
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.breeds)

    def __getitem__(self, idx):
        image = Image.open(self.root_dir + '/' + IDs[idx] + '.jpg')
        target = self.breeds[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    test_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'test' and epoch != num_epochs-1:
                break;
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects / len(datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            if phase == 'test':
                test_acc = epoch_acc

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    if report_test_acc:
        print('test Acc: {:4f}'.format(test_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# parameters

batch_size = 5
workers = 8
epochs = 50

use_gpu = torch.cuda.is_available()
do_transfer_learning = False
report_test_acc = False

# downloaded the data to my computer as i could not find it on the server
image_path = 'Data/dogs/'
label_path = 'Data/labels.csv'

labels = []
IDs = []

# load labels
with open(label_path) as csv_file:
    for rows in csv.reader(csv_file):
        IDs.append(rows[0])
        labels.append(rows[1])
del IDs[0], labels[0]
IDs = np.array(IDs)

# encode labels into integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# stratify and split the dataset
_, test_index = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=92748301).split(np.zeros(len(labels)), labels))
train_index, validation_index = next(
    StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=78547820).split(np.zeros(len(_)), labels[_]))

transformations = transforms.Compose([transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# create datasets
train_dataset = DogDataset(breeds=labels[train_index], filenames=IDs[train_index], root_dir=image_path, transform=transformations)
validation_dataset = DogDataset(breeds=labels[validation_index], filenames=IDs[validation_index], root_dir=image_path, transform=transformations)
test_dataset = DogDataset(breeds=labels[test_index], filenames=IDs[test_index], root_dir=image_path, transform=transformations)

datasets = {'train': train_dataset, 'val': validation_dataset, 'test':test_dataset}

# create loaders

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

dataloders = {'train': train_loader, 'val': validation_loader, 'test': test_loader}

# create models and change fc layer
alexnet_pretrained = models.alexnet(pretrained=True)
num_ftrs = alexnet_pretrained.classifier._modules['6'].in_features
alexnet_pretrained.classifier._modules['6'] = nn.Linear(num_ftrs, 120)

alexnet = models.alexnet()
alexnet.classifier._modules['6'] = nn.Linear(num_ftrs, 120)

model = alexnet

if do_transfer_learning:
    model = alexnet_pretrained

if use_gpu:
    model = model.cuda()

# define loss function, etc.

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=epochs)

