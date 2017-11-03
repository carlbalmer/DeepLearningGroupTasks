import csv
import pickle

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from dataset_helpers import DogsDataset, make_stratified_splits
import torch
import torchvision
from torch import nn

# set env
IMG_PATH = 'train/'
IMG_EXT = '.jpg'
TRAIN_DATA = 'labels.csv'

# load dataset
dog_dataset = DogsDataset(TRAIN_DATA, IMG_PATH, IMG_EXT, transforms.Compose([transforms.RandomSizedCrop(224),
                                                                             transforms.ToTensor(),
                                                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                                                  [0.229, 0.224, 0.225])
                                                                             ]))
train_index, val_index, test_index = make_stratified_splits(dog_dataset)
test_loader = DataLoader(dog_dataset, batch_size=1, sampler=SubsetRandomSampler(test_index), num_workers=1)

# define model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)
model = model.cuda()

# load weights
model.load_state_dict(torch.load('transfer_model.pt'))
model.eval()

feature_vectors = []
nbr_correct = 0
nbr_incorrect = 0
for data in test_loader:
    image, target = Variable(data[0].cuda()), Variable(data[1].cuda())
    output = model(image)

    if nbr_incorrect == nbr_correct >= 10:
        break
    elif (torch.max(output.data, 1)[1] == target.data).all() and nbr_correct < 10:
        nbr_correct += 1
    elif not (torch.max(output.data, 1)[1] == target.data).all() and nbr_incorrect < 10:
        nbr_incorrect += 1
    else:
        continue

    feature_vectors.append({'features': output.data.cpu().numpy(),
                            "encoded_label": target.data.cpu().cpu().numpy(),
                            "text_label": dog_dataset.mlb.inverse_transform(target.data.cpu().cpu().numpy()),
                            "correctly_classified": (torch.max(output.data, 1)[1] == target.data).all(),
                            })

pickle.dump(feature_vectors, open("feature_vectors.pkl",'wb'))
# load with  feature_vectors = pickle.load(open("feature_vectors.pkl", 'rb'))
