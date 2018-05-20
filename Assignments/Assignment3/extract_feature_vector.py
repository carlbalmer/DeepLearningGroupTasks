import csv
import pickle

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from .dataset_helpers import DogsDataset, make_stratified_splits
import torch
import torchvision
from torch import nn
import numpy as np

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
test_loader = DataLoader(dog_dataset, batch_size=1, sampler=SubsetRandomSampler(test_index))

# define model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)
model = model.cuda()

# load weights
model.load_state_dict(torch.load('transfer_model.pt'))
model.eval()

feature_vectors = []
feature_vectors.count
stuff = {}
stuff_r = {}
nbr_correct = 0
nbr_incorrect = 0
for data in test_loader: #for each data
  #for targetClass in range(5):
    image, target = Variable(data[0].cuda()), Variable(data[1].cuda())
    output = model(image)
    values,indices = torch.max(output.data.cpu(),1)
    indices = np.asscalar(indices.numpy())
    target = np.asscalar(target.data.cpu().numpy())
    #targetClass = 49 
    #prediction = (indices == targetClass).numpy()
    #gt = (target.data.cpu()==targetClass).numpy()
    #if the prediction is correct save to it's indices if the count
    #if the count is more than 10 skip
    #print('index',indices, 'data', target)
    if (target == indices):
            if indices in stuff:
                if len(stuff[indices]) < 10:
                    stuff[indices].append(values)
                    feature_vectors.append({'features': output.data.cpu().numpy(),
                            "encoded_label": target,
                            "text_label": dog_dataset.mlb.inverse_transform(target),
                            "correctly_classified": True,
                            })
                
            else:
                stuff[indices] = [values,]
    else:
        if indices in stuff_r:
                if len(stuff_r[indices]) < 10:
                    stuff_r[indices].append(values)
                    feature_vectors.append({'features': output.data.cpu().numpy(),
                            "encoded_label": target,
                            "text_label": dog_dataset.mlb.inverse_transform(target),
                            "correctly_classified": False,
                            })
                
        else:
            stuff_r[indices] = [values,]

    #print( len(stuff_r[key])  for key in stuff_r.keys())
    print( all(len(stuff_r[key]) >= 10 for key in stuff_r.keys()))
        #break;
                  
    
    '''if nbr_incorrect == nbr_correct >= 100:
        break
    elif ((prediction == gt) and prediction==1) and nbr_correct < 100:
        nbr_correct += 1
    #elif not (prediction == gt) and nbr_incorrect < 100:
        nbr_incorrect += 1
    else:
        continue'''

    '''feature_vectors.append({'features': output.data.cpu().numpy(),
                            "encoded_label": target.data.cpu().cpu().numpy(),
                            "text_label": dog_dataset.mlb.inverse_transform(target.data.cpu().cpu().numpy()),
                            "correctly_classified": ((prediction == gt) and prediction ==1),
                            })'''

pickle.dump(feature_vectors, open("feature_vectors_120.pkl",'wb'))
# load with  feature_vectors = pickle.load(open("feature_vectors.pkl", 'rb'))
