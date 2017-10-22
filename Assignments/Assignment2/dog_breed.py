import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import sklearn


data_dir  = 'train'
# load data and split it into train,validation and test

# X_train,X_test,Y_train,Y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)
#X_train_2,X_val,Y_train_2,Y_val = train_test_split(X_train,Y_train, train_size= 0.875,random_state=42)

alexnet_pretrained = models.alexnet(pretrained=True)
alexnet = models.alexnet()


num_ftrs = alexnet_pretrained.fc.in_features
alexnet_pretrained.fc = nn.Linear(num_ftrs, 120)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(alexnet_pretrained.parameters(), lr=0.001, momentum=0.9)