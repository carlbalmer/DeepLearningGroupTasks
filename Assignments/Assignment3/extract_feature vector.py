
# define model
import torch
import torchvision
from torch import nn

model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)

model = model.cuda()

#load weights
model.load_state_dict(torch.load('transfer_model.pt'))

print(model)