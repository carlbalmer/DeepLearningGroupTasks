import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from torch import np # Torch wrapper for Numpy

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.autograd import Variable

from sklearn.preprocessing import LabelEncoder

import numpy as np
import torchvision


# downloaded the data to my computer as i could not find it on the server

dog_images = []
dog_image_ids = []
labels = {}

'''# load data and split it into train,validation and test
#for file in os.listdir(train_path):
 #   img = Image.open(train_path+"/"+file)
  #  featurevector = numpy.array(img).flatten()
   # dog_images.append(featurevector)
   # dog_image_ids.append(os.path.splitext(file)[0])

reader = csv.reader(open(label_path))
for row in reader:
    labels[row[0]] = row[1]

X_train,X_test,Y_train,Y_test = train_test_split(dog_images,dog_image_ids,train_size=0.8,test_size=0.2,random_state=42)
X_train_2,X_val,Y_train_2,Y_val = train_test_split(X_train,Y_train, train_size= 0.875, test_size=0.125,random_state=42)
'''

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
        #print(tmp_df)
       # assert tmp_df['breed'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
#"Some images referenced in the CSV file were not found"
        
        self.mlb = LabelEncoder()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['id']
        self.y_train = self.mlb.fit_transform(tmp_df['breed'])#having problem shaping it in the right size
    
        #print(self.y_train[0])
        

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

dset_train = DogsDataset(TRAIN_DATA,IMG_PATH,IMG_EXT,transformations)

#split dset using kfold here and make multiple loaders




train_loader = DataLoader(dset_train,
                          batch_size=50,
                          shuffle=True,
                          #num_workers=4 # 1 for CUDA
                         # pin_memory=True # CUDA only
                         )
#make the net
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
optimizer = optim.SGD(model.fc.parameters(), lr= 1e-3, momentum=0.9)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        data, target = Variable(data.cuda()), Variable(target.long().cuda())
        
        optimizer.zero_grad()
        output = model(data)
        #print( output.size(), target.data)
        loss = criterion(output, target)        
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            
            
for epoch in range(1, 25):
    train(epoch)
