import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import csv
import numpy, os
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split

# downloaded the data to my computer as i could not find it on the server
train_path = '../../dogbreeds/train/'
label_path = '../../dogbreeds/labels.csv'

dog_images = []
dog_image_ids = []
labels = {}

# load data and split it into train,validation and test
for file in os.listdir(train_path):
    img = Image.open(train_path+"/"+file)
    featurevector = numpy.array(img).flatten()
    dog_images.append(featurevector)
    dog_image_ids.append(os.path.splitext(file)[0])

reader = csv.reader(open(label_path))
for row in reader:
    labels[row[0]] = row[1]

X_train,X_test,Y_train,Y_test = train_test_split(dog_images,dog_image_ids,train_size=0.8,test_size=0.2,random_state=42)
X_train_2,X_val,Y_train_2,Y_val = train_test_split(X_train,Y_train, train_size= 0.875, test_size=0.125,random_state=42)

alexnet_pretrained = models.alexnet(pretrained=True)
alexnet = models.alexnet()


num_ftrs = alexnet_pretrained.fc.in_features
alexnet_pretrained.fc = nn.Linear(num_ftrs, 120)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(alexnet_pretrained.parameters(), lr=0.001, momentum=0.9)