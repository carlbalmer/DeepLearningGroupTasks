import pickle
import sklearn
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pylab as py
from matplotlib.ticker import NullFormatter
from time import time
from sklearn.decomposition import PCA
(fig, subplots) = plt.subplots(2,2, figsize=(15, 8))
feautures = pickle.load(open("feature_vectors_120.pkl","rb"))

feature_vector = []
labels = []
names =[]
for feature in feautures:

    feature_vector.append(feature['features'])
    labels.append(feature['correctly_classified'])
    names.append(feature['encoded_label'])
#print(labels)
green = [i for i, x in enumerate(labels) if x]
red = [i for i, x in enumerate(labels) if not x]
x = np.array(feature_vector)
nsamples, nx, ny = x.shape
d2_train_dataset = x.reshape((nsamples,nx*ny))

ax = subplots[0][0]
model = TSNE(perplexity=50)
Y=model.fit_transform(d2_train_dataset)
names1 = set(names)
count = 0
for name in names1:
    if count > 10:
        break
    idx = [index for index, value in enumerate(names) if value == name]

    ax.scatter(Y[idx, 0], Y[idx, 1], s=200, marker=r"$ {} $".format(name), edgecolors='none', label=name)
    count += 1
ax.set_title("TSNE with perplexity 5")

#make  a scatter based on the class labels



ax = subplots[0][1]
pca = PCA(n_components=2)
Y2 =pca.fit_transform(d2_train_dataset)
ax.set_title("PCA")
count2=0
for name in names1:
    if count2 > 10:
        break
    idx = [index for index, value in enumerate(names) if value == name]

    ax.scatter(Y[idx, 0], Y[idx, 1], s=200, marker=r"$ {} $".format(name), edgecolors='none', label=name)
    count += 1

ax.axis('tight')



plt.show()


