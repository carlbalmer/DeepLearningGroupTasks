import pickle
import sklearn
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pylab as py
from matplotlib.ticker import NullFormatter
from time import time
from sklearn.decomposition import PCA
(fig, subplots) = plt.subplots(2,  figsize=(15, 15))
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

#ax = subplots[0][0]
model = TSNE(perplexity=20)
pca = PCA(n_components=2)
Y=model.fit_transform(d2_train_dataset)
Y2 =pca.fit_transform(d2_train_dataset)

'''

#make  a scatter based on the class labels
ax.scatter(Y[red, 0], Y[red, 1],c="r")
ax.scatter(Y[green,0],Y[green,1],c=(0.1, 0.2, 0.00005))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')


ax = subplots[0][1]



ax[0].scatter(Y2[red, 0], Y2[red, 1],c="r")
ax[0].scatter(Y2[green,0],Y2[green,1],c="g")
ax[0].xaxis.set_major_formatter(NullFormatter())
ax[0].yaxis.set_major_formatter(NullFormatter())
ax[0].axis('tight')'''
ax = subplots[0]
ax1 = subplots[1]
ax.set_title("PCA")
ax1.set_title("TSNE with perplexity 20")
#ax[0].xaxis.set_major_formatter(NullFormatter())
#ax[0].yaxis.set_major_formatter(NullFormatter())
#
names1 = set(names)
count = 0
for name in names1 :
    if count > 10:
        #break #uncomment to use least number of classes
        pass
    idx = [index for index,value in enumerate(names) if value == name]
    
    ax.scatter(Y2[idx,0], Y2[idx,1], s=200, marker=r"$ {} $".format(name), edgecolors='none', label=name)
    ax1.scatter(Y[idx,0], Y[idx,1], s=200, marker=r"$ {} $".format(name), edgecolors='none', label=name)
    count += 1
#plt.legend(numpoints=1)
    ax.axis('tight')
plt.show()