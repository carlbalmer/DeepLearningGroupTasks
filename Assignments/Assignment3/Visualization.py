import pickle
import sklearn
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from time import time
from sklearn.decomposition import PCA
(fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))
feautures = pickle.load(open("feature_vectors.pkl","rb"))

feature_vector = []
labels = []
for feature in feautures:

    feature_vector.append(feature['features'])
    labels.append(feature['correctly_classified'])

perplexities = [5, 30, 50, 100]
green = [i for i, x in enumerate(labels) if x]
red = [i for i, x in enumerate(labels) if not x]
x = np.array(feature_vector)
nsamples, nx, ny = x.shape
d2_train_dataset = x.reshape((nsamples,nx*ny))

ax = subplots[0][0]
model = TSNE(perplexity=5)
Y=model.fit_transform(d2_train_dataset)
ax.set_title("TSNE with perplexity 5")
ax.scatter(Y[red, 0], Y[red, 1],c="r")
ax.scatter(Y[green,0],Y[green,1],c="g")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')
ax = subplots[0][1]
pca = PCA(n_components=2)
Y2 =pca.fit_transform(d2_train_dataset)
ax.set_title("PCA")
ax.scatter(Y2[red, 0], Y2[red, 1],c="r")
ax.scatter(Y2[green,0],Y2[green,1],c="g")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.axis('tight')



plt.show()

