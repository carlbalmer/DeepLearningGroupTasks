import pickle
import matplotlib.pyplot as plt
import numpy as np

alphas = pickle.load(open("data/alphas.pkl", 'rb'))

plt.figure(0)
plt.imshow(alphas[0], vmin=0, vmax=2.6)
plt.colorbar()
plt.title("SpatialSExpLog - Layer 1 - 3-layer-CNN - CIFAR10")

plt.figure(1)
plt.imshow(alphas[1], vmin=0, vmax=2.6)
plt.colorbar()
plt.title("SpatialSExpLog - Layer 1 - 5-layer-CNN - DOGS")

plt.figure(2)
plt.imshow(alphas[2], vmin=0, vmax=2.6)
plt.colorbar()
plt.title("SpatialSExpLog - Layer 2 - 5-layer-CNN - DOGS")

plt.figure(3)
plt.imshow(alphas[3], vmin=0, vmax=2.6)
plt.colorbar()
plt.title("SpatialSExpLog - Layer 3 - 5-layer-CNN - DOGS")


plt.show()