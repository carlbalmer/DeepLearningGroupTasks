import numpy as np
from skimage.util.shape import  view_as_blocks,view_as_windows
import matplotlib.pyplot as plt
from scipy import misc

A = np.arange(4*4*3).reshape(4,4,3)

alp = np.random.normal(size=(A.shape))
h = alp*A
he = np.exp(h)
print('plot of image matrix')
plt.imshow(he)
plt.show()
B = view_as_windows(he, window_shape=(2, 2, 3), step=2)
#check case
print('plot of 1st block strided view')
plt.imshow(B[0,0].reshape(2,2,3))
plt.show()
print('hard checking numerical :')
print(np.log(B[0,0].reshape(2,2,3).sum()))
print(np.log(B[0,1].reshape(2,2,3).sum()))
print(np.log(B[1,0].reshape(2,2,3).sum()))
print(np.log(B[1,1].reshape(2,2,3).sum()))
#B.sum(axis=(5,4,3))
print('block views:')
print(np.log(B.sum(axis=(5,4,3))).reshape(2,2))

plt.imshow(np.log(B.sum(axis=(5,4,3))).reshape(2,2))

#f = misc.face()

#plt.imshow(f)
#plt.show()
