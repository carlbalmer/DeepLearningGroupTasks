import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interpolate

from util.misc import parse_file

input_filenames = ["data/3_layer_maxpool/17-12-18-14h-25m-58s/logs.txt",
                   "data/3_layer_sexplog/17-12-18-14h-02m-08s/logs.txt",
                   "data/3_layer_spatial/17-12-18-14h-14m-02s/logs.txt"
                   ]
acc = []
loss = []

for filename in input_filenames:
    a, l, _, _, _, _, _, _ = parse_file(filename)
    acc.append(a)
    loss.append(l)

# smooth curves
xnew = np.linspace(0,499,50) # set to 500 for zero smoothing

plt.figure(0)
plt.plot(xnew, interpolate.spline(range(500), acc[0], xnew), label='MAXpool')
plt.plot(xnew, interpolate.spline(range(500), acc[1], xnew), label='SExpLog')
plt.plot(xnew, interpolate.spline(range(500), acc[2], xnew), label='spatialSExpLog')

plt.legend()
plt.title("Comparison - DOGS - 5-layer-CNN")
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.figure(1)
plt.plot(xnew, interpolate.spline(range(500), loss[0], xnew), label='MAXpool')
plt.plot(xnew, interpolate.spline(range(500), loss[1], xnew), label='SExpLog')
plt.plot(xnew, interpolate.spline(range(500), loss[2], xnew), label='spatialSExpLog')

plt.legend()
plt.title("Comparison - DOGS - 5-layer-CNN")
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()
