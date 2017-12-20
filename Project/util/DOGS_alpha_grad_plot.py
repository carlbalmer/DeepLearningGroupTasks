import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interpolate

from util.misc import parse_file

input_filenames = ["data/3_layer_maxpool/17-12-18-14h-25m-58s/logs.txt",
                   "data/3_layer_sexplog/17-12-18-14h-02m-08s/logs.txt",
                   "data/3_layer_spatial/17-12-18-14h-14m-02s/logs.txt"
                   ]
alpha1 = []
alpha2 = []
alpha3 = []
grad1 = []
grad2 = []
grad3 = []

for filename in input_filenames:
    _, _, a1, g1, a2, g2, a3, g3 = parse_file(filename)
    alpha1.append(a1)
    alpha2.append(a2)
    alpha3.append(a3)
    grad1.append(g1)
    grad2.append(g2)
    grad3.append(g3)

# smooth curves
xnew = np.linspace(0,499,50) # set to 500 for zero smoothing

plt.figure(0)
plt.plot(xnew, interpolate.spline(range(500), alpha1[1], xnew), label='SExpLog L1', color='C1', linestyle='-')
plt.plot(xnew, interpolate.spline(range(500), alpha2[1], xnew), label='SExpLog L2', color='C1', linestyle='--')
plt.plot(xnew, interpolate.spline(range(500), alpha3[1], xnew), label='SExpLog L3', color='C1', linestyle=':')
plt.plot(xnew, interpolate.spline(range(500), alpha1[2], xnew), label='spatialSExpLog L1 (avg)', color='C2', linestyle='-')
plt.plot(xnew, interpolate.spline(range(500), alpha2[2], xnew), label='spatialSExpLog L2 (avg)', color='C2', linestyle='--')
plt.plot(xnew, interpolate.spline(range(500), alpha3[2], xnew), label='spatialSExpLog L3 (avg)', color='C2', linestyle=':')

plt.legend()
plt.title("Comparison - DOGS - 5-layer-CNN")
plt.ylabel('alpha')
plt.xlabel('epoch')

plt.figure(1)
plt.plot(xnew, interpolate.spline(range(500), grad1[1], xnew), label='SExpLog L1', color='C1', linestyle='-')
plt.plot(xnew, interpolate.spline(range(500), grad2[1], xnew), label='SExpLog L2', color='C1', linestyle='--')
plt.plot(xnew, interpolate.spline(range(500), grad3[1], xnew), label='SExpLog L3', color='C1', linestyle=':')
plt.plot(xnew, interpolate.spline(range(500), grad1[2], xnew), label='spatialSExpLog L1 (avg)', color='C2', linestyle='-')
plt.plot(xnew, interpolate.spline(range(500), grad2[2], xnew), label='spatialSExpLog L2 (avg)', color='C2', linestyle='--')
plt.plot(xnew, interpolate.spline(range(500), grad3[2], xnew), label='spatialSExpLog L3 (avg)', color='C2', linestyle=':')

plt.legend()
plt.title("Comparison - DOGS - 5-layer-CNN")
plt.ylabel('gradient')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(xnew, interpolate.spline(range(500), grad1[2], xnew), label='spatialSExpLog L1 (avg)', color='C2', linestyle='-')
plt.plot(xnew, interpolate.spline(range(500), grad2[2], xnew), label='spatialSExpLog L2 (avg)', color='C2', linestyle='--')
plt.plot(xnew, interpolate.spline(range(500), grad3[2], xnew), label='spatialSExpLog L3 (avg)', color='C2', linestyle=':')

plt.ylim((-0.001,0.001))
plt.legend()
plt.title("Comparison - DOGS - 5-layer-CNN")
plt.ylabel('gradient')
plt.xlabel('epoch')

plt.show()
