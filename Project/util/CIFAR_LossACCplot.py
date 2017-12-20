import matplotlib.pyplot as plt

from util.misc import parse_file

input_filenames = ["data/1_layer_maxpool_adam_0001/17-12-13-17h-24m-31s/logs.txt",
                   "data/1_layer_sexplog_adam_0001/17-12-13-15h-36m-06s/logs.txt",
                   "data/1_layer_spatial_adam_0001/17-12-18-00h-13m-20s/logs.txt"
                   ]
alpha = []
alpha_gradient = []

for filename in input_filenames:
    _, _, a, g = parse_file(filename)
    alpha.append(a)
    alpha_gradient.append(g)

plt.figure(0)
plt.plot(alpha[0], label='MAXpool')
plt.plot(alpha[1], label='SExpLog')
plt.plot(alpha[2], label='spatialSExpLog (avg)')

plt.legend()
plt.title("Comparison - CIFAR10 - 3-layer-CNN")
plt.ylabel('alpha')
plt.xlabel('epoch')

plt.figure(1)
plt.plot(alpha_gradient[0], label='MAXpool')
plt.plot(alpha_gradient[1], label='SExpLog')
plt.plot(alpha_gradient[2], label='spatialSExpLog (avg)')

plt.legend()
plt.title("Comparison - CIFAR10 - 3-layer-CNN")
plt.ylabel('gradient')
plt.xlabel('epoch')

plt.show()
