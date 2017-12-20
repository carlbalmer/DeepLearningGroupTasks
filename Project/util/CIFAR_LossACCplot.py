import matplotlib.pyplot as plt

from util.misc import parse_file

input_filenames = ["data/1_layer_maxpool_adam_0001_200epoch/17-12-20-14h-52m-35s/logs.txt",
                   "data/1_layer_sexplog_adam_0001_200epoch/17-12-20-14h-48m-12s/logs.txt",
                   "data/1_layer_spatial_adam_0001_200epoch/17-12-20-14h-50m-24s/logs.txt"
                   ]
ACC = []
loss = []

for filename in input_filenames:
    a, l, _, _, _, _, _, _ = parse_file(filename)
    ACC.append(a)
    loss.append(l)

plt.figure(0)
plt.plot(ACC[0], label='MAXpool')
plt.plot(ACC[1], label='SExpLog')
plt.plot(ACC[2], label='spatialSExpLog (avg)')

plt.legend()
plt.title("Comparison - CIFAR10 - 3-layer-CNN")
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.figure(1)
plt.plot(loss[0], label='MAXpool')
plt.plot(loss[1], label='SExpLog')
plt.plot(loss[2], label='spatialSExpLog (avg)')

plt.legend()
plt.title("Comparison - CIFAR10 - 3-layer-CNN")
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()
