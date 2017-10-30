import matplotlib.pyplot as plt
import numpy as np

inputFiles = [open("from_scratch", "r"),
              open("pretrained", "r")]

for file in inputFiles:
    train_acc = []
    val_acc = []
    loss = []

    for line in file:
        if "train Acc" in line:
            train_acc.append(float(line.split(':')[1]))
        if "valid Acc" in line:
            val_acc.append(float(line.split(':')[1]))
        if "7000" in line:
            loss.append(float(line.split(':')[2]))
    file.close()

    plt.figure(0)
    plt.plot(np.arange(len(train_acc)), train_acc, label='train ' + file.name)
    plt.plot(np.arange(len(val_acc)), val_acc, label='val ' +file.name)
    plt.legend()
    plt.title("ResNet18 - Train & Validation Acc")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.figure(1)
    plt.plot(np.arange(len(loss)), loss, label=file.name)
    plt.legend()
    plt.title("ResNet18 - Loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')



plt.show()
