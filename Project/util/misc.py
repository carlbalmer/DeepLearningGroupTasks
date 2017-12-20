"""
General purpose utility code from https://raw.githubusercontent.com/pytorch/vision/master/torchvision/datasets/utils.py
"""

import errno
import hashlib
import os
import os.path
import numpy as np


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy@k for the specified values of k
    :param output:
        The output of the model
    :param target:
        The GT for the corresponding output
    :param topk:
        Top@k return value. It can be a tuple (1,5) and it return Top1 and Top5
    :return:
        Top@k accuracy
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_file(filename):
    val_acc = []
    val_loss = []
    running_loss = []
    alpha = []
    alpha_gradient = []
    alpha2 = []
    alpha2_gradient = []
    alpha3 = []
    alpha3_gradient = []

    with open(filename, 'r') as log_file:
        for line in log_file:
            if "validate INFO: Epoch" in line:
                running_loss.append(float(line.split(' ')[9]))
            if "validate INFO:  * Acc@1" in line:
                val_acc.append(float(line.split(' ')[8]))
                val_loss.append(np.mean(running_loss))
                running_loss = []
            if "INFO:  * Alpha " in line or "validate INFO:  * Alpha1" in line:
                alpha.append(float(line.split(' ')[8]))
                alpha_gradient.append(float(line.split(' ')[10]))
            if "INFO:  * Alpha2" in line:
                alpha2.append(float(line.split(' ')[8]))
                alpha2_gradient.append(float(line.split(' ')[10]))
            if "INFO:  * Alpha3" in line:
                alpha3.append(float(line.split(' ')[8]))
                alpha3_gradient.append(float(line.split(' ')[10]))
    return val_acc, val_loss, alpha, alpha_gradient, alpha2, alpha2_gradient, alpha3, alpha3_gradient
