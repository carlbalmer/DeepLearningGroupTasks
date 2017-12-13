import torch
from torch import nn


class aSExpLog(nn.Module):
    def __init__(self, kernel_size):
        super(aSExpLog, self).__init__()
        self.kernel_size = kernel_size
        self.alpha = nn.Parameter(torch.ones(1, 1))
        self.pool = nn.AvgPool2d(kernel_size)

    def sexy(self, input_tensor):
        print(self.alpha)
        x = self.alpha.mul(input_tensor).exp()
        x = self.pool(x)
        x = x.log().mul(self.alpha.pow(-1))
        return x
