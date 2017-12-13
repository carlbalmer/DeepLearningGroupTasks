import torch
from torch import nn


class aSExpLog():
    def __init__(self, kernel_size, alpha):
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.pool = nn.AvgPool2d(kernel_size)

    def pool(self, input_tensor):
        print(self.alpha)
        return self.pool(self.alpha.mul(input_tensor).exp()).log().mul(self.alpha.pow(-1))
