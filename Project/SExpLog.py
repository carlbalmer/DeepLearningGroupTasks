import torch
import torch.nn as nn
import torch.nn.init as init
import SExpLogFunction


class SExpLog(nn.Module):
    def __init__(self,input_feautures, kernel_size,step):
        super(SExpLog,self).__init__()
        self.input_feautures = input_feautures
        self.kernel_size = kernel_size
        self.step = step
        #alphas shd be input feature size-kernel_size/s + 1
        height = 1+((input_feautures.size()[0]- kernel_size)/step)
        width = 1 + ((input_feautures.size()[1] - kernel_size) / step)
        self.alphas = nn.Parameter(torch.FloatTensor(width,height).zero_())
        #initialize alphas with one
        init.constant(self.alphas,1.0)

    def forward(self,input):
        return SExpLogFunction.apply(self.input_feautures,self.alphas)

