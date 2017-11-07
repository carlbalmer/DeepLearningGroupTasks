import torch.nn as nn
import torch.nn.init as init
import SExpLogFunction


class SExpLog(nn.Module):
    def __init__(self,input_feautures):
        super(SExpLog,self).__init__()
        self.input_feautures = input_feautures
        self.alphas = nn.Parameter(torch.Tensor(input_feautures))
        #initialize alphas with one
        init.constant(self.alphas,1)

    def forward(self,input):
        return SExpLogFunction.apply(self,self.input_feautures,2,self.alphas)

