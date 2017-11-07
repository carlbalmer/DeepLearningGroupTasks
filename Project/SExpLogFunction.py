import torch

from torch.autograd import Variable
from torch.autograd import Variable
from torch.autograd.function import Function, once_differentiable
from torch._thnn import type2backend
from . import _all


class SExpLogFunction(Funcion):

    @staticmethod
    def forward(ctx,input,kernel_size,alphas,padding = 0,stride=None):
        ctx.save_for_backward(input,alphas)
        ctx.input = input
        ctx.alphas = alphas
        ctx.kernel_size = kernel_size
        ctx.padding = padding
        ctx.stride = stride
        return input

        """
           In the backward pass we receive a Tensor containing the gradient of the loss
           with respect to the output, and we need to compute the gradient of the loss
           with respect to the input.
       """
    @staticmethod
    def backward(self,grad_output):

        grad_input = grad_output.clone()
        return grad_input




