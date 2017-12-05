import torch

from torch.autograd import Variable
from torch.autograd import Variable
from torch.autograd.function import Function, once_differentiable
from skimage.util.shape import  view_as_blocks,view_as_windows
from torch._thnn import type2backend
import numpy as np
#from . import _all  #what is this doiing? @sam


class SExpLogFunction(Function):

    @staticmethod
    def forward(ctx, input, alphas):
        ctx.save_for_backward(input,alphas)
        ctx.input = input
        ctx.alphas = alphas
        dims = input.shape[2]  # dims shd be taken from the 3rd dim of input

        output_size_x = input.shape[0]/ctx.kernel_size
        output_size_y = input.shape[1]/ctx.kernel_size
        #unfold takes in dim, size in each window determined by kernel, and step
        ctx.re = Variable(ctx.alphas * (ctx.input.unfold(0,ctx.kernel_size,ctx.step).unfold(1,ctx.kernel_size,ctx.step)\
            .contiguous().view(output_size_x,output_size_y,dims,-1)))
        #we have a 4 dim tensor, of which we operate the 4th dim to get colapse

        ctx.sexplog = Variable(torch.log1p(torch.sum(torch.exp(ctx.re),3))) # grad w.r.t sexplog()
        #sanipoint print functions history to validate grads

        #ctx.save_for_backward(input, alphas,ctx.sexplog)
        return ctx.sexplog
    """
           In the backward pass we receive a Tensor containing the gradient of the loss
           with respect to the output, and we need to compute the gradient of the loss
           with respect to the input.
       """
    @staticmethod
    def backward(ctx,grad_output):

        grad_input = grad_output * ctx.sexplog.backward()
        grad_alphas = grad_output * ctx.re.backward()
        return grad_input,grad_alphas




