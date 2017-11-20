import torch

from torch.autograd import Variable
from torch.autograd import Variable
from torch.autograd.function import Function, once_differentiable
from skimage.util.shape import  view_as_blocks,view_as_windows
from torch._thnn import type2backend
from . import _all


class SExpLogFunction(Funcion):

    @staticmethod
    def forward(ctx,input,kernel_size,alphas,dims):
        ctx.save_for_backward(input,alphas)
        ctx.input = input
        ctx.alphas = alphas
        ctx.kernel_size = kernel_size

        h = ctx.alphas*ctx.input
        he = np.exp(h)
        B = view_as_blocks(he, block_shape=(kernel_size,kernel_size,dims))
        output_size_x = input.shape[0]/kernel_size
        output_size_y = input.shape[1]/kernel_size
        return np.log(B.sum(axis=(5,4,3))).reshape(output_size_x,output_size_y)

        """
           In the backward pass we receive a Tensor containing the gradient of the loss
           with respect to the output, and we need to compute the gradient of the loss
           with respect to the input.
       """
    @staticmethod
    def backward(self,grad_output):

        grad_input = grad_output.clone()
        return grad_input




