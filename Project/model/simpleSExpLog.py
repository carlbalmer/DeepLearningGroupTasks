import torch
from torch import nn
from torch.autograd.function import Function


class simpleSExpLog(nn.Module):
    def __init__(self, kernel_size):
        super(simpleSExpLog, self).__init__()
        self.kernel_size = kernel_size

        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered s Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.

        self.alpha = nn.Parameter(torch.ones(1, 1))
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return simpleSExpLogFunction.apply(input, self.alpha, self.pool)


class simpleSExpLogFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, alpha, pool):
        ctx.save_for_backward(input, alpha)
        #alpha = torch.autograd.Variable(alpha)
        o = alpha.mul(input)
        o = o.exp()
        o = pool(o).data
        o = o.log()
        o = alpha.pow(-1).mul(o)
        output = o
        #output = pool(input.mul(alpha).exp()).log().mul(1/alpha)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


layer = simpleSExpLog(2)
layer.alpha = nn.Parameter(torch.DoubleTensor([-50]))
in_features = torch.autograd.Variable(torch.DoubleTensor([[[7,5,4,8],[8,3,3,10],[9,7,7,1],[1,8,6,8]]]))
output = layer.forward(in_features)
print(output)