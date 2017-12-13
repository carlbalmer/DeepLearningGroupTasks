import torch
from torch import nn
from torch.autograd import Variable, gradcheck
from torch.autograd.function import Function


class simpleSExpLog(nn.Module):
    def __init__(self, kernel_size):
        super(simpleSExpLog, self).__init__()
        self.kernel_size = kernel_size

        self.alpha = nn.Parameter(torch.ones(1, 1))
        self.pool = nn.AvgPool2d(kernel_size)

    def forward(self, input):
        return simpleSExpLogFunction.apply(input, self.alpha, self.pool)


class simpleSExpLogFunction(Function):
    @staticmethod
    def forward(ctx, input, alpha, pool):
        input = Variable(input, requires_grad=True)
        alpha = Variable(alpha, requires_grad=True)

        output = pool(alpha.mul(input).exp()).log().mul(alpha.pow(-1))
        ctx.input, ctx.alpha, ctx.output = input, alpha, output
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output, grad_output, None
        print("entered sexyBackward")
        print("retriving variables from ctx")
        input, alpha, output = ctx.input, ctx.alpha, ctx.output
        print(output.grad_fn)
        print("calling backward on output")
        output.backward()
        print("grads calculated")
        print(alpha.grad)
        print("returning grads")
        return input.grad, alpha.grad, None


#layer = simpleSExpLog(2)
#layer.alpha = nn.Parameter(torch.DoubleTensor([36]))
#in_features = torch.autograd.Variable(torch.DoubleTensor([[[7,5,4,8],[8,3,3,10],[9,7,7,1],[1,8,6,8]]]))
#output = layer.forward(in_features)
#layer.backward(torch.DoubleTensor([[1,1],[1,1]]))
#print(output)

input = (Variable(torch.randn(1,4,4).double(), requires_grad=True), Variable(torch.DoubleTensor([1]), requires_grad=True), nn.AvgPool2d(2))
test = gradcheck(simpleSExpLogFunction.apply, input, eps=1e-6, atol=1e-4)
print(test)