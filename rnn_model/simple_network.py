import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, learning_rate=0.001):
        super(SimpleNet, self).__init__()
        self.learning_rate = torch.tensor(learning_rate)
        self.w = torch.randn((), requires_grad=True)
        self.b = torch.randn((), requires_grad=True)
        self.y = torch.randn(())

    def forward(self, x):
        self.y = (torch.tanh(self.w*x+self.b))
        return self.y

    def backward(self, grad=torch.tensor(1.0), final_w_grad=torch.tensor(1.0), final_b_grad=torch.tensor(1.0)):
        if not isinstance(grad, torch.Tensor):
            grad=torch.tensor(grad)
        self.y.backward(grad)
        with torch.no_grad():
            self.w = self.w - self.learning_rate * final_w_grad
            self.b = self.b - self.learning_rate * final_b_grad
            self.w.requires_grad = True
            self.b.requires_grad = True

