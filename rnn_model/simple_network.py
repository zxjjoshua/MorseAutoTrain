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
        print("this is output ", self.y)
        return self.y

    def backward(self, grad=torch.tensor(1.0)):
        if not isinstance(grad, torch.Tensor):
            grad=torch.tensor(grad)
        self.w.retain_grad()
        self.b.retain_grad()
        # self.y.backward(grad, retain_graph = True)
        self.y.backward()
        with torch.no_grad():
            self.w = self.w - self.learning_rate * self.w.grad
            self.b = self.b - self.learning_rate * self.b.grad
            self.w.requires_grad = True
            self.b.requires_grad = True

