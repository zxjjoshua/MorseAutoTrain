import torch
import torch.nn as nn
import numpy as np


class SimpleNet(nn.Module):
    def __init__(self, learning_rate=0.001):
        super(SimpleNet, self).__init__()
        self.learning_rate = torch.tensor(learning_rate)
        self.w = torch.randn((), requires_grad=True)
        self.b = torch.randn((), requires_grad=True)
        self.y = torch.randn(())

    def forward(self, x):
        self.y = (torch.tanh(self.w*x+self.b))
        # print("this is output ", self.y)
        return self.y

    def backward(self):
        self.y.backward()
        return np.array([self.w.grad, self.b.grad])

    def update_weight(self, w_grad, b_grad):
        # w_grad, b_grad = inner_grad
        # w_grad = w_grad * outer_grad
        # b_grad = b_grad * outer_grad
        with torch.no_grad():
            self.w = self.w - self.learning_rate * w_grad
            self.b = self.b - self.learning_rate * b_grad
            self.w.requires_grad = True
            self.b.requires_grad = True

