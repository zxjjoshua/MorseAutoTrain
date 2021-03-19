import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleNet(nn.Module):
    def __init__(self, learning_rate=0.001):
        super(SimpleNet, self).__init__()

        self.fc1 = nn.Linear(1, 1)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.w, self.b = self.parameters()

    def forward(self, x):
        x = ((torch.tanh(self.fc1(x))) + 1.0) / 2.0
        return x



def back_propagate(network: SimpleNet, grad):
    network.