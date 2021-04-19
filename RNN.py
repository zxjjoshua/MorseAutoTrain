import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import numpy as np
from globals import GlobalVariable as gv
import sys
from utils import *

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

input_dim = gv.feature_size
output_dim = 3
hidden_dim = 64
numOfRNNLayers = 1
numOfEpoch = 100
batch_size = gv.batch_size
sequence_size = 100


def Comp_Loss(out):
    out_copy = torch.clone(
        out)  ## m by n by j, where m = # of batches, n = # of sequences in each batch, and j = output_dim
    batch_avg = torch.mean(out_copy, 1, True)  ## m by 1 by j
    # print(batch_avg.is_cuda)
    # print(torch.tensor([out.shape[1]]).is_cuda)
    tmp = torch.tensor([out.shape[1]]).to(device)
    # print(tmp.is_cuda)
    target = torch.repeat_interleave(batch_avg, tmp, dim=1)  ## m by n by j
    loss = torch.mean((out - target) ** 2)
    return loss


activation_relu = torch.nn.ReLU()
Loss_Function = Comp_Loss
Learning_Rate = 0.001


### RNN model
### 1. RNN layer
###    input: m by n by k, where m = # of batches, n = # of sequences in each batch, and k = # of features (12)
###    output: m by n by l, where m = # of batches, n = # of sequences in each batch, and l = hidden_dim
### 2. Linear layer
###    input: m by n by l, where m = # of batches, n = # of sequences in each batch, and l = hidden_dim
###    output: m by n by j, where m = # of batches, n = # of sequences in each batch, and j = output_dim
###


class RNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, numOfRNNLayers, dropout_threshold=0.2):
        super(RNNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.numOfRNNLayers = numOfRNNLayers
        self.rnn_layer = torch.nn.RNN(input_dim, hidden_dim, numOfRNNLayers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h = self.init_hidden(batch_size)
        h = h.to(device)
        out, hn = self.rnn_layer(x)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hn

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.numOfRNNLayers, batch_size, self.hidden_dim)
        return hidden


model = RNNet(input_dim, hidden_dim, output_dim, numOfRNNLayers)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)


def train_model(x):
    x = x.to(device)
    x.requires_grad = True
    model.train()
    epoch_list = []
    loss_list = []
    error_list = []
    # counter = 1
    # avg_loss = 0

    optimizer.zero_grad()
    # x.float().to(device)
    out, h = model(x.float())
    out = out.to(device)
    loss = Loss_Function(out)
    
    print("loss: ", loss.item())

    model_weights = wrap_model()
    gv.early_stopping_model_queue.append([loss.item(), model_weights])

    # early stopping and model saving
    if (len(gv.early_stopping_model_queue) == gv.early_stopping_patience and early_stop_triggered(gv.early_stopping_model_queue.popleft()[0], loss.item(), gv.early_stopping_threshold)):
        print("========================= early stopping triggered =========================")
        min_loss = min(list(gv.early_stopping_model_queue), key=lambda x: x[0])
        print("minimum loss: ", min_loss)
        popped_checkpoint = gv.early_stopping_model_queue.popleft()
        while popped_checkpoint > min_loss:
            popped_checkpoint = gv.early_stopping_model_queue.popleft()
        dump_model(popped_checkpoint[1], rnn=model)
        print("best model saved")
        sys.exit(0)

    loss.backward()
    # print(x)
    rnn_grad = x.grad
    # print(rnn_grad)
    optimizer.step()
    # avg_loss += loss.item()
    # epoch_list.append(counter)
    # loss_list.append(avg_loss / len(train_loader))
    # error_list.append(evaluate_model())
    # counter += 1
    # return epoch_list, loss_list, error_list
    return rnn_grad


def evaluate_model():
    # use current model to make prediction
    return 0


def draw_result(epoch_list, loss_list, error_list):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    line_1 = ax1.plot(epoch_list, loss_list, label="training loss")
    line_2 = ax2.plot(epoch_list, error_list, label="percentage error")
    ax1.set(xlabel="epoch", ylabel="MSE loss", title="MSE loss vs. Epoch")
    ax1.legend()
    ax2.set(xlabel="epoch", ylabel="percentage error", title="Percentage Error vs Epoch")
    ax2.legend()
    plt.subplots_adjust(wspace=0.5)
    plt.show()

    print("lowest MSE loss: {} at epoch {}".format(min(loss_list), loss_list.index(min(loss_list))))
    print("lowest percentage error: {} at epoch {}".format(min(error_list), error_list.index(min(error_list))))
