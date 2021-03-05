import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
% matplotlib inline

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

activation_relu = torch.nn.ReLU()
Loss_Function = torch.nn.MSELoss()
## TODO: write compactness loss function
Learning_Rate = 0.001
epochs = 100

input_dim = 12
output_dim = 3
hidden_dim = 64
numOfLayers = 2
numOfEpoch = 100
batch_size = 10


### RNN model
### 1. RNN layer
###    input: m by n by k, where m = # of batches, n = # of sequences in each batch, and k = # of features (12)
###    output: m by n by l, where m = # of batches, n = # of sequences in each batch, and l = hidden_dim
### 2. Linear layer
###    input: m by n by l, where m = # of batches, n = # of sequences in each batch, and l = hidden_dim
###    output: m by n by j, where m = # of batches, n = # of sequences in each batch, and j = output_dim
###
class RNNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, numOfLayers, dropout_threshold=0.2):
        super(RNNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.numOfLayers = numOfLayers
        self.rnn_layer = torch.nn.RNN(input_dim, hidden_dim, numOfLayers, batch_first=True, dropout=dropout_threshold)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h = self.init_hidden(batch_size)
        out, hn = self.rnn_layer(x, h)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hn

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


model = RNNet(input_dim, hidden_dim, output_dim, numOfLayers)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_model():
    model.train()
    epoch_list = []
    loss_list = []
    error_list = []
    counter = 1
    for epoch in tqdm(range(1, numOfEpoch + 1)):
        avg_loss = 0
        for x, label in train_loader:
            model.zero_grad()
            x.float().to(device)
            out, h = model(x.float())
            loss = Loss_Function(out, label.float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        epoch_list.append(counter)
        loss_list.append(avg_loss / len(train_loader))
        error_list.append(evaluate_model())
        counter += 1
    return epoch_list, loss_list, error_list


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