from utils import *

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



