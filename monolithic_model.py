from morse import Morse
from RNN import RNNet
from globals import GlobalVariable as gv

class MonolithicModel:

    def __init__(self, morse, rnn, data_loader):
        self.morse = morse
        self.rnn = rnn
        self.pos = 0
        self.data_loader = data_loader
        self.rnn_grad = None

    def forward(self):
        rnn_input = self.morse.forward()
        if rnn_input is not None:
            self.rnn_grad = self.rnn.forward(rnn_input)