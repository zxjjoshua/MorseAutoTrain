import numpy as np
import math
import morse_train
from processnode import ProcessNode
from globals import GlobalVariable as gv
import RNN
from target import Target as tg


def rand(a, b):
    return (b - a) * np.random.random() + a


def make_matrix(m, n, fill=0.0):  # 创造一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmod_derivative(x):
    return x * (1 - x)


class Train:

    def train(self, cases, labels, limit = 10000, learn = 0.05, correct = 0.1):
        for i in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells


        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights


        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate

        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
                for h in range(self.hidden_n):
                    for o in range(self.output_n):
                        self.output_weights[h][o] = rand(-2.0, 2.0)
                        # init correction matrix
                        self.input_correction = make_matrix(self.input_n, self.hidden_n)
                        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def forward(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]


def back_propagate(case, learn):
    # feed forward, get the predicted result
    # res = self.forward(case)

    # Morse_res = Morse.forward()
    # generate sequence
    # RNN_res=RNN.forward(Morse_res)

    # loss = calculate_loss(RNN_res)
    rnn_grad: np.array(4,3)
    # rnn_grad = calculate_loss_grad(loss)

    Morse_loss: np.array
    # Morse_grad = morse_train.back_propagate(RNN_grad, case)

    subtype=case[0][0]
    # 1 * 4
    final_grad=morse_train.back_propagate(subtype, case, rnn_grad)

    # update weights
    tg.a_b_setter(-learn*final_grad[0])
    tg.a_e_setter(-learn*final_grad[1])
    tg.benign_thresh_model_setter(-learn*final_grad[2], -learn*final_grad[3])
    tg.suspect_env_model_setter(-learn*final_grad[4], -learn*final_grad[5])


def back_propagate_batch(learn):
    node_list: list[ProcessNode]
    process_node_list = gv.processNodeSet
    file_node_list = gv.fileNodeSet
    # feed forward, get the predicted result
    # res = self.forward(case)

    # generate sequence
    cur_len=0
    cur_batch=[]
    remain_batch=[]
    for node_id in process_node_list:
        node=process_node_list[node_id]
        sequence = node.generate_sequence(gv.batch_size, gv.sequence_size)
        if len(sequence)+cur_len>gv.batch_size:
            need=gv.batch_size-cur_len
            cur_batch+=sequence[:need]
            cur_len=gv.batch_size
            remain_batch=sequence[need:]
        else:
            cur_batch+=sequence
            cur_len+=len(sequence)
        # batch_size * sequence_size * feature _size
        # batch_size: 100
        # sequence_size: 5
        # feature_size: 12

        # 100*5*12
        # 1*3
        if cur_len>=gv.batch_size:
            rnn_grad = RNN.train_model(sequence)
            # 100*5*12 * 12*64* 64*64 *  * *3
            # ? * ?

            final_grad = morse_train.back_propagate(sequence, node.get_event_type_list(), node.get_event_list(), rnn_grad)

            # update weights
            tg.a_b_setter(-learn * final_grad[0])
            tg.a_e_setter(-learn * final_grad[1])
            tg.benign_thresh_model_setter(final_grad[2])
            tg.suspect_env_model_setter(final_grad[3])
            cur_batch=remain_batch[::]
            remain_batch=[]

    cur_len = 0
    cur_batch = []
    remain_batch = []

    # tmp = 0
    # for node_id in file_node_list:
    #     if node_id not in process_node_list.keys():
    #         tmp += 1
    # print("not in: ", tmp)

    for node_id in file_node_list:
        node = file_node_list[node_id]
        sequence = node.generate_sequence()
        if len(sequence) + cur_len > gv.batch_size:
            need = gv.batch_size - cur_len
            cur_batch += sequence[:need]
            cur_len = gv.batch_size
            remain_batch = sequence[need:]
        else:
            cur_batch += sequence
            cur_len += len(sequence)
        # batch_size * sequence_size * feature _size
        # batch_size: 100
        # sequence_size: 5
        # feature_size: 12

        # 100*5*12
        # 1*3
        if cur_len >= gv.batch_size:
            rnn_grad = RNN.train_model(sequence)
            # 100*5*12 * 12*64* 64*64 *  * *3
            # ? * ?

            final_grad = morse_train.back_propagate(sequence, node.get_event_type_list(), node.get_event_list(),
                                                    rnn_grad)

            # update weights
            tg.a_b_setter(-learn * final_grad[0])
            tg.a_e_setter(-learn * final_grad[1])
            tg.benign_thresh_model_setter(final_grad[2])
            tg.suspect_env_model_setter(final_grad[3])
            cur_batch = remain_batch[::]
            remain_batch = []


