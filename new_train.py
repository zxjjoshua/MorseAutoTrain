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
from RNN import RNNet
from RNN import Comp_Loss
from logging import getLogger
import event.event_parser as ep
from event.event_parser import EventParser
from morse import Morse
from data_loader import DataLoader

def train_model():
    logger = getLogger("train mode")
    device = gv.device
    numOfEpoch = 100
    batch_size = gv.batch_size
    sequence_size = 100
    activation_relu = torch.nn.ReLU()
    Loss_Function = Comp_Loss
    Learning_Rate = 0.001


    # models initialization
    # data_loader = DataLoader(processNodeSet=gv.processNodeSet)
    morse = Morse(batch_size=gv.batch_size, sequence_size=gv.sequence_size, data_loader=gv.processNodeSet)
    event_parser = EventParser(morse)
    rnn = RNNet(input_dim=gv.feature_size, hidden_dim=64, output_dim=3, numOfRNNLayers=1)
    rnn = rnn.to(device)
    rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=Learning_Rate)


    f = open(gv.train_data, "r")
    i = 0
    max_event_per_epoch = 100
    event_num = 0
    while True:
        # print(len(gv.processNodeSet))
        line = f.readline()
        if not line:
            break
        # print(line, "and", line[0])
        if line[:4] == "data":
            record = readObj(f)
            if record.type == 1:
                # event type
                event_num += 1
                event_parser.parse(record, morse)
                # tr.back_propagate(record, 0.5)
                # data_rearrange.pre_process(record)

                # process batch-wise
                if event_num == max_event_per_epoch:
                    rnn_grad = None
                    # while data_loader.has_next():
                    input_tensor_list = morse.forward(0.05)
                    for input_tensor in input_tensor_list:
                        if input_tensor is not None:
                            input_tensor = input_tensor.to(device)
                            input_tensor.requires_grad = True
                            rnn.train()
                            rnn_optimizer.zero_grad()
                            rnn_out, rnn_h = rnn(input_tensor.float())
                            rnn_loss = Loss_Function(rnn_out)
                            rnn_loss.backward()
                            rnn_grad = input_tensor.grad
                            rnn_optimizer.step()
                            print("loss: ", rnn_loss.item())

                            if gv.early_stopping_on:
                                model_weights = wrap_model(morse)
                                gv.early_stopping_model_queue.append([rnn_loss.item(), model_weights])

                                # early stopping and model saving
                                if (
                                        len(gv.early_stopping_model_queue) == gv.early_stopping_patience and early_stop_triggered
                                    (gv.early_stopping_model_queue.popleft()[0], rnn_loss.item(),
                                     gv.early_stopping_threshold)):
                                    print(
                                        "========================= early stopping triggered =========================")
                                    min_loss = min(list(gv.early_stopping_model_queue), key=lambda x: x[0])
                                    print("minimum loss: ", min_loss)
                                    popped_checkpoint = gv.early_stopping_model_queue.popleft()
                                    while popped_checkpoint > min_loss:
                                        popped_checkpoint = gv.early_stopping_model_queue.popleft()
                                    dump_model(rnn=rnn, morse_model_weights=popped_checkpoint[1])
                                    print("best model saved")
                                    sys.exit(0)

                            event_num = 0

                            # integrated grads calculations and updates
                            simple_net_grad_tensor = morse.simple_net_grad_tensor.to(device)
                            morse_grad_tensor = morse.morse_grad_tensor.to(device)
                            simple_net_final_grad = torch.tensordot(rnn_grad, simple_net_grad_tensor,
                                                                    ([0, 1, 2], [0, 1, 2]))
                            final_morse_grad = torch.tensordot(rnn_grad, morse_grad_tensor, ([0, 1, 2], [0, 1, 2]))
                            morse.a_b_setter(-gv.learning_rate * final_morse_grad[0])
                            morse.a_e_setter(-gv.learning_rate * final_morse_grad[1])

                            # update SimpleNet's weights
                            morse.benign_thresh_model_setter(simple_net_final_grad[0],
                                                            simple_net_final_grad[1])
                            morse.suspect_env_model_setter(simple_net_final_grad[2],
                                                          simple_net_final_grad[3])

                    dump_model(morse=morse, rnn=rnn)

            elif record.type == -1:
                # file node
                if 0 < record.subtype < 5:
                    newNode = record.getFileNode(morse)
                    if not newNode:
                        logger.error("failed to get file node")
                        continue
                    if gv.exist_fileNode(newNode.id):
                        logger.error("duplicate file node: " + str(newNode.id))
                    else:
                        gv.set_fileNode(newNode.id, newNode)
                elif record.subtype == -1:
                    # common file
                    newNode = record.getFileNode(morse)
                    if not newNode:
                        logger.error("failed to get file node")
                        continue
                    if gv.exist_fileNode(newNode.id):
                        logger.error("duplicate file node: " + str(newNode.id))
                    else:
                        gv.set_fileNode(newNode.id, newNode)
                elif record.subtype == 5:
                    # process node
                    # if no params, this process is released
                    if not record.params:
                        gv.remove_processNode(record.Id)
                        continue
                    newNode = record.getProcessNode(morse)
                    if not newNode:
                        logger.error("failed to get process node")
                        continue
                    if gv.exist_processNode(newNode.id):
                        logger.error("duplicate process node: " + newNode.id)
                    else:
                        gv.set_processNode(newNode.id, newNode)
        i += 1
    f.close()

    return rnn_grad

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