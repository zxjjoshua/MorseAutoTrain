import numpy as np
import math

from numpy.core.numeric import tensordot
import morse_train
from processnode import ProcessNode
from globals import GlobalVariable as gv
import RNN
from morse import Morse
from RNN import RNNet
from event.event_parser import EventParser
import torch
from utils import *
from RNN import Comp_Loss
from logging import getLogger
import json

def predict_entry():
    logger = getLogger("test mode")
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
    rnn = RNNet(input_dim=gv.feature_size, hidden_dim=64, output_dim=3, numOfRNNLayers=1)
    morse, rnn = load_model(morse, rnn)
    rnn = rnn.to(device)
    event_parser = EventParser(morse)
    rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=Learning_Rate)
    f = open(gv.test_data, "r")
    i = 0
    max_event_per_epoch = 100
    event_num = 0
    out_batches = []
    while True:
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

                # process batch-wise
                if event_num == max_event_per_epoch:
                    tmp=predict(rnn)
                    out_batches.append(tmp)
                    event_num = 0
            elif record.type == -1:
                # file node
                if 0 < record.subtype < 5:
                    newNode = record.getFileNode(morse)
                    if not newNode:
                        logger.error("failed to get file node")
                        continue
                    if gv.exist_fileNode(newNode.id):
#                        logger.error("duplicate file node: " + str(newNode.id))
                        pass
                    else:
                        gv.set_fileNode(newNode.id, newNode)
                elif record.subtype == -1:
                    # common file
                    newNode = record.getFileNode(morse)
                    if not newNode:
                        logger.error("failed to get file node")
                        continue
                    if gv.exist_fileNode(newNode.id):
 #                       logger.error("duplicate file node: " + str(newNode.id))
                        pass
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

    return out_batches

def predict(rnn):
    process_node_list = gv.processNodeSet
    # generate sequence
    cur_len = 0
    cur_batch = []
    remain_batch = []
    out_batches = []

    f=open('./Data/morse.out', 'w')

    for node_id in process_node_list:
        node = process_node_list[node_id]
        sequence = node.generate_sequence(gv.batch_size, gv.sequence_size)
        need = gv.batch_size - cur_len
        if len(sequence) + cur_len > gv.batch_size:
            cur_batch += sequence[:need]
            cur_len = gv.batch_size
            remain_batch = sequence[need:]
        else:
            cur_batch += sequence[:need]
            cur_len += len(sequence)
        if cur_len >= gv.batch_size:
            input_tensor = torch.tensor(cur_batch)
            input_tensor = input_tensor.to(gv.device)
            input_tensor.requires_grad = True
            print(cur_batch)
            json.dump(cur_batch.tolist(), f)
            out, h = rnn(input_tensor.float())
            out_batches.append(out)
            cur_batch = remain_batch[::]
            cur_len = len(cur_batch)
            remain_batch = []

    return out_batches
