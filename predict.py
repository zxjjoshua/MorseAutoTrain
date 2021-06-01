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
from prepare_gold_labels import malicious_id


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
    malicious_batches = []

    snapshot_id = 0
    has_malicious = False

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
                if record.Id in malicious_id:
                    has_malicious = True
                    print("event " + str(record.Id) + " was captured in snapshot " + str(snapshot_id))
                # process batch-wise
                if event_num == max_event_per_epoch:
                    tmp_out, tmp_malicious = predict(rnn, snapshot_id, has_malicious)
                    snapshot_id += 1
                    if has_malicious:
                        # tmp_malicious should be n * 3
                        malicious_batches.append(tmp_malicious)
                        # malicious_batches should be m * n * 3, where m is the number of snapshots
                        #  containing malicious events
                    out_batches.append(tmp_out)
                    event_num = 0
                    has_malicious = False
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
    f.close()

    # dump malicious data points
    if len(malicious_batches) > 0:
        with open('./Data/malicious_data.out', 'w+') as out_f:
            json.dump(malicious_batches, out_f)
    return out_batches


def predict(rnn, snapshot_id, has_malicious):
    process_node_list = gv.processNodeSet
    # generate sequence
    cur_len = 0
    cur_batch = []
    remain_batch = []
    out_batches = []
    malicious_out = []
    cur_malicious_mark = []
    remain_malicious_mark = []

    if has_malicious:
        for node_id in process_node_list:
            node = process_node_list[node_id]
            sequence = node.generate_sequence(gv.batch_size, gv.sequence_size)
            tmp_malicious_mark = node.generate_malicious_mark(gv.batch_size, gv.sequence_size)
            need = gv.batch_size - cur_len
            if len(sequence) + cur_len > gv.batch_size:
                cur_batch += sequence[:need]
                cur_len = gv.batch_size
                cur_malicious_mark += tmp_malicious_mark[:need]
                remain_batch = sequence[need:]
                remain_malicious_mark = tmp_malicious_mark[need:]
            else:
                cur_batch += sequence[:need]
                cur_malicious_mark += tmp_malicious_mark[:need]
                cur_len += len(sequence)
            if cur_len >= gv.batch_size:
                input_tensor = torch.tensor(cur_batch)
                input_tensor = input_tensor.to(gv.device)
                input_tensor.requires_grad = True
                ''' morse output for visualize testing
                # tmp = []
                # for input in input_tensor:
                #     tmp.append(input.tolist())
                # print(tmp)
                # tmp_out_batches.append(tmp)
                '''
                out, h = rnn(input_tensor.float())
                out_batches.append(out)
                # print(cur_malicious_mark)
                for seq_idx in range(gv.batch_size):
                    for element_idx in range(gv.sequence_size):
                        if cur_malicious_mark[seq_idx][element_idx]:
                            malicious_out.append(out[seq_idx][element_idx])

                cur_batch = remain_batch[::]
                cur_malicious_mark = remain_malicious_mark[::]
                cur_len = len(cur_batch)
                remain_batch = []
                remain_malicious_mark = []
    else:
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
                ''' morse output for visualize testing
                # tmp = []
                # for input in input_tensor:
                #     tmp.append(input.tolist())
                # print(tmp)
                # tmp_out_batches.append(tmp)
                '''
                out, h = rnn(input_tensor.float())
                out_batches.append(out)
                cur_batch = remain_batch[::]
                cur_len = len(cur_batch)
                remain_batch = []
    # if has_malicious:
    #     f = open('./Data/morse_' + str(snapshot_id) + '.out', 'a+')
    #     json.dump(tmp_out_batches, f)
    #     f.close()
    return [out_batches, malicious_out]
