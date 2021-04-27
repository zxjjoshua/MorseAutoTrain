import pickle
import numpy as np
import torch
import math
from globals import GlobalVariable as gv
import os
from record import Record

loaded_model_weights = None

def dump_model(morse = None, rnn = None):
    '''
    save the trained model (morse, simplenet, rnn) to a binary file
    '''
    morse_model_weights = {
        "morse": {},
    }
    morse_model_weights["morse"]["a_b"] = morse.a_b
    morse_model_weights["morse"]["a_e"] = morse.a_e

    # save the morse model's weights
    with open(gv.morse_model_path, 'wb') as handle:
        pickle.dump(morse_model_weights, handle, pickle.HIGHEST_PROTOCOL)

    # save simplenet's weights
    torch.save(morse.benign_thresh_model.state_dict(), gv.benign_thresh_model_path)
    torch.save(morse.suspect_env_model.state_dict(), gv.suspect_env_model_path)

    # save rnn's weights
    torch.save(rnn.state_dict(), gv.rnn_model_path)

def load_model():
    '''
    load a trained model from a binary file
    '''
    with open(gv.morse_model_path, 'rb') as handle:
        loaded_model_weights = pickle.load(handle)

# def wrap_model():
#     '''
#     grap different sub models (morse, simplenet, rnn) together
#     '''
#     from morse import Morse as tg
#
#     model_weights = {
#         "morse": {},
#     }
#     model_weights["morse"]["a_b"] = tg.a_b
#     model_weights["morse"]["a_e"] = tg.a_e
#     # model_weights["simplenet"]["1"] = tg.benign_thresh_model
#     # model_weights["simplenet"]["2"] = tg.suspect_env_model
#     return model_weights

def save_pred_labels(pred_labels, file_path):
    '''
    save the prediction results of the test mode to a file
    '''
    with open(file_path, 'w') as f:
        for line in pred_labels:
            f.write(line+"\n")

def evaluate_classification(pred_labels, gold_labels):
    total = len(pred_labels)
    # positive: benign
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(gold_labels)):
        if (gold_labels[i] == pred_labels[i]):
            if (pred_labels[i] == 'benign'):
                tp += 1
            elif (pred_labels[i] == 'malicious'):
                tn += 1
        else:
            if (pred_labels[i] == 'benign'):
                fp += 1
            elif (pred_labels[i] == 'malicious'):
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp +fn)
    f1 = 2 * precision * recall / (precision + recall)

    print("======= evaluation results =======")
    print("precision: ", precision)
    print("recall: ", recall)
    print("accuracy: ", accuracy)
    print("f1: ", f1)

    return precision, recall, accuracy, f1

def early_stop_triggered(loss1, loss2, threshold):
    return loss2 > loss1 * threshold

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

# data reading
def readObj(f):
    event = Record()
    params = []
    while True:
        line = f.readline()
        # print(line)
        if not line:
            break
        # print(line)
        if line[0] == "}":
            break
        line = pruningStr(line)
        if line == "paras {":
            while True:
                data = f.readline()
                data = pruningStr(data)
                if data == "}":
                    break
                words = data.split(": ")
                # print(words)
                params.append(pruningStr(words[1]))
        else:
            words = line.split(": ")
            # print(words)
            if words[0] == "ID":
                event.Id = int(pruningStr(words[1]))
            elif words[0] == "type":
                event.type = int(pruningStr(words[1]))
            elif words[0] == "time":
                event.time = int(pruningStr(words[1]))
            elif words[0] == "subtype":
                event.subtype = int(words[1])
            elif words[0] == "size":
                event.size = int(words[1])
            elif words[0] == "srcId":
                event.srcId = int(words[1])
            elif words[0] == "desId":
                event.desId = int(words[1])
    if params:
        event.params = params
    return event


def pruningStr(line):
    if not line:
        return line
    start = 0
    while start < len(line):
        if line[start] == ' ' or line[start] == '\t':
            start += 1
        else:
            break
    if line[start] == "\"":
        start += 1
    if line[-1] == "\"":
        line = line[:-1]
    if line[-1] == '\n':
        return line[start:-1]
    return line[start:]