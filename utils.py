import pickle

import torch

from globals import GlobalVariable as gv
import os

loaded_model_weights = None

def dump_model(morse_model_weights=None, rnn=None):
    '''
    save the trained model (morse, simplenet, rnn) to a binary file
    '''
    from target import Target as tg
    if morse_model_weights is None:
        morse_model_weights = wrap_model()

    # save the morse model's weights
    with open(gv.morse_model_path, 'wb') as handle:
        pickle.dump(morse_model_weights, handle, pickle.HIGHEST_PROTOCOL)

    # save simplenet's weights
    torch.save(tg.benign_thresh_model.state_dict(), gv.benign_thresh_model_path)
    torch.save(tg.suspect_env_model.state_dict(), gv.suspect_env_model_path)

    # save rnn's weights
    torch.save(rnn.state_dict(), gv.rnn_model_path)

def load_model():
    '''
    load a trained model from a binary file
    '''
    with open(gv.morse_model_path, 'rb') as handle:
        loaded_model_weights = pickle.load(handle)

def wrap_model():
    '''
    grap different sub models (morse, simplenet, rnn) together 
    '''
    from target import Target as tg
      
    from RNN import model as rnn  
    model_weights = {
        "morse": {},
    }
    model_weights["morse"]["a_b"] = tg.a_b
    model_weights["morse"]["a_e"] = tg.a_e
    # model_weights["simplenet"]["1"] = tg.benign_thresh_model
    # model_weights["simplenet"]["2"] = tg.suspect_env_model
    return model_weights

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
