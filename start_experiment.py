import os
import fire
import json
from globals import GlobalVariable as gv
import torch
import logging
import argparse
from new_train import train_model
import time
from predict import predict_entry
from utils import save_hyperparameters
from utils import save_evaluation_results

import numpy as np
import json

def start_experiment(config="config.json"):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"
    parser = argparse.ArgumentParser(description="train or test the model")
    parser.add_argument("--batch_size", nargs='?', default=5, type=int)
    parser.add_argument("--learning_rate", nargs='?', default=0.001, type=float)
    parser.add_argument("--sequence_length", nargs='?', default=5, type=int)
    parser.add_argument("--feature_dimension", nargs='?', default=12, type=int)
    parser.add_argument("--device", nargs='?', default="cuda", type=str)
    parser.add_argument("--train_data", nargs='?', default="EventData/north_korea_apt_attack_data_debug.out", type=str)
    parser.add_argument("--test_data", nargs='?', default="EventData/north_korea_apt_attack_data_debug.out", type=str)
    parser.add_argument("--validation_data", nargs='?', default="EventData/north_korea_apt_attack_data_debug.out", type=str)
    parser.add_argument("--model_save_path", nargs='?', default="trainedModels", type=str)
    parser.add_argument("--mode", nargs="?", default="train", type=str)
    parser.add_argument("--early_stopping_on", nargs="?", default="off", type=str)
    parser.add_argument("--early_stopping_patience", nargs="?", default=10, type=int)
    parser.add_argument("--early_stopping_threshold", nargs="?", default=10, type=int)
    parser.add_argument("--classify_boundary_threshold", nargs="?", default=1e-11, type=float)
    parser.add_argument("--load_model_from", nargs="?", default=None, type=str)
    parser.add_argument("--data_saved_path", nargs="?", default="Data", type=str)
    gv.project_path = os.getcwd()

    args = parser.parse_args()
    if args.early_stopping_on == "on":
        gv.early_stopping_on = True
    else:
        gv.early_stopping_on = False
    gv.learning_rate = args.learning_rate
    gv.batch_size = args.batch_size
    gv.sequence_size = args.sequence_length
    gv.feature_size = args.feature_dimension
    if torch.cuda.is_available():
        gv.device = torch.device(args.device)
    gv.train_data = args.train_data
    gv.test_data = args.test_data
    gv.validation_data = args.validation_data
    gv.model_save_path = args.model_save_path
    gv.early_stopping_patience = args.early_stopping_patience
    gv.early_stopping_threshold = args.early_stopping_threshold
    gv.mode = args.mode

    if (gv.mode == "train"):
        paths_setting(str(int(time.time())))
        logging.basicConfig(level=logging.INFO,
                            filename='debug.log',
                            filemode='w+',
                            format='%(asctime)s %(levelname)s:%(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
        save_hyperparameters(args, "train")
        train_model()
    elif (gv.mode == "test"):
        if args.load_model_from is None:
            raise ValueError("A path must be given to load the trained model from")
        gv.load_model_from = args.load_model_from
        test_id = paths_setting(args.load_model_from, mode="test")
        save_hyperparameters(args, "test")
        out_batches = predict_entry()
        '''losses = []
        for out_batch in out_batches:
            out_copy = torch.clone(out_batch)  ## m by n by j, where m = # of batches, n = # of sequences in each batch, and j = output_dim
            
            
            batch_avg = torch.mean(out_copy, 1, True)  ## m by 1 by j
            # print(batch_avg.is_cuda)
            # print(torch.tensor([out.shape[1]]).is_cuda)
            tmp = torch.tensor([out_batch.shape[1]])
            # print(tmp.is_cuda)
            batch_avg = batch_avg.to(gv.device)
            tmp = tmp.to(gv.device)
            target = torch.repeat_interleave(batch_avg, tmp, dim=1)  ## m by n by j
            loss = (out_batch - target) ** 2
            losses += torch.mean(loss, dim=1)
        # calculate the final accuracy of classification using labels from test data
        pred_labels = []
        for loss in losses:
            if loss <= args.classify_boundary_threshold:
                pred_labels.append("benign")
            else:
                pred_labels.append("malicious")
        # print(pred_labels)
        from utils import evaluate_classification
        from prepare_gold_labels import prepare_gold_labels
        gold_labels = prepare_gold_labels()
        print("================================================")
        print(f"TrainID: {args.load_model_from}")
        print(f"TestID: {test_id}")
        print(f"classification boundary threshold: {args.classify_boundary_threshold}")
        precision, recall, accuracy, f1 = evaluate_classification(pred_labels, gold_labels)
        save_evaluation_results(precision, recall, accuracy, f1)
'''
        print(len(out_batches))
        tmp_batches=[] 
        with open(args.data_saved_path+"/data.txt", "w") as fp:
            for out_batch in out_batches:
                print(type(out_batch))
                out_copy=torch.clone(out_batch)
                tmp_batches.append(out_copy.tolist())
            json.dump(tmp_batches, fp)
        


def paths_setting(save_models_dirname, mode="train"):
    test_id = ""
    # if mode == "test":
    #     test_dir = os.path.join(gv.model_save_path, save_models_dirname, "tests")
    #     if not os.path.exists(test_dir):
    #         os.makedirs(test_dir)
    #     test_id = str(int(time.time()))
    #     save_models_dirname = os.path.join(test_dir, test_id)
    #     if not os.path.exists(save_models_dirname):
    #         os.makedirs(save_models_dirname)
    gv.save_models_dirname = save_models_dirname
    if not os.path.exists(os.path.join(gv.model_save_path, gv.save_models_dirname)):
        os.makedirs(os.path.join(gv.model_save_path, gv.save_models_dirname))
    gv.morse_model_path = os.path.join(gv.model_save_path, gv.save_models_dirname, gv.morse_model_filename)
    gv.benign_thresh_model_path = os.path.join(gv.model_save_path, gv.save_models_dirname,
                                               gv.benign_thresh_model_filename)
    gv.suspect_env_model_path = os.path.join(gv.model_save_path, gv.save_models_dirname, gv.suspect_env_model_filename)
    gv.rnn_model_path = os.path.join(gv.model_save_path, gv.save_models_dirname, gv.rnn_model_filename)

    return test_id


if __name__ == '__main__':
    start_experiment()
