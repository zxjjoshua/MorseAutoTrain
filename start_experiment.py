import os
import fire
import json
from globals import GlobalVariable as gv
import torch
import logging
import argparse

def start_experiment(config="config.json"):
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"
    parser = argparse.ArgumentParser(description="train or test the model")
    parser.add_argument("--batch_size", nargs='?', default=5, type=int)
    parser.add_argument("--sequence_length", nargs='?', default=5, type=int)
    parser.add_argument("--feature_dimension", nargs='?', default=12, type=int)
    parser.add_argument("--device", nargs='?', default="cuda", type=str)
    parser.add_argument("--train_data", nargs='?', default="./EventData/north_korea_apt_attack_data_debug.out", type=str)
    parser.add_argument("--test_data", nargs='?', default="./EventData/north_korea_apt_attack_data_debug.out", type=str)
    parser.add_argument("--validation_data", nargs='?', default="./EventData/north_korea_apt_attack_data_debug.out", type=str)
    parser.add_argument("--model_save_path", nargs='?', default="./trainedModels", type=str)
    parser.add_argument("--mode", nargs="?", default="train", type=str)
    parser.add_argument("--early_stopping_patience", nargs="?", default=10, type=int)
    parser.add_argument("--early_stopping_threshold", nargs="?", default=10, type=int)

    args = parser.parse_args()
    
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
    from data_read import dataRead
    # with open(config, 'r') as config_json:
    #     config = json.load(config_json)
    # gv.batch_size = config["batch_size"]
    # gv.sequence_size = config["sequence_length"]
    # gv.feature_size = config["feature_dimension"]
    # if torch.cuda.is_available():
    #     gv.device = torch.device(config["device"])
    # gv.train_data = config["train_data"]
    # gv.model_save_path = config["model_save_path"]

    
    logging.basicConfig(level=logging.INFO,
                        filename='debug.log',
                        filemode='w+',
                        format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    dataRead()



if __name__ == '__main__':
    start_experiment()