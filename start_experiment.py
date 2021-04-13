import os
import fire
import json
from globals import GlobalVariable as gv
import torch
from data_read import dataRead
import logging

def start_experiment(config="config.json"):
    with open(config, 'r') as config_json:
        config = json.load(config_json)
    gv.batch_size = config["batch_size"]
    gv.sequence_size = config["sequence_length"]
    gv.feature_size = config["feature_dimension"]
    if torch.cuda.is_available():
        gv.device = torch.device(config["device"])
    gv.train_data = config["train_data"]

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"
    logging.basicConfig(level=logging.INFO,
                        filename='debug.log',
                        filemode='w+',
                        format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    dataRead()



if __name__ == '__main__':
    fire.Fire(start_experiment)