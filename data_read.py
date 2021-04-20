from record import Record
from logging import getLogger
import event.event_parser as ep
import logging
from globals import GlobalVariable as gv
import train as tr
import data_rearrange
import train
import os

logger = getLogger("dataRead")


# processNodeSet: Dict[int, ProcessNode]={}
# fileNodeSet: Dict[int, FileNode]={}
# global processNodeSet
# global fileNodeSet
def predict():
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
                ep.EventParser.parse(record)

                # process batch-wise
                if event_num == max_event_per_epoch:
                    out_batches += train.predict()
                    event_num = 0

            elif record.type == -1:
                # file node
                if 0 < record.subtype < 5:
                    newNode = record.getFileNode()
                    if not newNode:
                        logger.error("failed to get file node")
                        continue
                    if gv.exist_fileNode(newNode.id):
                        logger.error("duplicate file node: " + str(newNode.id))
                    else:
                        gv.set_fileNode(newNode.id, newNode)
                elif record.subtype == -1:
                    # common file
                    newNode = record.getFileNode()
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
                    newNode = record.getProcessNode()
                    if not newNode:
                        logger.error("failed to get process node")
                        continue
                    if gv.exist_processNode(newNode.id):
                        logger.error("duplicate process node: " + newNode.id)
                    else:
                        gv.set_processNode(newNode.id, newNode)
        i += 1

    return out_batches

def dataRead():
    f = open(gv.train_data, "r")
    i = 0
    max_event_per_epoch = 100
    event_num = 0
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
                ep.EventParser.parse(record)
                # tr.back_propagate(record, 0.5)
                # data_rearrange.pre_process(record)

                # process batch-wise
                if event_num == max_event_per_epoch:
                    train.back_propagate_batch(0.05)
                    event_num = 0

            elif record.type == -1:
                # file node
                if 0 < record.subtype < 5:
                    newNode = record.getFileNode()
                    if not newNode:
                        logger.error("failed to get file node")
                        continue
                    if gv.exist_fileNode(newNode.id):
                        logger.error("duplicate file node: " + str(newNode.id))
                    else:
                        gv.set_fileNode(newNode.id, newNode)
                elif record.subtype == -1:
                    # common file
                    newNode = record.getFileNode()
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
                    newNode = record.getProcessNode()
                    if not newNode:
                        logger.error("failed to get process node")
                        continue
                    if gv.exist_processNode(newNode.id):
                        logger.error("duplicate process node: " + newNode.id)
                    else:
                        gv.set_processNode(newNode.id, newNode)
        i += 1
    f.close()
    # data_rearrange.post_train()


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


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"
    logging.basicConfig(level=logging.INFO,
                        filename='debug.log',
                        filemode='w+',
                        format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    # dataRead(gv.train_data)
