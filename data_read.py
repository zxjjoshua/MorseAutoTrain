



from globals import GlobalVariable as gv
import train as tr
import data_rearrange
import train
import os




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







if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"
    logging.basicConfig(level=logging.INFO,
                        filename='debug.log',
                        filemode='w+',
                        format='%(asctime)s %(levelname)s:%(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    # dataRead(gv.train_data)
