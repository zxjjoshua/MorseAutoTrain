from record import Record
from filenode import FileNode
from processnode import ProcessNode
from event.event_processor import *
from globals import GlobalVariable as gv
from morse import Morse
import numpy as np
import morse_train

logger = getLogger("EventParser")


class EventParser:

    def __init__(self, morse: Morse):
        self.succ_count = 0
        self.fail_count = 0
        self.event_processor = EventProcessor(morse)

    def parse(self, record: Record, morse: Morse = None) -> np.ndarray:
        '''
        parse record data, and convert it to np.ndarray((1,12)),
        if failed, np.zeros([1, 12]) will be returned.
        :param record: Record
        :return: morse_res, np.ndarray((1,12))
        '''
        vector = np.zeros([4, 4])
        morse_res = np.zeros([1, 12])
        src_id = record.srcId
        des_id = record.desId
        event_id = record.Id
        subtype = record.subtype
        src_node = None
        des_node = None

        if record.subtype == 1:
            pass
        elif record.subtype == 2:
            pass
        elif record.subtype == 3:
            pass
        elif record.subtype == 4:
            vector = self.process2file_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.read_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 5:
            vector = self.process2file_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.read_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 6:
            vector = self.process2file_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.read_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 7:
            vector = self.process2file_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.read_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 8:
            vector = self.file2process_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.write_process(vector)
            src_node = gv.get_fileNode(src_id)
            des_node = gv.get_processNode(des_id)
            pass
        elif record.subtype == 9:
            vector = self.file2process_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.write_process(vector)
            src_node = gv.get_fileNode(src_id)
            des_node = gv.get_processNode(des_id)
            pass
        elif record.subtype == 10:
            vector = self.file2process_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.write_process(vector)
            src_node = gv.get_fileNode(src_id)
            des_node = gv.get_processNode(des_id)
            pass
        elif record.subtype == 11:
            vector = self.file2process_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.write_process(vector)
            src_node = gv.get_fileNode(src_id)
            des_node = gv.get_processNode(des_id)
            pass
        elif record.subtype == 12:
            pass
        elif record.subtype == 13:
            pass
        elif record.subtype == 14:
            vector = self.process2process_parser(record, morse)
            # print("this is vector ",vector)
            if vector is None:
                return
            morse_res = self.event_processor.exec_process(vector, morse)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_processNode(des_id)
            pass
        elif record.subtype == 15:
            pass
        elif record.subtype == 16:
            pass
        elif record.subtype == 17:
            pass
        elif record.subtype == 18:
            pass
        elif record.subtype == 19:
            pass
        elif record.subtype == 20:
            pass
        elif record.subtype == 21:
            pass
        elif record.subtype == 22:
            pass
        elif record.subtype == 23:
            pass
        elif record.subtype == 24:
            pass
        elif record.subtype == 25:
            pass
        elif record.subtype == 26:
            pass
        elif record.subtype == 27:
            vector = self.process2file_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.read_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 28:
            pass
        elif record.subtype == 29:
            pass
        elif record.subtype == 30:
            pass
        elif record.subtype == 31:
            pass
        elif record.subtype == 32:
            pass
        elif record.subtype == 33:
            pass
        elif record.subtype == 34:
            pass
        elif record.subtype == 35:
            vector = self.process2file_parser(record, morse)
            if vector is None:
                return
            morse_res = self.event_processor.load_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 36:
            pass
        elif record.subtype == 37:
            pass
        morse_res = np.array(morse_res)
        if src_node and des_node:
            # print(vector)
            # print("src_node: ", src_node.seq_len, "des_node: ", des_node.seq_len, "subtype: ", record.subtype)

            # get morse_grad and simple_net grad, and do multiplication to get morse_simple_net_grad
            # simple_net_grad: np.array(4)
            # morse_grad: np.array(12, 4)
            # morse_simple_net_grad: np.array(12, 4)
            simple_net_grad=gv.get_morse_grad(event_id)
            morse_grad=morse_train.get_morse_grad(record.subtype, vector, self.event_processor)
            morse_grad=np.array(morse_grad)
            morse_simple_net_grad = np.transpose(np.array([morse_grad[:, 2], morse_grad[:, 2], morse_grad[:, 3], morse_grad[:, 3]]))*simple_net_grad
            morse_grad=morse_grad[:, 0:2]
            src_node.state_update(morse_res, subtype, vector, morse_grad, morse_simple_net_grad, event_id)
            des_node.state_update(morse_res, subtype, vector, morse_grad, morse_simple_net_grad, event_id)
            gv.succ_count += 1
        else:
            gv.fail_count += 1
            # print(gv.fail_count, "src_node: ",record.srcId, "des_node: ", record.desId, "subtype: ",
            # record.subtype, record.Id)
        # print(event_id)

        gv.set_event_by_id(event_id, morse_res)
        # print(type(morse_res))
        return morse_res

    def file2process_parser(self, record: Record, morse: Morse = None) -> np.ndarray((4,4)):
        id = record.Id
        time = record.time
        subtype = record.subtype
        srcNode: FileNode
        destNode: ProcessNode
        if not gv.exist_fileNode(record.srcId):
            if record.srcId == -1:
                return None
            logger.error("file to process, can't find srcNode " + str(record.srcId))
            return None
        else:
            srcNode = gv.get_fileNode(record.srcId)
        if not gv.exist_processNode(record.desId):
            logger.error("file to process, can't find desNode " + str(record.desId))
            return None
        else:
            destNode = gv.get_processNode(record.desId)
        if not srcNode or not destNode:
            logger.error(
                "file to process, can't find srcNode or destNode " + ' ' + str(record.srcId) + ' ' + str(record.desId))
            return None

        eventArray = [id, time, subtype, 0]
        srcArray = srcNode.get_matrix_array(4)
        desArray = destNode.get_matrix_array(4)

        params = [morse.get_attenuate_benign(), morse.get_attenuate_susp_env(),
                  morse.get_benign_possibility(srcArray[1]).cpu().detach().numpy(),
                  morse.get_susp_possibility(srcArray[1]).cpu().detach().numpy()]
        benign_grad = morse.get_benign_thresh_grad()
        susp_grad = morse.get_susp_thresh_grad()
        gv.add_morse_grad(id, np.concatenate([benign_grad, susp_grad]))
        # print("params: ", params[2].detach().numpy())
        return np.array([eventArray, params, srcArray, desArray])

    def process2file_parser(self, record: Record, morse: Morse) -> np.ndarray((4,4)):
        id = record.Id
        time = record.time
        subtype = record.subtype
        srcNode: ProcessNode = None
        destNode: FileNode = None
        if not gv.exist_processNode(record.srcId):
            logger.error("process to file, can't find srcNode " + str(record.srcId))
            return None
        else:
            srcNode = gv.get_processNode(record.srcId)
        if not gv.exist_fileNode(record.desId):
            if record.desId == -1:
                return None
            logger.error("process to file, can't find desNode" + ' ' + str(record.desId))
            return None
        else:
            destNode = gv.get_fileNode(record.desId)
        if not srcNode or not destNode:
            logger.error(
                "process to file, can't find desNode or destNode " + str(record.srcId) + ' ' + str(record.desId))
            return None

        eventArray = [id, time, subtype, 0]
        srcArray = srcNode.get_matrix_array(4)
        desArray = destNode.get_matrix_array(4)

        params = [morse.get_attenuate_benign(), morse.get_attenuate_susp_env(),
                  morse.get_benign_possibility(srcArray[1]).cpu().detach().numpy(),
                  morse.get_susp_possibility(srcArray[1]).cpu().detach().numpy()]
        benign_grad = morse.get_benign_thresh_grad()
        susp_grad = morse.get_susp_thresh_grad()
        gv.add_morse_grad(id, np.concatenate([benign_grad, susp_grad]))
        return np.array([eventArray, params, srcArray, desArray])

    def process2process_parser(self, record: Record, morse: Morse) -> np.ndarray((4,4)):
        id = record.Id
        time = record.time
        subtype = record.subtype
        srcNode: ProcessNode = None
        destNode: ProcessNode = None
        if not gv.exist_processNode(record.srcId):
            logger.error("process to process, can't find srcNode " + str(record.srcId))
            return None
        else:
            srcNode = gv.get_processNode(record.srcId)
        if not gv.exist_processNode(record.desId):
            logger.error("process to process, can't find desNode" + ' ' + str(record.desId))
            return None
        else:
            destNode = gv.get_processNode(record.desId)
        if not srcNode or not destNode:
            logger.error(
                "process to process, can't find desNode or destNode " + str(record.srcId) + ' ' + str(record.desId))
            return None

        eventArray = [id, time, subtype, 0]
        srcArray = srcNode.get_matrix_array(4)
        desArray = destNode.get_matrix_array(4)
        params = [morse.get_attenuate_benign(), morse.get_attenuate_susp_env(), morse.get_benign_possibility(srcArray[1]).item(),
                  morse.get_susp_possibility(srcArray[1]).item()]
        benign_grad = morse.get_benign_thresh_grad()
        susp_grad = morse.get_susp_thresh_grad()
        gv.add_morse_grad(id, np.concatenate([benign_grad, susp_grad]))
        return np.array([eventArray, params, srcArray, desArray])

    def file2file_parser(self, record: Record, morse: Morse) -> np.ndarray((4,4)):
        id = record.Id
        time = record.time
        subtype = record.subtype
        srcNode: FileNode = None
        destNode: FileNode = None
        if not gv.exist_fileNode(record.srcId):
            logger.error("file to file, can't find srcNode" + ' ' + str(record.srcId))
            return None
        else:
            srcNode = gv.get_fileNode(record.srcId)
        if not gv.exist_fileNode(record.desId):
            logger.error("file to file, can't find desNode" + ' ' + str(record.desId))
            return None
        else:
            destNode = gv.get_fileNode(record.desId)
        if not srcNode or not destNode:
            logger.error(
                "file to file, can't find desNode or destNode" + ' ' + str(record.srcId) + ' ' + str(record.desId))
            return None

        eventArray = [id, time, subtype, 0]
        srcArray = srcNode.get_matrix_array(4)
        desArray = destNode.get_matrix_array(4)

        params = [morse.get_attenuate_benign(), morse.get_attenuate_susp_env(), morse.get_benign_possibility(srcArray[1]),
                  morse.get_susp_possibility(srcArray[1])] + [0] * (4 - len(record.params))
        benign_grad = morse.get_benign_thresh_grad()
        susp_grad = morse.get_susp_thresh_grad()
        gv.add_morse_grad(id, np.concatenate([benign_grad, susp_grad]))
        return np.array([eventArray, params, srcArray, desArray])

