from record import Record
from filenode import FileNode
from processnode import ProcessNode
from event.event_processor import *
from globals import GlobalVariable as gv
from target import Target as tg
import numpy as np

logger = getLogger("EventParser")


class EventParser:
    succ_count = 0
    fail_count = 0

    @staticmethod
    def parse(record: Record):
        vector= np.zeros([4,4])
        morse_res = np.zeros([1,12])
        src_id = record.srcId
        des_id = record.desId
        event_id=record.Id
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
            vector = EventParser.process2file_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.read_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 5:
            vector = EventParser.process2file_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.read_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 6:
            vector = EventParser.process2file_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.read_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 7:
            vector = EventParser.process2file_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.read_process(vector)
            src_node = gv.get_processNode(src_id)
            des_node = gv.get_fileNode(des_id)
            pass
        elif record.subtype == 8:
            vector = EventParser.file2process_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.write_process(vector)
            src_node = gv.get_fileNode(src_id)
            des_node = gv.get_processNode(des_id)
            pass
        elif record.subtype == 9:
            vector = EventParser.file2process_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.write_process(vector)
            src_node = gv.get_fileNode(src_id)
            des_node = gv.get_processNode(des_id)
            pass
        elif record.subtype == 10:
            vector = EventParser.file2process_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.write_process(vector)
            src_node = gv.get_fileNode(src_id)
            des_node = gv.get_processNode(des_id)
            pass
        elif record.subtype == 11:
            vector = EventParser.file2process_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.write_process(vector)
            src_node = gv.get_fileNode(src_id)
            des_node = gv.get_processNode(des_id)
            pass
        elif record.subtype == 12:
            pass
        elif record.subtype == 13:
            pass
        elif record.subtype == 14:
            vector = EventParser.process2process_parser(record)
            # print("this is vector ",vector)
            if vector is None:
                return
            morse_res = EventProcessor.exec_process(vector)
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
            vector = EventParser.process2file_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.read_process(vector)
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
            vector = EventParser.process2file_parser(record)
            if vector is None:
                return
            morse_res = EventProcessor.load_process(vector)
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
            src_node.state_update(morse_res,subtype, vector, event_id)
            des_node.state_update(morse_res,subtype, vector, event_id)
            gv.succ_count+=1
        else:
            gv.fail_count+=1
            # print(gv.fail_count, "src_node: ",record.srcId, "des_node: ", record.desId, "subtype: ", record.subtype, record.Id)
        # print(event_id)

        gv.set_event_by_id(event_id, morse_res)
        # print(type(morse_res))
        return morse_res

    @staticmethod
    def file2process_parser(record: Record) -> np.array:
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


        params = [tg.get_attenuate_benign(), tg.get_attenuate_susp_env(), tg.get_benign_possibility(srcArray[1]).detach().numpy(),
                  tg.get_susp_possibility(srcArray[1]).detach().numpy()]
        benign_grad=tg.get_benign_thresh_grad()
        susp_grad=tg.get_susp_thresh_grad()
        gv.add_morse_grad(id, benign_grad+susp_grad)
        # print("params: ", params[2].detach().numpy())
        return np.array([eventArray, params, srcArray, desArray])

    @staticmethod
    def process2file_parser(record: Record) -> np.array:
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

        params = [tg.get_attenuate_benign(), tg.get_attenuate_susp_env(), tg.get_benign_possibility(srcArray[1]).detach().numpy(),
                  tg.get_susp_possibility(srcArray[1]).detach().numpy()]
        benign_grad = tg.get_benign_thresh_grad()
        susp_grad = tg.get_susp_thresh_grad()
        gv.add_morse_grad(id, benign_grad + susp_grad)
        return np.array([eventArray, params, srcArray, desArray])

    @staticmethod
    def process2process_parser(record: Record)->np.array:
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
        params = [tg.get_attenuate_benign(), tg.get_attenuate_susp_env(), tg.get_benign_possibility(srcArray[1]).item(),
                  tg.get_susp_possibility(srcArray[1]).item()]
        benign_grad = tg.get_benign_thresh_grad()
        susp_grad = tg.get_susp_thresh_grad()
        gv.add_morse_grad(id, benign_grad + susp_grad)
        return np.array([eventArray, params, srcArray, desArray])

    @staticmethod
    def file2file_parser(record: Record) -> np.array:
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

        params = [tg.get_attenuate_benign(), tg.get_attenuate_susp_env(), tg.get_benign_possibility(srcArray[1]),
                  tg.get_susp_possibility(srcArray[1])] + [0] * (4 - len(record.params))
        benign_grad = tg.get_benign_thresh_grad()
        susp_grad = tg.get_susp_thresh_grad()
        gv.add_morse_grad(id, benign_grad + susp_grad)
        return np.array([eventArray, params, srcArray, desArray])
