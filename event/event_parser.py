from record import Record
import numpy as np
# from DataRead import processNodeSet, fileNodeSet
import data_read
from logging import getLogger
from filenode import FileNode
from processnode import ProcessNode
from event.event_processor import *
import logging
from globals import GlobalVariable as gv

logger=getLogger("EventParser")

class EventParser:
    @staticmethod
    def parse(record: Record):
        vector: np.array
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
            EventProcessor.read_process(vector)
            pass
        elif record.subtype == 5:
            pass
        elif record.subtype == 6:
            pass
        elif record.subtype == 7:
            pass
        elif record.subtype == 8:
            EventParser.file2process_parser(record)
            pass
        elif record.subtype == 9:
            pass
        elif record.subtype == 10:
            pass
        elif record.subtype == 11:
            pass
        elif record.subtype == 12:
            pass
        elif record.subtype == 13:
            pass
        elif record.subtype == 14:
            vector = EventParser.process2process_parser(record)
            if vector is None:
                return
            EventProcessor.exec_process(vector)
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
            EventParser.process2file_parser(record)
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
            EventProcessor.load_process(vector)
            pass
        elif record.subtype == 36:
            pass
        elif record.subtype == 37:
            pass


    @staticmethod
    def file2process_parser(record):
        id = record.Id
        time = record.time
        subtype = record.subtype
        srcNode: FileNode
        destNode: ProcessNode
        if not gv.exist_fileNode(record.srcId):
            logger.error("file to process, can't find srcNode "+ str(record.srcId))
            return None
        else:
            srcNode = gv.get_fileNode(record.srcId)
        if not gv.exist_processNode(record.desId):
            logger.error("file to process, can't find desNode "+ str(record.desId))
            return None
        else:
            destNode = gv.get_processNode(record.desId)
        if not srcNode or not destNode:
            logger.error("file to process, can't find srcNode or destNode "+' '+str(record.srcId)+' '+ str(record.desId))
            return None

        eventArray = [id, time, subtype, 0]
        params = record.params+[0]*(4-len(record.params))
        srcArray = srcNode.getMatrixArray(4)
        desArray = destNode.getMatrixArray(4)
        return np.array([eventArray, params, srcArray, desArray])


    @staticmethod
    def process2file_parser(record):
        id = record.Id
        time = record.time
        subtype = record.subtype
        srcNode: ProcessNode = None
        destNode: FileNode = None
        if not gv.exist_processNode(record.srcId):
            logger.error("process to file, can't find srcNode "+ str(record.srcId))
            return None
        else:
            srcNode = gv.get_processNode(record.srcId)
        if not gv.exist_fileNode(record.desId):
            logger.error("process to file, can't find desNode"+ ' '+ str(record.desId))
            return None
        else:
            destNode = gv.get_fileNode(record.desId)
        if not srcNode or not destNode:
            logger.error("process to file, can't find desNode or destNode "+ str(record.srcId)+' '+ str(record.desId))
            return None

        eventArray = [id, time, subtype, 0]
        params = record.params + [0] * (4 - len(record.params))
        srcArray = srcNode.getMatrixArray(4)
        desArray = destNode.getMatrixArray(4)
        return np.array([eventArray, params, srcArray, desArray])

    @staticmethod
    def process2process_parser(record):
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
        params = record.params + [0] * (4 - len(record.params))
        srcArray = srcNode.getMatrixArray(4)
        desArray = destNode.getMatrixArray(4)
        return np.array([eventArray, params, srcArray, desArray])

    @staticmethod
    def file2file_parser(record):
        id = record.Id
        time = record.time
        subtype = record.subtype
        srcNode: FileNode = None
        destNode: FileNode = None
        if not gv.exist_fileNode(record.srcId):
            logger.error("file to file, can't find srcNode"+' '+  str(record.srcId))
            return None
        else:
            srcNode = gv.get_fileNode(record.srcId)
        if not gv.exist_fileNode(record.desId):
            logger.error("file to file, can't find desNode"+' '+  str(record.desId))
            return None
        else:
            destNode=gv.get_fileNode(record.desId)
        if not srcNode or not destNode:
            logger.error("file to file, can't find desNode or destNode"+' '+  str(record.srcId)+' '+  str(record.desId))
            return None

        eventArray = [id, time, subtype, 0]
        params = record.params + [0] * (4 - len(record.params))
        srcArray = srcNode.getMatrixArray(4)
        desArray = destNode.getMatrixArray(4)
        return np.array([eventArray, params, srcArray, desArray])






