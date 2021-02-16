from typing import Dict, Any
import numpy as np
from record import Record
from logging import getLogger
import event.event_parser as ep
import logging
from processnode import ProcessNode
from filenode.file_node import FileNode
from globals import GlobalVariable as gv

logger=getLogger("dataRead")
# processNodeSet: Dict[int, ProcessNode]={}
# fileNodeSet: Dict[int, FileNode]={}
# global processNodeSet
# global fileNodeSet
processNodeSet={}
fileNodeSet={}


def dataRead(fileName):
    f = open(fileName, "r")
    i=0
    while True:
        line = f.readline()
        if not line:
            break
        # print(line, "and", line[0])
        if line[:4]=="data":
            record=readObj(f)
            if record.type==1:
                # event type
                ep.EventParser.parse(record)
                pass
            elif record.type==-1:
                # file node
                if record.subtype>0 and record.subtype<5:
                    newNode=record.getFileNode()
                    if not newNode:
                        logger.error("failed to get file node")
                        continue
                    if gv.exist_fileNode(newNode.id):
                        logger.error("duplicate file node: " + str(newNode.id))
                    else:
                        gv.set_fileNode(newNode.id, newNode)
                elif record.subtype==5:
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
                        logger.error("duplicate process node: "+newNode.id)
                    else:
                        gv.set_processNode(newNode.id, newNode)
        i+=1
        # if i>5000:
        #     break

def readObj(f):
    event=Record()
    params=[]
    while True:
        line=f.readline()
        # print(line)
        if not line:
            break
        # print(line)
        if line[0]=="}":
            break
        line = pruningStr(line)
        if line=="paras {":
            while True:
                data=f.readline()
                data=pruningStr(data)
                if data=="}":
                    break
                words=data.split(": ")
                # print(words)
                params.append(pruningStr(words[1]))
        else:
            words=line.split(": ")
            # print(words)
            if words[0]=="ID":
                event.Id=int(pruningStr(words[1]))
            elif words[0]=="type":
                event.type=int(pruningStr(words[1]))
            elif words[0]=="time":
                event.time=int(pruningStr(words[1]))
            elif words[0]=="subtype":
                event.subtype=int(words[1])
            elif words[0]=="size":
                event.size=int(words[1])
            elif words[0]=="srcId":
                event.srcId=int(words[1])
            elif words[0]=="desId":
                event.desId = int(words[1])
    if params:
        event.params=params
    return event


def remove_file_node(id: int):
    if id in fileNodeSet:
        fileNodeSet.pop(id)

def remove_process_node(id: int):
    if id in processNodeSet:
        processNodeSet.pop(id)

def pruningStr(line):
    if not line:
        return line
    start=0
    while start<len(line):
        if line[start]==' ' or line[start]=='\t':
            start+=1
        else:
            break
    if line[start]=="\"":
        start+=1
    if line[-1]=="\"":
        line=line[:-1]
    if line[-1]=='\n':
        return line[start:-1]
    return line[start:]





if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, filename='debug.log')

    dataRead("./EventData/north_korea_apt_attack_data_debug.out")

