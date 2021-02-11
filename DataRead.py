import numpy as np
from record import Record
from logging import getLogger

processNodeSet={}
fileNodeSet={}
logger=getLogger("dataRead")

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
                pass

            elif record.type==-1 and record.subtype==5 and not record.params:
                # event
                pass
            elif record.type==-1:
                if record.subtype>0 and record.subtype<5:
                    newNode=record.getFileNode()
                    if not newNode:
                        logger.error("failed to get file node")
                        continue
                    if newNode.id in fileNodeSet:
                        logger.error("duplicate file node: " + str(newNode.id))
                    else:
                        fileNodeSet[newNode.id]=newNode
                elif record.subtype==5:
                    newNode = record.getProcessNode()
                    if not newNode:
                        logger.error("failed to get process node")
                        break
                        continue
                    if newNode.id in processNodeSet:
                        logger.error("duplicate process node: "+newNode.id)
                    else:
                        processNodeSet[newNode.id] = newNode
        i+=1
        # if i>10000:
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
                event.srcId=words[1]
            elif words[0]=="desId":
                event.desId = words[1]
    if event.Id==99712:
        logger.info(params)
    if params:
        event.params=params
    return event


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
    dataRead("./EventData/debug.out")

