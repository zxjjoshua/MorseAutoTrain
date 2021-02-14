from processnode import ProcessNode
from filenode import FileNode, CommonFile, InetSocketFile, PipFile, UnixSocketFile
from logging import getLogger
import traceback

logger=getLogger("Record")

class Record:

    def __init__(self):
        self.Id:int
        self.time:int
        self.type:int
        self.subtype:int
        self.size:int
        self.desId:int
        self.srcId:int
        self.params:list

        self.Id = -1
        self.time = -1
        self.subtype = -1
        self.type=-1
        self.size = -1
        self.desId=-1
        self.srcId=-1
        self.params = []


    # def __init__(self, id, time, subtype, size, params):
    #     self.Id=id
    #     self.time=time
    #     self.subtype=subtype
    #     self.size=size
    #     self.params=params

    def getFileNode(self)->FileNode:
        if self.type!=-1:
            return None
        try:
            if self.subtype==1:
                if (len(self.params)!=1):
                    return None
                res=CommonFile(self.Id, self.time, self.type, self.subtype, self.params[0])
                return res
            elif self.subtype==2:
                if (len(self.params) != 1):
                    return None
                res = UnixSocketFile(self.Id, self.time, self.type, self.subtype, self.params[0])
                return res
            elif self.subtype==3:
                if (len(self.params) != 3):
                    return None
                res = InetSocketFile(self.Id, self.time, self.type, self.subtype, self.params[0], self.params[1], self.params[2])
                return res
            elif self.subtype==4:
                if (len(self.params) != 2):
                    return None
                res = PipFile(self.Id, self.time, self.type, self.subtype, self.params[0], self.params[1])
                return res
            else:
                print("unexpected filenode subtype", self.subtype)
        except Exception as err:
            logger.error("get file node failed")
            msg=str(self.Id)+" "+str(self.time)+" "+str(self.type)+" "+str(self.subtype)+" "+str(self.params)
            logger.error(msg)
            traceback.print_exc()
            # print("get file node failed")

        return None





    def getProcessNode(self):
        if self.type!=-1:
            return None
        if self.subtype!=5:
            return None
        if len(self.params)!=4:
            logger.error("lack of params")
            # print(self.Id, self.type, self.subtype, self.params)
            return None
        processNode=ProcessNode(self.Id, self.time, self.type, self.subtype, self.params[0],self.params[1],self.params[2],self.params[3])
        return processNode



