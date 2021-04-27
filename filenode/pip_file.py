from filenode.file_node import FileNode as fn
from morse import Morse as tg

class PipFile(fn):
    def __init__(self, id: int, time: int, type: int, subtype: int, fd1: str, fd2: str, iTag, cTag):
        super(PipFile, self).__init__(id, time, type, subtype)
        self.fd1=fd1
        self.fd2=fd2

        self.iTag = iTag
        self.cTag = cTag
