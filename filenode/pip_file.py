from filenode.file_node import FileNode as fn
from target import Target as tg

class PipFile(fn):
    def __init__(self, id: int, time: int, type: int, subtype: int, fd1: str, fd2: str):
        super(PipFile, self).__init__(id, time, type, subtype)
        self.fd1=fd1
        self.fd2=fd2

        self.iTag = tg.get_itag_benign()
        self.cTag = tg.get_itag_benign()
