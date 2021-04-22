from filenode.file_node import FileNode as fn
from morse import Morse as tg


class CommonFile(fn):
    def __init__(self, id: int, time: int, type: int, subtype: int, fileName: str):
        super(CommonFile, self).__init__(id, time, type, subtype)
        self.fileName=fileName
        self.iTag = tg.get_itag_benign()
        self.cTag = tg.get_itag_benign()
