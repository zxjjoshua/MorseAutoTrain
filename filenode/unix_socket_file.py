from filenode.file_node import FileNode as fn
from morse import Morse as tg

class UnixSocketFile(fn):
    def __init__(self, id: int, time: int, type: int, subtype: int, unixSocketFd: str):
        super(UnixSocketFile, self).__init__(id, time, type, subtype)
        self.unixSocketFd = unixSocketFd

        self.iTag = tg.get_itag_benign()
        self.cTag = tg.get_ctag_benign()
