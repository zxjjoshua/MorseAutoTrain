from filenode.file_node import FileNode as fn
from morse import Morse

class UnixSocketFile(fn):
    def __init__(self, id: int, time: int, type: int, subtype: int, unixSocketFd: str, morse: Morse):
        super(UnixSocketFile, self).__init__(id, time, type, subtype)
        self.unixSocketFd = unixSocketFd

        self.iTag = morse.get_itag_benign()
        self.cTag = morse.get_ctag_benign()
