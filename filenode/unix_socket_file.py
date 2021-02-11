from filenode.file_node import FileNode as fn
class UnixSocketFile(fn):
    def __init__(self, id: int, time: int, type: int, subtype: int, unixSocketFd: str):
        super(UnixSocketFile,self).__init__(id, time, type, subtype)
        self.unixSocketFd=unixSocketFd