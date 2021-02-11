from filenode.file_node import FileNode as fn
class InetSocketFile(fn):
    def __init__(self, id: int, time: int, type: int, subtype: int, inetSocketFd: str, ip: str, port:int):
        super(InetSocketFile, self).__init__(id, time, type, subtype)
        self.inetSocketFd=inetSocketFd
        self.ip=ip
        self.port=port