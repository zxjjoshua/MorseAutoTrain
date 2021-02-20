from filenode.file_node import FileNode as fn
from target import Target as tg

class InetSocketFile(fn):
    def __init__(self, id: int, time: int, type: int, subtype: int, inetSocketFd: str, ip: str, port:int):
        super(InetSocketFile, self).__init__(id, time, type, subtype)
        self.inetSocketFd=inetSocketFd
        self.ip=ip
        self.port=port

        # for localhost connection or local network connection, we set them as benign

        if ip :
            self.iTag = tg.get_itag_benign()
            self.cTag = tg.get_ctag_benign()
        else:
            self.iTag = tg.get_itag_susp_env()
            self.cTag = tg.get_itag_susp_env()