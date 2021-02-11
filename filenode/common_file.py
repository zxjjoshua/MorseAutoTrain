from filenode.file_node import FileNode as fn
class CommonFile(fn):
    def __init__(self, id: int, time: int, type: int, subtype: int, fileName: str):
        super(CommonFile, self).__init__(id, time, type, subtype)
        self.fileName=fileName