class FileNode:
    def __init__(self, id: int, time: int, type: int, subtype: int):
        self.id=id
        self.time=time
        self.type=type
        self.subtype=subtype

        self.iTag: float = 0.0
        self.cTag: float = 0.0

    def getMatrixArray(self, padding: 4):
        if padding < 4:
            return None
        return [self.subtype, 0.0, self.iTag, self.cTag] + [0] * (padding - 4)


