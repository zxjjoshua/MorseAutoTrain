class ProcessNode:
    def __init__(self, id: int, time: int, type: int, subtype: int, pid: int, ppid: int, cmdLine: str, processName: str):
        self.id=id
        self.timtime=time
        self.type=type
        self.subtype=subtype
        self.pid=pid
        self.ppid = ppid
        self.cmdLine = cmdLine
        self.processName=processName

        self.sTag: float = 0.0
        self.iTag: float = 0.0
        self.cTag: float = 0.0



