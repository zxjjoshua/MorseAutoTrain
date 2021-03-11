from typing import Dict, Optional
# from processnode import ProcessNode as pn
import processnode as pn
import filenode as fn
import record

class GlobalVariable:
    fileNodeSet: Dict[int, fn.FileNode] = {}
    processNodeSet: Dict[int, pn.ProcessNode] = {}
    processNodePidMap: Dict[int, int] = {}
    event_set: Dict[int, record.Record] = {}

    # batch processing


    # # init value
    # stag_benign = 0.5
    # stag_suspect_env = 0.25
    # stag_dangerous = 0.2
    # itag_benign = 0.5
    # itag_suspect_env = 0.25
    # itag_dangerous = 0.2
    # ctag_benign = 0.5
    # ctag_suspect_env = 0.25
    # ctag_dangerous = 0.2
    #
    # # threshold
    # benign = 0.5
    # suspect_env = 0.25
    #
    # # decay and attenuation
    # a_b = 0.1
    # a_e = 0.05

    # -------------- fileNodeSet and processNodeSet ----------------- #
    @classmethod
    def get_fileNode(cls, index: int) -> Optional[fn.FileNode]:
        return cls.fileNodeSet[index] if index in cls.fileNodeSet else None

    @classmethod
    def set_fileNode(cls, index: int, node: fn.FileNode):
        cls.fileNodeSet[index] = node

    @classmethod
    def exist_fileNode(cls, index: int) -> bool:
        return True if index in cls.fileNodeSet else False

    @classmethod
    def remove_fileNode(cls, index: int):
        if index in cls.fileNodeSet:
            cls.fileNodeSet.pop(index)

    @classmethod
    def get_processNode(cls, index: int) -> pn.ProcessNode:
        return cls.processNodeSet[index] if index in cls.processNodeSet else None

    @classmethod
    def set_processNode(cls, index: int, node: pn.ProcessNode):
        cls.processNodeSet[index] = node

    @classmethod
    def exist_processNode(cls, index: int) -> bool:
        return True if index in cls.processNodeSet else False

    @classmethod
    def remove_processNode(cls, index: int):
        if index in cls.processNodeSet:
            cls.processNodeSet.pop(index)

    # -------------- processNodePidMap ------------- #
    @classmethod
    def get_processNode_by_pid(cls, pid) -> int:
        if pid in cls.processNodePidMap:
            return cls.processNodePidMap[pid]
        return -1

    # -------------- event map ------------------ #
    @classmethod
    def get_event_by_id(cls, id: int)-> record.Record:
        if id in cls.event_set:
            return cls.event_set[id]
        return None

    @classmethod
    def set_event_by_id(cls, id: int, record: record.Record):
        cls.event_set[id]=record

    # -------------- tag getters ------------------ #
    # ---  deprecated

    # @classmethod
    # def get_benign_thresh(cls) -> float:
    #     return cls.benign
    #
    # @classmethod
    # def get_stag_benign(cls) -> float:
    #     return cls.stag_benign
    #
    # @classmethod
    # def get_itag_benign(cls) -> float:
    #     return cls.itag_benign
    #
    # @classmethod
    # def get_ctag_benign(cls) -> float:
    #     return cls.ctag_benign
    #
    # @classmethod
    # def get_stag_susp_env(cls) -> float:
    #     return cls.stag_suspect_env
    #
    # @classmethod
    # def get_itag_susp_env(cls) -> float:
    #     return cls.itag_suspect_env
    #
    # @classmethod
    # def get_ctag_susp_env(cls) -> float:
    #     return cls.ctag_suspect_env
    #
    # @classmethod
    # def get_stag_dangerous(cls) -> float:
    #     return cls.stag_dangerous
    #
    # @classmethod
    # def get_itag_dangerous(cls) -> float:
    #     return cls.itag_dangerous
    #
    # @classmethod
    # def get_ctag_dangerous(cls) -> float:
    #     return cls.ctag_dangerous
    #
    # @classmethod
    # def get_attenuate_susp_env(cls) -> float:
    #     return cls.a_e
    #
    # @classmethod
    # def get_attenuate_benign(cls) -> float:
    #     return cls.a_b
    #
    # # ------------------ tag setters -------------- #
    #
    # @classmethod
    # def set_stag_benign(cls, val):
    #     cls.stag_benign = val
    #
    # @classmethod
    # def set_itag_benign(cls, val):
    #     cls.itag_benign = val
    #
    # @classmethod
    # def set_ctag_benign(cls, val):
    #     cls.ctag_benign = val
    #
    # @classmethod
    # def set_stag_susp_env(cls, val):
    #     cls.stag_suspect_env = val
    #
    # @classmethod
    # def set_itag_susp_env(cls, val):
    #     cls.itag_suspect_env = val
    #
    # @classmethod
    # def set_ctag_susp_env(cls, val):
    #     cls.ctag_suspect_env = val
    #
    # @classmethod
    # def set_stag_dangerous(cls, val):
    #     cls.stag_dangerous = val
    #
    # @classmethod
    # def set_itag_dangerous(cls, val):
    #     cls.itag_dangerous = val
    #
    # @classmethod
    # def set_itag_dangerous(cls, val):
    #     cls.itag_dangerous = val
