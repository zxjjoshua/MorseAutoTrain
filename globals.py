from typing import Dict, Optional
# from processnode import ProcessNode as pn
import processnode as pn
import filenode as fn
import record
import numpy as np


class GlobalVariable:
    fileNodeSet: Dict[int, fn.FileNode] = {}
    processNodeSet: Dict[int, pn.ProcessNode] = {}
    processNodePidMap: Dict[int, int] = {}
    event_set: Dict[int, np.ndarray] = {}

    simple_net_grad_set: Dict[int, np.ndarray] = {}
    # event_id: [benign_thresh_w_grad, benign_thresh_b_grad, susp_thresh_w_grad, susp_thresh_b_grad]

    # batch processing
    node_list={}
    batch_size = 50
    sequence_size = 5
    feature_size = 12

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
    def get_event_by_id(cls, id: int)-> np.ndarray:
        # print(id)
        if id in cls.event_set:
            return cls.event_set[id]
        return None

    @classmethod
    def set_event_by_id(cls, id: int, morese_result: np.ndarray):
        cls.event_set[id]=morese_result

    # -------------- node list ------------------ #
    @classmethod
    def get_node_list(cls):
        return cls.node_list

    @classmethod
    def empty_node_list(cls):
        cls.node_list=[]

    @classmethod
    def add_node_list(cls, node: object):
        cls.node_list.append(node)

    # -------------- grad dict ------------------ #
    @classmethod
    def add_morse_grad(cls, event_id: int, grad_list: np.ndarray((1,4))):
        if event_id in cls.simple_net_grad_set:
            print("globals  morse_grad_add() : duplicate event id")
        cls.simple_net_grad_set[event_id]=grad_list

    @classmethod
    def get_morse_grad(cls, event_id)->np.ndarray((1,4)):
        if event_id not in cls.simple_net_grad_set:
            print("globals  morse_grad_get() : no such event")
        return cls.simple_net_grad_set[event_id]

    # -------------- testing ------------------ #
    succ_count=0
    fail_count=0