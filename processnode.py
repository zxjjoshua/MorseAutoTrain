__all__ = ['ProcessNode']

from target import Target as tg
import numpy as np
import torch
# from globals import GlobalVariable as gv


class ProcessNode:
    def __init__(self, id: int, time: int, type: int, subtype: int, pid: int, ppid: int, cmdLine: str,
                 processName: str):
        self.id = id
        self.time = time
        self.type = type
        self.subtype = subtype
        self.pid = pid
        self.ppid = ppid
        self.cmdLine = cmdLine
        self.processName = processName

        self.event_list = []
        self.event_id_list = []
        self.event_type_list = []
        self.state_list = []
        self.grad_list=[]
        self.cur_state = np.zeros([2,3])
        self.seq_len = 0

        # init tags
        self.sTag: float = 0.0
        # benign
        self.iTag: float = 0.0
        #
        self.cTag: float = 0.0

        if self.ppid == -1:
            # unknown parent
            self.sTag = tg.get_stag_dangerous()
        elif self.ppid == 0:
            # process generated by root
            self.sTag = tg.get_stag_benign()
        else:
            from globals import GlobalVariable as gv
            parent_id = gv.get_processNode_by_pid(self.ppid)
            parent_node = gv.get_processNode(parent_id)
            if not parent_node:
                # parent node not exist or has been released, then this node is not valid
                self.sTag = tg.get_stag_dangerous()
                self.iTa = tg.get_itag_dangerous()
                self.cTag = tg.get_ctag_dangerous()
            else:
                self.sTag = parent_node.sTag
                self.iTag = parent_node.iTag
                self.cTag = parent_node.cTag

    def get_matrix_array(self, padding: 4):
        if padding < 4:
            return None

        return [self.subtype, self.sTag, self.iTag, self.cTag] + [0] * (padding - 4)

    def add_event(self, event_id: int, event_type: int):
        # print(event_id)
        self.event_list.append(event_id)
        self.event_type_list.append(event_type)

    def get_event_list(self) -> list:
        return self.event_list
    
    def get_event_id_list(self) -> list:
        return self.event_id_list

    def get_event_type_list(self) -> list:
        return self.event_type_list

    def state_update(self,state: np.array, event_type: int, event: np.array, event_id: int=None):
        from globals import GlobalVariable as gv
        # print(event)
        if event_id is not None:
            self.cur_state = state
            self.state_list.append(state)
            self.event_list.append(event)

            self.event_id_list.append(event_id)
            grad = gv.get_morse_grad(event_id)
            self.grad_list.append(grad)
            self.event_type_list.append(event_type)
            self.seq_len += 1

    def generate_sequence(self, batch_size=100, sequence_size=5):
        """
        :param batch_size: how many sequences in a batch
        :param sequence_size: how long a sequence is
        :return: a batch of sequences
        """
        if self.seq_len < sequence_size:
            return []
        res = []
        grad_res=[]
        total_len = min(batch_size, self.seq_len - sequence_size + 1)
        for i in range(total_len):
            res.append(self.state_list[i:i + sequence_size])
            grad_res.append(self.grad_list[i:i + sequence_size])
        # if total_len < batch_size:
        #     res += [[]] * (batch_size - total_len)
        return [res, grad_res]

    def generate_simple_net_grad_sequence(self, batch_size=100, sequence_size=5):
        """
        :param batch_size: how many sequences in a batch
        :param sequence_size: how long a sequence is
        :return: a batch of sequences
        """
        if self.seq_len < sequence_size:
            return []
        res = []
        total_len = min(batch_size, self.seq_len - sequence_size + 1)
        for i in range(total_len):
            res.append(self.grad_list[i:i + sequence_size])
        # if total_len < batch_size:
        #     res += [[]] * (batch_size - total_len)
        return res
