import numpy as np
import torch


class FileNode:
    def __init__(self, id: int, time: int, type: int, subtype: int):
        self.id=id
        self.time=time
        self.type=type
        self.subtype=subtype

        self.sTag: float = -1.0
        self.iTag: float = 0.0
        self.cTag: float = 0.0

        self.event_list = []
        self.event_id_list = []
        self.event_type_list = []
        self.state_list = []
        self.cur_state = np.zeros([2, 3])
        self.seq_len = 0


    def get_matrix_array(self, padding: 4):
        if padding < 4:
            return None

        return [self.subtype, 0.0, self.iTag, self.cTag] + [0] * (padding - 4)

    def add_event(self, event_id: int, event_type: int):
        self.event_list.append(event_id)
        self.event_type_list.append(event_type)

    def get_event_list(self)->list:
        return self.event_list

    def get_event_type_list(self) -> list:
        return self.event_type_list

    def state_update(self,state: np.array, event_type: int, event: np.array, event_id: int=None):
        self.cur_state = state
        self.state_list.append(state)
        self.event_list.append(event)
        if event_id is not None:
            self.event_id_list.append(event_id)
        self.event_type_list.append(event_type)
        self.seq_len += 1

    def generate_sequence(self, length:5):
        if self.seq_len<length:
            return []
        res=[]
        for i in range(len(self.state_list)-length):
            res.append(self.state_list[i:i+length+1])
        return torch.tensor(res)
