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
        self.morse_grad_list = []
        self.simple_net_grad_list = []
        # grad list stores grad of morse
        self.cur_state = np.zeros([2, 3])
        self.seq_len = 0


    def get_matrix_array(self, padding: 4):
        if padding < 4:
            return None

        return [self.subtype, 0.0, self.iTag, self.cTag] + [0] * (padding - 4)

    def add_event(self, event_id: int, event_type: int):
        self.event_list.append(event_id)
        self.event_type_list.append(event_type)

    def get_event_list(self) -> list:
        return self.event_list

    def get_event_id_list(self) -> list:
        return self.event_id_list

    def get_event_type_list(self) -> list:
        return self.event_type_list

    def state_update(self, state: np.array, event_type: int, event: np.array, morse_grad: np.ndarray, simple_net_grad: np.ndarray, event_id: int = None):
        if event_id is not None:
            self.cur_state = state
            # cur_state(12)

            self.state_list.append(state)
            self.event_list.append(event)
            # event(4,4)

            self.event_id_list.append(event_id)
            self.morse_grad_list.append(morse_grad)
            # morse_grad(12, 2)

            self.simple_net_grad_list.append(simple_net_grad)
            # simple_net_grad(12, 4)

            self.event_type_list.append(event_type)
            self.seq_len += 1

    def generate_sequence(self, batch_size=100, sequence_size=5):
        """
        :param batch_size: how many sequences in a batch
        :param sequence_size: how long a sequence is
        :return: a batch of sequences
        """
        if self.seq_len < sequence_size:
            return [[], [], []]
        res = []
        morse_grad_res = []
        simple_net_grad_res = []
        total_len = min(batch_size, self.seq_len - sequence_size + 1)
        for i in range(total_len):
            res.append(self.state_list[i:i + sequence_size])
            morse_grad_res.append(self.morse_grad_list[i:i + sequence_size])
            simple_net_grad_res.append(self.simple_net_grad_list[i:i + sequence_size])
        # if total_len < batch_size:
        #     res += [[]] * (batch_size - total_len)
        print(np.shape(morse_grad_res))
        return [res, morse_grad_res, simple_net_grad_res]
