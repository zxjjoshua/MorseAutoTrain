__all__ = ['ProcessNode']

from morse import Morse
import numpy as np
import torch


# from globals import GlobalVariable as gv


class ProcessNode:
    def __init__(self, id: int, time: int, type: int, subtype: int, pid: int, ppid: int, cmdLine: str,
                 processName: str, morse: Morse = None):
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
        self.morse_grad_list = []
        self.simple_net_grad_list = []
        # grad list stores grad of morse
        self.cur_state = np.zeros([2, 3])
        self.seq_len = 0

        # init tags
        self.sTag: float = 0.0
        # benign
        self.iTag: float = 0.0
        #
        self.cTag: float = 0.0

        if self.ppid == -1:
            # unknown parent
            self.sTag = morse.get_stag_dangerous()
        elif self.ppid == 0:
            # process generated by root
            self.sTag = morse.get_stag_benign()
        else:
            from globals import GlobalVariable as gv
            parent_id = gv.get_processNode_by_pid(self.ppid)
            parent_node = gv.get_processNode(parent_id)
            if not parent_node:
                # parent node not exist or has been released, then this node is not valid
                self.sTag = morse.get_stag_dangerous()
                self.iTa = morse.get_itag_dangerous()
                self.cTag = morse.get_ctag_dangerous()
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

    def generate_sequence_and_grad(self, batch_size=100, sequence_size=5):
        """
        :param batch_size: how many sequences in a batch
        :param sequence_size: how long a sequence is
        :return: a batch of sequences and their grads
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
        # print(np.shape(morse_grad_res))
        return [res, morse_grad_res, simple_net_grad_res]

    def generate_sequence(self, batch_size=100, sequence_size=5):
        """
        :param batch_size: how many sequences in a batch
        :param sequence_size: how long a sequence is
        :return: a batch of sequences
        """
        if self.seq_len < sequence_size:
            need_padding_len=sequence_size-self.seq_len
            padding_element=-np.ones(12)
            return [self.state_list+ [padding_element]*need_padding_len]
        res = []
        total_len = min(batch_size, self.seq_len - sequence_size + 1)
        for i in range(total_len):
            res.append(self.state_list[i:i + sequence_size])
        return res

    def generate_simple_net_grad_sequence(self, batch_size=100, sequence_size=5):
        """
        :param batch_size: how many sequences in a batch
        :param sequence_size: how long a sequence is
        :return: a batch of sequences
        """
        if self.seq_len < sequence_size:
            return [[], []]
        res = []
        total_len = min(batch_size, self.seq_len - sequence_size + 1)
        for i in range(total_len):
            res.append(self.grad_list[i:i + sequence_size])
        # if total_len < batch_size:
        #     res += [[]] * (batch_size - total_len)
        return res

    def generate_malicious_mark(self, batch_size=100, sequence_size=5):
        """
        :param batch_size: how many sequences in a batch
        :param sequence_size: how long a sequence is
        :return: malicious marker list, a boolean sequence list. This method is used in visualize testing, to locate where malicious points are
        """
        from prepare_gold_labels import malicious_id

        res = []
        total_len = min(batch_size, self.seq_len - sequence_size + 1)
        if total_len<=0:
            total_len=1

        malicious_marker_list = [False] * self.seq_len
        has_malicious=False

        for i in range(self.seq_len):
            if self.event_id_list[i] in malicious_id:
                malicious_marker_list[i]=True
                has_malicious=True

        if not has_malicious:
            return [[False]*sequence_size for _ in range(total_len)]
        print(malicious_marker_list)
        if self.seq_len < sequence_size:
            return [[malicious_marker_list+[False]*(sequence_size-self.seq_len)]]

        for i in range(total_len):
            res.append(malicious_marker_list[i:i + sequence_size])
        return res