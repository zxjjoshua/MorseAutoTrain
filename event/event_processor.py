import numpy as np
import logging
from logging import getLogger

logger=getLogger("EventProcessor")
benign = 0.5
suspect = 0.5
a_b=0.1
a_e=0.05


class EventProcessor:

    @staticmethod
    def read_process(vector: np.array):
        # 2x4 4x4 4x3 -> 2x3
        # tags structure
        # srcNode: sTag iTag cTag
        # desNode: sTag iTag cTag
        left_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags=np.dot(np.dot(left_matrix,vector),right_matrix)
        tags[0][1:3]=np.min(tags[:, 1:3], axis=0)
        return tags
        pass

    @staticmethod
    def write_process(vector: np.array):
        left_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = np.dot(np.dot(left_matrix, vector), right_matrix)
        if tags[0][0]>= benign:
            attenuation=np.array([[0, a_b, a_b], [0, 0, 0]])
            tags[0][1:3]=np.min(tags+attenuation, axis=0)[1:3]
            pass
        elif tags[0][0]>=suspect:
            attenuation = np.array([[0, a_e, a_e], [0, 0, 0]])
            tags[0][1:3] = np.min(tags + attenuation, axis=0)[1:3]
            pass
        else:
            tags[0][1:3] = np.min(tags[:, 1:3], axis=0)
            pass
        pass

    @staticmethod
    def create_process(vector: np.array):
        left_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = np.dot(np.dot(left_matrix, vector), right_matrix)
        tags[0][1:3] = tags[1][:, 1:3]

        pass



