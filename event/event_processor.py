import numpy as np
import logging
from logging import getLogger

logger=getLogger("EventProcessor")



class EventProcessor:

    benign = 0.5
    suspect_env = 0.25
    a_b = 0.1
    a_e = 0.05

    @staticmethod
    def read_process(vector: np.array):
        # 2x4 4x4 4x3 -> 2x3
        # tags structure
        # srcNode: sTag iTag cTag
        # desNode: sTag iTag cTag
        vector.astype(np.float)
        left_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags=np.dot(np.dot(left_matrix,vector),right_matrix)
        tags[0][1:3]=np.min(tags[:, 1:3], axis=0)
        return tags
        pass

    @staticmethod
    def write_process(vector: np.array):
        vector.astype(np.float)
        left_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = np.dot(np.dot(left_matrix, vector), right_matrix)
        if tags[0][0]>= EventProcessor.benign:
            attenuation=np.array([[0, EventProcessor.a_b, EventProcessor.a_b], [0, 0, 0]])
            tags[0][1:3]=np.min(tags+attenuation, axis=0)[1:3]
            pass
        elif tags[0][0]>=EventProcessor.suspect_env:
            attenuation = np.array([[0, EventProcessor.a_e, EventProcessor.a_e], [0, 0, 0]])
            tags[0][1:3] = np.min(tags + attenuation, axis=0)[1:3]
            pass
        else:
            tags[0][1:3] = np.min(tags[:, 1:3], axis=0)
            pass
        pass

    @staticmethod
    def create_process(vector: np.array):
        vector.astype(np.float)
        left_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = np.dot(np.dot(left_matrix, vector), right_matrix)
        tags[0][1:3] = tags[1][:, 1:3]

        pass

    @staticmethod
    def load_process(vector: np.array):
        # print("load event: ")
        # print("before processing")
        # print(vector)
        tags = vector[2:, 1:].astype(np.float)
        res = tags + np.array([[np.minimum(tags[0, 0], tags[1, 0]) - tags[0, 0],
                               np.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                               np.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                              [0.0, 0.0, 0.0]])
        # print("after processing")
        # print(res)

    @staticmethod
    def exec_process(vector: np.array):
        # print("execve event: ")
        # print("before processing")
        # print(vector)
        tags = vector[2:, 1:].astype(np.float)
        if tags[0, 0] >= EventProcessor.benign:
            res = tags + np.array([[tags[1, 1] - tags[0, 0],
                                   1.0 - tags[0, 1],
                                   1.0 - tags[0, 2]],
                                  [0, 0, 0]])
        elif tags[0, 0] >= EventProcessor.suspect_env:
            res = tags + np.array([[np.minimum(EventProcessor.suspect_env, tags[1, 1]) - tags[0, 0],
                                   np.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                                   np.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                                  [0, 0, 0]])
        else:
            res = tags + np.array([[tags[1, 1] - tags[0, 0],
                                   np.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                                   np.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                                  [0, 0, 0]])
        # print("after processing")
        # print(res)

    @staticmethod
    def inject_process(vector: np.array):
        pass
