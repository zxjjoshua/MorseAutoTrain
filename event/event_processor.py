import numpy as np
from logging import getLogger
from target import Target as tg

logger=getLogger("EventProcessor")


class EventProcessor:
    # init value
    # stag_benign = 0.5
    # stag_suspect_env = 0.25
    # stag_dangerous = 0.2
    # itag_benign = 0.5
    # itag_suspect_env = 0.25
    # itag_dangerous = 0.2
    # ctag_benign = 0.5
    # ctag_suspect_env = 0.25
    # ctag_dangerous = 0.2

    # threshold
    # benign = 0.5
    # suspect_env = 0.25

    # decay and attenuation
    # a_b = 0.1
    # a_e = 0.05



    # benign :0.75-0.5
    # suspicious 0.5-0.25
    # dangerous 0.25-0


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

        a_e, a_b=tg.get_attenuate_susp_env(), tg.get_attenuate_benign()
        stag_benign_thresh=tg.get_stag_benign()
        stag_susp_env_thresh = tg.get_stag_susp_env()

        if tags[0][0]>= stag_benign_thresh:
            attenuation=np.array([[0, a_b, a_b], [0, 0, 0]])
            tags[0][1:3]=np.min(tags+attenuation, axis=0)[1:3]
            pass
        elif tags[0][0]>=stag_susp_env_thresh:
            attenuation = np.array([[0, a_e, a_e], [0, 0, 0]])
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

        stag_benign_thresh = tg.get_stag_benign()
        stag_susp_env_thresh = tg.get_stag_susp_env()

        tags = vector[2:, 1:].astype(np.float)
        if tags[0, 0] >= stag_benign_thresh:
            res = tags + np.array([[tags[1, 1] - tags[0, 0],
                                   1.0 - tags[0, 1],
                                   1.0 - tags[0, 2]],
                                  [0, 0, 0]])
        elif tags[0, 0] >= stag_susp_env_thresh:
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
