import numpy as np

class EventProcessor:

    benign = 0.5
    suspect_env = 0.25

    @staticmethod
    def read_process(vector: np.array):
        pass

    @staticmethod
    def load_process(vector: np.array):
        print("load event: ")
        print("before processing")
        print(vector)
        tags = vector[2:, 1:].astype(np.float)
        res = tags + np.array([[np.minimum(tags[0, 0], tags[1, 0]) - tags[0, 0],
                               np.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                               np.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                              [0.0, 0.0, 0.0]])
        print("after processing")
        print(res)

    @staticmethod
    def exec_process(vector: np.array):
        print("execve event: ")
        print("before processing")
        print(vector)
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
        print("after processing")
        print(res)

    @staticmethod
    def inject_process(vector: np.array):
        pass
