import numpy as np
from logging import getLogger
from morse import Morse
import jax.numpy as jnp
from jax import jit
from jax import jacfwd, jacrev
import jax
from torch import Tensor

logger=getLogger("EventProcessor")
length = 2 * 3


class EventProcessor:

    def __init__(self, morse: Morse):
        self.morse = morse
        self.jrev = jit(jacrev(self.write_process))

    def read_process(self, vector: jnp.array):
        # 2x4 4x4 4x3 -> 2x3
        # tags structure
        # srcNode: sTag iTag cTag
        # desNode: sTag iTag cTag
        # print(vector)
        for i, l in enumerate(vector):
            for j, t in enumerate(l): 
                if isinstance(t, Tensor):
                    # print(type(vector))
                    # print(type(jax.ops.index[i, j]))
                    vector[i][j] = float(t.cpu().detach().numpy())
                    # jax.ops.index_update(vector, jax.ops.index[i,j], float(t.cpu().detach().numpy()))
                elif isinstance(t, np.ndarray):
                    vector[i][j] = float(t)
                    # jax.ops.index_update(vector, jax.ops.index[i,j], t.astype(float))
        vector = vector.astype('float64')
        # print("vector", vector.dtype)
        left_matrix = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        tags = jnp.dot(jnp.dot(left_matrix, vector), right_matrix)
        final_tags = (jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags[:, 1:3], axis=0))).reshape(1,length)

        tags.reshape(1, length)
        res_tags = jnp.append(tags, final_tags)
        return res_tags


    def write_process(
            self,
            vector: jnp.array
    ):
        benign_thresh = vector[1][2]
        susp_thresh = vector[1][3]
        a_b = vector[1][0]
        a_e = vector[1][1]

        # print(vector)
        vector.astype(float)
        left_matrix = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # print("left_matrix: ", left_matrix.shape)
        # print("vector: ", vector.shape)
        # print("right_matrix: ", right_matrix.shape)
        # print(vector)
        # some values in the vector are torch tensor but not a number, so they are replaced by their data, losing the autograd trace
        
        # print("left_matrix", left_matrix)
        # print("vector", vector)
        # print("vector", vector)
        # vector = jnp.array(vector)
        
        for i, l in enumerate(vector):
            for j, t in enumerate(l): 
                if isinstance(t, Tensor):
                    # print(type(vector))
                    # print(type(jax.ops.index[i, j]))
                    vector[i][j] = float(t.cpu().detach().numpy())
                    # jax.ops.index_update(vector, jax.ops.index[i,j], float(t.cpu().detach().numpy()))
                elif isinstance(t, np.ndarray):
                    vector[i][j] = float(t)
                    # jax.ops.index_update(vector, jax.ops.index[i,j], t.astype(float))

        # print("left_matrix", left_matrix)
        # print("vector", vector)
        # print(left_matrix.dtype)
        # print(vector.dtype)
        # for l in vector:
        #     for t in l:
        #         print(t.dtype)
        # print(type(vector))
        vector = vector.astype('float64')
        tmp = jnp.dot(left_matrix, vector)
        tags = jnp.dot(tmp, right_matrix)

        benign_mul = benign_thresh+susp_thresh
        susp_mul = (1-benign_thresh) + susp_thresh
        dangerous_mul = (1-benign_thresh) + (1- susp_thresh)

        # print("a_b", type(a_b))
        # print("a_e", type(a_e))
        if isinstance(a_b, Tensor):
            a_b = float(a_b.cpu().detach().numpy())
        if isinstance(a_e, Tensor):
            a_e = float(a_e.cpu().detach().numpy())
        attenuation_b = jnp.array([[0, a_b, a_b], [0, 0, 0]])
        attenuation_e = jnp.array([[0, a_e, a_e], [0, 0, 0]])

        tag_benign = (jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags + attenuation_b, axis=0)[1:3])).reshape(1,length)
        tag_susp_env = (jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags + attenuation_e, axis=0)[1:3])).reshape(1,length)
        tag_dangerous = (jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags[:, 1:3], axis=0))).reshape(1,length)

        possible_tags = jnp.concatenate([tag_benign, tag_susp_env, tag_dangerous])
        tags_probability = jax.nn.softmax(jnp.array([benign_mul, susp_mul, dangerous_mul]))

        final_tags = jnp.dot(tags_probability, possible_tags)

        # if tags[0][0] >= benign:
        #     attenuation = jnp.array([[0, a_b, a_b], [0, 0, 0]])
        #     jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags + attenuation, axis=0)[1:3])
        # elif tags[0][0] >= suspect_env:
        #     attenuation = jnp.array([[0, a_e, a_e], [0, 0, 0]])
        #     jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags + attenuation, axis=0)[1:3])
        # else:
        #     jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags[:, 1:3], axis=0))
        res_tags = jnp.append(tags.reshape(1,length), final_tags)
        return res_tags

    # @staticmethod
    # def create_process(vector: np.array):
    #     vector.astype(np.float)
    #     left_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    #     right_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    #     tags = np.dot(np.dot(left_matrix, vector), right_matrix)
    #     tags[0][1:3] = tags[1][1:3]
    #
    #     pass


    def create_process(self, vector: jnp.array):
        vector.astype(np.float)
        left_matrix = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = jnp.dot(jnp.dot(left_matrix, vector), right_matrix)
        final_tags = tags
        final_tags[0][1:3] = tags[1][:, 1:3]

        res_tags = jnp.append(tags.reshape(1, length), final_tags)

        return res_tags

    # @staticmethod
    # def load_process(vector: np.array):
    #     # print("load event: ")
    #     # print("before processing")
    #     # print(vector)
    #     tags = vector[2:, 1:].astype(np.float)
    #     res = tags + np.array([[np.minimum(tags[0, 0], tags[1, 0]) - tags[0, 0],
    #                            np.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
    #                            np.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
    #                           [0.0, 0.0, 0.0]])
    #     # print("after processing")
    #     # print(res)


    def load_process(self, vector: jnp.array):
        # print("load event: ")
        # print("before processing")
        # print(vector)
        tags = vector[2:, 1:].astype(np.float)
        final_tags = tags + jnp.array([[jnp.minimum(tags[0, 0], tags[1, 0]) - tags[0, 0],
                                 jnp.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                                 jnp.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                                [0.0, 0.0, 0.0]])
        res_tags = jnp.append(tags.reshape(1,length), final_tags.reshape(1, length))
        return res_tags

    # @staticmethod
    # def exec_process(vector: np.array):
    #     # print("execve event: ")
    #     # print("before processing")
    #     # print(vector)
    #
    #     stag_benign_thresh = tg.get_stag_benign()
    #     stag_susp_env_thresh = tg.get_stag_susp_env()
    #
    #     tags = vector[2:, 1:].astype(np.float)
    #     if tags[0, 0] >= stag_benign_thresh:
    #         res = tags + np.array([[tags[1, 1] - tags[0, 0],
    #                                1.0 - tags[0, 1],
    #                                1.0 - tags[0, 2]],
    #                               [0, 0, 0]])
    #     elif tags[0, 0] >= stag_susp_env_thresh:
    #         res = tags + np.array([[np.minimum(EventProcessor.suspect_env, tags[1, 1]) - tags[0, 0],
    #                                np.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
    #                                np.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
    #                               [0, 0, 0]])
    #     else:
    #         res = tags + np.array([[tags[1, 1] - tags[0, 0],
    #                                np.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
    #                                np.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
    #                               [0, 0, 0]])
    #     # print("after processing")
    #     # print(res)


    def exec_process(
            self,
            vector: jnp.array
    ):
        benign_thresh = vector[1][2]
        susp_thresh = vector[1][3]
        a_b = vector[1][0]
        a_e = vector[1][1]

        # print(vector.shape)
        tags = vector[2:, 1:].astype(np.float)

        benign_mul = benign_thresh + susp_thresh
        susp_mul = (1 - benign_thresh) + susp_thresh
        dangerous_mul = (1 - benign_thresh) + (1 - susp_thresh)

        tag_benign = tags + jnp.array([[tags[1, 1] - tags[0, 0],
                                     1.0 - tags[0, 1],
                                     1.0 - tags[0, 2]],
                                    [0, 0, 0]])

        tag_susp_env = tags + jnp.array([[jnp.minimum(self.morse.get_susp_thresh(), tags[1, 1]) - tags[0, 0],
                                     jnp.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                                     jnp.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                                    [0, 0, 0]])

        tag_dangerous = tags + jnp.array([[tags[1, 1] - tags[0, 0],
                                     jnp.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                                     jnp.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                                    [0, 0, 0]])
        tag_benign.reshape(1,length)
        tag_susp_env.reshape(1, length)
        tag_dangerous.reshape(1,length)

        possible_tags = jnp.stack([tag_benign, tag_susp_env, tag_dangerous])
        tags_probability = jax.nn.softmax(jnp.array([benign_mul, susp_mul, dangerous_mul]))
        
        # print(tags_probability.shape)
        # print(possible_tags.shape)
        final_tags = jnp.tensordot(tags_probability, possible_tags, axes=([0], [0]))
        # final_tags = jnp.dot(tags_probability, possible_tags)

        # if tags[0, 0] >= benign:
        #     res = tags + jnp.array([[tags[1, 1] - tags[0, 0],
        #                              1.0 - tags[0, 1],
        #                              1.0 - tags[0, 2]],
        #                             [0, 0, 0]])
        # elif tags[0, 0] >= suspect_env:
        #     res = tags + jnp.array([[jnp.minimum(suspect_env, tags[1, 1]) - tags[0, 0],
        #                              jnp.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
        #                              jnp.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
        #                             [0, 0, 0]])
        # else:
        #     res = tags + jnp.array([[tags[1, 1] - tags[0, 0],
        #                              jnp.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
        #                              jnp.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
        #                             [0, 0, 0]])

        res_tags = jnp.append(tags.reshape(1, length), final_tags)
        return res_tags

    def inject_process(self, vector: jnp.array):
        pass

    def get_read_grad(self, vector: jnp.array):
        vector = vector.astype('float64')
        grad = self.jrev(vector)
        # grad = jit(jacrev(EventProcessor.write_process)(vector))
        # [12 * 4 * 4]
        return grad[:, 1, :]
        # [12 * 4]

    def get_write_grad(self, vector: jnp.array
                       ):
        vector = vector.astype('float64')
        grad = self.jrev(vector)
        # [12 * 4 * 4]
        return grad[:,1,:]
        # [12 * 4]

    def get_create_grad(self, vector: jnp.array):
        vector = vector.astype('float64')
        grad = self.jrev(vector)
        # [12 * 4 * 4]
        return grad[:, 1, :]
        # [12 * 4]

    def get_load_grad(self, vector: jnp.array):
        vector = vector.astype('float64')
        grad = self.jrev(vector)
        # [12 * 4 * 4]
        return grad[:, 1, :]
        # [12 * 4]

    def get_exec_grad(self, vector: jnp.array):
        vector = vector.astype('float64')
        grad = self.jrev(vector)
        # [12 * 4 * 4]
        return grad[:, 1, :]
        # [12 * 4]

