import numpy as np
from logging import getLogger
from target import Target as tg
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax

logger=getLogger("EventProcessor")
length = 2 * 3


class EventProcessor:

    @staticmethod
    def read_process(vector: jnp.array):
        # 2x4 4x4 4x3 -> 2x3
        # tags structure
        # srcNode: sTag iTag cTag
        # desNode: sTag iTag cTag
        vector.astype(np.float)
        left_matrix = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = jnp.dot(jnp.dot(left_matrix, vector), right_matrix)
        final_tags = (jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags[:, 1:3], axis=0))).reshape(1,length)

        tags.reshape(1, length)
        res_tags = jnp.append(tags, final_tags)
        return res_tags


    @staticmethod
    def write_process(
            vector: jnp.array
    ):
        benign = vector[1][2]
        suspect_env = vector[1][3]
        a_b = vector[1][0]
        a_e = vector[1][1]


        vector.astype(np.float)
        left_matrix = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = jnp.dot(jnp.dot(left_matrix, vector), right_matrix)

        benign_thresh = tg.benign_thresh_model(tags[0][0])
        susp_thresh = tg.suspect_env_model(tags[0][0])

        benign_mul = benign_thresh+susp_thresh
        susp_mul = (1-benign_thresh) + susp_thresh
        dangerous_mul = (1-benign_thresh) + (1- susp_thresh)

        attenuation = jnp.array([[0, a_b, a_b], [0, 0, 0]])

        tag_benign = (jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags + attenuation, axis=0)[1:3])).reshape(1,length)
        tag_susp_env = (jax.ops.index_update(tags, jax.ops.index[0, 1:3], jnp.min(tags + attenuation, axis=0)[1:3])).reshape(1,length)
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

    @staticmethod
    def create_process(vector: jnp.array):
        vector.astype(np.float)
        left_matrix = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = jnp.dot(jnp.dot(left_matrix, vector), right_matrix)
        final_tags = tags
        final_tags[0][1:3] = tags[1][:, 1:3]


        res_tags = jnp.concatenate(tags.reshape(1, length), final_tags)

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

    @staticmethod
    def load_process(vector: jnp.array):
        # print("load event: ")
        # print("before processing")
        # print(vector)
        tags = vector[2:, 1:].astype(np.float)
        final_tags = tags + jnp.array([[jnp.minimum(tags[0, 0], tags[1, 0]) - tags[0, 0],
                                 jnp.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                                 jnp.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                                [0.0, 0.0, 0.0]])
        res_tags = jnp.concatenate(tags.reshape(1,length), final_tags.reshape(1, length))
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

    @staticmethod
    def exec_process(
            vector: jnp.array
    ):
        benign = vector[1][2]
        suspect_env = vector[1][3]
        a_b = vector[1][0]
        a_e = vector[1][1]

        tags = vector[2:, 1:].astype(np.float)

        benign_thresh = tg.benign_thresh_model(tags[0][0])
        susp_thresh = tg.suspect_env_model(tags[0][0])

        benign_mul = benign_thresh + susp_thresh
        susp_mul = (1 - benign_thresh) + susp_thresh
        dangerous_mul = (1 - benign_thresh) + (1 - susp_thresh)

        tag_benign = tags + jnp.array([[tags[1, 1] - tags[0, 0],
                                     1.0 - tags[0, 1],
                                     1.0 - tags[0, 2]],
                                    [0, 0, 0]])

        tag_susp_env = tags + jnp.array([[jnp.minimum(suspect_env, tags[1, 1]) - tags[0, 0],
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

        final_tags = jnp.dot(tags_probability, possible_tags)

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

    @staticmethod
    def inject_process(vector: jnp.array):
        pass


def get_read_grad(vector: jnp.array):
    return jacrev(EventProcessor.read_process)(vector)


def get_write_grad(vector: jnp.array
                   ):
    return jacrev(EventProcessor.write_process)(vector)
    # jacobian matrix
    # write_process -> 4x3 matrix [previous tags, updated tags]


def get_create_grad(vector: jnp.array):
    return jacrev(EventProcessor.create_process)(vector)


def get_load_grad(vector: jnp.array):
    return jacrev(EventProcessor.load_process)(vector)


def get_exec_grad(vector: jnp.array):
    return jacrev(EventProcessor.exec_process)(vector)

