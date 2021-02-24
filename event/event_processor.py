import numpy as np
import logging
from logging import getLogger
import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev
import jax

logger=getLogger("EventProcessor")



class EventProcessor:

    # benign = 0.5
    # suspect_env = 0.25
    # a_b = 0.1
    # a_e = 0.05


    @staticmethod
    def read_process(vector: jnp.array):
        # 2x4 4x4 4x3 -> 2x3
        # tags structure
        # srcNode: sTag iTag cTag
        # desNode: sTag iTag cTag
        left_matrix = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags=jnp.dot(jnp.dot(left_matrix,vector),right_matrix)
        jax.ops.index_update(tags, jax.ops.index[0,1:3], jnp.min(tags[:, 1:3], axis=0))
        return tags

    @staticmethod
    def write_process(
        vector: jnp.array, 
        benign: float=0.5, 
        suspect_env: float=0.25, 
        a_b: float=0.1,
        a_e: float=0.05
        ):

        left_matrix = jnp.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = jnp.dot(jnp.dot(left_matrix, vector), right_matrix)
        if tags[0][0]>= benign:
            attenuation=jnp.array([[0, a_b, a_b], [0, 0, 0]])
            jax.ops.index_update(tags, jax.ops.index[0,1:3], jnp.min(tags+attenuation, axis=0)[1:3])
            return tags
        elif tags[0][0]>=suspect_env:
            attenuation = jnp.array([[0, a_e, a_e], [0, 0, 0]])
            jax.ops.index_update(tags, jax.ops.index[0,1:3], jnp.min(tags + attenuation, axis=0)[1:3])
            return tags
        else:
            jax.ops.index_update(tags, jax.ops.index[0,1:3], jnp.min(tags[:, 1:3], axis=0))
            return tags

    @staticmethod
    def create_process(vector: np.array):
        vector.astype(np.float)
        left_matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        right_matrix = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tags = np.dot(np.dot(left_matrix, vector), right_matrix)
        tags[0][1:3] = tags[1][:, 1:3]

        pass

    @staticmethod
    def load_process(vector: jnp.array):
        # print("load event: ")
        # print("before processing")
        # print(vector)
        tags = vector[2:, 1:]
        res = tags + jnp.array([[jnp.minimum(tags[0, 0], tags[1, 0]) - tags[0, 0],
                               jnp.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                               jnp.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                              [0.0, 0.0, 0.0]])
        return res
        # print("after processing")
        # print(res)

    @staticmethod
    def exec_process(
        vector: jnp.array, 
        benign: float=0.5, 
        suspect_env: float=0.25
        ):
        tags = vector[2:, 1:]
        if tags[0, 0] >= benign:
            res = tags + jnp.array([[tags[1, 1] - tags[0, 0],
                                   1.0 - tags[0, 1],
                                   1.0 - tags[0, 2]],
                                  [0, 0, 0]])
        elif tags[0, 0] >= suspect_env:
            res = tags + jnp.array([[jnp.minimum(suspect_env, tags[1, 1]) - tags[0, 0],
                                   jnp.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                                   jnp.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                                  [0, 0, 0]])
        else:
            res = tags + jnp.array([[tags[1, 1] - tags[0, 0],
                                   jnp.minimum(tags[0, 1], tags[1, 1]) - tags[0, 1],
                                   jnp.minimum(tags[0, 2], tags[1, 2]) - tags[0, 2]],
                                  [0, 0, 0]])
        return res
        # print("after processing")
        # print(res)

    @staticmethod
    def inject_process(vector: np.array):
        pass
