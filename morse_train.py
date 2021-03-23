import target as tg
import numpy as np
import event.event_processor as ep


def back_propagate(event_type: int, vector: np.array, rnn_grad: np.array):
    if event_type == 1:
        pass
    elif event_type == 4:
        morse_grad: np.array
        morse_grad = ep.get_read_grad(vector)

    elif event_type == 5:
        pass
    elif event_type == 6:
        pass
    # write process
    elif event_type == 8:
        morse_grad: np.array
        morse_grad = ep.get_write_grad(vector)
        # 1*12 * 4*4
        final_grad = get_grad_from_morse_and_rnn(morse_grad, rnn_grad)
        return final_grad
    else:
        pass


def get_grad_from_morse_and_rnn(morse_grad: np.array(4, 3, 4, 4), rnn_grad: np.array(4, 3)):
    rnn_r_num, rnn_c_num = rnn_grad.shape
    res = np.zeros(4, 4)
    for m in range(rnn_r_num):
        for n in range(rnn_c_num):
            rnn_grad = rnn_grad[m][n]
            for i in range():
                for j in range():
                    # morse_grad[m][n]
                    res[i][j] += morse_grad[m][n][i][j] * rnn_grad[m][n]
    return res
