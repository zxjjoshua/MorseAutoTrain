import target as tg
import numpy as np
import event.event_processor as ep


def back_propagate(event_type: int, vector: np.array, loss: np.array, rnn_grad: np.array):
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
        final_grad = get_grad_from_morse_and_rnn(morse_grad, rnn_grad)
        return final_grad
    else:
        pass


def get_grad_from_morse_and_rnn(morse_grad: np.array(int, int, int, int), rnn_grad: np.array(int, int, int)):
    '''
    morse: w1 * x = y
    rnn: w2 * y = z
    ex: y.shape: [100, 5, 12]
    w1.shape: [4]
    rnn_grad.shape: [100, 5, 12]
    morse_grad.shape: [100, 5, 12, 4]
    final_grad = w1.grad * y.grad
    so for each dimension of the 4, rnn_grad's [100, 5, 12] element-wise multiplies morse_grad's [100, 5, 12] then sum
    '''
    rnn_r_num, rnn_c_num = rnn_grad.shape
    res = np.tensordot(morse_grad, rnn_grad, ([0,1,2], [0,1,2]))
    # res = np.zeros(4, 4)
    # for m in range(rnn_r_num):
    #     for n in range(rnn_c_num):
    #         rnn_grad = rnn_grad[m][n]
    #         for i in range():
    #             for j in range():
    #                 # morse_grad[m][n]
    #                 res[i][j] += morse_grad[m][n][i][j] * rnn_grad[m][n]
    return res
