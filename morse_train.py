import morse as tg
import numpy as np
import event.event_processor as ep
from globals import GlobalVariable as gv


def back_propagate(batch, event_type_list, event_list, rnn_grad):
    morse_grad = []
    for sequence_idx in range(len(batch)):
        sequence = batch[sequence_idx]
        tmp_grad = []
        for index in range(len(sequence)):
            try:
                event_idx = sequence_idx + index
                event_type = event_type_list[event_idx]
                event = event_list[event_idx]
                tmp_grad.append(get_morse_grad(event_type, event))
            except:
                print(len(batch),len(sequence), len(event_type_list), len(event_list), sequence_idx, index, sequence_idx + index)
            # [12 * 4]
        morse_grad.append(tmp_grad[::])

    # morse_grad 100 * 5 * 12 * 4

    return get_grad_from_morse_and_rnn(morse_grad, rnn_grad)


def get_morse_grad(event_type, event, ep):
    morse_grad = np.ones([12, 4])
    if event_type == 1:
        pass
    elif event_type == 4 or event_type == 5 or event_type == 6 or event_type == 7:
        morse_grad = ep.get_read_grad(event)
    # write process
    elif event_type == 8 or event_type == 9 or event_type == 10 or event_type == 11:
        morse_grad = ep.get_write_grad(event)
    else:
        pass
    return morse_grad


def get_grad_from_morse_and_rnn(morse_grad: np.array, rnn_grad: np.array):
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
    # rnn_r_num, rnn_c_num = rnn_grad.shape
    # print(morse_grad, rnn_grad)
    res = np.tensordot(morse_grad, rnn_grad, ([0, 1, 2], [0, 1, 2]))
    # res: [1 * 4]
    # res = np.zeros(4, 4)
    # for m in range(rnn_r_num):
    #     for n in range(rnn_c_num):
    #         rnn_grad = rnn_grad[m][n]
    #         for i in range():
    #             for j in range():
    #                 # morse_grad[m][n]
    #                 res[i][j] += morse_grad[m][n][i][j] * rnn_grad[m][n]
    return res
