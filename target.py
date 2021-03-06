import rnn_model.simple_network as simple_net
import numpy as np


class Target:
    # init value
    stag_benign = 0.5
    stag_suspect_env = 0.25
    stag_dangerous = 0.2
    itag_benign = 0.5
    itag_suspect_env = 0.25
    itag_dangerous = 0.2
    ctag_benign = 0.5
    ctag_suspect_env = 0.25
    ctag_dangerous = 0.2

    # threshold
    benign = 0.5
    suspect_env = 0.25

    # decay and attenuation
    a_b = 0.1
    a_e = 0.05

    # scaler w and b 
    benign_thresh_model = simple_net.SimpleNet()
    # scaler w and b
    suspect_env_model = simple_net.SimpleNet()

    # -------------- tag getters ------------------ #

    @classmethod
    def get_benign_thresh(cls) -> float:
        return cls.benign

    @classmethod
    def get_susp_thresh(cls) -> float:
        return cls.suspect_env

    @classmethod
    def get_stag_benign(cls) -> float:
        return cls.stag_benign

    @classmethod
    def get_itag_benign(cls) -> float:
        return cls.itag_benign

    @classmethod
    def get_ctag_benign(cls) -> float:
        return cls.ctag_benign

    @classmethod
    def get_stag_susp_env(cls) -> float:
        return cls.stag_suspect_env

    @classmethod
    def get_itag_susp_env(cls) -> float:
        return cls.itag_suspect_env

    @classmethod
    def get_ctag_susp_env(cls) -> float:
        return cls.ctag_suspect_env

    @classmethod
    def get_stag_dangerous(cls) -> float:
        return cls.stag_dangerous

    @classmethod
    def get_itag_dangerous(cls) -> float:
        return cls.itag_dangerous

    @classmethod
    def get_ctag_dangerous(cls) -> float:
        return cls.ctag_dangerous

    @classmethod
    def get_attenuate_susp_env(cls) -> float:
        return cls.a_e

    @classmethod
    def get_attenuate_benign(cls) -> float:
        return cls.a_b

    # ------------------ tag setters -------------- #

    @classmethod
    def set_stag_benign(cls, val):
        cls.stag_benign = val

    @classmethod
    def set_itag_benign(cls, val):
        cls.itag_benign = val

    @classmethod
    def set_ctag_benign(cls, val):
        cls.ctag_benign = val

    @classmethod
    def set_stag_susp_env(cls, val):
        cls.stag_suspect_env = val

    @classmethod
    def set_itag_susp_env(cls, val):
        cls.itag_suspect_env = val

    @classmethod
    def set_ctag_susp_env(cls, val):
        cls.ctag_suspect_env = val

    @classmethod
    def set_stag_dangerous(cls, val):
        cls.stag_dangerous = val

    @classmethod
    def set_itag_dangerous(cls, val):
        cls.itag_dangerous = val

    @classmethod
    def set_itag_dangerous(cls, val):
        cls.itag_dangerous = val

    # ------------------ model getters-------------- #
    @classmethod
    def get_benign_possibility(cls, stag: float):
        return cls.benign_thresh_model(stag)

    @classmethod
    def get_susp_possibility(cls, stag: float):
        return cls.suspect_env_model(stag)

    @classmethod
    def get_benign_thresh_grad(cls)-> np.ndarray((1,2)):
        return cls.benign_thresh_model.backward()

    @classmethod
    def get_susp_thresh_grad(cls) -> np.ndarray((1,2)):
        return cls.suspect_env_model.backward()

    @classmethod
    def benign_thresh_backward(cls, grad: float):
        cls.benign_thresh_model.backward(grad)

    @classmethod
    def susp_thresh_backward(cls, grad: float):
        cls.suspect_env_model.backward(grad)

    # ------------------ weights setters ----------- # 

    @classmethod
    def a_b_setter(cls, final_a_b_grad):
        cls.a_b = cls.a_b + final_a_b_grad
    
    @classmethod
    def a_e_setter(cls, final_a_e_grad):
        cls.a_e = cls.a_e + final_a_e_grad

    @classmethod
    def benign_thresh_model_setter(cls, w_grad, b_grad):
        cls.benign_thresh_model.update_weight(w_grad, b_grad)

    @classmethod
    def suspect_env_model_setter(cls, w_grad, b_grad):
        cls.suspect_env_model.update_weight(w_grad, b_grad)