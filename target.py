

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

    # -------------- tag getters ------------------ #

    @classmethod
    def get_benign_thresh(cls) -> float:
        return cls.benign

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