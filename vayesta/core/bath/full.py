import numpy as np
from vayesta.core.bath.bath import Bath


class Full_Bath_RHF(Bath):
    def __init__(self, fragment, dmet_bath, occtype, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
        if occtype not in ("occupied", "virtual"):
            raise ValueError("Invalid occtype: %s" % occtype)
        self.occtype = occtype

    @property
    def c_env(self):
        if self.occtype == "occupied":
            return self.dmet_bath.c_env_occ
        if self.occtype == "virtual":
            return self.dmet_bath.c_env_vir

    def get_bath(self, *args, **kwargs):
        nao = self.c_env.shape[0]
        return self.c_env, np.zeros((nao, 0))


class Full_Bath_UHF(Full_Bath_RHF):
    def get_bath(self, *args, **kwargs):
        nao = self.c_env[0].shape[0]
        return self.c_env, tuple(2 * [np.zeros((nao, 0))])
