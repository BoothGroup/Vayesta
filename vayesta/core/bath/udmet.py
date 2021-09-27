import numpy as np

from .dmet import DMET_Bath

class UDMET_Bath(DMET_Bath):

    def get_occupied_bath(self, *args, **kwargs):
        """Inherited bath classes can overwrite this to return additional occupied bath orbitals."""
        nao = self.mol.nao_nr()
        return np.zeros((2, nao, 0)), self.c_env_occ

    def get_virtual_bath(self, *args, **kwargs):
        """Inherited bath classes can overwrite this to return additional virtual bath orbitals."""
        nao = self.mol.nao_nr()
        return np.zeros((2, nao, 0)), self.c_env_vir

    def make_dmet_bath(self, c_env, dm1=None, **kwargs):
        if dm1 is None: dm1 = self.mf.make_rdm1()
        results = []
        for s, spin in enumerate(('alpha', 'beta')):
            self.log.info("Making %s-DMET bath", spin)
            # Use restricted DMET bath routine for each spin:
            results.append(super().make_dmet_bath(c_env[s], dm1=2*dm1[s], **kwargs))
        return tuple(zip(*results))
