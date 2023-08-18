from vayesta.core.util import einsum, dot
from vayesta.core.bath.bno import BNO_Threshold
import numpy as np




class Bosonic_Bath_Truncator:
    def __init__(self, target_m0, fragment):
        self.target_m0 = target_m0
        self.fragment = fragment

    @property
    def cluster_excitations(self):
        co = self.fragment.get_overlap('cluster[occ]|mo[occ]')
        cv = self.fragment.get_overlap('cluster[vir]|mo[vir]')
        ov_ss = einsum("Ii,Aa->IAia", co, cv).reshape(-1, co.shape[1] * cv.shape[1])
        return np.hstack((ov_ss, ov_ss))

    def gen_bosons(self, bno_threshold=None):
        # Generate full local fermionic excitation space.
        clus_ov = self.cluster_excitations
        # Remove any contributions within the fermionic excitation space of the fragment.
        m0_env = self.target_m0 - dot(dot(self.target_m0, clus_ov), clus_ov.T)

        # Now we can construct the bosonic bath by diagonlising these contributions.
        contribs = dot(m0_env, m0_env.T)
        occup, c = np.linalg.eigh(contribs)
        # select based on the eigenvalues.
        boson_number = bno_threshold.get_number(occup, electron_total=None)




