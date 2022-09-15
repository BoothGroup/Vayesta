import numpy as np
from vayesta.core.util import *
import dataclasses

def ClusterHamiltonian(fragment, mf, log=None, **kwargs):
    if np.ndim(mf.mo_coeff[0]) == 1:
        return RClusterHamiltonian(fragment, mf, log=None, **kwargs)
    return UClusterHamiltonian(fragment, mf, log=None, **kwargs)

class RClusterHamiltonian:
    @dataclasses.dataclass
    class Options(OptionsBase):
        pass

    def __init__(self, fragment, mf, log=None, **kwargs):

        self.mf = mf
        self.fragment = fragment
        self.log = (log or fragment.log)
        # --- Options:
        self.opts = self.Options()
        self.opts.update(**kwargs)
        self.log.info("Parameters of %s:" % self.__class__.__name__)
        self.log.info(break_into_lines(str(self.opts), newline='\n    '))
        self.v_ext = None

    @property
    def cluster(self):
        return self.fragment.cluster

    def get_clus_info(self):
        nelec = (self.cluster.nocc_active, self.cluster.nocc_active)
        nao = self.cluster.norb_active
        mo_coeff = self.cluster.c_active
        mo_occ = np.zeros((nao,))
        mo_occ[:nelec[0]] = 2.0
        return nelec, nao, mo_coeff, mo_occ

    def get_hamils(self, eris=None):
        if eris is None: eris = self.get_eris()
        return self.get_heff(eris), eris

    def get_heff(self, eris, fock=None, with_vext=True):
        if fock is None:
            fock = self.fragment.get_fock()
        mo_energy = np.diag(fock)

        occ = np.s_[:self.cluster.nocc_active]
        v_act = 2 * einsum('iipq->pq', eris[occ, occ]) - einsum('iqpi->pq', eris[occ, :, :, occ])
        h_eff = fock - v_act
        # This should be equivalent to:
        # core = np.s_[:self.nocc_frozen]
        # dm_core = 2*np.dot(self.mo_coeff[:,core], self.mo_coeff[:,core].T)
        # v_core = self.mf.get_veff(dm=dm_core)
        # h_eff = np.linalg.multi_dot((self.c_active.T, self.base.get_hcore()+v_core, self.c_active))
        if with_vext and self.v_ext is not None:
            h_eff += self.v_ext
        return h_eff, mo_energy

    def get_eris(self, *args, **kwargs):
        with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
            coeff = self.cluster.c_active
            eris = self.fragment.base.get_eris_array(coeff)
        return eris

    def to_pyscf_mf(self):
        pass

class UClusterHamiltonian(RClusterHamiltonian):
    pass