import numpy as np
import scipy.linalg


import pyscf.scf

from vayesta.core.scmf.scmf import SCMF
from vayesta.core.foldscf import FoldedSCF
from vayesta.lattmod import LatticeRHF

from dyson import Lehmann, AufbauPrinciple

class QSEGF_RHF(SCMF):
    """Quasiparticle self-consistent embedded Green's function"""
    name = "QSEGF_RHF"

    def __init__(self, emb, static_potential_conv_tol=1e-5, static_potential_init=None, global_static_potential=True, eta=1e-2, damping=0, sc=False, store_hist=True, store_scfs=False, *args, **kwargs):
        """ 
        Initialize QPEWDMET 
        
        Parameters
        ----------
        emb : REGF object 
            Green's function Embedding object on which self consitency is based
        static_potential_conv_tol : float
            Convergence threshold for static potential
        eta : float
            Broadening factor for static potential
        damping : float
            Damping factor for Fock matrix update
        sc : bool
            Use self-consistent determination of MOs for new Fock matrix
        store_hist : bool
            Store history throughout SCMF calculation (for debugging purposes)
        static_potential_init : ndarray
            Inital static potential
        """
        
        self.emb = emb
        self.sc_fock = emb.get_fock()
        self.static_self_energy = np.zeros_like(self.sc_fock)
        self.sc = sc
        self.eta = eta # Broadening factor
        self.static_potential = None
        self.static_potential_last = None
        self.static_potential_conv_tol = static_potential_conv_tol
        self.store_hist = store_hist 
        self.static_potential_init = static_potential_init
        self.store_scfs = store_scfs
        self.global_static_potential = global_static_potential
        
        super().__init__(emb, *args, **kwargs)

        if self.store_hist:
            self.static_potential_hist = []
            self.static_potential_frag_hist = []
            self.fock_hist = []
            self.static_gap_hist = []
            self.dynamic_gap_hist = []  
            self.mo_coeff_hist = [self.emb.mo_coeff.copy()]
            self.gf_hist = []
            self.se_hist = []

            self.mom_hist = []

        if self.store_scfs and self.sc:
            self.scfs = []

        self.damping = damping

        if self.static_potential_init is not None:
            
            e, mo_coeff = self.fock_scf(self.static_potential_init)
            self.emb.update_mf(mo_coeff)

            dm1 = self.mf.make_rdm1()
            # Check symmetry - needs fixing
            try:
                self.emb.check_fragment_symmetry(dm1)
            except SymmetryError:
                self.log.error("Symmetry check failed in %s", self.name)
                self.converged = False

    def update_mo_coeff(self, mo_coeff, mo_occ, diis=None):
            
        """
        Get new MO coefficients for a SCMF iteration.

        TODO: Should GF and SE be stored in AO basis since the MO basis may change between iterations?

        Parameters
        ----------
        mf : PySCF compatible SCF object
            Mean-field object.
        diis : pyscf.lib.diis.DIIS object, optional
            DIIS object.

        Returns
        -------
        mo_coeff : ndarray
            New MO coefficients.
        """

        if self.global_static_potential:
            self.static_potential = self.emb.mo_coeff @ self.emb.self_energy.as_static_potential(self.emb.mf.mo_energy, eta=self.eta)  @ self.emb.mo_coeff.T
        else:
            raise NotImplementedError()
        
        v_old = self.static_potential.copy()
        sc = self.emb.mf.get_ovlp() @ self.emb.mo_coeff
        if self.global_static_potential:
            self.static_potential = self.emb.mo_coeff @ self.emb.self_energy.as_static_potential(self.emb.mf.mo_energy, eta=self.eta)  @ self.emb.mo_coeff.T
        self.static_potential = self.emb.mf.get_ovlp() @ self.static_potential @ self.emb.mf.get_ovlp()
        if diis is not None:
            self.static_potential = diis.update(self.static_potential)
        
        new_fock = self.emb.get_fock() + self.static_potential
        self.sc_fock = self.damping * self.emb.get_fock() + (1-self.damping) * new_fock
        #self.sc_fock = self.sc_fock + (1-self.damping) * self.static_potential
        self.gf = self.emb.gf

        if self.sc:
            #e, mo_coeff = self.fock_scf(self.static_potential)
            raise NotImplementedError()
        else:
            e, mo_coeff = scipy.linalg.eigh(self.sc_fock, self.emb.get_ovlp())

        nelec = self.emb.mf.mol.nelectron
        chempot_qp = (e[nelec//2-1] + e[nelec//2] ) / 2
        self.gf_qp = Lehmann(e, np.eye(len(e)), chempot=chempot_qp)
        nelec_gf_qp = 2*np.trace(self.gf_qp.occupied().moment(0))
        self.emb.log.info("QP GF nelec = %f"%nelec_gf_qp)
        #assert np.allclose(np.trace(self.gf_qp.occupied().moment(0)), nelec)
        

        gap = lambda gf: gf.physical().virtual().energies[0] - gf.physical().occupied().energies[-1]
        dynamic_gap = gap(self.gf)
        static_gap = gap(self.gf_qp)
        self.log.info("Dynamic Gap = %f"%dynamic_gap)
        self.log.info("Static Gap = %f"%static_gap)
            
        if self.store_hist:        
            #self.static_potential_frag_hist.append(v_frag.copy())
            self.static_potential_hist.append(self.static_potential.copy())
            self.fock_hist.append(self.sc_fock.copy())
            self.static_gap_hist.append(static_gap)
            self.dynamic_gap_hist.append(dynamic_gap)
            self.mo_coeff_hist.append(mo_coeff.copy())
            self.gf_hist.append(self.gf.copy())
            self.se_hist.append(self.emb.self_energy.copy())

        return mo_coeff


class QSEGF_MOD(object):

    def __init__(self, mf, maxiter=50, eta=1e-2, **opts):

        self.mol = mf.mol
        self.hcore = mf.get_hcore().copy()
        self.mf = mf
        self.maxiter = maxiter
        self.eta = eta
        self.opts = opts
        self.mfs = []
        self.embs = []
        self.stat_pots = []
    
    def kernel(self):


        self.static_potential_ao = np.zeros_like(self.hcore)

        for it in range(self.maxiter):

            mf_cls = type(self.mf)
            mf_it = mf_cls(self.mf.mol)
            mf_it.mol.h1e += self.static_potential_ao
            mf_it.kernel()
            assert mf_it.converged
            emb = vayesta.egf.EGF(mf_it, **self.opts)
            nimages = [nsite//nfrag, 1, 1]
            emb.symmetry.set_translations(nimages)
            with emb.site_fragmentation() as f:
                f.add_atomic_fragment(range(nfrag))
            emb.qsEGF(maxiter=1, eta=self.eta)
            emb.kernel()

            static_potential = emb.self_energy.as_static_potential(mf_it.mo_energy, eta=self.eta)
            self.static_potential_ao = mf_it.mo_coeff @ static_potential @ mf_it.mo_coeff.T

            self.mfs.append(mf_it)
            self.embs.append(emb)
            self.stat_pots.append(self.static_potential_ao)