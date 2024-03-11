import numpy as np
import scipy.linalg

import pyscf.scf

import vayesta
from vayesta.core.scmf.scmf import SCMF
from vayesta.core.foldscf import FoldedSCF
from vayesta.lattmod import LatticeRHF
from vayesta.core.qemb.self_energy import make_self_energy_1proj, make_self_energy_2proj
from dyson import Lehmann, AuxiliaryShift

class QPEWDMET_RHF(SCMF):
    """ Quasi-particle self-consistent energy weighted density matrix embedding """
    name = "QP-EWDMET"

    def __init__(self, emb, proj=2, static_potential_conv_tol=1e-5, eta=1e-2, damping=0, sc=True, aux_shift=False, aux_shift_frag=False, store_hist=True, store_scfs=False, use_sym=False, static_potential_init=None, se_degen_tol=1e-6, se_eval_tol=1e-6, drop_non_causal=False, *args, **kwargs):
        """ 
        Initialize QPEWDMET 
        
        Parameters
        ----------
        emb : QEmbedding o
            Embedding object on which self consitency is based
        proj : int 
            Number of fragment projectors applied to cluster self-energy
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
        use_sym : bool
            Use fragment symmetry
        static_potential_init : ndarray
            Inital static potential
        """
        
        self.sc_fock = emb.get_fock()
        self.static_self_energy = np.zeros_like(self.sc_fock)
        self.sc = sc
        self.eta = eta # Broadening factor
        self.self_energy = None
        self.static_potential = None
        self.static_potential_last = None
        self.static_potential_conv_tol = static_potential_conv_tol
        self.proj = proj
        self.store_hist = store_hist 
        self.use_sym = use_sym
        self.static_potential_init = static_potential_init
        self.store_scfs = store_scfs
        self.se_degen_tol = se_degen_tol
        self.se_eval_tol = se_eval_tol
        self.drop_non_causal = drop_non_causal
        self.aux_shift = aux_shift
        self.aux_shift_frag = aux_shift_frag
        
        super().__init__(emb, *args, **kwargs)

        if self.store_hist:
            self.static_potential_hist = []
            self.static_potential_frag_hist = []
            self.fock_hist = []
            self.static_gap_hist = []
            self.dynamic_gap_hist = []  
            self.mo_coeff_hist = []

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

        if self.static_potential is not None:
            self.static_potential_last = self.static_potential.copy()

        self.fock = self.emb.get_fock()
        couplings = []
        energies = []
        self.static_potential = np.zeros_like(self.fock)
        self.static_self_energy = np.zeros_like(self.fock)

        if self.proj == 1:
            self.self_energy, self.static_self_energy, self.static_potential = make_self_energy_1proj(self.emb, use_sym=self.use_sym, eta=self.eta,aux_shift_frag=self.aux_shift_frag, se_degen_tol=self.se_degen_tol, se_eval_tol=self.se_eval_tol)
        elif self.proj == 2:
            self.self_energy, self.static_self_energy, self.static_potential = make_self_energy_2proj(self.emb, use_sym=self.use_sym, eta=self.eta)
        else:
            return NotImpementedError()
        phys = self.emb.mo_coeff.T @ self.fock @ self.emb.mo_coeff + self.static_self_energy
        gf = Lehmann(*self.self_energy.diagonalise_matrix_with_projection(phys), chempot=self.self_energy.chempot)
        dm = gf.occupied().moment(0) * 2.0
        nelec_gf = np.trace(dm)
        self.emb.log.info('Number of electrons in GF: %f'%nelec_gf)
        if self.aux_shift:
            aux = AuxiliaryShift(self.fock+self.static_self_energy, self.self_energy, self.emb.mf.mol.nelectron, occupancy=2, log=self.log)
            aux.kernel()
            self.self_energy = aux.get_self_energy()
            gf = aux.get_greens_function()
            dm = gf.occupied().moment(0) * 2.0
            nelec_gf = np.trace(dm)
            self.emb.log.info('Number of electrons in (shifted) GF: %f'%nelec_gf)
        gap = lambda gf: gf.physical().virtual().energies[0] - gf.physical().occupied().energies[-1]
        dynamic_gap = gap(gf)

        v_old = self.static_potential.copy()
        if diis is not None:
            self.static_potential = diis.update(self.static_potential)

        new_fock = self.fock + self.static_potential
        self.sc_fock = self.damping * self.fock + (1-self.damping) * new_fock
        #self.sc_fock = self.sc_fock + (1-self.damping) * self.static_potential


        if self.sc:
            e, mo_coeff = self.fock_scf(self.static_potential)
        else:
            e, mo_coeff = scipy.linalg.eigh(self.sc_fock, self.emb.get_ovlp())
        
        gf_static = Lehmann(e, mo_coeff, chempot=gf.chempot)
        static_gap = gap(gf_static)
        if self.store_hist:        
            #self.static_potential_frag_hist.append(v_frag.copy())
            self.static_potential_hist.append(self.static_potential.copy())
            self.fock_hist.append(self.sc_fock.copy())
            self.static_gap_hist.append(static_gap)
            self.dynamic_gap_hist.append(dynamic_gap)
            self.mo_coeff_hist.append(mo_coeff.copy())

        self.log.info("Dynamic Gap = %f"%dynamic_gap)
        self.log.info("Static Gap = %f"%static_gap)
        return mo_coeff

    def fock_scf(self, v):
        """
            Relax density in presence of new static potential

            Parameters
            ----------
            v : ndarray
                Static potential
            
            Returns
            -------
            mo_coeff : ndarray
                New MO coefficients.
        """

        #mf = LatticeRHF(self.emb.mf.mol)
        mf_class = type(self.emb.mf)
        if mf_class is FoldedSCF:
            # Temporary work around for k-SCF
            raise NotImplementedError("QP-EWDMET with Fock re-diagonalisation not implemented for FoldedSCF")
        mf = mf_class(self.emb.mf.mol)
        #mf.get_fock_old = mf.get_fock
        #def get_fock(*args, **kwargs):
        #    return mf.get_fock_old(*args, **kwargs) + self.static_potential

        def get_hcore(*args, **kwargs):
            return self.emb.mf.get_hcore() + v
        mf.get_hcore = get_hcore
        e_tot = mf.kernel()


        # def get_fock(dm):
        #     dm_ao = np.linalg.multi_dot((mf.mo_coeff, dm, mf.mo_coeff.T))
        #     fock = mf.get_fock(dm=dm_ao)
        #     return np.linalg.multi_dot((mf.mo_coeff.T, fock, mf.mo_coeff))

        # # Use the DensityRelaxation class to relax the density matrix such
        # # that it is self-consistent with the Fock matrix, and the number of
        # # electrons is correct
        # solver = DensityRelaxation(get_fock, se, mol.nelectron)
        # solver.conv_tol = 1e-10
        # solver.max_cycle_inner = 30
        # solver.kernel()2

        if mf.converged:
            self.log.info("SCF converged, energy: {:.6f}".format(e_tot))
        else:
            self.log.warning("SCF NOT converged, energy: {:.6f}".format(e_tot))

        if self.store_scfs:
            self.scfs.append(mf)
        return mf.mo_energy, mf.mo_coeff

    def check_convergence(self, e_tot, dm1, e_last=None, dm1_last=None, etol=None, dtol=None):
        _, de, ddm = super().check_convergence(e_tot, dm1, e_last=e_last, dm1_last=dm1_last, etol=etol, dtol=dtol)
        dv = np.inf
        if self.static_potential_last is not None:
            dv = np.abs(self.static_potential - self.static_potential_last).sum()
            if dv < self.static_potential_conv_tol:
                return True, dv, ddm
        return False, dv, ddm

    def get_greens_function(self):
        """ 
        Calculate the dynamic and static Green's function in the Lehmann representation
        
        Returns
        -------
        gf : Lehmann
            Dynamic Green's function from Block Lanczos
        gf_qp : Lehmann
            Static Green's function from Fock matrix + Klein potential
         """


        # Shift final auxiliaries to ensure right particle number
        fock = self.fock + self.static_self_energy
        nelec = self.emb.mf.mol.nelectron
        shift = AuxiliaryShift(fock, self.self_energy, nelec, occupancy=2, log=self.emb.log)
        shift.kernel()
        se_shifted = shift.get_self_energy()
        vayesta.log.info('Final (shifted) auxiliaries: {} ({}o, {}v)'.format(se_shifted.naux, se_shifted.occupied().naux, se_shifted.virtual().naux))
        self.se_shifted = se_shifted
        # Find the Green's function

        gf = Lehmann(*se_shifted.diagonalise_matrix_with_projection(fock), chempot=se_shifted.chempot)
        dm = gf.occupied().moment(0) * 2.0
        nelec_gf = np.trace(dm)
        if not np.isclose(nelec_gf, gf.occupied().weights(occupancy=2).sum()):
            vayesta.log.warning('Number of electrons in final (shifted) GF: %f'%nelec_gf)
        else:
            vayesta.log.info('Number of electrons in final (shifted) GF with dynamical self-energy: %f'%nelec_gf)
        
        if not np.isclose(nelec_gf, float(nelec)):
            vayesta.log.warning('Number of electrons in final (shifted) GF: %f'%nelec_gf)

        #qp_ham = self.emb.get_fock() + self.static_potential
        qp_ham = self.fock + self.static_self_energy + self.static_potential
        qp_e, qp_c = np.linalg.eigh(qp_ham)
        self.qpham = qp_ham
        qp_mu = (qp_e[nelec//2-1] + qp_e[nelec//2] ) / 2
        self.qpmu = qp_mu
        gf_qp = Lehmann(qp_e, qp_c, chempot=qp_mu)

        return gf, gf_qp