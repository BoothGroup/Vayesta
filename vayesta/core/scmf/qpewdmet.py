import numpy as np
import scipy.linalg

import vayesta
from vayesta.core.scmf.scmf import SCMF
from vayesta.core.foldscf import FoldedSCF
from vayesta.lattmod import LatticeRHF

from dyson import Lehmann, FCI, MBLGF, MixedMBLGF, NullLogger, AuxiliaryShift
import pyscf.scf



class QPEWDMET_RHF(SCMF):
    """ Quasi-particle self-consistent energy weighted density matrix embedding """
    name = "QP-EWDMET"

    def __init__(self, emb, with_static=True, proj=2, v_conv_tol=1e-5, eta=1e-2, damping=0, sc=True, store_hist=True, *args, **kwargs):
        """ 
        Initialize QPEWDMET 
        
        Parameters
        ----------
        emb : QEmbedding o
            Embedding object on which self consitency is based
        proj : int 
            Number of fragment projectors applied to cluster self-energy
        v_conv_tol : float
            Convergence threshold in Klein potential
        eta : float
            Broadening factor for Klein potential
        damping : float
            Damping factor for Fock matrix update
        sc : bool
            Use self-consistent determination of MOs for new Fock matrix
        store_hist : bool
            Store history throughout SCMF calculation (for debugging purposes)
        """
        
        self.sc_fock = emb.get_fock()
        self.sc = sc
        self.eta = eta # Broadening factor
        self.se = None
        self.v = None
        self.v_last = None
        self.v_conv_tol = v_conv_tol
        self.proj = proj
        self.store_hist = store_hist 
        self.with_static = with_static 
        

        if self.store_hist:
            self.v_hist = []
            self.v_frag_hist = []
            self.fock_hist = []
            self.static_gap_hist = []
            self.dynamic_gap_hist = []  
            self.mo_coeff_hist = []

        super().__init__(emb, *args, **kwargs)

        self.damping = damping

    def update_mo_coeff(self, mf, diis=None):
        
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

        if self.v is not None:
            self.v_last = self.v.copy()
        couplings = []
        energies = []
        self.v = np.zeros_like(self.sc_fock)
        for f in self.emb.fragments:
            # Calculate self energy from cluster moments
            th, tp = f.results.moms
            vth, vtp = th.copy(), tp.copy()
            solverh = MBLGF(th, log=NullLogger())
            solverp = MBLGF(tp, log=NullLogger())
            solver = MixedMBLGF(solverh, solverp)
            solver.kernel()


            se = solver.get_self_energy()
            
            energies_f = se.energies
            couplings_f = se.couplings

            ovlp = self.emb.get_ovlp()
            cf = f.c_frag.T @ ovlp @ f.cluster.c_active  # Projector onto fragment (frag|cls)
            cfc = cf.T @ cf

            fock = self.emb.get_fock()
            fock_cls = f.cluster.c_active.T @ fock @ f.cluster.c_active
            e_cls = np.diag(fock_cls)
            
            
            #v += np.linalg.multi_dot((ca, v_frag, ca.T))
            v_cls = se.as_static_potential(e_cls, eta=self.eta) # Single particle potential from Klein Functional (used to update MF for the self-consistnecy)
            if self.with_static:
                v_cls += th[1] + tp[1] - fock_cls # Static self energy

            if self.proj == 2:
                v_frag = np.linalg.multi_dot((cf, v_cls, cf.T))
                self.v += np.linalg.multi_dot((f.c_frag, v_frag, f.c_frag.T))

                couplings.append(f.c_frag @ cf @ se.couplings)
                energies.append(se.energies)

            elif self.proj == 1:

                v_frag = cfc @ v_cls  
                v_frag = 0.5 * (v_frag + v_frag.T)
                self.v += f.cluster.c_active @ v_frag @ f.cluster.c_active.T

                sym_coup = np.einsum('pa,qa->apq', np.dot(cfc, se.couplings) , se.couplings) 
                sym_coup = 0.5 * (sym_coup + sym_coup.transpose(0,2,1))
                couplings_cf = np.zeros((sym_coup.shape[0]*2, sym_coup.shape[1]))
                for a in range(sym_coup.shape[0]):
                    m = sym_coup[a]
                    val, vec = np.linalg.eigh(m)
                    w = vec[:,-2:] @ np.diag(np.sqrt(val[-2:], dtype=np.complex64))
                    w = np.array(w, dtype=np.float64)
                    couplings_cf[2*a:2*a+2] = w.T
                
                couplings.append(f.cluster.c_active @ couplings_cf.T)
                energies.append(np.repeat(se.energies, 2)) 

        couplings = np.hstack(couplings)
        energies = np.concatenate(energies)
        self.se = Lehmann(energies, couplings)

        gap = lambda e: e[len(e)//2] - e[len(e)//2-1]
        dynamic_gap = gap(self.se.energies)

        v_old = self.v.copy()
        if diis is not None:
            self.v = diis.update(self.v)

        new_fock = self.emb.get_fock() + self.v
        self.sc_fock = self.damping * self.sc_fock + (1-self.damping) * new_fock
        #self.sc_fock = self.sc_fock + (1-self.damping) * self.v
        
        static_gap = gap(energies)


        if self.sc:
            #mf = LatticeRHF(self.emb.mf.mol)
            mf_class = type(self.emb.mf)
            if mf_class is FoldedSCF:
                # Temporary work around for k-SCF
                raise NotImplementedError("QP-EWDMET with Fock re-diagonalisation not implemented for FoldedSCF")
            mf = mf_class(self.emb.mf.mol)
            mf.get_fock_old = mf.get_fock
            def get_fock(*args, **kwargs):
                return mf.get_fock_old(*args, **kwargs) + self.v
            mf.get_fock = get_fock
            e_tot = mf.kernel()

            mo_coeff = mf.mo_coeff
            scf_conv = mf.converged
            static_gap = gap(mf.mo_energy) 
            self.log.info("SCF converged: {}, energy: {:.6f}".format(scf_conv, e_tot))
        else:
            e, mo_coeff = scipy.linalg.eigh(self.sc_fock, self.emb.get_ovlp())
            static_gap = gap(e)
        if self.store_hist:        
            self.v_frag_hist.append(v_frag.copy())
            self.v_hist.append(self.v.copy())
            self.fock_hist.append(self.sc_fock.copy())
            self.static_gap_hist.append(static_gap)
            self.dynamic_gap_hist.append(dynamic_gap)
            self.mo_coeff_hist.append(mo_coeff.copy())

        self.log.info("Dynamic Gap = %f"%dynamic_gap)
        self.log.info("Static Gap = %f"%static_gap)
        return mo_coeff

    def check_convergence(self, e_tot, dm1, e_last=None, dm1_last=None, etol=None, dtol=None):
        _, de, ddm = super().check_convergence(e_tot, dm1, e_last=e_last, dm1_last=dm1_last, etol=etol, dtol=dtol)
        dv = np.inf
        if self.v_last is not None:
            dv = np.abs(self.v - self.v_last).sum()
            if dv < self.v_conv_tol:
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
        nelec = self.emb.mf.mol.nelectron
        shift = AuxiliaryShift(self.emb.get_fock(), self.se, nelec, occupancy=2, log=NullLogger())
        shift.kernel()
        se_shifted = shift.get_self_energy()
        vayesta.log.info('Final (shifted) auxiliaries: {} ({}o, {}v)'.format(se_shifted.naux, se_shifted.occupied().naux, se_shifted.virtual().naux))
        self.se_shifted = se_shifted
        # Find the Green's function
        fock = self.sc_fock if self.with_static else self.emb.mf.get_fock()
        gf = Lehmann(*se_shifted.diagonalise_matrix_with_projection(fock), chempot=se_shifted.chempot)
        dm = gf.occupied().moment(0) * 2.0
        nelec_gf = np.trace(dm)
        if not np.isclose(nelec_gf, gf.occupied().weights(occupancy=2).sum()):
            vayesta.log.warning('Number of electrons in final (shifted) GF: %f'%nelec_gf)
        else:
            vayesta.log.info('Number of electrons in final (shifted) GF with dynamical self-energy: %f'%nelec_gf)
        
        if not np.isclose(nelec_gf, float(nelec)):
            vayesta.log.warning('Number of electrons in final (shifted) GF: %f'%nelec_gf)

        # TODO: Also, get the energy, IP, EA and gap from GF in the presence of the auxiliaries, as 
        # well as Fermi liquid parameters from the self-energy

        # Find qp-Green's function (used to define the self-consistent bath space
        # (and bath effective interactions if not using an interacting bath)

        #qp_ham = self.emb.get_fock() + self.v
        qp_ham = self.sc_fock
        qp_e, qp_c = np.linalg.eigh(qp_ham)
        self.qpham = qp_ham
        qp_mu = (qp_e[nelec//2-1] + qp_e[nelec//2] ) / 2
        self.qpmu = qp_mu
        gf_qp = Lehmann(qp_e, qp_c, chempot=qp_mu)

        return gf, gf_qp