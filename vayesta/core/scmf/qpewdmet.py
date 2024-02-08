import numpy as np
import scipy.linalg

import vayesta
from vayesta.core.scmf.scmf import SCMF
from vayesta.core.foldscf import FoldedSCF
from vayesta.lattmod import LatticeRHF

from dyson import Lehmann, FCI, MBLGF, MixedMBLGF, NullLogger, AuxiliaryShift
import pyscf.scf

def get_unique(array, atol=1e-15):
    
    # Find elements of a sorted float array which are unique up to a tolerance
    
    assert len(array.shape) == 1
    
    i = 0
    slices = []
    while i < len(array):
        j = 1
        idxs = [i]
        while i+j < len(array):
            if np.abs(array[i] - array[i+j]) < atol:
                idxs.append(i+j)
                j += 1
            else: 
                break
        i = i + j
        slices.append(np.s_[idxs[0]:idxs[-1]+1])
    new_array = np.array([array[s].mean() for s in slices])
    return new_array, slices

class QPEWDMET_RHF(SCMF):
    """ Quasi-particle self-consistent energy weighted density matrix embedding """
    name = "QP-EWDMET"

    def __init__(self, emb, with_static=True, proj=2, v_conv_tol=1e-5, eta=1e-2, damping=0, sc=True, store_hist=True, store_scfs=False, use_sym=False, v_init=None, se_degen_tol=1e-6, se_eval_tol=1e-6, drop_non_causal=False, *args, **kwargs):
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
        use_sym : bool
            Use fragment symmetry
        v_init : ndarray
            Inital static potential
        """
        
        self.sc_fock = emb.get_fock()
        self.static_self_energy = np.zeros_like(self.sc_fock)
        self.sc = sc
        self.eta = eta # Broadening factor
        self.se = None
        self.v = None
        self.v_last = None
        self.v_conv_tol = v_conv_tol
        self.proj = proj
        self.store_hist = store_hist 
        self.use_sym = use_sym
        self.v_init = v_init
        self.store_scfs = store_scfs
        self.se_degen_tol = se_degen_tol
        self.se_eval_tol = se_eval_tol
        self.drop_non_causal = drop_non_causal
        
        super().__init__(emb, *args, **kwargs)

        if self.store_hist:
            self.v_hist = []
            self.v_frag_hist = []
            self.fock_hist = []
            self.static_gap_hist = []
            self.dynamic_gap_hist = []  
            self.mo_coeff_hist = []

            self.mom_hist = []

        if self.store_scfs and self.sc:
            self.scfs = []

        self.damping = damping

        if self.v_init is not None:
            
            e, mo_coeff = self.fock_scf(self.v_init)
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

        if self.v is not None:
            self.v_last = self.v.copy()

        self.fock = self.emb.get_fock()
        couplings = []
        energies = []
        self.v = np.zeros_like(self.fock)
        self.static_se = np.zeros_like(self.fock)

        fragments = self.emb.get_fragments(sym_parent=None) if self.use_sym else self.emb.get_fragments()
        for i, f in enumerate(fragments):
            # Calculate self energy from cluster moments
            th, tp = f.results.moms

            if self.store_hist:
                thc = np.einsum('pP,qQ,nPQ->npq', f.cluster.c_active, f.cluster.c_active, th)
                tpc = np.einsum('pP,qQ,nPQ->npq', f.cluster.c_active, f.cluster.c_active, tp)
                self.mom_hist.append((thc,tpc))

            vth, vtp = th.copy(), tp.copy()
            solverh = MBLGF(th, log=NullLogger())
            solverp = MBLGF(tp, log=NullLogger())
            solver = MixedMBLGF(solverh, solverp)
            solver.kernel()


            se = solver.get_self_energy()

            energies_f = se.energies
            couplings_f = se.couplings

            ovlp = self.emb.get_ovlp()
            mc = f.get_overlap('mo|cluster')
            mf = f.get_overlap('mo|frag')
            fc = f.get_overlap('frag|cluster')
            cfc = fc.T @ fc

            
            fock_cls = mc.T @ self.fock @ mc
            e_cls = np.diag(fock_cls)
            
            
            #v += np.linalg.multi_dot((ca, v_frag, ca.T))
            v_cls = se.as_static_potential(e_cls, eta=self.eta) # Single particle potential from Klein Functional (used to update MF for the self-consistnecy)
            
            static_se_cls = th[1] + tp[1] - fock_cls # Static self energy

            if self.proj == 2:
                v_frag = fc @ v_cls @ fc.T
                static_se_frag = fc @ static_se_cls @ fc.T
                self.v += f.c_frag @ v_frag @ f.c_frag.T
                self.static_self_energy += mf @ static_se_frag @ mf.T
                couplings.append(mf @ fc @ se.couplings)
                energies.append(se.energies)

                if self.use_sym:
                    # FIX
                    for child in f.get_symmetry_children():
                        self.v += np.linalg.multi_dot((child.c_frag, v_frag, child.c_frag.T))
                        self.static_self_energy += np.linalg.multi_dot((child.c_frag, static_se_frag, child.c_frag.T))
                        couplings.append(child.c_frag @ fc @ se.couplings)
                        energies.append(se.energies)

                    

            elif self.proj == 1:

                v_frag = cfc @ v_cls  
                v_frag = 0.5 * (v_frag + v_frag.T)
                
                static_self_energy_frag = cfc @ static_se_cls
                static_self_energy_frag = 0.5 * (static_self_energy_frag + static_self_energy_frag.T)

                sym_coup = np.einsum('pa,qa->apq', np.dot(cfc, se.couplings) , se.couplings) 
                sym_coup = 0.5 * (sym_coup + sym_coup.transpose(0,2,1))
                
                rank = 2 # rank / multiplicity 
                tol = 1e-12
                couplings_cf, energies_cf = [], []
                for a in range(sym_coup.shape[0]):
                    m = sym_coup[a]
                    val, vec = np.linalg.eigh(m)
                    idx = np.abs(val) > tol
                    assert (np.abs(val) > tol).sum() <= rank
                    w = vec[:,idx] @ np.diag(np.sqrt(val[idx], dtype=np.complex64))
                    couplings_cf.append(w)
                    energies_cf += [se.energies[a] for e in range(idx.sum())]

                couplings_cf = np.hstack(couplings_cf)
                self.v += f.cluster.c_active @ v_frag @ f.cluster.c_active.T
                self.static_self_energy += mc @ static_self_energy_frag @ mc.T

                couplings.append(mc @ couplings_cf)
                energies.append(energies_cf)    

                if self.use_sym:
                    for fc in f.get_symmetry_children():
                        self.v += fc.cluster.c_active @ v_frag @ fc.cluster.c_active.T
                        self.static_self_energy += fc.cluster.c_active @ static_self_energy_frag @ fc.cluster.c_active.T
                        couplings.append(fc.cluster.c_active @ couplings_cf.T)
                        energies.append(np.repeat(se.energies, 2)) 



        couplings = np.hstack(couplings)
        energies = np.concatenate(energies)
        self.se = Lehmann(energies, couplings)
        
        if self.proj == 1:
            self.se = self.remove_se_degeneracy(self.se, dtol=self.se_degen_tol, etol=self.se_eval_tol)

        gap = lambda e: e[len(e)//2] - e[len(e)//2-1]
        dynamic_gap = gap(self.se.energies)

        v_old = self.v.copy()
        if diis is not None:
            self.v = diis.update(self.v)

        new_fock = self.fock + self.v
        self.sc_fock = self.damping * self.fock + (1-self.damping) * new_fock
        #self.sc_fock = self.sc_fock + (1-self.damping) * self.v


        if self.sc:
            e, mo_coeff = self.fock_scf(self.v)
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


    def remove_se_degeneracy(self, se, dtol=1e-8, etol=1e-6):
        self.log.info("Removing degeneracy in self-energy - degenerate energy tol=%e   evec tol=%e"%(dtol, etol))
        e, v = se.energies, se.couplings
        e_new, slices = get_unique(e, atol=dtol)#
        self.log.info("Number of energies = %d,  unique = %d"%(len(e),len(e_new)))
        energies, couplings = [], []
        for i, s in enumerate(slices):
            mat = np.einsum('pa,qa->pq', v[:,s], v[:,s]).real
            val, vec = np.linalg.eigh(mat)
            if self.drop_non_causal:
                idx = val > etol
            else:
                idx = np.abs(val) > etol
            if np.sum(val[idx] < -etol) > 0:
                self.log.warning("Large negative eigenvalues - non-causal self-energy")
            w = vec[:,idx] @ np.diag(np.sqrt(val[idx], dtype=np.complex64))
            couplings.append(w)
            energies += [e_new[i] for _ in range(idx.sum())]

            self.log.info("    | E = %e << %s"%(e_new[i],e[s]))
            self.log.info("       evals: %s"%val)
            self.log.info("       kept:  %s"%(val[idx]))   
        couplings = np.hstack(couplings).real
        return Lehmann(np.array(energies), np.array(couplings))

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
        #    return mf.get_fock_old(*args, **kwargs) + self.v

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
        fock = self.fock + self.static_self_energy
        nelec = self.emb.mf.mol.nelectron
        shift = AuxiliaryShift(fock, self.se, nelec, occupancy=2, log=NullLogger())
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

        #qp_ham = self.emb.get_fock() + self.v
        qp_ham = self.fock + self.static_self_energy + self.v
        qp_e, qp_c = np.linalg.eigh(qp_ham)
        self.qpham = qp_ham
        qp_mu = (qp_e[nelec//2-1] + qp_e[nelec//2] ) / 2
        self.qpmu = qp_mu
        gf_qp = Lehmann(qp_e, qp_c, chempot=qp_mu)

        return gf, gf_qp