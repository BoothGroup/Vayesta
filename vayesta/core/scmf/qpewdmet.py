import numpy as np
import scipy.linalg

from vayesta.core.scmf.scmf import SCMF

from dyson import Lehmann, FCI, MBLGF, MixedMBLGF, NullLogger, AuxiliaryShift




class QPEWDMET_RHF(SCMF):

    name = "QP-EWDMET"

    def __init__(self, emb, eta=1e-2, *args, **kwargs):
        self.sc_fock = emb.get_fock()
        self.eta = eta # Broadening factor
        self.se = None
        self.v = None
        super().__init__(emb, *args, **kwargs)

    def update_mo_coeff(self, mf, diis=None):
        couplings = []
        energies = []
        vs = []
        for f in self.emb.fragments:
        

            th, tp = f.results.moms
        
            vth, vtp = th.copy(), tp.copy()
            solverh = MBLGF(th, log=NullLogger())
            solverp = MBLGF(tp, log=NullLogger())
            solver = MixedMBLGF(solverh, solverp)
            solver.kernel()


            se = solver.get_self_energy()
            
            energies_f = se.energies
            couplings_f = se.couplings


            c = np.linalg.multi_dot((f.c_frag.T, f.cluster.c_active))  # (frag|cls)
    
            F = self.emb.get_fock()
            Fc = np.einsum('pP,qQ,pq->PQ', f.cluster.c_active, f.cluster.c_active, F)
            e_cls = np.diag(Fc)
            
            v_cls = se.as_static_potential(e_cls, eta=self.eta)
            # Rotate into the fragment basis and tile for all fragments
            c_frag_canon = np.linalg.multi_dot((f.c_frag.T, f.cluster.c_active))
            v_frag = np.linalg.multi_dot((c_frag_canon, v_cls, c_frag_canon.T))  # (frag|frag)
            vs.append(v_frag)
    
            se.couplings = np.dot(c, se.couplings) 
            energies.append(se.energies)
            couplings.append(se.couplings)


        vcouplings = scipy.linalg.block_diag(*couplings)

        energies = np.concatenate(energies)
        se = Lehmann(energies, vcouplings)

        self.se = se

        # TODO: Will this only work for the 1D model? Check tiling with other models.
        self.v = scipy.linalg.block_diag(*vs) # (site|site)


        self.sc_fock += self.v
        e, mo_coeff = np.linalg.eigh(self.sc_fock)
        return mo_coeff

    def get_greens_function(self):

        # Shift final auxiliaries to ensure right particle number
        nelec = self.emb.mf.mol.nelectron
        shift = AuxiliaryShift(self.emb.get_fock(), self.se, nelec, occupancy=2, log=NullLogger())
        shift.kernel()
        se_shifted = shift.get_self_energy()
        print('Final (shifted) auxiliaries: {} ({}o, {}v)'.format(se_shifted.naux, se_shifted.occupied().naux, se_shifted.virtual().naux))
        self.se_shifted = se_shifted
        # Find the Green's function
        gf = Lehmann(*se_shifted.diagonalise_matrix_with_projection(self.emb.mf.get_fock()), chempot=se_shifted.chempot)
        dm = gf.occupied().moment(0) * 2.0
        nelec_gf = np.trace(dm)
        assert(np.isclose(nelec_gf, gf.occupied().weights(occupancy=2).sum()))
        print('Number of electrons in final (shifted) GF with dynamical self-energy: {}'.format(nelec_gf))
        assert(np.isclose(nelec_gf, float(nelec)))

        # TODO: Also, get the energy, IP, EA and gap from GF in the presence of the auxiliaries, as 
        # well as Fermi liquid parameters from the self-energy

        # Find qp-Green's function (used to define the self-consistent bath space
        # (and bath effective interactions if not using an interacting bath)

        qp_ham = self.emb.get_fock() + self.v
        qp_e, qp_c = np.linalg.eigh(qp_ham)
        self.qpham = qp_ham
        qp_mu = (qp_e[nelec//2-1] + qp_e[nelec//2] ) / 2
        self.qpmu = qp_mu
        gf_qp = Lehmann(qp_e, qp_c, chempot=qp_mu)

        return gf, gf_qp