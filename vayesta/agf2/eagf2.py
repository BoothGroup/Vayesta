import logging
import dataclasses

import numpy as np

import pyscf
import pyscf.lib
import pyscf.agf2

import vayesta
import vayesta.ewf
from vayesta.ewf import helper
from vayesta.core import QEmbeddingMethod
from vayesta.core.util import OptionsBase, time_string
from vayesta.agf2.fragment import EAGF2Fragment
from vayesta.agf2.ragf2 import RAGF2, RAGF2Options

try:
    from mpi4py import MPI
    timer = MPI.Wtime
except ImportError:
    from timeit import default_timer as timer

#TODO bath_type = 'ALL', 'NONE'
#TODO ragf2 as solver


@dataclasses.dataclass
class EAGF2Options(OptionsBase):
    ''' Options for EAGF2 calculations
    '''

    # --- Fragment settings
    fragment_type: str = 'Lowdin-AO'
    iao_minao: str = 'auto'

    # --- Bath settings
    bath_type: str = 'MP2-BNO'    # 'MP2-BNO', 'EWDMET' == 'POWER'
    nmom_bath: int = 2
    bno_threshold: float = 1e-8
    bno_threshold_factor: float = 1.0
    dmet_threshold: float = 1e-4
    ewdmet_threshold: float = 1e-4

    # --- Solver settings
    solver_options: dict = dataclasses.field(default_factory=dict)

    # --- Other
    strict: bool = False
    orthogonal_mo_tol = 1e-9


@dataclasses.dataclass
class EAGF2Results:
    ''' Results for EAGF2 calculations
    '''

    e_corr: float = None
    e_1b: float = None
    e_2b: float = None
    gf: pyscf.agf2.GreensFunction = None
    se: pyscf.agf2.SelfEnergy = None


#TODO: combine with vayesta.agf2.ragf2.RAGF2 functionality
#TODO: take fock loop opts from solver_options
def fock_loop(
        mf, gf, se,
        max_cycle_inner=50,
        max_cycle_outer=20,
        conv_tol_nelec=1e-7,
        conv_tol_rdm1=1e-7,
):
    '''
    Perform a Fock loop with Fock builds in AO basis
    '''

    nmo = gf.nphys
    nelec = mf.mol.nelectron
    diis = pyscf.lib.diis.DIIS()
    converged = False

    def get_fock():
        rdm1 = gf.make_rdm1()
        rdm1 = np.linalg.multi_dot((mf.mo_coeff, rdm1, mf.mo_coeff.T))
        fock = mf.get_fock(dm=rdm1)
        fock = np.linalg.multi_dot((mf.mo_coeff.T, fock, mf.mo_coeff))
        return fock

    fock = get_fock()
    rdm1_prev = np.zeros_like(fock)

    for niter1 in range(max_cycle_outer):
        se, opt = pyscf.agf2.chempot.minimize_chempot(
                se, fock, nelec, x0=se.chempot,
                tol=conv_tol_nelec*1e-2,
                maxiter=max_cycle_inner,
        )

        for niter2 in range(max_cycle_inner):
            w, v = se.eig(fock)
            se.chempot, nerr = pyscf.agf2.chempot.binsearch_chempot((w, v), nmo, nelec)
            gf = pyscf.agf2.GreensFunction(w, v[:nmo], chempot=se.chempot)

            fock = get_fock()
            rdm1 = gf.make_rdm1()
            fock = diis.update(fock, xerr=None)

            derr = np.max(np.absolute(rdm1 - rdm1_prev))
            rdm1_prev = rdm1.copy()

            if derr < conv_tol_rdm1:
                break

        if derr < conv_tol_rdm1 and abs(nerr) < conv_tol_nelec:
            converged = True
            break

    return gf, se, fock


class EAGF2(QEmbeddingMethod):

    Fragment = EAGF2Fragment

    def __init__(self, mf, options=None, log=None, **kwargs):
        ''' Embedded AGF2 calculation

        Parameters
        ----------
        mf : pyscf.scf ojbect
            Converged mean-field object.
        '''

        super().__init__(mf, log=log)
        t0 = timer()

        # --- Quiet logger for AGF2 calculations on the clusters
        self.quiet_log = logging.Logger('quiet')
        self.quiet_log.setLevel(logging.CRITICAL)

        self.opts = options
        if self.opts is None:
            self.opts = EAGF2Options(**kwargs)
        else:
            self.opts = self.opts.replace(kwargs)
        self.log.info("EAGF2 parameters:")
        for key, val in self.opts.items():
            if key == 'solver_options':
                continue
            self.log.info("  > %-24s %r", key + ":", val)

        solver_options = RAGF2Options(**self.opts.solver_options)
        self.opts.solver_options = solver_options
        self.log.info("RAGF2 solver parameters:")
        for key, val in self.opts.solver_options.items():
            self.log.info("  > %-24s %r", key + ":", val)

        # --- Check input
        if not mf.converged:
            if self.opts.strict:
                raise RuntimeError("Mean-field calculation not converged.")
            else:
                self.log.error("Mean-field calculation not converged.")
        self.bno_threshold = bno_threshold

        # Orthogonalize insufficiently orthogonal MOs
        # (For example as a result of k2gamma conversion with low cell.precision)
        c = self.mo_coeff.copy()
        assert np.all(c.imag == 0), "max|Im(C)|= %.2e" % abs(c.imag).max()
        ctsc = np.linalg.multi_dot((c.T, self.get_ovlp(), c))
        nonorth = abs(ctsc - np.eye(ctsc.shape[-1])).max()
        self.log.info("Max. non-orthogonality of input orbitals= %.2e%s", nonorth,
                      " (!!!)" if nonorth > 1e-5 else "")
        if self.opts.orthogonal_mo_tol and nonorth > self.opts.orthogonal_mo_tol:
            t0 = timer()
            self.log.info("Orthogonalizing orbitals...")
            self.mo_coeff = helper.orthogonalize_mo(c, self.get_ovlp())
            change = abs(np.diag(np.linalg.multi_dot((self.mo_coeff.T, self.get_ovlp(), c)))-1)
            self.log.info("Max. orbital change= %.2e%s", change.max(),
                          " (!!!)" if change.max() > 1e-4 else "")
            self.log.timing("Time for orbital orthogonalization: %s", time_string(timer()-t0))

        # Prepare fragments
        t1 = timer()
        fragkw = {}
        if self.opts.fragment_type.upper() == 'IAO':
            if self.opts.iao_minao == 'auto':
                self.opts.iao_minao = helper.get_minimal_basis(self.mol.basis)
                self.log.warning("Minimal basis set '%s' for IAOs was selected automatically.",
                                 self.opts.iao_minao)
            self.log.info("Computational basis= %s", self.mol.basis)
            self.log.info("Minimal basis=       %s", self.opts.iao_minao)
            fragkw['minao'] = self.opts.iao_minao
        self.init_fragmentation(self.opts.fragment_type, **fragkw)
        self.log.timing("Time for fragment initialization: %s", time_string(timer() - t1))

        self.log.timing("Time for EAGF2 setup: %s", time_string(timer() - t0))

        self.cluster_results = {}
        self.result = None
        self.e_corr = 0.0


    def __repr__(self):
        keys = ['mf']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])


    @property
    def e_tot(self):
        """Total energy."""
        return self.e_mf + self.e_corr


    def kernel(self):
        ''' Run EAGF2
        '''

        t0 = timer()

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")

        nelec_frags = sum([f.sym_factor*f.nelectron for f in self.loop()])
        self.log.info("Total number of mean-field electrons over all fragments= %.8f", nelec_frags)
        if abs(nelec_frags - np.rint(nelec_frags)) > 1e-4:
            self.log.warning("Number of electrons not integer!")

        nmo = self.mf.mo_occ.size
        rdm1 = np.zeros((nmo, nmo))
        t_occ = np.zeros((2, nmo, nmo))  #TODO higher moments?
        t_vir = np.zeros((2, nmo, nmo))

        for x, frag in enumerate(self.fragments):
            self.log.info("Now running %s", frag)
            self.log.info("************%s", len(str(frag))*"*")
            self.log.changeIndentLevel(1)

            result = frag.kernel()
            self.cluster_results[frag.id] = result

            self.log.info("E(corr) = %20.12f", result.e_corr)
            if not result.converged:
                self.log.error("%s is not converged", frag)
            else:
                self.log.info("%s is done.", frag)

            ovlp = frag.mf.get_ovlp()
            c = pyscf.lib.einsum('pa,pq,qi->ai', result.c_active.conj(), ovlp, frag.mf.mo_coeff)

            rdm1 += pyscf.lib.einsum('pq,pi,qj->ij', result.rdm1, c.conj(), c)
            t_occ += pyscf.lib.einsum('...pq,pi,qj->...ij', result.t_occ, c.conj(), c)
            t_vir += pyscf.lib.einsum('...pq,pi,qj->...ij', result.t_vir, c.conj(), c)

            self.log.changeIndentLevel(-1)

        gf2 = RAGF2(self.mf,
                eri=np.zeros((1, 1, 1, 1)),
                log=self.quiet_log,
                options=self.opts.solver_options,
        )

        se_occ = gf2._build_se_from_moments(t_occ)
        se_vir = gf2._build_se_from_moments(t_vir)
        gf2.se = pyscf.agf2.aux.combine(se_occ, se_vir)

        mo_coeff = self.mf.mo_coeff
        rdm1_ao = np.linalg.multi_dot((mo_coeff, rdm1, mo_coeff.T.conj()))
        fock_ao = self.mf.get_fock(dm=rdm1_ao)
        fock = np.linalg.multi_dot((mo_coeff.T.conj(), fock_ao, mo_coeff))

        gf2.gf = gf2.se.get_greens_function(fock)

        if self.opts.solver_options.fock_loop:
            gf2.gf, gf2.se, fock = fock_loop(self.mf, gf2.gf, gf2.se)

        gf2.e_1b  = 0.5 * np.sum(rdm1 * (gf2.h1e + fock))
        gf2.e_1b += gf2.e_nuc
        gf2.e_2b  = gf2.energy_2body()

        result = EAGF2Results(
                e_corr=gf2.e_corr,
                e_1b=gf2.e_1b,
                e_2b=gf2.e_2b,
                gf=gf2.gf,
                se=gf2.se,
        )
        self.result = result

        with pyscf.lib.temporary_env(gf2, log=self.log):
            gf2.print_excitations()
            gf2.print_energies(output=True)

        self.log.info("Total wall time:  %s", time_string(timer() - t0))
        self.log.info("All done.")

    run = kernel


    def print_clusters(self):
        """Print fragments of calculations."""
        self.log.info("%3s  %20s  %8s  %4s", "ID", "Name", "Solver", "Size")
        for frag in self.loop():
            self.log.info("%3d  %20s  %8s  %4d", frag.id, frag.name, frag.solver, frag.size)



if __name__ == '__main__':
    bno_threshold = 0.0
    fragment_type = 'Lowdin-AO'

    mol = pyscf.gto.Mole()
    mol.atom = 'He 0 0 0; He 0 0 1'
    mol.basis = '6-31g'
    mol.verbose = 0
    mol.max_memory = 1e9
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.run()

    ip_mf = -mf.mo_energy[mf.mo_occ > 0].max()
    ea_mf = mf.mo_energy[mf.mo_occ == 0].min()

    eagf2 = EAGF2(mf,
            bno_threshold=bno_threshold,
            fragment_type=fragment_type,
            bath_type='MP2-BNO',
    )
    for i in range(mol.natm):
        frag = eagf2.make_atom_fragment(i)
        frag.make_bath()
    eagf2.run()

    vayesta.log.setLevel(logging.OUTPUT)
    agf2 = RAGF2(mf)
    agf2.kernel()
    vayesta.log.setLevel(logging.INFO)
