import logging
import dataclasses

import numpy as np

import pyscf
import pyscf.agf2

import vayesta
import vayesta.ewf
from vayesta.ewf import helper
from vayesta.core.util import Options, time_string
from vayesta.core.qmethod import QEmbeddingMethod
from vayesta.agf2.fragment import EAGF2Fragment
from vayesta.agf2.ragf2 import RAGF2
from vayesta.agf2 import util

try:
    from mpi4py import MPI
    timer = MPI.Wtime
except ImportError:
    from timeit import default_timer as timer


@dataclasses.dataclass
class EAGF2Options(Options):
    ''' Options for EAGF2 calculations
    '''

    # --- Fragment settings
    fragment_type: str = 'IAO'
    iao_minao: str = 'auto'

    # --- Bath settings
    dmet_threshold: float = 1e-4
    orthogonal_mo_tol: float = False
    bath_type: str = 'dmet+mp2'
    nmom_bath: int = 0
    wtol_bath: float = 1e-14

    # --- Solver settings
    solver_options: dict = dataclasses.field(default_factory=dict)

    # --- Other
    strict: bool = False


@dataclasses.dataclass
class EAGF2Results:
    ''' Results for EAGF2 calculations
    '''

    bno_threshold: float = None
    e_corr: float = None
    e_1b: float = None
    e_2b: float = None
    gf: pyscf.agf2.GreensFunction = None
    se: pyscf.agf2.SelfEnergy = None


class EAGF2(QEmbeddingMethod):

    FRAGMENT_CLS = EAGF2Fragment

    def __init__(self, mf, bno_threshold=1e-8, options=None, log=None, **kwargs):
        ''' Embedded AGF2 calculation

        Parameters
        ----------
        mf : pyscf.scf ojbect
            Converged mean-field object.
        '''

        super().__init__(mf, log=log)
        t0 = timer()

        self.opts = options
        if self.opts is None:
            self.opts = EAGF2Options(**kwargs)
        else:
            self.opts = self.opts.replace(kwargs)
        self.log.info("EAGF2 parameters:")
        for key, val in self.opts.items():
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
        self.log.info("Max. non-orthogonality of input orbitals= %.2e%s", nonorth, " (!!!)" if nonorth > 1e-5 else "")
        if self.opts.orthogonal_mo_tol and nonorth > self.opts.orthogonal_mo_tol:
            t0 = timer()
            self.log.info("Orthogonalizing orbitals...")
            self.mo_coeff = helper.orthogonalize_mo(c, self.get_ovlp())
            change = abs(np.diag(np.linalg.multi_dot((self.mo_coeff.T, self.get_ovlp(), c)))-1)
            self.log.info("Max. orbital change= %.2e%s", change.max(), " (!!!)" if change.max() > 1e-4 else "")
            self.log.timing("Time for orbital orthogonalization: %s", time_string(timer()-t0))

        # Prepare fragments
        #if self.local_orbital_type in ("IAO", "LAO"):
        t1 = timer()
        fragkw = {}
        if self.opts.fragment_type.upper() == 'IAO':
            if self.opts.iao_minao == 'auto':
                self.opts.iao_minao = helper.get_minimal_basis(self.mol.basis)
                self.log.warning("Minimal basis set '%s' for IAOs was selected automatically.",  self.opts.iao_minao)
            self.log.info("Computational basis= %s", self.mol.basis)
            self.log.info("Minimal basis=       %s", self.opts.iao_minao)
            fragkw['minao'] = self.opts.iao_minao
        self.init_fragmentation(self.opts.fragment_type, **fragkw)
        self.log.timing("Time for fragment initialization: %s", time_string(timer() - t1))

        self.log.timing("Time for EAGF2 setup: %s", time_string(timer() - t0))

        self.iteration = 0
        self.cluster_results = {}
        self.results = []
        self.e_corr = 0.0


    def __repr__(self):
        keys = ['mf', 'bno_threshold']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])


    @property
    def e_tot(self):
        """Total energy."""
        return self.e_mf + self.e_corr


    def kernel(self, bno_threshold=None):
        ''' Run EAGF2

        Parameters
        ----------
        bno_threshold : float or list, optional
            Bath natural orbital threhsold. Default: 1e-8
        '''

        t0 = timer()

        bno_threshold = bno_threshold or self.bno_threshold
        if np.ndim(bno_threshold) == 0:
            bno_threshold = [bno_threshold]
        bno_threshold = np.sort(np.asarray(bno_threshold))

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")

        self.lo = pyscf.lo.orth_ao(self.mol, "lowdin")

        nelec_frags = sum([f.sym_factor*f.nelectron for f in self.loop()])
        self.log.info("Total number of mean-field electrons over all fragments= %.8f", nelec_frags)
        if abs(nelec_frags - np.rint(nelec_frags)) > 1e-4:
            self.log.warning("Number of electrons not integer!")


        for i, bno_thr in enumerate(bno_threshold):
            self.log.info("Now running BNO threshold = %.2e", bno_thr)
            self.log.info("************************************")

            se = []
            fock = []

            for x, frag in enumerate(self.fragments):
                self.log.info("Now running %s", frag)
                self.log.info("************%s", len(str(frag))*"*")
                self.log.changeIndentLevel(1)

                result = frag.kernel(bno_threshold=bno_thr)
                self.cluster_results[(frag.id, bno_thr)] = result

                if not result.converged:
                    self.log.error("%s is not converged", frag)
                else:
                    self.log.info("%s is done.", frag)

                self.log.changeIndentLevel(-1)

                fock.append(result.fock)
                se.append(result.se)

            fock = util.block_diagonal(fock)
            se = pyscf.agf2.SelfEnergy(
                    np.concatenate([x.energy for x in se]),
                    util.block_diagonal([x.coupling for x in se]),
            )

            c = self.mf.mo_coeff
            s = self.mf.get_ovlp()
            if self.opts.fragment_type.lower() == 'iao':
                coeff = np.linalg.multi_dot((c.T.conj(), s, self.iao_coeff))
            elif self.opts.fragment_type.lower() == 'lowdin-ao':
                coeff = np.linalg.multi_dot((c.T.conj(), s, self.lao_coeff))

            fock = np.linalg.multi_dot((coeff, fock, coeff.T.conj()))
            se.coupling = np.dot(coeff, se.coupling)

            class FakeRAGF2(RAGF2):
                def __init__(self, mf):
                    tmplog = logging.getLogger('tmp')
                    tmplog.setLevel(logging.ERROR)
                    eri = mf._eri
                    RAGF2.__init__(self, mf, eri=eri, log=tmplog)

                def get_fock(self, *args, **kwargs):
                    rdm1 = RAGF2.make_rdm1(self, *args, **kwargs)
                    mo_coeff = self.mf.mo_coeff
                    rdm1 = np.linalg.multi_dot((mo_coeff, rdm1, mo_coeff.T.conj()))
                    vj, vk = pyscf.scf._vhf.incore(self.eri, rdm1)
                    fock = self.mf.get_hcore() + vj - 0.5 * vk
                    fock = np.linalg.multi_dot((mo_coeff.T.conj(), fock, mo_coeff))
                    return fock

            gf2 = FakeRAGF2(self.mf)
            gf, se = gf2.fock_loop(gf=se.get_greens_function(fock), se=se)

            rdm1 = gf.make_rdm1()
            h1e = np.linalg.multi_dot((c.T.conj(), self.mf.get_hcore(), c))
            e_1b = 0.5 * np.sum(rdm1 * (h1e + fock))
            e_1b += self.mf.mol.energy_nuc()
            e_2b = RAGF2.energy_2body(None, gf=gf, se=se)
            e_corr = e_1b + e_2b - self.mf.e_tot

            e_ip = -gf.get_occupied().energy.max()
            e_ea = gf.get_virtual().energy.min()

            result = EAGF2Results(
                    bno_threshold=bno_thr,
                    e_corr=e_corr,
                    e_1b=e_1b,
                    e_2b=e_2b,
                    gf=gf,
                    se=se,
            )
            self.results.append(result)

        self.log.info("E(corr) = %20.12f", e_corr)
        self.log.info("E(1b)   = %20.12f", e_1b)
        self.log.info("E(2b)   = %20.12f", e_2b)
        self.log.info("E(tot)  = %20.12f", e_1b + e_2b)

        self.log.info("IP      = %20.12f", e_ip)
        self.log.info("EA      = %20.12f", e_ea)
        self.log.info("Gap     = %20.12f", e_ip + e_ea)

        self.log.info("Total wall time:  %s", time_string(timer() - t0))
        self.log.info("All done.")

    run = kernel


    def print_clusters(self):
        """Print fragments of calculations."""
        self.log.info("%3s  %20s  %8s  %4s", "ID", "Name", "Solver", "Size")
        for frag in self.loop():
            self.log.info("%3d  %20s  %8s  %4d", frag.id, frag.name, frag.solver, frag.size)



if __name__ == '__main__':
    bno_threshold = [1e-7, 1e-6, 1e-5]
    fragment_type = 'IAO'
    #fragment_type = 'Lowdin-AO'

    mol = pyscf.gto.Mole()
    mol.atom = 'Li 0 0 0; H 0 0 1.4'
    #mol.atom = 'N 0 0 0; N 0 0 0.8'
    #mol.atom = 'O 0 0 0; H 0 0 1; H 0 1 0'
    mol.basis = 'aug-cc-pvdz'
    #mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.max_memory = 1e9
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.run()

    ip_mf = -mf.mo_energy[mf.mo_occ > 0].max()
    ea_mf = mf.mo_energy[mf.mo_occ == 0].min()

    eagf2 = EAGF2(mf, bno_threshold=bno_threshold, fragment_type=fragment_type)
    for i in range(mol.natm):
        frag = eagf2.make_atom_fragment(i)
        frag.make_bath()
    eagf2.run()


    vayesta.log.setLevel(logging.ERROR)
    agf2 = RAGF2(mf)
    agf2.kernel()
    vayesta.log.setLevel(logging.INFO)

    vayesta.log.info('         %14s %14s %s', 'RHF', 'AGF2', ' '.join(['%14s' % res.bno_threshold for res in eagf2.results]))
    vayesta.log.info('max(N) = %14d %14d %s',     mol.nao,              mol.nao,    ' '.join(['%14d' % max([eagf2.cluster_results[(frag.id, res.bno_threshold)].n_active for frag in eagf2.fragments]) for res in eagf2.results]))
    vayesta.log.info('E(1b)  = %14.8f %14.8f %s', mf.e_tot,             agf2.e_1b,  ' '.join(['%14.8f' % res.e_1b for res in eagf2.results]))
    vayesta.log.info('E(2b)  = %14.8f %14.8f %s', 0.0,                  agf2.e_2b,  ' '.join(['%14.8f' % res.e_2b for res in eagf2.results]))
    vayesta.log.info('E(tot) = %14.8f %14.8f %s', mf.e_tot,             agf2.e_tot, ' '.join(['%14.8f' % (res.e_1b+res.e_2b) for res in eagf2.results]))
    vayesta.log.info('IP     = %14.8f %14.8f %s', ip_mf,                agf2.e_ip,  ' '.join(['%14.8f' % -res.gf.get_occupied().energy.max() for res in eagf2.results]))
    vayesta.log.info('EA     = %14.8f %14.8f %s', ea_mf,                agf2.e_ea,  ' '.join(['%14.8f' % res.gf.get_virtual().energy.min() for res in eagf2.results]))
    vayesta.log.info('Gap    = %14.8f %14.8f %s', ip_mf+ea_mf, agf2.e_ip+agf2.e_ea, ' '.join(['%14.8f' % (res.gf.get_virtual().energy.min()-res.gf.get_occupied().energy.max()) for res in eagf2.results]))
