
import os.path
import functools
from datetime import datetime
import dataclasses

import numpy as np
import scipy
import scipy.linalg

from vayesta.core.util import *
from vayesta.core import QEmbeddingMethod


from vayesta.ewf import helper
from .fragment import DMETFragment, DMETFragmentExit
from .sdp_sc import perform_SDP_fit

from timeit import default_timer as timer


@dataclasses.dataclass
class DMETOptions(Options):
    """Options for EWF calculations."""
    # --- Fragment settings
    fragment_type: str = 'IAO'
    localize_fragment: bool = False     # Perform numerical localization on fragment orbitals
    iao_minao : str = 'auto'            # Minimal basis for IAOs
    # --- Bath settings
    dmet_threshold: float = 1e-4
    orbfile: str = None                 # Filename for orbital coefficients
    # If multiple bno thresholds are to be calculated, we can project integrals and amplitudes from a previous larger cluster:
    project_eris: bool = False          # Project ERIs from a pervious larger cluster (corresponding to larger eta), can result in a loss of accuracy especially for large basis sets!
    project_init_guess: bool = True     # Project converted T1,T2 amplitudes from a previous larger cluster
    orthogonal_mo_tol: float = False
    #Orbital file
    plot_orbitals: str = False          # {True, False, 'dmet-exit'}
    plot_orbitals_dir: str = 'orbitals'
    plot_orbitals_kwargs: dict = dataclasses.field(default_factory=dict)
    # --- Solver settings
    solver_options: dict = dataclasses.field(default_factory=dict)
    make_rdm1: bool = False
    pop_analysis: str = False          # Do population analysis
    eom_ccsd: bool = False              # Perform EOM-CCSD in each cluster by default
    eomfile: str = 'eom-ccsd'           # Filename for EOM-CCSD states
    # Counterpoise correction of BSSE
    bsse_correction: bool = True
    bsse_rmax: float = 5.0              # In Angstrom
    # -- Self-consistency
    maxiter: int = 30
    sc_energy_tol: float = 1e-6
    sc_mode: int = 0
    # --- Other
    energy_partitioning: str = 'first-occ'
    strict: bool = False                # Stop if cluster not converged


@dataclasses.dataclass
class DMETResults:
    cluster_sizes: np.ndarray = None
    e_corr: float = None


VALID_SOLVERS = [None, "", "MP2", "CISD", "CCSD", 'TCCSD', "CCSD(T)", 'FCI', "FCI-spin0", "FCI-spin1"]

class DMET(QEmbeddingMethod):

    FRAGMENT_CLS = DMETFragment

    def __init__(self, mf, bno_threshold=1e-8, solver='CCSD', options=None, log=None, **kwargs):
        """Density matrix embedding theory (DMET) calculation object.

        Parameters
        ----------

        """

        super().__init__(mf, log=log)
        t_start = timer()
        if options is None:
            options = DMETOptions(**kwargs)
        else:
            options = options.replace(kwargs)
        # Options logic
        if options.pop_analysis:
            options.make_rdm1 = True
        self.opts = options
        self.log.info("DMET parameters:")
        for key, val in self.opts.items():
            self.log.info('  > %-24s %r', key + ':', val)

        # --- Check input
        if not mf.converged:
            if self.opts.strict:
                raise RuntimeError("Mean-field calculation not converged.")
            else:
                self.log.error("Mean-field calculation not converged.")
        self.bno_threshold = bno_threshold
        if solver not in VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)
        self.solver = solver

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
        t0 = timer()
        fragkw = {}
        if self.opts.fragment_type.upper() == 'IAO':
            if self.opts.iao_minao == 'auto':
                self.opts.iao_minao = helper.get_minimal_basis(self.mol.basis)
                self.log.warning("Minimal basis set '%s' for IAOs was selected automatically.",  self.opts.iao_minao)
            self.log.info("Computational basis= %s", self.mol.basis)
            self.log.info("Minimal basis=       %s", self.opts.iao_minao)
            fragkw['minao'] = self.opts.iao_minao
        self.init_fragmentation(self.opts.fragment_type, **fragkw)
        self.log.timing("Time for fragment initialization: %s", time_string(timer()-t0))

        self.log.timing("Time for EWF setup: %s", time_string(timer()-t_start))

        # Intermediate and output attributes
        #self.e_corr = 0.0           # Correlation energy
        #self.e_pert_t = 0.0         # CCSD(T) correction
        #self.e_delta_mp2 = 0.0      # MP2 correction

        # Population analysis
        self.pop_mf = None
        #self.pop_mf_chg = None

        self.iteration = 0
        self.cluster_results = {}
        self.results = []
        self.e_corr = 0.0

    def __repr__(self):
        keys = ['mf', 'bno_threshold', 'solver']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])

    @property
    def e_tot(self):
        """Total energy."""
        return self.e_mf + self.e_corr


    def pop_analysis(self, dm1, mo_coeff=None, filename=None, filemode='a', verbose=True):
        """
        Parameters
        ----------
        dm1 : (N, N) array
            If `mo_coeff` is None, AO representation is assumed!
        """
        if mo_coeff is not None:
            dm1 = einsum('ai,ij,bj->ab', mo_coeff, dm1, mo_coeff)
        c_loc = self.lo
        cs = np.dot(c_loc.T, self.get_ovlp())
        pop = einsum('ia,ab,ib->i', cs, dm1, cs)
        # Get atomic charges
        elecs = np.zeros(self.mol.natm)
        for i, label in enumerate(self.mol.ao_labels(fmt=None)):
            elecs[label[0]] += pop[i]
        chg = self.mol.atom_charges() - elecs

        if not verbose:
            return pop, chg

        if filename is None:
            write = lambda *args : self.log.info(*args)
            write("Population analysis")
            write("*******************")
        else:
            f = open(filename, filemode)
            write = lambda fmt, *args : f.write((fmt+'\n') % args)
            tstamp = datetime.now()
            self.log.info("[%s] Writing population analysis to file \"%s\"", tstamp, filename)
            write("[%s] Population analysis" % tstamp)
            write("*%s*********************" % (26*"*"))

        #shellslices = self.mol.aoslice_by_atom()[:,:2]
        aoslices = self.mol.aoslice_by_atom()[:,2:]
        aolabels = self.mol.ao_labels()

        for atom in range(self.mol.natm):
            write("> Charge of atom %d%-6s= % 11.8f (% 11.8f electrons)", atom, self.mol.atom_symbol(atom), chg[atom], elecs[atom])
            aos = aoslices[atom]
            for ao in range(aos[0], aos[1]):
                label = aolabels[ao]
                write("    %4d %-16s= % 11.8f" % (ao, label, pop[ao]))
            #for sh in range(self.mol.nbas):
            #    # Loop over AOs in shell

        if filename is not None:
            f.close()
        return pop, chg

    def kernel(self, bno_threshold=None):
        """Run DMET calculation.
        """
        t_start = timer()

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")

        maxiter = self.opts.maxiter
        # View this as a single number for now, if needed at all...
        bno_thr = bno_threshold

        #rdm = self.mf.make_rdm1()
        fock = self.get_fock()
        vcorr = np.zeros_like(fock)

        exit = False
        for iteration in range(1, maxiter + 1):
            self.iteration = iteration
            self.log.info("Now running iteration= %2d", iteration)
            self.log.info("****************************************************")

            mo_energy, mo_coeff = self.mf.eig(fock + vcorr, self.get_ovlp())
            mo_occ = self.mf.get_occ(mo_energy, mo_coeff)
            rdm = self.mf.make_rdm1(mo_coeff, mo_occ)
            fock = self.mf.get_fock(dm=rdm)

            for x, frag in enumerate(self.fragments):
                msg = "Now running %s" % (frag)
                self.log.info(msg)
                self.log.info(len(msg) * "*")
                self.log.changeIndentLevel(1)
                try:
                    result = frag.kernel(rdm, bno_threshold=bno_thr)
                except DMETFragmentExit:
                    exit = True
                    self.log.info("Exiting %s", frag)
                    self.log.changeIndentLevel(-1)
                    continue

                self.cluster_results[frag.id] = result
                if not result.converged:
                    self.log.error("%s is not converged!", frag)
                else:
                    self.log.info("%s is done.", frag)
                self.log.changeIndentLevel(-1)
                if exit:
                    break
            if exit:
                break
            # Now for the DMET self-consistency! This is where we start needing extra functionality compared to EWF.
            self.log.info("Now running DMET correlation potential fitting")

            impurity_projectors = [None] * len(self.fragments)
            hl_rdms = [None] * len(self.fragments)

            for x, frag in enumerate(self.fragments):
                c = frag.c_frag#self.cluster_results[frag.id].c_frag
                impurity_projectors[x] = [c]
                # Project AO rdm into fragment space.
                hl_rdms[x] = np.dot(c.T, np.dot(self.cluster_results[frag.id].dm1, c)) / 2
            vcorr = perform_SDP_fit(self.mol.nelec[0], fock, impurity_projectors, hl_rdms, self.get_ovlp(), self.log)
        else:
            if self.opts.sc_mode:
                self.log.error("Self-consistency not reached!")

    def print_results(self, results):
        self.log.info("Energies")
        self.log.info("********")
        fmt = "%-20s %+16.8f Ha"
        for i, frag in enumerate(self.loop()):
            e_corr = results["e_corr"][i]
            self.log.output(fmt, 'E(corr)[' + frag.trimmed_name() + ']=', e_corr)
        self.log.output(fmt, 'E(corr)=', self.e_corr)
        self.log.output(fmt, 'E(MF)=', self.e_mf)
        self.log.output(fmt, 'E(nuc)=', self.mol.energy_nuc())
        self.log.output(fmt, 'E(tot)=', self.e_tot)


    def get_energies(self):
        """Get total energy."""
        return [(self.e_mf + r.e_corr) for r in self.results]

    def print_clusters(self):
        """Print fragments of calculations."""
        self.log.info("%3s  %20s  %8s  %4s", "ID", "Name", "Solver", "Size")
        for frag in self.loop():
            self.log.info("%3d  %20s  %8s  %4d", frag.id, frag.name, frag.solver, frag.size)
