import os.path
import functools
from datetime import datetime
import dataclasses

import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.lo
import pyscf.scf
import pyscf.pbc
import pyscf.pbc.tools

from vayesta.core.util import *
from vayesta.core import QEmbeddingMethod

from . import helper
from .fragment import EWFFragment as Fragment

try:
    from mpi4py import MPI
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.Get_size()
    timer = MPI.Wtime
except ImportError:
    MPI = False
    MPI_rank = 0
    MPI_size = 1
    from timeit import default_timer as timer


@dataclasses.dataclass
class EWFResults:
    bno_threshold: float = None
    cluster_sizes: np.ndarray = None
    e_corr: float = None


VALID_SOLVERS = [None, "", "MP2", "CISD", "CCSD", 'TCCSD', "CCSD(T)", 'FCI', "FCI-spin0", "FCI-spin1"]

class EWF(QEmbeddingMethod):

    Fragment = Fragment

    @dataclasses.dataclass
    class Options(QEmbeddingMethod.Options):
        """Options for EWF calculations."""
        # --- Fragment settings
        #fragment_type: str = 'IAO'
        localize_fragment: bool = False     # Perform numerical localization on fragment orbitals
        iao_minao : str = 'auto'            # Minimal basis for IAOs
        # --- Bath settings
        bath_type: str = 'MP2-BNO'
        orbfile: str = None                 # Filename for orbital coefficients
        # If multiple bno thresholds are to be calculated, we can project integrals and amplitudes from a previous larger cluster:
        project_eris: bool = False          # Project ERIs from a pervious larger cluster (corresponding to larger eta), can result in a loss of accuracy especially for large basis sets!
        project_init_guess: bool = True     # Project converted T1,T2 amplitudes from a previous larger cluster
        orthogonal_mo_tol: float = False
        # --- Solver settings
        make_rdm1: bool = False
        make_rdm2: bool = False
        #solve_lambda: bool = 'auto'         # If False, use T-amplitudes inplace of Lambda-amplitudes
        t_as_lambda: bool = False           # If True, use T-amplitudes inplace of Lambda-amplitudes
        dm_with_frozen: bool = False        # Add frozen parts to cluster DMs
        eom_ccsd: list = dataclasses.field(default_factory=list)  # Perform EOM-CCSD in each cluster by default
        eom_ccsd_nroots: int = 5            # Perform EOM-CCSD in each cluster by default
        eomfile: str = 'eom-ccsd'           # Filename for EOM-CCSD states
        # Counterpoise correction of BSSE
        bsse_correction: bool = True
        bsse_rmax: float = 5.0              # In Angstrom
        # -- Self-consistency
        sc_maxiter: int = 30
        sc_energy_tol: float = 1e-6
        sc_mode: int = 0
        nelectron_target: int = None
        # --- Other
        #energy_partitioning: str = 'first-occ'
        strict: bool = False                # Stop if cluster not converged


    def __init__(self, mf, bno_threshold=1e-8, solver='CCSD', options=None, log=None, **kwargs):
        """Embedded wave function (EWF) calculation object.

        Parameters
        ----------
        mf : pyscf.scf object
            Converged mean-field object.
        solver : str, optional
            Solver for embedding problem. Default: 'CCSD'.
        **kwargs :
            See class `Options` for additional options.
        """

        t_start = timer()
        super().__init__(mf, options=options, log=log, **kwargs)

        # Options
        self.log.info("Parameters of %s:", self.__class__.__name__)
        self.log.info(break_into_lines(str(self.opts), newline='\n    '))

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

        #self._mo_coeff = self.get_init_mo_coeff()

        self.iteration = 0
        self.cluster_results = {}
        self.results = []
        #self.e_corr = 0.0
        self.log.timing("Time for EWF setup: %s", time_string(timer()-t_start))

    def __repr__(self):
        keys = ['mf', 'bno_threshold', 'solver']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])


    def get_fock_for_energy(self, *args, **kwargs):
        """Overriding this allows to use a different Fock-matrix for the energy evaluation."""
        return self.get_fock(*args, **kwargs)

    #def init_fragments(self):
    #    if self.opts.fragment_type.upper() == "IAO":
    #        #self.C_ao, self.C_env, self.iao_labels = self.make_iao_coeffs(minao=self.opts.iao_minao)
    #        minao=self.opts.iao_minao
    #        self.C_ao, self.C_env = self.make_iao_coeffs(minao=minao)
    #        self.iao_labels = self.get_iao_labels(minao=minao)
    #        # Only for printing:
    #        self.get_iao_occupancy(self.C_ao, minao=minao)
    #        self.ao_labels = self.iao_labels
    #    elif self.opts.fragment_type.upper() == "LOWDIN-AO":
    #        self.C_ao, self.lao_labels = self.make_lowdin_ao()
    #        self.ao_labels = self.lao_labels

    #    locmethod = self.opts.localize_fragment
    #    if locmethod:
    #        self.log.debug("Localize fragment orbitals with %s method", locmethod)

    #        #orbs = {self.ao_labels[i] : self.C_ao[:,i:i+1] for i in range(self.C_ao.shape[-1])}
    #        #orbs = {"A" : self.C_ao}
    #        #create_orbital_file(self.mol, "%s.molden" % self.local_orbital_type, orbs)
    #        coeffs = self.C_ao
    #        names = [("%d-%s-%s-%s" % l).rstrip("-") for l in self.ao_labels]
    #        #create_orbital_file(self.mol, self.local_orbital_type, coeffs, names, directory="fragment")
    #        create_orbital_file(self.mol, self.opts.fragment_type, coeffs, names, directory="fragment", filetype="cube")

    #        t0 = timer()
    #        if locmethod in ("BF", "ER", "PM"):
    #            localizer = getattr(pyscf.lo, locmethod)(self.mol)
    #            localizer.init_guess = None
    #            #localizer.pop_method = "lowdin"
    #            C_loc = localizer.kernel(self.C_ao, verbose=4)
    #        elif locmethod == "LAO":
    #            #centers = [l[0] for l in self.mol.ao_labels(None)]
    #            centers = [l[0] for l in self.ao_labels]
    #            self.log.debug("Atom centers: %r", centers)
    #            C_loc = localize_ao(self.mol, self.C_ao, centers)

    #        #C_loc = locfunc(self.mol).kernel(self.C_ao, verbose=4)
    #        self.log.timing("Time for orbital localization: %s", time_string(timer()-t0))
    #        assert C_loc.shape == self.C_ao.shape
    #        # Check that all orbitals kept their fundamental character
    #        chi = np.einsum("ai,ab,bi->i", self.C_ao, self.get_ovlp(), C_loc)
    #        self.log.info("Diagonal of AO-Loc(AO) overlap: %r", chi)
    #        self.log.info("Smallest value: %.3g" % np.amin(chi))
    #        #assert np.all(chi > 0.5)
    #        self.C_ao = C_loc

    #        #orbs = {"A" : self.C_ao}
    #        #orbs = {self.ao_labels[i] : self.C_ao[:,i:i+1] for i in range(self.C_ao.shape[-1])}
    #        #create_orbital_file(self.mol, "%s-local.molden" % self.local_orbital_type, orbs)
    #        #raise SystemExit()

    #        coeffs = self.C_ao
    #        names = [("%d-%s-%s-%s" % l).rstrip("-") for l in self.ao_labels]
    #        #create_orbital_file(self.mol, self.local_orbital_type, coeffs, names, directory="fragment-localized")
    #        create_orbital_file(self.mol, self.opts.fragment_type, coeffs, names, directory="fragment-localized", filetype="cube")

    def get_init_mo_coeff(self, mo_coeff=None):
        """Orthogonalize insufficiently orthogonal MOs.

        (For example as a result of k2gamma conversion with low cell.precision)
        """
        if mo_coeff is None: mo_coeff = self.mo_coeff
        c = mo_coeff.copy()
        ovlp = self.get_ovlp()
        assert np.all(c.imag == 0), "max|Im(C)|= %.2e" % abs(c.imag).max()
        err = abs(dot(c.T, ovlp, c) - np.eye(c.shape[-1])).max()
        if err > 1e-5:
            self.log.error("Orthogonality error of MOs= %.2e !!!", err)
        else:
            self.log.debug("Orthogonality error of MOs= %.2e", err)
        if self.opts.orthogonal_mo_tol and err > self.opts.orthogonal_mo_tol:
            t0 = timer()
            self.log.info("Orthogonalizing orbitals...")
            c_orth = helper.orthogonalize_mo(c, ovlp)
            change = abs(einsum('ai,ab,bi->i', c_orth, ovlp, c)-1)
            self.log.info("Max. orbital change= %.2e%s", change.max(), " (!!!)" if change.max() > 1e-4 else "")
            self.log.timing("Time for orbital orthogonalization: %s", time_string(timer()-t0))
            c = c_orth
        return c

    @property
    def e_tot(self):
        """Total energy."""
        return self.get_e_tot()

    @property
    def e_corr(self):
        """Correlation energy."""
        return self.get_e_corr()

    def get_e_tot(self):
        return self.e_mf + self.get_e_corr()

    def get_e_corr(self):
        e_corr = 0.0
        for f in self.fragments:
            e_corr += f.results.e_corr
        return e_corr

    #def get_e_corr_new(self):
    #    e_corr = 0.0
    #    t1 = self.get_t12(calc_t2=False)[0]
    #    for f in self.fragments:
    #        e_corr += (f.results.e1b + f.results.e2b_conn)
    #        ro, rv = f.get_rot_to_mf()

    #        #e_corr_2 += einsum('jb,pj,qb,pq->', t1, ro, rv, f.results.e2b_disc)
    #        cc = pyscf.cc.CCSD(self.mf)
    #        eris = cc.ao2mo()
    #        gov = eris.ovvo[:]
    #        gov = 2*gov - gov.transpose(3, 1, 2, 0)
    #        rf = dot(f.c_frag.T, self.get_ovlp(), self.mo_coeff_occ)
    #        e_corr += einsum('xa,JB,xI,aA,IABJ->', f.results.t1_pf, t1, rf, rv, gov)
    #    return e_corr

    def get_wf_cisd(self, intermediate_norm=False, c0=None):
        c0_target = c0

        c0 = 1.0
        c1 = np.zeros((self.nocc, self.nvir))
        c2 = np.zeros((self.nocc, self.nocc, self.nvir, self.nvir))
        ovlp = self.get_ovlp()
        # Add fragment WFs in intermediate normalization
        for f in self.fragments:
            c1f, c2f = f.results.c1/f.results.c0, f.results.c2/f.results.c0
            #c1f, c2f = f.results.c1, f.results.c2
            c1f = f.project_amplitude_to_fragment(c1f, c_occ=f.c_active_occ)
            c2f = f.project_amplitude_to_fragment(c2f, c_occ=f.c_active_occ)
            ro = np.linalg.multi_dot((f.c_active_occ.T, ovlp, self.mo_coeff_occ))
            rv = np.linalg.multi_dot((f.c_active_vir.T, ovlp, self.mo_coeff_vir))
            c1 += einsum('ia,iI,aA->IA', c1f, ro, rv)
            #c2f = (c2f + c2f.transpose(1,0,3,2))/2
            c2 += einsum('ijab,iI,jJ,aA,bB->IJAB', c2f, ro, ro, rv, rv)

        # Symmetrize
        c2 = (c2 + c2.transpose(1,0,3,2))/2

        # Restore standard normalization
        if not intermediate_norm:
            #c0 = self.fragments[0].results.c0
            norm = np.sqrt(c0**2 + 2*np.dot(c1.flatten(), c1.flatten()) + 2*np.dot(c2.flatten(), c2.flatten())
                    - einsum('jiab,ijab->', c2, c2))
            c0 = c0/norm
            c1 /= norm
            c2 /= norm
            # Check normalization
            norm = (c0**2 + 2*np.dot(c1.flatten(), c1.flatten()) + 2*np.dot(c2.flatten(), c2.flatten())
                    - einsum('jiab,ijab->', c2, c2))
            assert np.isclose(norm, 1.0)

            if c0_target is not None:
                norm12 = (2*np.dot(c1.flatten(), c1.flatten()) + 2*np.dot(c2.flatten(), c2.flatten())
                        - einsum('jiab,ijab->', c2, c2))
                if norm12 > 1e-10:
                    print('norm12= %.6e' % norm12)
                    fac12 = np.sqrt((1.0-c0_target**2)/norm12)
                    print('fac12= %.6e' % fac12)
                    c0 = c0_target
                    c1 *= fac12
                    c2 *= fac12

                    # Check normalization
                    norm = (c0**2 + 2*np.dot(c1.flatten(), c1.flatten()) + 2*np.dot(c2.flatten(), c2.flatten())
                            - einsum('jiab,ijab->', c2, c2))
                    assert np.isclose(norm, 1.0)

        return c0, c1, c2

    # -------------------------------------------------------------------------------------------- #

    # TODO: Reimplement
    #def make_local_nonorth_iao_orbitals(self, ao_indices, minao="minao"):
    #    C_occ = self.mo_coeff[:,self.mo_occ>0]
    #    C_ao = pyscf.lo.iao.iao(self.mol, C_occ, minao=minao)

    #    ao_labels = np.asarray(self.mol.ao_labels())[ao_indices]
    #    refmol = pyscf.lo.iao.reference_mol(self.mol, minao=minao)
    #    iao_labels = refmol.ao_labels()
    #    assert len(iao_labels) == C_ao.shape[-1]

    #    loc = np.isin(iao_labels, ao_labels)
    #    self.log.debug("Local NonOrth IAOs: %r", (np.asarray(iao_labels)[loc]).tolist())
    #    nlocal = np.count_nonzero(loc)
    #    self.log.debug("Number of local IAOs=%3d", nlocal)

    #    C_local = C_ao[:,loc]
    #    # Orthogonalize locally
    #    #S = self.mf.get_ovlp()
    #    S = self.get_ovlp()
    #    C_local = pyscf.lo.vec_lowdin(C_local, S)

    #    # Add remaining space
    #    # Transform to MO basis
    #    C_local_mo = np.linalg.multi_dot((self.mo_coeff.T, S, C_local))
    #    # Get eigenvectors of projector into complement
    #    P_local = np.dot(C_local_mo, C_local_mo.T)
    #    norb = self.mo_coeff.shape[-1]
    #    P_env = np.eye(norb) - P_local
    #    e, C = np.linalg.eigh(P_env)
    #    assert np.all(np.logical_or(abs(e) < 1e-10, abs(e)-1 < 1e-10))
    #    mask_env = (e > 1e-10)
    #    assert (np.sum(mask_env) + nlocal == norb)
    #    # Transform back to AO basis
    #    C_env = np.dot(self.mo_coeff, C[:,mask_env])

    #    # Test orthogonality
    #    C = np.hstack((C_local, C_env))
    #    assert np.allclose(C.T.dot(S).dot(C) - np.eye(norb), 0)

    #    return C_local, C_env

    # -------------------------------------------------------------------------------------------- #


    # TODO: Reimplement PMO
    #def make_atom_fragment(self, atoms, name=None, check_atoms=True, **kwargs):
    #    """
    #    Parameters
    #    ---------
    #    atoms : list of int/str or int/str
    #        Atom labels of atoms in fragment.
    #    name : str
    #        Name of fragment.
    #    """
    #    # Atoms may be a single atom index/label
    #    #if not isinstance(atoms, (tuple, list, np.ndarray)):
    #    if np.isscalar(atoms):
    #        atoms = [atoms]

    #    # Check if atoms are valid labels of molecule
    #    atom_labels_mol = [self.mol.atom_symbol(atomid) for atomid in range(self.mol.natm)]
    #    if isinstance(atoms[0], str) and check_atoms:
    #        for atom in atoms:
    #            if atom not in atom_labels_mol:
    #                raise ValueError("Atom with label %s not in molecule." % atom)

    #    # Get atom indices/labels
    #    if isinstance(atoms[0], (int, np.integer)):
    #        atom_indices = atoms
    #        atom_labels = [self.mol.atom_symbol(i) for i in atoms]
    #    else:
    #        atom_indices = np.nonzero(np.isin(atom_labels_mol, atoms))[0]
    #        atom_labels = atoms
    #    assert len(atom_indices) == len(atom_labels)

    #    # Generate cluster name if not given
    #    if name is None:
    #        name = ",".join(atom_labels)

    #    # Indices refers to AOs or IAOs, respectively

    #    # Non-orthogonal AOs
    #    if self.opts.fragment_type == "AO":
    #        # Base atom for each AO
    #        ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
    #        indices = np.nonzero(np.isin(ao_atoms, atoms))[0]
    #        C_local, C_env = self.make_local_ao_orbitals(indices)

    #    # Lowdin orthonalized AOs
    #    elif self.opts.fragment_type == "LAO":
    #        lao_atoms = [lao[1] for lao in self.lao_labels]
    #        indices = np.nonzero(np.isin(lao_atoms, atom_labels))[0]
    #        C_local, C_env = self.make_local_lao_orbitals(indices)

    #    # Orthogonal intrinsic AOs
    #    elif self.opts.fragment_type == "IAO":
    #        iao_atoms = [iao[0] for iao in self.iao_labels]
    #        iao_indices = np.nonzero(np.isin(iao_atoms, atom_indices))[0]
    #        C_local, C_env = self.make_local_iao_orbitals(iao_indices)

    #    # Non-orthogonal intrinsic AOs
    #    elif self.opts.fragment_type == "NonOrth-IAO":
    #        ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
    #        indices = np.nonzero(np.isin(ao_atoms, atom_labels))[0]
    #        C_local, C_env = self.make_local_nonorth_iao_orbitals(indices, minao=self.opts.iao_minao)

    #    # Projected molecular orbitals
    #    # (AVAS paper)
    #    elif self.opts.fragment_type == "PMO":
    #        #ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
    #        #indices = np.nonzero(np.isin(ao_atoms, atoms))[0]

    #        # Use atom labels as AO labels
    #        self.log.debug("Making occupied projector.")
    #        Po = self.get_ao_projector(atom_labels, basis=kwargs.pop("basis_proj_occ", None))
    #        self.log.debug("Making virtual projector.")
    #        Pv = self.get_ao_projector(atom_labels, basis=kwargs.pop("basis_proj_vir", None))
    #        self.log.debug("Done.")

    #        o = (self.mo_occ > 0)
    #        v = (self.mo_occ == 0)
    #        C = self.mo_coeff
    #        So = np.linalg.multi_dot((C[:,o].T, Po, C[:,o]))
    #        Sv = np.linalg.multi_dot((C[:,v].T, Pv, C[:,v]))
    #        eo, Vo = np.linalg.eigh(So)
    #        ev, Vv = np.linalg.eigh(Sv)
    #        rev = np.s_[::-1]
    #        eo, Vo = eo[rev], Vo[:,rev]
    #        ev, Vv = ev[rev], Vv[:,rev]
    #        self.log.debug("Non-zero occupied eigenvalues:\n%r", eo[eo>1e-10])
    #        self.log.debug("Non-zero virtual eigenvalues:\n%r", ev[ev>1e-10])
    #        #tol = 1e-8
    #        tol = 0.1
    #        lo = eo > tol
    #        lv = ev > tol
    #        Co = np.dot(C[:,o], Vo)
    #        Cv = np.dot(C[:,v], Vv)
    #        C_local = np.hstack((Co[:,lo], Cv[:,lv]))
    #        C_env = np.hstack((Co[:,~lo], Cv[:,~lv]))
    #        self.log.debug("Number of local orbitals: %d", C_local.shape[-1])
    #        self.log.debug("Number of environment orbitals: %d", C_env.shape[-1])

    #    frag = self.make_fragment(name, C_local, C_env, atoms=atom_indices, **kwargs)

    #    # TEMP
    #    #ao_indices = get_ao_indices_at_atoms(self.mol, atomids)
    #    ao_indices = helper.atom_labels_to_ao_indices(self.mol, atom_labels)
    #    frag.ao_indices = ao_indices

    #    return frag

    def tailor_all_fragments(self):
        for frag in self.fragments:
            for frag2 in frag.loop_fragments(exclude_self=True):
                frag.add_tailor_fragment(frag2)

    def kernel(self, bno_threshold=None):
        """Run EWF.

        Parameters
        ----------
        bno_threshold : float or list, optional
            Bath natural orbital threshold. Default: 1e-8.
        """

        if MPI: MPI_comm.Barrier()
        t_start = timer()

        if bno_threshold is None: bno_threshold = self.bno_threshold
        if np.ndim(bno_threshold) == 0: bno_threshold = [bno_threshold]
        bno_threshold = np.sort(np.asarray(bno_threshold))

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")

        # TODO: clean this up
        if self.opts.orbfile:
            filename = "%s.txt" % self.opts.orbfile
            tstamp = datetime.now()
            nfo = self.iao_coeff.shape[-1]
            #ao_labels = ["-".join(x) for x in self.mol.ao_labels(None)]
            ao_labels = ["-".join([str(xi) for xi in x]) for x in self.mol.ao_labels(None)]
            iao_labels = ["-".join([str(xi) for xi in x]) for x in self.iao_labels]
            #iao_labels = ["-".join(x) for x in self.iao_labels]
            self.log.info("[%s] Writing fragment orbitals to file \"%s\"", tstamp, filename)
            with open(filename, "a") as f:
                f.write("[%s] Fragment Orbitals\n" % tstamp)
                f.write("*%s-------------------\n" % (26*"-"))
                # Header
                fmtline = "%20s" + nfo*"   %20s" + "\n"
                f.write(fmtline % ("AO", *iao_labels))
                fmtline = "%20s" + nfo*"   %+20.8e" + "\n"
                # Loop over AO
                for i in range(self.iao_coeff.shape[0]):
                    f.write(fmtline % (ao_labels[i], *self.iao_coeff[i]))

        if self.is_rhf:
            nelec_frags = sum([f.sym_factor*f.nelectron for f in self.loop()])
            self.log.info("Total number of mean-field electrons over all fragments= %.8f", nelec_frags)
            if abs(nelec_frags - np.rint(nelec_frags)) > 1e-4:
                self.log.warning("Number of electrons not integer!")
        else:
            nelec_frags = (sum([f.sym_factor*f.nelectron[0] for f in self.loop()]),
                           sum([f.sym_factor*f.nelectron[1] for f in self.loop()]))
            self.log.info("Total number of mean-field electrons over all fragments= %.8f , %.8f", *nelec_frags)
            if abs(nelec_frags[0] - np.rint(nelec_frags[0])) > 1e-4 or abs(nelec_frags[1] - np.rint(nelec_frags[1])) > 1e-4:
                self.log.warning("Number of electrons not integer!")

        exit = False
        for i, bno_thr in enumerate(bno_threshold):
            e_corr_1 = 0.0
            e_corr_last = 0.0
            # Self consistency loop
            maxiter = (self.opts.sc_maxiter if self.opts.sc_mode else 1)
            for iteration in range(1, maxiter+1):
                self.iteration = iteration
                if self.opts.sc_mode:
                    self.log.info("Now running BNO threshold= %.2e - Iteration= %2d", bno_thr, iteration)
                    self.log.info("====================================================")
                else:
                    self.log.info("Now running BNO threshold= %.2e", bno_thr)
                    self.log.info("===================================")

                for x, frag in enumerate(self.fragments):
                    if MPI_rank != (x % MPI_size):
                        continue
                    # Do not calculate symmetry children
                    if frag.sym_parent is not None:
                        self.log.debugv("Skipping %s with symmetry parent %s", frag.name, frag.sym_parent.name)
                        self.cluster_results[(frag.id, bno_thr)] = frag.results
                        continue

                    mpi_info = (" on MPI process %d" % MPI_rank) if MPI_size > 1 else ""
                    msg = "Now running %s%s" % (frag, mpi_info)
                    self.log.info(msg)
                    self.log.info(len(msg)*"-")
                    self.log.changeIndentLevel(1)
                    try:
                        result = frag.kernel(bno_threshold=bno_thr)
                    except Fragment.Exit:
                        exit = True
                        self.log.info("Exiting %s", frag)
                        self.log.changeIndentLevel(-1)
                        continue

                    self.cluster_results[(frag.id, bno_thr)] = result
                    if not result.converged:
                        self.log.error("%s is not converged!", frag)
                    else:
                        self.log.info("%s is done.", frag)
                    self.log.changeIndentLevel(-1)
                if exit:
                    break

                e_corr = sum([self.cluster_results[(f.id, bno_thr)].e_corr for f in self.fragments])
                if iteration == 1:
                    e_corr_1 = e_corr
                de = (e_corr - e_corr_last)
                e_corr_last = e_corr
                if self.opts.sc_mode:
                    self.log.info("Iteration %d: E(corr)= % 12.8f Ha (dE= % 12.8f Ha)", iteration, e_corr, de)
                else:
                    self.log.info("E(corr)= % 12.8f Ha", e_corr)
                if (self.opts.sc_mode and (abs(de) < self.opts.sc_energy_tol)):
                    self.log.info("Self-consistency reached in %d iterations", iteration)
                    break
                e_corr0 = e_corr
            else:
                if self.opts.sc_mode:
                    self.log.error("Self-consistency not reached!")
            if exit:
                break

            if self.opts.sc_mode:
                self.log.info("E(corr)[SC]= % 12.8f Ha  E(corr)[1]= % 12.8f Ha  (diff= % 12.8f Ha)", e_corr, e_corr_1, (e_corr-e_corr_1))

            result = EWFResults(bno_threshold=bno_thr, e_corr=e_corr)
            self.results.append(result)
        if exit:
            return

        self.log.info("Fragment Correlation Energies")
        self.log.info("-----------------------------")
        self.log.info("%13s:" + self.nfrag*" %16s", "BNO threshold", *[f.id_name for f in self.fragments])
        # TODO
        fmt = "%13.2e:" + self.nfrag*" %13.8f Ha"
        #fmt0 = self.nfrag*" %13.8f Ha"
        for bno_thr in bno_threshold[::-1]:
            #for n in range(0, self.nfrag, 5):
            #    energies = [self.cluster_results[(f.id, bno_thr)].e_corr for f in self.fragments[n:n+5])
            #    if n == 0:
            #        fmt = ("%13.2e:" % bno_thr) + fmt0
            #    else:
            #        fmt = 13*" " + fmt0
            #    self.log.info(fmt, *energies)
            self.log.info(fmt, bno_thr, *[self.cluster_results[(f.id, bno_thr)].e_corr for f in self.fragments])

        bno_min = np.min(bno_threshold)
        #self.e_corr = sum([results[(f.id, bno_min)].e_corr for f in self.fragments])
        #self.e_corr = self.results[0].e_corr
        self.log.output('E(nuc)=  %s', energy_string(self.mol.energy_nuc()))
        self.log.output('E(MF)=   %s', energy_string(self.e_mf))
        self.log.output('E(corr)= %s', energy_string(self.e_corr))
        self.log.output('E(tot)=  %s', energy_string(self.e_tot))

        #attributes = ["converged", "e_corr", "e_delta_mp2", "e_pert_t"]

        #results = self.collect_results(*attributes)
        #if MPI_rank == 0 and not np.all(results["converged"]):
        #    self.log.critical("The following fragments did not converge:")
        #    for i, frag in enumerate(self.loop()):
        #        if not results["converged"][i]:
        #            self.log.critical("%3d %s solver= %s", frag.id, frag.name, frag.solver)
        #    if self.opts.strict:
        #        raise RuntimeError("Not all fragments converged")

        #self.e_corr = sum(results["e_corr"])
        ##self.e_pert_t = sum(results["e_pert_t"])
        ##self.e_pert_t2 = sum(results["e_pert_t2"])
        #self.e_delta_mp2 = sum(results["e_delta_mp2"])

        #self.e_corr_full = sum(results["e_corr_full"])

        #if MPI_rank == 0:
        #    self.print_results(results)

        #if MPI: MPI_comm.Barrier()
        self.log.info("Total wall time:  %s", time_string(timer()-t_start))
        self.log.info("All done.")


    def collect_results(self, *attributes):
        """Use MPI to collect results from all fragments."""

        #self.log.debug("Collecting attributes %r from all clusters", (attributes,))
        fragments = self.fragments

        if MPI:
            def reduce_fragment(attr, op=MPI.SUM, root=0):
                res = MPI_comm.reduce(np.asarray([getattr(f, attr) for f in fragments]), op=op, root=root)
                return res
        else:
            def reduce_fragment(attr):
                res = np.asarray([getattr(f, attr) for f in fragments])
                return res

        results = {}
        for attr in attributes:
            results[attr] = reduce_fragment(attr)

        return results

    #def show_cluster_sizes(self, results, show_largest=True):
    #    self.log.info("Cluster Sizes")
    #    self.log.info("*************")
    #    fmtstr = "  * %3d %-10s  :  active=%4d  frozen=%4d  ( %5.1f %%)"
    #    imax = [0]
    #    for i, frag in enumerate(self.loop()):
    #        nactive = results["nactive"][i]
    #        nfrozen = results["nfrozen"][i]
    #        self.log.info(fmtstr, frag.id, frag.trimmed_name(10), nactive, nfrozen, 100.0*nactive/self.nmo)
    #        if i == 0:
    #            continue
    #        if nactive > results["nactive"][imax[0]]:
    #            imax = [i]
    #        elif nactive == results["nactive"][imax[0]]:
    #            imax.append(i)

    #    if show_largest and self.nfrag > 1:
    #        self.log.info("Largest Cluster")
    #        self.log.info("***************")
    #        for i in imax:
    #            x = self.fragments[i]
    #            nactive = results["nactive"][i]
    #            nfrozen = results["nfrozen"][i]
    #            self.log.info(fmtstr, x.id, x.trimmed_name(10), nactive, nfrozen, 100.0*nactive/self.nmo)


    #def print_results(self, results):
    #    self.show_cluster_sizes(results)

    #    self.log.info("Fragment Energies")
    #    self.log.info("*****************")
    #    self.log.info("CCSD / CCSD+dMP2 / CCSD+dMP2+(T)")
    #    fmtstr = "  * %3d %-10s  :  %+16.8f Ha  %+16.8f Ha  %+16.8f Ha"
    #    for i, frag in enumerate(self.loop()):
    #        e_corr = results["e_corr"][i]
    #        e_pert_t = results["e_pert_t"][i]
    #        e_delta_mp2 = results["e_delta_mp2"][i]
    #        self.log.info(fmtstr, frag.id, frag.trimmed_name(10), e_corr, e_corr+e_delta_mp2, e_corr+e_delta_mp2+e_pert_t)

    #    self.log.info("  * %-14s  :  %+16.8f Ha  %+16.8f Ha  %+16.8f Ha", "total", self.e_corr, self.e_corr+self.e_delta_mp2, self.e_corr+self.e_delta_mp2+self.e_pert_t)
    #    self.log.info("E(corr)= %+16.8f Ha", self.e_corr)
    #    self.log.info("E(tot)=  %+16.8f Ha", self.e_tot)

    #def print_results(self, results):
    #    self.log.info("Energies")
    #    self.log.info("========")
    #    fmt = "%-20s %+16.8f Ha"
    #    for i, frag in enumerate(self.loop()):
    #        e_corr = results["e_corr"][i]
    #        self.log.output(fmt, 'E(corr)[' + frag.trimmed_name() + ']=', e_corr)
    #    self.log.output(fmt, 'E(corr)=', self.e_corr)
    #    self.log.output(fmt, 'E(MF)=', self.e_mf)
    #    self.log.output(fmt, 'E(nuc)=', self.mol.energy_nuc())
    #    self.log.output(fmt, 'E(tot)=', self.e_tot)

    def print_results(self, results):
        self.log.info("Energies")
        self.log.info("========")
        fmt = "%-20s %s"
        for i, frag in enumerate(self.loop()):
            e_corr = results["e_corr"][i]
            self.log.output('E(corr)[' + frag.trimmed_name() + ']= %s', energy_string(e_corr))
        self.log.output('E(corr)= %s', energy_string(self.e_corr))
        self.log.output('E(MF)=   %s', energy_string(self.e_mf))
        self.log.output('E(nuc)=  %s', energy_string(self.mol.energy_nuc()))
        self.log.output('E(tot)=  %s', energy_string(self.e_tot))

    def get_energies(self):
        """Get total energy."""
        return [(self.e_mf + r.e_corr) for r in self.results]

    #def get_cluster_sizes(self)
    #    sizes = np.zeros((self.nfrag, self.ncalc), dtype=np.int)
    #    for i, frag in enumerate(self.loop()):
    #        sizes[i] = frag.n_active
    #    return sizes

    def print_clusters(self):
        """Print fragments of calculations."""
        self.log.info("%3s  %20s  %8s  %4s", "ID", "Name", "Solver", "Size")
        for frag in self.loop():
            self.log.info("%3d  %20s  %8s  %4d", frag.id, frag.name, frag.solver, frag.size)

REWF = EWF
