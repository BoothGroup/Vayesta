# Standard libaries
import os
import os.path
from collections import OrderedDict
import functools
from datetime import datetime
from timeit import default_timer as timer
import dataclasses
import copy

# External libaries
import numpy as np
import scipy
import scipy.linalg

# Internal libaries
import pyscf
import pyscf.pbc
from pyscf.pbc.tools import cubegen

# Local modules
from vayesta.core.util import *
from vayesta.core import QEmbeddingFragment

from . import ewf
from .solver import get_solver_class
from .dmet_bath import make_dmet_bath, project_ref_orbitals
from .mp2_bath import make_mp2_bno
from . import helper
from . import psubspace


@dataclasses.dataclass
class EWFFragmentOptions(Options):
    """Attributes set to `NotSet` inherit their value from the parent EWF object."""
    dmet_threshold : float = NotSet
    make_rdm1 : bool = NotSet
    eom_ccsd : bool = NotSet
    plot_orbitals : bool = NotSet
    solver_options : dict = NotSet
    bsse_correction : bool = NotSet
    bsse_rmax : float = NotSet
    energy_partitioning : str = NotSet
    bno_threshold_factor : float = 1.0


class EWFFragment(QEmbeddingFragment):

    def __init__(self, base, fid, name, c_frag, c_env, fragment_type, sym_factor=1, atoms=None, log=None,
            solver=None, bno_threshold=None, options=None, **kwargs):
        """
        Parameters
        ----------
        base : EWF
            Base EWF object.
        fid : int
            Unique ID of fragment.
        name :
            Name of fragment.
        """

        super().__init__(base, fid, name, c_frag, c_env, fragment_type, sym_factor=sym_factor, atoms=atoms, log=log)

        if options is None:
            options = EWFFragmentOptions(**kwargs)
        else:
            options = options.replace(kwargs)
        options = options.replace(self.base.opts, select=NotSet)
        self.opts = options
        for key, val in self.opts.items():
            self.log.infov('  * %-24s %r', key + ':', val)

        if solver is None:
            solver = self.base.solver
        if solver not in ewf.VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)
        self.solver = solver
        self.log.infov('  * %-24s %r', 'Solver:', self.solver)

       # Bath natural orbital (BNO) threshold
        if bno_threshold is None:
            bno_threshold = self.base.bno_threshold
        if np.ndim(bno_threshold) == 0:
            bno_threshold = [bno_threshold]
        assert len(bno_threshold) == len(self.base.bno_threshold)
        self.bno_threshold = self.opts.bno_threshold_factor*np.asarray(bno_threshold)
        # Sort such that most expensive calculation (smallest threshold) comes first
        # (allows projecting down ERIs and initial guess for subsequent calculations)
        self.bno_threshold.sort()

        # Intermediate and output attributes:
        self.nactive = 0
        self.nfrozen = 0
        # Intermediate values
        self.c_no_occ = self.c_no_vir = None
        self.n_no_occ = self.n_no_vir = None
        # Save correlation energies for different BNO thresholds
        self.e_corrs = len(self.bno_threshold)*[None]
        self.n_active = len(self.bno_threshold)*[None]
        self.iteration = 0
        # Output values
        self.converged = False
        #self.e_corr = 0.0
        self.e_delta_mp2 = 0.0
        self.e_pert_t = 0.0
        self.e_corr_dmp2 = 0.0
        # For EMO-CCSD
        self.eom_ip_energy = None
        self.eom_ea_energy = None

        # BSSE energies
        self.e_bsse = len(self.bno_threshold)*[None]


    def __repr__(self):
        keys = ['id', 'name', 'fragment_type', 'sym_factor', 'atoms', 'aos']    # QEmbeddingFragment
        keys += ['solver', 'bno_threshold']                                     # EWFFragment
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])


    @property
    def e_corr(self):
        """Best guess for correlation energy, using the lowest BNO threshold."""
        idx = np.argmin(self.bno_threshold)
        return self.e_corrs[idx]


    def kernel(self, solver=None, bno_threshold=None):
        """Construct bath orbitals and run solver.

        Parameters
        ----------
        solver : str
            Method ["MP2", "CISD", "CCSD", "CCSD(T)", "FCI"]
        bno_threshold : list
            List of bath natural orbital (BNO) thresholds.
        """

        solver = solver or self.solver
        if bno_threshold is None:
            bno_threshold = self.bno_threshold

        t0_bath = t0 = timer()
        self.log.info("Making DMET Bath")
        self.log.info("****************")
        self.log.changeIndentLevel(1)
        c_dmet, c_env_occ, c_env_vir = self.make_dmet_bath(tol=self.opts.dmet_threshold)
        self.log.timing("Time for DMET bath:  %s", time_string(timer()-t0))
        self.log.changeIndentLevel(-1)

        # Add fragment and DMET orbitals for cube file plots
        if self.opts.plot_orbitals:
            os.makedirs(self.base.opts.plot_orbitals_dir, exist_ok=True)
            name = "%s.cube" % os.path.join(self.base.opts.plot_orbitals_dir, self.name)
            self.cubefile = cubegen.CubeFile(self.mol, filename=name, **self.base.opts.plot_orbitals_kwargs)
            self.cubefile.add_orbital(self.c_frag.copy())
            self.cubefile.add_orbital(c_dmet.copy(), dset_idx=1001)

        # Add additional orbitals to cluster [optional]
        #c_dmet, c_env_occ, c_env_vir = self.additional_bath_for_cluster(c_dmet, c_env_occ, c_env_vir)

        # Diagonalize cluster DM to separate cluster occupied and virtual
        self.c_cluster_occ, self.c_cluster_vir = self.diagonalize_cluster_dm(c_dmet, tol=2*self.opts.dmet_threshold)
        self.log.info("Cluster orbitals:  n(occ)= %3d  n(vir)= %3d", self.c_cluster_occ.shape[-1], self.c_cluster_vir.shape[-1])

        # Add cluster orbitals to plot
        #if self.opts.plot_orbitals:
        #    self.cubefile.add_orbital(C_occclst.copy(), dset_idx=2001)
        #    self.cubefile.add_orbital(C_virclst.copy(), dset_idx=3001)

        # Primary MP2 bath orbitals
        # TODO NOT MAINTAINED
        #if True:
        #    if self.opts.prim_mp2_bath_tol_occ:
        #        self.log.info("Adding primary occupied MP2 bath orbitals")
        #        C_add_o, C_rest_o, *_ = self.make_mp2_bath(C_occclst, C_virclst, "occ",
        #                c_occenv=C_occenv, c_virenv=C_virenv, tol=self.opts.prim_mp2_bath_tol_occ,
        #                mp2_correction=False)
        #    if self.opts.prim_mp2_bath_tol_vir:
        #        self.log.info("Adding primary virtual MP2 bath orbitals")
        #        C_add_v, C_rest_v, *_ = self.make_mp2_bath(C_occclst, C_virclst, "vir",
        #                c_occenv=C_occenv, c_virenv=C_virenv, tol=self.opts.prim_mp2_bath_tol_occ,
        #                mp2_correction=False)
        #    # Combine
        #    if self.opts.prim_mp2_bath_tol_occ:
        #        C_bath = np.hstack((C_add_o, C_bath))
        #        C_occenv = C_rest_o
        #    if self.opts.prim_mp2_bath_tol_vir:
        #        C_bath = np.hstack((C_bath, C_add_v))
        #        C_virenv = C_rest_v

        #    # Re-diagonalize cluster DM to separate cluster occupied and virtual
        #    C_occclst, C_virclst = self.diagonalize_cluster_dm(C_bath)
        #self.C_bath = C_bath

        if c_env_occ.shape[-1] > 0:
            self.log.info("Making Occupied BNOs")
            self.log.info("********************")
            t0 = timer()
            self.log.changeIndentLevel(1)
            self.c_no_occ, self.n_no_occ = make_mp2_bno(
                    self, "occ", self.c_cluster_occ, self.c_cluster_vir, c_env_occ, c_env_vir)
            self.log.timing("Time for occupied BNOs:  %s", time_string(timer()-t0))
            if len(self.n_no_occ) > 0:
                self.log.info("Occupied BNO histogram:")
                for line in helper.plot_histogram(self.n_no_occ):
                    self.log.info(line)
            self.log.changeIndentLevel(-1)
        else:
            self.c_no_occ = c_env_occ
            self.n_no_occ = np.zeros((0,))

        if c_env_vir.shape[-1] > 0:
            self.log.info("Making Virtual BNOs")
            self.log.info("*******************")
            t0 = timer()
            self.log.changeIndentLevel(1)
            self.c_no_vir, self.n_no_vir = make_mp2_bno(
                    self, "vir", self.c_cluster_occ, self.c_cluster_vir, c_env_occ, c_env_vir)
            self.log.timing("Time for virtual BNOs:   %s", time_string(timer()-t0))
            if len(self.n_no_vir) > 0:
                self.log.info("Virtual BNO histogram:")
                for line in helper.plot_histogram(self.n_no_vir):
                    self.log.info(line)
            self.log.changeIndentLevel(-1)
        else:
            self.c_no_vir = c_env_vir
            self.n_no_vir = np.zeros((0,))


        # Plot orbitals
        if self.opts.plot_orbitals:
            # Save state of cubefile, in case a replot of the same data is required later:
            self.cubefile.save_state("%s.pkl" % self.cubefile.filename)
            self.cubefile.write()
        self.log.timing("Time for bath:  %s", time_string(timer()-t0_bath))

        init_guess = eris = None
        for icalc, bno_thr in enumerate(bno_threshold):
            self.log.info("Run %2d - BNO threshold= %.1e", icalc, bno_thr)
            self.log.info("*******************************")
            self.log.changeIndentLevel(1)

            if True:
                e_corr, n_active, init_guess, eris = self.run_bno_threshold(solver, bno_thr, init_guess=init_guess, eris=eris)
                self.log.info("BNO threshold= %.1e :  E(corr)= %+14.8f Ha", bno_thr, e_corr)
            else:
                try:
                    e_corr, n_active, init_guess, eris = self.run_bno_threshold(solver, bno_thr, init_guess=init_guess, eris=eris)
                    self.log.info("BNO threshold= %.1e :  E(corr)= %+14.8f Ha", bno_thr, e_corr)
                except Exception as e:
                    self.log.error("Exception for BNO threshold= %.1e:\n%r", bno_thr, e)
                    e_corr = n_active = 0

            self.e_corrs[icalc] = e_corr
            self.n_active[icalc] = n_active
            self.log.changeIndentLevel(-1)

        self.log.info("Fragment Correlation Energies")
        self.log.info("*****************************")
        for i in range(len(bno_threshold)):
            icalc = -(i+1)
            bno_thr = bno_threshold[icalc]
            n_active = self.n_active[icalc]
            e_corr = self.e_corrs[icalc]
            if n_active > 0:
                self.log.info("  * BNO threshold= %.1e :  n(active)= %4d  E(corr)= %+14.8f Ha", bno_thr, n_active, e_corr)
            else:
                self.log.info("  * BNO threshold= %.1e :  <Exception during calculation>", bno_thr)



    def run_bno_threshold(self, solver, bno_thr, init_guess=None, eris=None):
        #self.e_delta_mp2 = e_delta_occ + e_delta_vir
        #self.log.debug("MP2 correction = %.8g", self.e_delta_mp2)

        assert (self.c_no_occ is not None)
        assert (self.c_no_vir is not None)

        self.log.info("Occupied BNOs:")
        c_nbo_occ, c_frozen_occ = self.apply_bno_threshold(self.c_no_occ, self.n_no_occ, bno_thr)
        self.log.info("Virtual BNOs:")
        c_nbo_vir, c_frozen_vir = self.apply_bno_threshold(self.c_no_vir, self.n_no_vir, bno_thr)

        # Canonicalize orbitals
        c_active_occ = self.canonicalize_mo(self.c_cluster_occ, c_nbo_occ)[0]
        c_active_vir = self.canonicalize_mo(self.c_cluster_vir, c_nbo_vir)[0]

        # Combine, important to keep occupied orbitals first!
        # Put frozen (occenv, virenv) orbitals to the front and back
        # and active orbitals (occact, viract) in the middle
        c_occ = np.hstack((c_frozen_occ, c_active_occ))
        c_vir = np.hstack((c_active_vir, c_frozen_vir))
        nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
        mo_coeff = np.hstack((c_occ, c_vir))

        # Check occupations
        n_occ = self.get_mo_occupation(c_occ)
        if not np.allclose(n_occ, 2, atol=2*self.opts.dmet_threshold):
            raise RuntimeError("Incorrect occupation of occupied orbitals:\n%r" % n_occ)
        n_vir = self.get_mo_occupation(c_vir)
        if not np.allclose(n_vir, 0, atol=2*self.opts.dmet_threshold):
            raise RuntimeError("Incorrect occupation of virtual orbitals:\n%r" % n_vir)
        mo_occ = np.asarray(nocc*[2] + nvir*[0])

        nocc_frozen = c_frozen_occ.shape[-1]
        nvir_frozen = c_frozen_vir.shape[-1]
        nfrozen = nocc_frozen + nvir_frozen
        nactive = c_active_occ.shape[-1] + c_active_vir.shape[-1]

        self.log.info("Orbitals for Fragment %3d", self.id)
        self.log.info("************************")
        self.log.info("  * Occupied: active= %4d  frozen= %4d  total= %4d", c_active_occ.shape[-1], nocc_frozen, c_occ.shape[-1])
        self.log.info("  * Virtual:  active= %4d  frozen= %4d  total= %4d", c_active_vir.shape[-1], nvir_frozen, c_vir.shape[-1])
        self.log.info("  * Total:    active= %4d  frozen= %4d  total= %4d", nactive, nfrozen, mo_coeff.shape[-1])

        # --- Do nothing if solver is not set
        if not solver:
            self.log.info("Solver set to None. Skipping calculation.")
            self.converged = True
            return 0, nactive, None, None

        #self.log.info("RUNNING %s SOLVER", solver)
        #self.log.info((len(solver)+15)*"*")
        #self.log.changeIndentLevel(1)

        # --- Project initial guess and integrals from previous cluster calculation with smaller eta:
        # Use initial guess from previous calculations
        if self.base.opts.project_init_guess and init_guess is not None:
            # Projectors for occupied and virtual orbitals
            p_occ = np.linalg.multi_dot((init_guess.pop("c_occ").T, self.base.get_ovlp(), c_active_occ))
            p_vir = np.linalg.multi_dot((init_guess.pop("c_vir").T, self.base.get_ovlp(), c_active_vir))
            t1, t2 = init_guess.pop("t1"), init_guess.pop("t2")
            t1, t2 = helper.transform_amplitudes(t1, t2, p_occ, p_vir)
            init_guess["t1"] = t1
            init_guess["t2"] = t2
        else:
            init_guess = None
        # If superspace ERIs were calculated before, they can be transformed and used again
        if self.base.opts.project_eris and eris is not None:
            t0 = timer()
            self.log.debug("Projecting previous ERIs onto subspace")
            eris = psubspace.project_eris(eris, c_active_occ, c_active_vir, ovlp=self.base.get_ovlp())
            self.log.timing("Time to project ERIs:  %s", time_string(timer()-t0))
        else:
            eris = None

        # Create solver object
        t0 = timer()
        csolver = get_solver_class(solver)(self, mo_coeff, mo_occ, nocc_frozen=nocc_frozen, nvir_frozen=nvir_frozen,
                eris=eris, options=self.opts.solver_options)
        csolver.kernel(init_guess=init_guess)
        self.log.timing("Time for %s solver:  %s", solver, time_string(timer()-t0))
        self.converged = csolver.converged
        self.e_corr_full = csolver.e_corr
        # ERIs and initial guess for next calculations
        if self.base.opts.project_eris:
            eris = csolver._eris
        else:
            eris = None
        if self.base.opts.project_init_guess:
            init_guess = {"t1" : csolver.t1, "t2" : csolver.t2, "c_occ" : c_active_occ, "c_vir" : c_active_vir}
        else:
            init_guess = None

        p1, p2 = self.project_amplitudes_to_fragment(csolver._solver, csolver.c1, csolver.c2)
        e_corr = self.get_fragment_energy(csolver._solver, p1, p2, eris=csolver._eris)
        # Population analysis
        if self.opts.make_rdm1 and csolver.dm1 is not None:
            try:
                self.pop_analysis(csolver.dm1)
            except Exception as e:
                self.log.error("Exception in population analysis: %s", e)
        # EOM analysis
        if self.opts.eom_ccsd in (True, "IP"):
            self.eom_ip_energy, _ = self.eom_analysis(csolver, "IP")
        if self.opts.eom_ccsd in (True, "EA"):
            self.eom_ea_energy, _ = self.eom_analysis(csolver, "EA")

        #self.log.changeIndentLevel(-1)

        return e_corr, nactive, init_guess, eris

    def apply_bno_threshold(self, c_no, n_no, bno_thr):
        """Split natural orbitals (NO) into bath and rest."""
        n_bno = sum(n_no >= bno_thr)
        n_rest = len(n_no)-n_bno
        n_in, n_cut = np.split(n_no, [n_bno])
        # Logging
        fmt = "  %4s: N= %4d  max= % 9.3g  min= % 9.3g  sum= % 9.3g ( %7.3f %%)"
        if n_bno > 0:
            self.log.info(fmt, "Bath", n_bno, max(n_in), min(n_in), np.sum(n_in), 100*np.sum(n_in)/np.sum(n_no))
        else:
            self.log.info(fmt[:13], "Bath", 0)
        if n_rest > 0:
            self.log.info(fmt, "Rest", n_rest, max(n_cut), min(n_cut), np.sum(n_cut), 100*np.sum(n_cut)/np.sum(n_no))
        else:
            self.log.info(fmt[:13], "Rest", 0)

        c_bno, c_rest = np.hsplit(c_no, [n_bno])
        return c_bno, c_rest


    # Register frunctions of dmet_bath.py as methods
    make_dmet_bath = make_dmet_bath


    def add_tailor_fragment(self, frag):
        raise NotImplementedError()


    def additional_bath_for_cluster(self, c_bath, c_occenv, c_virenv):
        """Add additional bath orbitals to cluster (fragment+DMET bath)."""
        # NOT MAINTAINED
        raise NotImplementedError()
        if self.power1_occ_bath_tol is not False:
            c_add, c_occenv, _ = make_mf_bath(self, c_occenv, "occ", bathtype="power",
                    tol=self.power1_occ_bath_tol)
            self.log.info("Adding %d first-order occupied power bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_add, c_bath))
        if self.power1_vir_bath_tol is not False:
            c_add, c_virenv, _ = make_mf_bath(self, c_virenv, "vir", bathtype="power",
                    tol=self.power1_vir_bath_tol)
            self.log.info("Adding %d first-order virtual power bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_bath, c_add))
        # Local orbitals:
        if self.local_occ_bath_tol is not False:
            c_add, c_occenv = make_local_bath(self, c_occenv, tol=self.local_occ_bath_tol)
            self.log.info("Adding %d local occupied bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_add, c_bath))
        if self.local_vir_bath_tol is not False:
            c_add, c_virenv = make_local_bath(self, c_virenv, tol=self.local_vir_bath_tol)
            self.log.info("Adding %d local virtual bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_bath, c_add))
        return c_bath, c_occenv, c_virenv


    def project_amplitudes_to_fragment(self, cm, c1, c2, **kwargs):
        """Wrapper for project_amplitude_to_fragment, where the mo coefficients are extracted from a MP2 or CC object."""

        act = cm.get_frozen_mask()
        occ = cm.mo_occ[act] > 0
        vir = cm.mo_occ[act] == 0
        c = cm.mo_coeff[:,act]
        c_occ = c[:,occ]
        c_vir = c[:,vir]

        p1 = p2 = None
        if c1 is not None:
            p1 = self.project_amplitude_to_fragment(c1, c_occ, c_vir, **kwargs)
        if c2 is not None:
            p2 = self.project_amplitude_to_fragment(c2, c_occ, c_vir, **kwargs)
        return p1, p2


    def project_amplitude_to_fragment(self, c, c_occ=None, c_vir=None, partitioning=None, symmetrize=False):
        """Get local contribution of amplitudes."""

        if np.ndim(c) not in (2, 4):
            raise NotImplementedError()
        if partitioning is None:
            part = self.opts.energy_partitioning
        else:
            part = partitioning
        if part not in ('first-occ', 'first-vir', 'democratic'):
            raise ValueError("Unknown partitioning of amplitudes: %s" % part)

        # Projectors into fragment occupied and virtual space
        if part in ("first-occ", "democratic"):
            assert c_occ is not None
            fo = self.get_fragment_projector(c_occ)
        if part in ("first-vir", "democratic"):
            assert c_vir is not None
            fv = self.get_fragment_projector(c_vir)
        # Inverse projectors needed
        if part == "democratic":
            ro = np.eye(fo.shape[-1]) - fo
            rv = np.eye(fv.shape[-1]) - fv

        if np.ndim(c) == 2:
            if part == "first-occ":
                p = einsum("xi,ia->xa", fo, c)
            elif part == "first-vir":
                p = einsum("ia,xa->ix", c, fv)
            elif part == "democratic":
                p = einsum("xi,ia,ya->xy", fo, c, fv)
                p += einsum("xi,ia,ya->xy", fo, c, rv) / 2.0
                p += einsum("xi,ia,ya->xy", ro, c, fv) / 2.0
            return p

        # ndim == 4:

        if part == "first-occ":
            p = einsum("xi,ijab->xjab", fo, c)
        elif part == "first-vir":
            p = einsum("ijab,xa->ijxb", c, fv)
        elif part == "democratic":

            def project(p1, p2, p3, p4):
                p = einsum("xi,yj,ijab,za,wb->xyzw", p1, p2, c, p3, p4)
                return p

            # Factors of 2 due to ij,ab <-> ji,ba symmetry
            # Denominators 1/N due to element being shared between N clusters

            # Quadruple F
            # ===========
            # This is fully included
            p = project(fo, fo, fv, fv)
            # Triple F
            # ========
            # This is fully included
            p += 2*project(fo, fo, fv, rv)
            p += 2*project(fo, ro, fv, fv)
            # Double F
            # ========
            # P(FFrr) [This wrongly includes: 1x P(FFaa), instead of 0.5x - correction below]
            p +=   project(fo, fo, rv, rv)
            p += 2*project(fo, ro, fv, rv)
            p += 2*project(fo, ro, rv, fv)
            p +=   project(ro, ro, fv, fv)
            # Single F
            # ========
            # P(Frrr) [This wrongly includes: P(Faar) (where r could be a) - correction below]
            p += 2*project(fo, ro, rv, rv) / 4.0
            p += 2*project(ro, ro, fv, rv) / 4.0

            # Corrections
            # ===========
            # Loop over all other clusters x
            for x in self.loop_fragments(exclude_self=True):

                xo = x.get_fragment_projector(c_occ)
                xv = x.get_fragment_projector(c_vir)

                # Double correction
                # -----------------
                # Correct for wrong inclusion of P(FFaa)
                # The case P(FFaa) was included with prefactor of 1 instead of 1/2
                # We thus need to only correct by "-1/2"
                p -=   project(fo, fo, xv, xv) / 2.0
                p -= 2*project(fo, xo, fv, xv) / 2.0
                p -= 2*project(fo, xo, xv, fv) / 2.0
                p -=   project(xo, xo, fv, fv) / 2.0

                # Single correction
                # -----------------
                # Correct for wrong inclusion of P(Faar)
                # This corrects the case P(Faab) but overcorrects P(Faaa)!
                p -= 2*project(fo, xo, xv, rv) / 4.0
                p -= 2*project(fo, xo, rv, xv) / 4.0 # If r == x this is the same as above -> overcorrection
                p -= 2*project(fo, ro, xv, xv) / 4.0 # overcorrection
                p -= 2*project(xo, xo, fv, rv) / 4.0
                p -= 2*project(xo, ro, fv, xv) / 4.0 # overcorrection
                p -= 2*project(ro, xo, fv, xv) / 4.0 # overcorrection

                # Correct overcorrection
                # The additional factor of 2 comes from how often the term was wrongly included above
                p += 2*2*project(fo, xo, xv, xv) / 4.0
                p += 2*2*project(xo, xo, fv, xv) / 4.0

        # Note that the energy should be invariant to symmetrization
        if symmetrize:
            p = (p + p.transpose(1,0,3,2)) / 2

        return p


    def get_fragment_energy(self, cm, p1, p2, eris):
        """
        Parameters
        ----------
        cm : pyscf[.pbc].cc.CCSD or pyscf[.pbc].mp.MP2
            PySCF coupled cluster or MP2 object. This function accesses
            `cc.get_frozen_mask()` and `cm.mo_occ`.
        p1 : (nOcc, nVir) array
            Locally projected C1 amplitudes.
        p2 : (nOcc, nOcc, nVir, nVir) array
            Locally projected C2 amplitudes.
        eris :
            PySCF eris object as returned by cm.ao2mo()

        Returns
        -------
        e_frag : float
            Fragment energy contribution.
        """
        # MP2
        if p1 is None:
            e1 = 0
        # CC
        else:
            act = cm.get_frozen_mask()
            occ = cm.mo_occ[act] > 0
            vir = cm.mo_occ[act] == 0
            f = eris.fock[occ][:,vir]
            e1 = 2*np.sum(f * p1)

        if hasattr(eris, "ovvo"):
            eris_ovvo = eris.ovvo
        # MP2 only has eris.ovov - are these the same integrals?
        else:
            no, nv = p2.shape[1:3]
            eris_ovvo = eris.ovov[:].reshape(no,nv,no,nv).transpose(0, 1, 3, 2).conj()
        e2 = 2*einsum('ijab,iabj', p2, eris_ovvo)
        e2 -=  einsum('ijab,jabi', p2, eris_ovvo)

        self.log.info("Energy components: E1= % 16.8f Ha, E2=% 16.8f Ha", e1, e2)
        if e1 > 1e-4 and 10*e1 > e2:
            self.log.warning("WARNING: Large E1 component!")

        e_frag = self.sym_factor * (e1 + e2)
        return e_frag


    def pop_analysis(self, dm1, filename=None, mode="a", sig_tol=0.01):
        """Perform population analsis for the given density-matrix and compare to the MF."""
        if filename is None:
            filename = "%s-%s.txt" % (self.base.opts.popfile, self.name)

        sc = np.dot(self.base.get_ovlp(), self.base.lo)
        dm1 = np.linalg.multi_dot((sc.T, dm1, sc))
        pop, chg = self.mf.mulliken_pop(dm=dm1, s=np.eye(dm1.shape[-1]))
        pop_mf = self.base.pop_mf
        chg_mf = self.base.pop_mf_chg

        tstamp = datetime.now()

        self.log.info("[%s] Writing cluster population analysis to file \"%s\"", tstamp, filename)
        with open(filename, mode) as f:
            f.write("[%s] Population analysis\n" % tstamp)
            f.write("*%s*********************\n" % (26*"*"))

            # per orbital
            for i, s in enumerate(self.mf.mol.ao_labels()):
                dmf = (pop[i]-pop_mf[i])
                sig = (" !" if abs(dmf)>=sig_tol else "")
                f.write("  orb= %4d %-16s occ= %10.5f dHF= %+10.5f%s\n" %
                        (i, s, pop[i], dmf, sig))
            # Charge per atom
            f.write("[%s] Atomic charges\n" % tstamp)
            f.write("*%s****************\n" % (26*"*"))
            for ia in range(self.mf.mol.natm):
                symb = self.mf.mol.atom_symbol(ia)
                dmf = (chg[ia]-chg_mf[ia])
                sig = (" !" if abs(dmf)>=sig_tol else "")
                f.write("  atom= %3d %-3s charge= %10.5f dHF= %+10.5f%s\n" %
                        (ia, symb, chg[ia], dmf, sig))

        return pop, chg

    def eom_analysis(self, csolver, kind, filename=None, mode="a", sort_weight=True, r1_min=1e-2):
        kind = kind.upper()
        assert kind in ("IP", "EA")

        if filename is None:
            filename = "%s-%s.txt" % (self.base.opts.eomfile, self.name)

        sc = np.dot(self.base.get_ovlp(), self.base.lo)
        if kind == "IP":
            e, c = csolver.ip_energy, csolver.ip_coeff
        else:
            e, c = csolver.ea_energy, csolver.ea_coeff
        nroots = len(e)
        eris = csolver._eris
        cc = csolver._solver

        self.log.info("EOM-CCSD %s energies= %r", kind, e[:5].tolist())
        tstamp = datetime.now()
        self.log.info("[%s] Writing detailed cluster %s-EOM analysis to file \"%s\"", tstamp, kind, filename)

        with open(filename, mode) as f:
            f.write("[%s] %s-EOM analysis\n" % (tstamp, kind))
            f.write("*%s*****************\n" % (26*"*"))

            for root in range(nroots):
                r1 = c[root][:cc.nocc]
                qp = np.linalg.norm(r1)**2
                f.write("  %s-EOM-CCSD root= %2d , energy= %+16.8g , QP-weight= %10.5g\n" %
                        (kind, root, e[root], qp))
                if qp < 0.0 or qp > 1.0:
                    self.log.error("Error: QP-weight not between 0 and 1!")
                r1lo = einsum("i,ai,al->l", r1, eris.mo_coeff[:,:cc.nocc], sc)

                if sort_weight:
                    order = np.argsort(-r1lo**2)
                    for ao, lab in enumerate(np.asarray(self.mf.mol.ao_labels())[order]):
                        wgt = r1lo[order][ao]**2
                        if wgt < r1_min*qp:
                            break
                        f.write("  * Weight of %s root %2d on OrthAO %-16s = %10.5f\n" %
                                (kind, root, lab, wgt))
                else:
                    for ao, lab in enumerate(ao_labels):
                        wgt = r1lo[ao]**2
                        if wgt < r1_min*qp:
                            continue
                        f.write("  * Weight of %s root %2d on OrthAO %-16s = %10.5f\n" %
                                (kind, root, lab, wgt))

        return e, c


    def get_fragment_bsse(self, rmax=None, nimages=5, unit='A'):
        self.log.info("Counterpoise Calculation")
        self.log.info("************************")
        # Currently only PBC
        #if not self.boundary_cond == 'open':
        #    raise NotImplementedError()
        if rmax is None:
            rmax = self.opts.bsse_rmax

        # Atomic calculation with atomic basis functions:
        #mol = self.mol.copy()
        #atom = mol.atom[self.atoms]
        #self.log.debugv("Keeping atoms %r", atom)
        #mol.atom = atom
        #mol.a = None
        #mol.build(False, False)

        natom0, e_mf0, e_cm0, dm = self.counterpoise_calculation(rmax=0.0, nimages=0)
        assert natom0 == len(self.atoms)
        self.log.debugv("Counterpoise: E(atom)= % 16.8f Ha", e_cm0)

        #natom_list = []
        #e_mf_list = []
        #e_cm_list = []
        r_values = np.hstack((np.arange(1.0, int(rmax)+1, 1.0), rmax))
        #for r in r_values:
        r = rmax
        natom, e_mf, e_cm, dm = self.counterpoise_calculation(rmax=r, dm0=dm)
        self.log.debugv("Counterpoise: n(atom)= %3d  E(mf)= %16.8f Ha  E(%s)= % 16.8f Ha", natom, e_mf, self.solver, e_cm)

        e_bsse = self.sym_factor*(e_cm - e_cm0)
        self.log.debugv("Counterpoise: E(BSSE)= % 16.8f Ha", e_bsse)
        return e_bsse


    def counterpoise_calculation(self, rmax, dm0=None, nimages=5, unit='A'):
        mol = self.make_counterpoise_mol(rmax, nimages=nimages, unit=unit, output='pyscf-cp.txt')
        # Mean-field
        #mf = type(self.mf)(mol)
        mf = pyscf.scf.RHF(mol)
        mf.conv_tol = self.mf.conv_tol
        #if self.mf.with_df is not None:
        #    self.log.debugv("Setting GDF")
        #    self.log.debugv("%s", type(self.mf.with_df))
        #    # ONLY GDF SO FAR!
        # TODO: generalize
        if self.base.kdf is not None:
            auxbasis = self.base.kdf.auxbasis
        elif self.mf.with_df is not None:
            auxbasis = self.mf.with_df.auxbasis
        else:
            auxbasis=None
        if auxbasis:
            mf = mf.density_fit(auxbasis=auxbasis)
        # TODO:
        #use dm0 as starting point
        mf.kernel()
        dm0 = mf.make_rdm1()
        # Embedded calculation with same options
        ecc = ewf.EWF(mf, solver=self.solver, bno_threshold=self.bno_threshold, options=self.base.opts)
        ecc.make_atom_cluster(self.atoms, options=self.opts)
        ecc.kernel()

        return mol.natm, mf.e_tot, ecc.e_tot, dm0
