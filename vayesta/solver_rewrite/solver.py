import dataclasses
from timeit import default_timer as timer
import copy

import numpy as np
import scipy
import scipy.optimize

from pyscf import scf

from vayesta.core.util import *


class ClusterSolver:
    """Base class for cluster solver"""

    @dataclasses.dataclass
    class Options(OptionsBase):
        pass

    def __init__(self, mf, fragment, cluster, log=None, **kwargs):
        """
        Arguments
        ---------
        """
        self.mf = mf
        self.fragment = fragment
        self.cluster = cluster
        self.log = (log or fragment.log)
        # --- Options:
        self.opts = self.Options()
        self.opts.update(**kwargs)
        self.log.info("Parameters of %s:" % self.__class__.__name__)
        self.log.info(break_into_lines(str(self.opts), newline='\n    '))

        # Additional external potential
        self.v_ext = None

        # --- Results
        self.converged = False
        self.e_corr = 0
        self.wf = None
        self.dm1 = None
        self.dm2 = None

    @property
    def base(self):
        return self.fragment.base

    def get_fock(self):
        c_active = self.cluster.c_active
        return dot(c_active.T, self.base.get_fock(), c_active)

    def kernel(self, *args, **kwargs):
        """Set up everything for a calculation on the CAS and pass this to the solver-specific kernel that runs on this
        information."""
        mf = self.make_clus_mf()
        return self.kernel_solver(mf)

    def kernel_solver(self, mf_clus, eris_energy=None):
        raise AbstractMethodError()

    @property
    def _scf_class(self):
        return scf.RHF

    def make_clus_hamil(self, screening=None):
        nelec, nao, mo_coeff, mo_occ = self.get_clus_info()
        bare_eris = self.get_eris()
        heff, mo_energy = self.get_heff(bare_eris)
        if screening is None:
            seris = bare_eris
        else:
            seris = None
        return nelec, nao, mo_coeff, mo_occ, mo_energy, heff, bare_eris, seris

    def make_clus_mf(self, screening=None):
        nelec, nao, mo_coeff, mo_occ, mo_energy, heff, bare_eris, seris = self.make_clus_hamil(screening)
        clusmf = self._scf_class(self.mf.mol)
        clusmf.get_hcore = lambda *args: heff
        fock = self.get_fock()
        clusmf.get_fock = lambda *args: fock
        clusmf.mo_coeff = mo_coeff
        clusmf.mo_occ = mo_occ
        clusmf.mo_energy = mo_energy
        raise NotImplementedError("Unfinished...")
        return clusmf

    def make_clus_mf(self, screening=None):




        bare_eris = self.get_eris()
        heff, mo_energy = self.get_heff(bare_eris)
        clusmol, mo_coeff, mo_occ = self.get_clus_info()

        clusmf = self._scf_class(clusmol)
        clusmf.get_hcore = lambda *args: heff
        clusmf.get_ovlp = lambda *args: np.eye(clusmol.nao)
        fock = self.get_fock()
        clusmf.get_fock = lambda *args: fock
        clusmf.mo_coeff = mo_coeff
        clusmf.mo_occ = mo_occ
        clusmf.mo_energy = mo_energy

        if screening is None:
            clusmf._eri = bare_eris
        else:
            pass
        return clusmf

    def get_clus_info(self):
        nelec = (self.cluster.nocc_active, self.cluster.nocc_active)
        nao = self.cluster.norb_active
        mo_coeff = self.cluster.c_active
        mo_occ = np.zeros((nao,))
        mo_occ[:nelec[0]] = 2.0
        return nelec, nao, mo_coeff, mo_occ

    def get_heff(self, eris, fock=None, with_vext=True):
        if fock is None:
            fock = self.get_fock()
        mo_energy = np.diag(fock)

        occ = np.s_[:self.cluster.nocc_active]
        v_act = 2 * einsum('iipq->pq', eris[occ, occ]) - einsum('iqpi->pq', eris[occ, :, :, occ])
        h_eff = fock - v_act
        # This should be equivalent to:
        # core = np.s_[:self.nocc_frozen]
        # dm_core = 2*np.dot(self.mo_coeff[:,core], self.mo_coeff[:,core].T)
        # v_core = self.mf.get_veff(dm=dm_core)
        # h_eff = np.linalg.multi_dot((self.c_active.T, self.base.get_hcore()+v_core, self.c_active))
        if with_vext and self.v_ext is not None:
            h_eff += self.v_ext
        return h_eff, mo_energy

    def get_eris(self, *args, **kwargs):
        with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
            coeff = self.cluster.c_active
            eris = self.base.get_eris_array(coeff)
        return eris

    def get_projector(self, c_frag):
        r_frag = dot(self.cluster.c_active.T, self.mf.get_ovlp(), c_frag)
        p_frag = np.dot(r_frag, r_frag.T)
        return p_frag

    def optimize_cpt(self, nelectron, c_frag, cpt_guess=0, atol=1e-6, rtol=1e-6, cpt_radius=0.5):
        """Enables chemical potential optimization to match a number of electrons in the fragment space.

        Parameters
        ----------
        nelectron: float
            Target number of electrons.
        c_frag: array
            Fragment orbitals.
        cpt_guess: float, optional
            Initial guess for fragment chemical potential. Default: 0.
        atol: float, optional
            Absolute electron number tolerance. Default: 1e-6.
        rtol: float, optional
            Relative electron number tolerance. Default: 1e-6
        cpt_radius: float, optional
            Search radius for chemical potential. Default: 0.5.

        Returns
        -------
        results:
            Solver results.
        """

        kernel_orig = self.kernel
        ovlp = self.mf.get_ovlp()
        # Make projector into fragment space

        p_frag = self.get_projector(c_frag)

        class CptFound(RuntimeError):
            """Raise when electron error is below tolerance."""
            pass

        def kernel(self, *args, eris=None, **kwargs):
            result = None
            err = None
            cpt_opt = None
            iterations = 0
            init_guess = {}
            err0 = None

            # Avoid calculating the ERIs multiple times:
            if eris is None:
                eris = self.get_eris()

            def electron_err(cpt):
                nonlocal result, err, err0, cpt_opt, iterations, init_guess
                # Avoid recalculation of cpt=0.0 in SciPy:
                if (cpt == 0) and (err0 is not None):
                    self.log.debugv("Chemical potential %f already calculated - returning error= %.8f", cpt, err0)
                    return err0

                kwargs.update(init_guess)
                self.log.debugv("kwargs keys for solver: %r", kwargs.keys())

                replace = {}
                if cpt:
                    v_ext_0 = (self.v_ext if self.v_ext is not None else 0)
                    if self.is_rhf:
                        replace['v_ext'] = v_ext_0 - cpt*p_frag
                    else:
                        if v_ext_0 == 0:
                            v_ext_0 = (v_ext_0, v_ext_0)
                        replace['v_ext'] = (v_ext_0[0] - cpt*p_frag[0], v_ext_0[1] - cpt*p_frag[1])

                self.reset()
                with replace_attr(self, **replace):
                    results = kernel_orig(eris=eris, **kwargs)
                if not self.converged:
                    raise ConvergenceError()
                dm1 = self.wf.make_rdm1()
                if self.is_rhf:
                    ne_frag = einsum('xi,ij,xj->', p_frag, dm1, p_frag)
                else:
                    ne_frag = (einsum('xi,ij,xj->', p_frag[0], dm1[0], p_frag[0])
                             + einsum('xi,ij,xj->', p_frag[1], dm1[1], p_frag[1]))

                err = (ne_frag - nelectron)
                self.log.debug("Fragment chemical potential= %+12.8f Ha:  electrons= %.8f  error= %+.3e", cpt, ne_frag, err)
                iterations += 1
                if abs(err) < (atol + rtol*nelectron):
                    cpt_opt = cpt
                    raise CptFound()
                # Initial guess for next chemical potential
                #init_guess = results.get_init_guess()
                init_guess = self.get_init_guess()
                return err

            # First run with cpt_guess:
            try:
                err0 = electron_err(cpt_guess)
            except CptFound:
                self.log.debug("Chemical potential= %.6f leads to electron error= %.3e within tolerance (atol= %.1e, rtol= %.1e)", cpt_guess, err, atol, rtol)
                return result

            # Not enough electrons in fragment space -> raise fragment chemical potential:
            if err0 < 0:
                lower = cpt_guess
                upper = cpt_guess+cpt_radius
            # Too many electrons in fragment space -> lower fragment chemical potential:
            else:
                lower = cpt_guess-cpt_radius
                upper = cpt_guess

            self.log.debugv("Estimated bounds: %.3e %.3e", lower, upper)
            bounds = np.asarray([lower, upper], dtype=float)

            for ntry in range(5):
                try:
                    cpt, res = scipy.optimize.brentq(electron_err, a=bounds[0], b=bounds[1], xtol=1e-12, full_output=True)
                    if res.converged:
                        raise RuntimeError("Chemical potential converged to %+16.8f, but electron error is still %.3e" % (cpt, err))
                except CptFound:
                    break
                # Could not find chemical potential in bracket:
                except ValueError:
                    bounds *= 2
                    self.log.warning("Interval for chemical potential search too small. New search interval: [%f %f]", *bounds)
                    continue
                # Could not convergence in bracket:
                except ConvergenceError:
                    bounds /= 2
                    self.log.warning("Solver did not converge. New search interval: [%f %f]", *bounds)
                    continue
                raise RuntimeError("Invalid state: electron error= %.3e" % err)
            else:
                errmsg = ("Could not find chemical potential within interval [%f %f]!" % (bounds[0], bounds[1]))
                self.log.critical(errmsg)
                raise RuntimeError(errmsg)

            self.log.info("Chemical potential optimized in %d iterations= %+16.8f Ha", iterations, cpt_opt)
            return result

        # Replace kernel:
        self.kernel = kernel.__get__(self)

class UClusterSolver(ClusterSolver):

    def get_fock(self):
        c_active = self.cluster.c_active
        fock = self.base.get_fock()
        return (dot(c_active[0].T, fock[0], c_active[0]),
                dot(c_active[1].T, fock[1], c_active[1]))
        raise RuntimeError

    @property
    def _scf_class(self):
        return scf.UHF

    def get_clus_info(self):
        clusmol = self.mf.mol.__class__()
        clusmol.nelec = self.cluster.nocc_active
        na, nb = self.cluster.norb_active
        # NB if the number of alpha and beta active orbitals is different this approach can't work.
        assert (na == nb)
        clusmol.nao = na
        clusmol.build()
        mo_coeff = np.zeros((2, clusmol.nao, clusmol.nao))
        mo_coeff[0] = np.eye(clusmol.nao)
        mo_coeff[1] = np.eye(clusmol.nao)

        mo_occ = np.zeros((2, clusmol.nao))
        mo_occ[0, :clusmol.nelec[0]] = 1
        mo_occ[1, :clusmol.nelec[1]] = 1

        return clusmol, mo_coeff, mo_occ

    def get_heff(self, eris, fock=None, with_vext=True):
        if fock is None:
            fock = self.get_fock()
        mo_energy = np.diagonal(fock, 1, 2)

        oa = np.s_[:self.cluster.nocc_active[0]]
        ob = np.s_[:self.cluster.nocc_active[1]]
        gaa, gab, gbb = eris
        va = (einsum('iipq->pq', gaa[oa, oa]) + einsum('pqii->pq', gab[:, :, ob, ob])  # Coulomb
              - einsum('ipqi->pq', gaa[oa, :, :, oa]))  # Exchange
        vb = (einsum('iipq->pq', gbb[ob, ob]) + einsum('iipq->pq', gab[oa, oa])  # Coulomb
              - einsum('ipqi->pq', gbb[ob, :, :, ob]))  # Exchange
        h_eff = (fock[0] - va, fock[1] - vb)
        if with_vext and self.v_ext is not None:
            h_eff = ((h_eff[0] + self.v_ext[0]),
                     (h_eff[1] + self.v_ext[1]))
        return h_eff, mo_energy

    def get_eris(self, *args, **kwargs):
        with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
            coeff = self.cluster.c_active
            eris = (self.base.get_eris_array(coeff[0]),
                    self.base.get_eris_array((coeff[0], coeff[0], coeff[1], coeff[1])),
                    self.base.get_eris_array(coeff[1]))
        return eris

    def get_projector(self, c_frag):
        ovlp = self.mf.get_ovlp()
        r_frag = (dot(self.cluster.c_active[0].T, ovlp, c_frag[0]),
                  dot(self.cluster.c_active[1].T, ovlp, c_frag[1]))
        p_frag = (np.dot(r_frag[0], r_frag[0].T),
                  np.dot(r_frag[1], r_frag[1].T))
        return p_frag

class EBClusterSolver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        polaritonic_shift: bool = True

    def kernel(self, *args, **kwargs):
        """Set up everything for a calculation on the CAS and pass this to the solver-specific kernel that runs on this
        information."""
        self.set_polaritonic_shift(self.fragment.bos_freqs, self.fragment.couplings)
        mf = self.make_clus_mf()
        couplings = self.get_polaritonic_shifted_couplings(self.fragment.bos_freqs, self.fragment.couplings)

        return self.kernel_solver(mf, self.fragment.bos_freqs, couplings)

    def kernel_solver(self, mf_clus, freqs, couplings):
        raise AbstractMethodError()

    @property
    def polaritonic_shift(self):
        try:
            return self._polaritonic_shift
        except AttributeError as e:
            self.log.critical("Polaritonic shift not yet set.")
            raise e

    def get_heff(self, eris, fock=None, with_vext=True):
        heff, mo_energy = super().get_heff(eris, fock, with_vext)

        if self.opts.polaritonic_shift:
            fock_shift = self.get_polaritonic_fock_shift(self.fragment.couplings)
            if not np.allclose(fock_shift[0], fock_shift[1]):
                self.log.critical("Polaritonic shift breaks cluster spin symmetry; please either use an unrestricted"
                                  "formalism or bosons without polaritonic shift.")
            heff = heff + fock_shift[0]
        return heff, mo_energy

    def set_polaritonic_shift(self, freqs, couplings):
        no = self.cluster.nocc_active
        if isinstance(no, int):
            noa = nob = no
        else:
            noa, nob = no
        self._polaritonic_shift = np.multiply(freqs ** (-1), einsum("npp->n", couplings[0][:, :noa, :noa]) +
                                              einsum("npp->n", couplings[1][:, :nob, :nob]))
        self.log.info("Applying Polaritonic shift gives energy change of %e",
                      -sum(np.multiply(self._polaritonic_shift ** 2, freqs)))

    def get_polaritonic_fock_shift(self, couplings):
        return tuple([- einsum("npq,n->pq", x + x.transpose(0, 2, 1), self.polaritonic_shift) for x in couplings])

    def get_polaritonic_shifted_couplings(self, freqs, couplings):
        temp = np.multiply(self.polaritonic_shift, freqs) / (2 * self.cluster.nocc_active)
        couplings = tuple([x - einsum("pq,n->npq", np.eye(x.shape[1]), temp) for x in couplings])
        if not np.allclose(couplings[0], couplings[1]):
            self.log.critical("Polaritonic shifted bosonic fermion-boson couplings break cluster spin symmetry; please"
                              "use an unrestricted formalism.")
            raise RuntimeError()
        return couplings[0]

    def get_eb_dm_polaritonic_shift(self):
        shift = self.polaritonic_shift
        if isinstance(self.dm1, tuple):
            # UHF calculation
            return tuple([-einsum("n,pq->pqn", shift, x) for x in self.dm1])
        else:
            return (-einsum("n,pq->pqn", shift, self.dm1 / 2),) * 2


# If we wanted an abstract base class for unrestricted electron-boson this would be all that was required; in practice
# can do this in actual solvers.
class UEBClusterSolver(EBClusterSolver, UClusterSolver):

    def get_heff(self, eris, fock=None, with_vext=True):
        heff, mo_energy = super().get_heff(eris, fock, with_vext)

        if self.opts.polaritonic_shift:
            fock_shift = self.get_polaritonic_fock_shift(self.fragment.couplings)
            heff = tuple([x + y for x, y in zip(heff, fock_shift)])
        return heff, mo_energy

    def get_polaritonic_shifted_couplings(self, freqs, couplings):
        temp = np.multiply(self.polaritonic_shift, freqs) / sum(self.cluster.nocc_active)
        return tuple([x - einsum("pq,n->npq", np.eye(x.shape[1]), temp) for x in couplings])
