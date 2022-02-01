# Standard library
import dataclasses

# NumPy
import numpy as np

# PySCF
from pyscf import ao2mo, agf2, lib
from pyscf.agf2 import mpi_helper, _agf2

# Vayesta
import vayesta
from vayesta.core.util import time_string, OptionsBase, NotSet
from vayesta.eagf2.ragf2 import RAGF2Options, RAGF2, _ao2mo_3c, _ao2mo_4c

# Timings
if mpi_helper.mpi == None:
    from timeit import default_timer as timer
else:
    timer = mpi_helper.mpi.Wtime


@dataclasses.dataclass
class UAGF2Options(RAGF2Options):
    """Options for UAGF2 calculations.
    """

    pass


class UAGF2(RAGF2):
    
    Options = UAGF2Options

    def _active_slices(self, nmo, frozen):
        """Get slices for frozen occupied, active, frozen virtual spaces.
        """

        focc = tuple(slice(None, f[0]) for f in frozen)
        fvir = tuple(slice(n-f[1], None) for n, f in zip(nmo, frozen))
        act = tuple(slice(f[0], n-f[1]) for n, f in zip(nmo, frozen))

        return focc, fvir, act


    def _get_h1e(self, mo_coeff):
        """Get the core Hamiltonian.
        """

        return (
            super()._get_h1e(mo_coeff[0]),
            super()._get_h1e(mo_coeff[1]),
        )


    def _get_frozen_veff(self, mo_coeff):
        """Get Veff due to the frozen density.
        """

        if self.veff is not None:
            self.log.info("Veff due to frozen density passed by kwarg")
        else:
            dm_froz = [None, None]
            for s, spin in self.spins:
                #FIXME only if needed
                self.log.info("Calculating Veff (%s) due to frozen density" % spin)
                c_focc = mo_coeff[s][:, self.focc[s]]
                dm_froz[s] = np.dot(c_focc, c_focc.T.conj()) * 2

            veff = list(self.mf.get_veff(dm=dm_froz))
            for s, spin in self.spins:
                veff[s] = np.linalg.multi_dot((mo_coeff[s].T.conj(), veff[s], mo_coeff[s]))

            self.veff = tuple(veff)

        return self.veff


    def _print_sizes(self):
        """Print the system sizes.
        """

        self.log.info("                 %6s %6s %6s", 'active', 'frozen', 'total')

        for s, spin in self.spins:
            nmo, nocc, nvir = self.nmo[s], self.nocc[s], self.nvir[s]
            frozen, nact, nfroz = self.frozen[s], self.nact[s], self.nfroz[s]

            self.log.info("Occupied %s MOs:  %6d %6d %6d", spin, nocc-frozen[0], frozen[0], nocc)
            self.log.info("Virtual %s MOs:   %6d %6d %6d", spin, nvir-frozen[1], frozen[1], nvir)
            self.log.info("General %s MOs:   %6d %6d %6d", spin, nact,           nfroz,     nmo)


    def _build_moments(self, eo, ev, xija, os_factor=None, ss_factor=None):
        """Build the occupied or virtual self-energy moments.
        """

        os_factor = os_factor or self.opts.os_factor
        ss_factor = ss_factor or self.opts.ss_factor
        facs = {'os_factor': os_factor, 'ss_factor': ss_factor}

        if self.opts.nmom_lanczos == 0 and not self.opts.diagonal_se:
            if isinstance(xija[0], tuple):
                t = _agf2.build_mats_dfuagf2_incore(*xija, eo, ev, **facs)
            else:
                t = _agf2.build_mats_uagf2_incore(xija, eo, ev, **facs)
        else:
            raise NotImplementedError

        return t


    def _combine_se(self, se_occ, se_vir, gf=None):
        """Combine the occupied and virtual self-energies.
        """

        se = [agf2.aux.combine(se_occ[s], se_vir[s]) for s, spin in self.spins]

        if self.opts.nmom_projection is not None:
            gf = gf or self.gf
            fock = self.get_fock(gf=gf, with_frozen=False)
            for s, spin in self.spins:
                se[s] = se[s].compress(n=(self.opts.nmom_projection, None), phys=fock[s])

        for s, spin in self.spins:
            se[s].remove_uncoupled(tol=self.opts.weight_tol)

        return tuple(se)


    def ao2mo(self):
        """Get the ERIs in MO basis.
        """

        t0 = timer()

        if getattr(self.mf, 'with_df', None) is not None:
            self.log.info("ERIs will be density fitted")
            if self.mf.with_df._cderi is None:
                self.mf.with_df.build()
            if not isinstance(self.mf.with_df._cderi, np.ndarray):
                raise ValueError("DF _cderi object is not an array (%s)" % self.mf.with_df._cderi)

            self.eri = [None for s, spin in self.spins]
            for s, spin in self.spins:
                mo_coeff = self.mo_coeff[s][:, self.act[s]]

                self.eri[s] = np.asarray(lib.unpack_tril(self.mf.with_df._cderi, axis=-1))
                self.eri[s] = _ao2mo_3c(self.eri[s], mo_coeff, mo_coeff)

                self.log.timing(
                        "Time for AO->MO (L|%s%s):  %s",
                        spin, spin, time_string(timer() - t0),
                )

            self.eri = tuple(self.eri)

        else:
            self.log.info("ERIs will be four-centered")

            self.eri = [[None for s1, spin1 in self.spins] for s2, spin2 in self.spins]
            for s1, spin1 in self.spins:
                for s2, spin2 in self.spins:
                    mo_coeff_1 = self.mo_coeff[s1][:, self.act[s1]]
                    mo_coeff_2 = self.mo_coeff[s2][:, self.act[s2]]
                    coeffs = (mo_coeff_1, mo_coeff_1, mo_coeff_2, mo_coeff_2)
                    shape = (self.nact[s1],) * 2 + (self.nact[s2],) * 2

                    self.eri[s1][s2] = ao2mo.incore.general(self.mf._eri, coeffs, compact=False)
                    self.eri[s1][s2] = self.eri[s1][s2].reshape(shape)

                    self.log.timing(
                            "Time for AO->MO (%s%s|%s%s):  %s",
                            spin1, spin1, spin2, spin2, time_string(timer() - t0),
                    )

            self.eri = (tuple(self.eri[0]), tuple(self.eri[1]))

        return self.eri


    def build_self_energy(self, gf, se_prev=None):
        """Build the self-energy using a given Green's function.
        """

        t0 = timer()

        cx = tuple(np.eye(n) for n in self.nact)
        ci = tuple(g.get_occupied().coupling for g in gf)
        ca = tuple(g.get_virtual().coupling for g in gf)

        ei = tuple(g.get_occupied().energy for g in gf)
        ea = tuple(g.get_virtual().energy for g in gf)

        nmom = self.opts.nmom_lanczos
        t_occ = [None for s, spin in self.spins]
        t_vir = [None for s, spin in self.spins]
        se_occ = [None for s, spin in self.spins]
        se_vir = [None for s, spin in self.spins]

        for s1, spin1 in self.spins:
            s2 = (s1 + 1) % 2
            self.log.info("Building the %s self-energy" % spin1)
            self.log.info("**************************")

            if self.opts.non_dyson:
                xo = slice(None, self.nocc[s1]-self.frozen[s1][0])
                xO = slice(None, self.nocc[s2]-self.frozen[s2][0])
                xv = slice(self.nvir[s1]-self.frozen[s1][0], None)
                xV = slice(self.nvir[s2]-self.frozen[s2][0], None)
            else:
                xo = xO = xv = xV = slice(None)

            ei1 = (ei[s1], ei[s2])
            ea1 = (ea[s1], ea[s2])


            if getattr(self.mf, 'with_df', None) is not None:
                qxi = _ao2mo_3c(self.eri[s1], cx[s1][:, xo], ci[s1])
                qja = _ao2mo_3c(self.eri[s1], ci[s1], ca[s1])
                qJA = _ao2mo_3c(self.eri[s2], ci[s2], ca[s2])
                xija = (qxi, qxi)
                xiJA = (qja, qJA)
                dtype = qxi.dtype
            else:
                xija = _ao2mo_4c(self.eri[s1][s1][xo], None, ci[s1], ci[s1], ca[s1])
                xiJA = _ao2mo_4c(self.eri[s1][s2][xo], None, ci[s1], ci[s2], ca[s2])
                dtype = xiJA.dtype

            self.log.timing("Time for MO->QMO (xi|ja):  %s", time_string(timer() - t0))
            t0 = timer()

            t_occ[s1] = np.zeros((2*nmom+2, self.nact[s1], self.nact[s1]), dtype=dtype)
            t_occ[s1][:, xo, xo] = self._build_moments(ei1, ea1, (xija, xiJA))
            del xija, xiJA


            if getattr(self.mf, 'with_df', None) is not None:
                qxa = _ao2mo_3c(self.eri[s1], cx[s1][:, xv], ca[s1])
                qbi = _ao2mo_3c(self.eri[s1], ca[s1], ci[s1])
                qBI = _ao2mo_3c(self.eri[s2], ca[s2], ci[s2])
                xabi = (qxa, qxa)
                xaBI = (qbi, qBI)
                dtype = qxi.dtype
            else:
                xabi = _ao2mo_4c(self.eri[s1][s1][xo], None, ca[s1], ca[s1], ci[s1])
                xaBI = _ao2mo_4c(self.eri[s1][s2][xo], None, ca[s1], ca[s2], ci[s2])
                dtype = xaBI.dtype

            self.log.timing("Time for MO->QMO (xa|bi):  %s", time_string(timer() - t0))
            t0 = timer()

            t_vir[s1] = np.zeros((2*nmom+2, self.nact[s1], self.nact[s1]), dtype=dtype)
            t_vir[s1][:, xv, xv] = self._build_moments(ea1, ei1, (xabi, xaBI))
            del xabi, xaBI


            for i in range(2*nmom+2):
                self.log.debug(
                        "Trace of n=%d moments (%s):  Occupied = %.5g  Virtual = %.5g",
                        i, spin1, np.trace(t_occ[s1][i]).real, np.trace(t_vir[s1][i]).real,
                )


            # === Occupied:

            self.log.info("Occupied %s self-energy:" % spin1)
            with self.log.withIndentLevel(1):
                nija = ei[0].size * (ei[0].size*ea[0].size + ei[1].size*ea[1].size)
                self.log.debug("Number of ija:  %s", nija)

                w = np.linalg.eigvalsh(t_occ[s1][0])
                wmin, wmax = w.min(), w.max()
                (self.log.warning if wmin < 1e-8 else self.log.debug)(
                        'Eigenvalue range:  %.5g -> %.5g', wmin, wmax,
                )

                se_occ[s1] = self._build_se_from_moments(t_occ[s1], chempot=gf[s1].chempot)
                self.log.info("Built %d occupied auxiliaries", se_occ[s1].naux)


            # === Virtual:

            self.log.info("Virtual %s self-energy:" % spin1)
            with self.log.withIndentLevel(1):
                nabi = ea[0].size * (ea[0].size*ei[0].size + ea[1].size*ei[1].size)
                self.log.debug("Number of abi:  %s", nabi)

                w = np.linalg.eigvalsh(t_vir[s1][0])
                wmin, wmax = w.min(), w.max()
                (self.log.warning if wmin < 1e-8 else self.log.debug)(
                        'Eigenvalue range:  %.5g -> %.5g', wmin, wmax,
                )

                se_vir[s1] = self._build_se_from_moments(t_vir[s1], chempot=gf[s1].chempot)
                self.log.info("Built %d virtual auxiliaries", se_vir[s1].naux)


        wt = lambda v: np.sum(v * v)
        for s, spin in self.spins:
            nh = self.nocc[s] - self.frozen[s][0]
            self.log.infov("Total weights of coupling (%s) blocks:", spin)
            self.log.infov("        %6s  %6s", "2h1p", "1h2p")
            self.log.infov(
                    "    1h  %6.4f  %6.4f",
                    wt(se_occ[s].coupling[:nh]),
                    wt(se_vir[s].coupling[:nh]),
            )
            self.log.infov(
                    "    1p  %6.4f  %6.4f",
                    wt(se_occ[s].coupling[nh:]),
                    wt(se_vir[s].coupling[nh:]),
            )


        se = self._combine_se(se_occ, se_vir, gf=gf)

        for s, spin in self.spins:
            self.log.debugv("Auxiliary energies (%s):", spin)
            with self.log.withIndentLevel(1):
                for p0, p1 in lib.prange(0, se[s].naux, 6):
                    self.log.debugv("%12.6f " * (p1-p0), *se[s].energy[p0:p1])
            self.log.info("Number of auxiliaries built:  %s", se[s].naux)
            self.log.timing("Time for self-energy build:  %s", time_string(timer() - t0))

        return tuple(se)


    def run_diis(self, se, gf, diis, se_prev=None):
        """Update the self-energy using DIIS and apply damping.
        """

        t = np.array([(
            s.get_occupied().moment(range(2*self.opts.nmom_lanczos+2)),
            s.get_virtual().moment(range(2*self.opts.nmom_lanczos+2)),
        ) for s in se])

        self.log.debug("Summed trace of moments:")
        for s, spin in self.spins:
            self.log.debug(" > Initial (%s) :  %.5g", spin, np.einsum("onii->", t[s]))

        if self.opts.damping and se_prev:
            t = np.array([(
                s.get_occupied().moment(range(2*self.opts.nmom_lanczos+2)),
                s.get_virtual().moment(range(2*self.opts.nmom_lanczos+2)),
            ) for s in se_prev])

            t *= (1.0 - self.opts.damping)
            t += self.opts.damping * t_prev

            for s, spin in self.spins:
                self.log.debug(" > Damping (%s) :  %.5g", spin, np.einsum("onii->", t[s]))

        t = diis.update(t)

        for s, spin in self.spins:
            self.log.debug(" > DIIS (%s)    :  %.5g", spin, np.einsum("onii->", t[s]))

        se_occ, se_vir = [None, None], [None, None]
        for s, spin in self.spins:
            se_occ[s] = self._build_se_from_moments(t[s, 0], chempot=se[s].chempot)
            se_vir[s] = self._build_se_from_moments(t[s, 1], chempot=se[s].chempot)

        se = self._combine_se(se_occ, se_vir, gf=gf)

        return se


    def build_init_greens_function(self):
        """Build the mean-field Green's function.
        """

        gf = [None, None]
        for s, spin in self.spins:
            chempot = 0.5 * (
                    + self.mo_energy[s][self.mo_occ[s] > 0].max()
                    + self.mo_energy[s][self.mo_occ[s] == 0].min()
            )

            e = self.mo_energy[s][self.act[s]]
            v = np.eye(self.nact[s])
            gf[s] = agf2.GreensFunction(e, v, chempot=chempot)
            nelec = np.trace(gf[s].make_rdm1(occupancy=1))

            self.log.debug("Built G0 (%s) with μ(MF) = %.5g", spin, chempot)
            self.log.info("Number of active electrons in G0 (%s):  %s", spin, nelec)

        return tuple(gf)


    def solve_dyson(self, se=None, gf=None, fock=None):
        """Solve the Dyson equation.
        """

        se = se or self.se
        gf = gf or self.gf

        if fock is None:
            fock = self.get_fock(gf=gf, with_frozen=False)

        w_α, v_α = super().solve_dyson(se=se[0], gf=gf[0], fock=fock[0])
        w_β, v_β = super().solve_dyson(se=se[1], gf=gf[1], fock=fock[1])

        return ((w_α, w_β), (v_α, v_β))


    def fock_loop(self, gf=None, se=None, fock=None, project_gf=True, return_fock=False):
        """Do the self-consistent Fock loop.
        """

        t0 = timer()
        gf = gf or self.gf
        se = se or self.se

        nα = self.nocc[0] - self.frozen[0][0]
        nβ = self.nocc[1] - self.frozen[1][0]
        nelec = (nα, nβ)
        if fock is None:
            fock = self.get_fock(gf=gf, with_frozen=False)

        if not self.opts.fock_loop:
            # Just solve Dyson eqn
            self.log.info("Solving Dyson equation")
            w, v = self.solve_dyson(se=se, gf=gf, fock=fock)
            gf = [None, None]

            for s, spin in self.spins:
                if project_gf:
                    gf[s] = agf2.GreensFunction(w[s], v[s][:self.nact[s]], chempot=se[s].chempot)
                else:
                    gf[s] = agf2.GreensFunction(w[s], v[s], chempot=se[s].chempot)
                gf[s].chempot = se[s].chempot = agf2.chempot.binsearch_chempot(
                        (w[s], v[s]),
                        self.nact[s],
                        nelec[s],
                        occupancy=1,
                )[0]

            if return_fock:
                return gf, se, True, fock
            else:
                return gf, se, True

        self.log.info('Fock loop')
        self.log.info('*********')

        diis = self.DIIS(space=self.opts.fock_diis_space, min_space=self.opts.fock_diis_min_space)
        converged = False

        for s, spin in self.spins:
            self.log.debug("Target number of electrons (%s):  %d", spin, nelec[s])
        self.log.infov('%12s %9s %12s %12s', 'Iteration', 'Cycles', 'Nelec error', 'DM change')

        se = list(se)
        gf = list(gf)
        fock_prev = tuple(np.zeros_like(f) for f in fock)
        rdm1_prev = tuple(np.zeros_like(f) for f in fock)

        for niter1 in range(1, self.opts.max_cycle_outer+1):
            for s, spin in self.spins:
                se[s], opt = agf2.chempot.minimize_chempot(
                        se[s], fock[s], nelec[s],
                        occupancy=1,
                        tol=self.opts.conv_tol_nelec*self.opts.conv_tol_nelec_factor,
                        maxiter=self.opts.max_cycle_inner,
                )

            for niter2 in range(1, self.opts.max_cycle_inner+1):
                w, v = self.solve_dyson(se=se, fock=fock)

                nerr = 0.0
                for s, spin in self.spins:
                    se[s].chempot, nerr_s = agf2.chempot.binsearch_chempot(
                            (w[s], v[s]),
                            self.nact[s],
                            nelec[s],
                            occupancy=1,
                    )
                    nerr += nerr_s
                    gf[s] = agf2.GreensFunction(w[s], v[s][:self.nact[s]], chempot=se[s].chempot)

                fock_prev = tuple(f.copy() for f in fock)
                fock = self.get_fock(gf=gf, with_frozen=False)

                if self.opts.fock_damping:
                    fock = tuple(
                            f * (1.0 - self.opts.fock_damping) + fp * self.opts.fock_damping
                            for f, fp in zip(fock, fock_prev)
                    )

                rdm1 = self.make_rdm1(gf=gf, with_frozen=False)
                fock = tuple(diis.update(np.array(fock), xerr=None))

                derr = sum(
                    np.max(np.absolute(rdm1[s] - rdm1_prev[s]))
                    for s, spin in self.spins
                )
                rdm1_prev = tuple(d.copy() for d in rdm1)

                self.log.debugv(
                        "%12s %9s %12.4g %12.4g",
                        '(*) %d'%niter1, '-> %d'%niter2, nerr, derr,
                )

                if abs(derr) < self.opts.conv_tol_rdm1:
                    break

            self.log.infov("%12d %9d %12.4g %12.4g", niter1, niter2, nerr, derr)

            if abs(derr) < self.opts.conv_tol_rdm1 and abs(nerr) < self.opts.conv_tol_nelec:
                converged = True
                break

        if not project_gf:
            for s, spin in self.spins:
                gf[s] = agf2.GreensFunction(w[s], v[s], chempot=se[s].chempot)

        (self.log.info if converged else self.log.warning)("Converged = %r", converged)
        for s, spin in self.spins:
            self.log.info("μ (%s) = %.9g", spin, se[s].chempot)
        self.log.timing('Time for fock loop:  %s', time_string(timer() - t0))
        for s, spin in self.spins:
            self.log.debugv("QMO energies (%s):", spin)
            for p0, p1 in lib.prange(0, gf[s].naux, 6):
                self.log.debugv("%12.6f " * (p1-p0), *gf[s].energy[p0:p1])

        if not return_fock:
            return tuple(gf), tuple(se), converged
        else:
            return tuple(gf), tuple(se), converged, fock


    def get_fock(self, gf=None, rdm1=None, with_frozen=True, fock_last=None):
        """Get the Fock matrix including all frozen contributions.
        """

        if self.opts.fock_basis.lower() == 'ao':
            raise NotImplementedError
        elif self.opts.fock_basis.lower() == "adc":
            raise NotImplementedError

        t0 = timer()

        gf = gf or self.gf
        if rdm1 is None:
            rdm1 = self.make_rdm1(gf=gf, with_frozen=False)
        eri = self.eri

        vj = [[np.zeros((n, n)) for n in self.nact] for s, spin in self.spins]
        vk = [[np.zeros((n, n)) for n in self.nact] for s, spin in self.spins]

        for s1, spin1 in self.spins:
            s2 = (s1 + 1) % 2

            if getattr(self.mf, 'with_df', None) is None:
                for i0, i1 in mpi_helper.prange(0, self.nmo[s1], self.nmo[s1]):
                    i = slice(i0, i1)

                    vj[s1][s1][i] += lib.einsum('ijkl,kl->ij', eri[s1][s1][i], rdm1[s1])
                    vk[s1][s1][i] += lib.einsum('iklj,kl->ij', eri[s1][s1][i], rdm1[s1])

                    vj[s1][s2][i] += lib.einsum('ijkl,kl->ij', eri[s1][s2][i], rdm1[s2])

            else:
                naux = eri[s1].shape[0]
                for q0, q1 in mpi_helper.prange(0, naux, naux):
                    q = slice(q0, q1)

                    tmp = lib.einsum('Qik,kl->Qil', eri[s1][q], rdm1[s1])
                    vj[s1][s1] += lib.einsum('Qij,Qkk->ij', eri[s1][q], tmp)
                    vk[s1][s1] += lib.einsum('Qlj,Qil->ij', eri[s1][q], tmp)

                    tmp = lib.einsum('Qkl,kl->Q', eri[s2][q], rdm1[s2])
                    vj[s1][s2] += lib.einsum('Qij,Q->ij', eri[s1][q], tmp)

            mpi_helper.barrier()
            mpi_helper.allreduce_safe_inplace(vj[s1][s1])
            mpi_helper.allreduce_safe_inplace(vj[s1][s2])
            mpi_helper.allreduce_safe_inplace(vk[s1][s1])
            mpi_helper.allreduce_safe_inplace(vk[s1][s2])

        fock = [None, None]
        for s1, spin1 in self.spins:
            s2 = (s1 + 1) % 2

            fock[s1] = vj[s1][s1] + vj[s1][s2] - vk[s1][s1]
            if self.veff[s1] is not None:
                fock[s1] += self.veff[s1][self.act[s1], self.act[s1]]
            fock[s1] += self.h1e[s1][self.act[s1], self.act[s1]]

            if with_frozen:
                fock_ref = np.diag(self.mo_energy[s1])
                fock_ref[self.act[s1], self.act[s1]] = fock[s1]
                fock[s1] = fock_ref

        return fock


    def make_rdm1(self, gf=None, with_frozen=True):
        """Get the 1RDM.
        """

        gf = gf or self.gf
        rdm1 = [gf[0].make_rdm1(occupancy=1), gf[1].make_rdm1(occupancy=1)]

        if with_frozen:
            sc = tuple(
                np.dot(self.mf.get_ovlp(), self.mo_coeff[s])
                for s, spin in self.spins
            )
            rdm1_ref = list(self.mf.make_rdm1(self.mo_coeff, self.mo_occ))

            for s, spin in self.spins:
                rdm1_ref_s = np.linalg.multi_dot((sc[s].T, rdm1_ref[s], sc[s]))
                rdm1_ref_s[self.act[s], self.act[s]] = rdm1_ref_s
                rdm1[s] = rdm1_ref_s

        return tuple(rdm1)


    make_rdm2 = lambda *args, **kwargs: NotImplemented


    def energy_mp2(self, mo_energy=None, se=None, flip=False):
        """Calculate the MP2 energy.
        """

        se = se or self.se
        mo_energy = mo_energy if mo_energy is not None else self.mo_energy

        e_mp2 = 0.0

        for s, spin in self.spins:
            if not flip:
                mo = mo_energy[s][self.act[s]] < se[s].chempot
                se_part = se[s].get_virtual()
            else:
                mo = mo_energy[s][self.act[s]] >= se[s].chempot
                se_part = se[s].get_occupied()

            v_se = se_part.coupling[mo]
            Δ = lib.direct_sum('x,k->xk', mo_energy[s][self.act[s]][mo], -se_part.energy)

            e_mp2 += np.sum(v_se * v_se.conj() / Δ).real

        return e_mp2


    def energy_1body(self, gf=None, e_nuc=None):
        """Calculate the one-body energy.
        """

        rdm1 = self.make_rdm1(gf=gf, with_frozen=True)
        fock = self.get_fock(gf=gf, with_frozen=True)
        h1e = self.h1e

        e_1b = 0.0

        for s, spin in self.spins:
            e_1b += 0.5 * np.sum(rdm1[s] * (h1e[s] + fock[s]))

        e_1b += e_nuc if e_nuc is not None else self.e_nuc

        return e_1b


    def energy_2body(self, gf=None, se=None, flip=False):
        """Calculate the two-body energy.
        """

        gf = gf or self.gf
        se = se or self.se

        e_2b = 0.0

        for s, spin in self.spins:
            if not flip:
                gf_part = gf[s].get_occupied()
                se_part = se[s].get_virtual()
            else:
                gf_part = gf[s].get_virtual()
                se_part = se[s].get_occupied()

            for i in range(gf_part.naux):
                v_gf = gf_part.coupling[:, i]
                v_se = se_part.coupling
                v_dyson = v_se * v_gf[:, None]
                Δ = gf_part.energy[i] - se_part.energy

                e_2b += np.sum(np.dot(v_dyson / Δ[None], v_dyson.T.conj())).real

        e_2b *= 2.0

        return e_2b


    population_analysis = lambda *args, **kwargs: NotImplemented

    dip_moment = lambda *args, **kwargs: NotImplemented

    dump_cube = lambda *args, **kwargs: NotImplemented


    def print_excitations(self, gf=None, title="Excitations"):
        """Print the excitations and some information on their character.
        """

        gf = gf or self.gf

        for s, spin in self.spins:
            super().print_excitations(gf=gf[s], title="%s (%s)" % (title, spin))


    def _convergence_checks(self, se=None, se_prev=None, e_prev=None):
        """Return a list of [energy, 0th moment, 1st moment] changes
        between iterations to check convergence progress.
        """

        se = se or self.se
        e_prev = e_prev or self.mf.e_tot

        deltas = [0.0, 0.0, 0.0]

        for s, spin in self.spins:
            sp = se_prev if se_prev is None else se_prev[s]
            ds = super()._convergence_checks(se=se[s], se_prev=sp, e_prev=e_prev)

            deltas[0] += ds[0]
            deltas[1] += ds[1]
            deltas[2] = max(ds[2], deltas[2])

        return deltas


    @property
    def spins(self):
        """Enumerate spin channels."""
        return [(0, "α"), (1, "β")]


    @property
    def qmo_energy(self):
        return tuple(self.g.energy for g in self.gf)

    @property
    def qmo_coeff(self):
        return tuple(np.dot(mo, g) for mo, g in zip(self.mo_coeff, self.gf))

    @property
    def qmo_occ(self):
        return tuple(
                np.concatenate([
                    np.linalg.norm(self.gf.get_occupied().coupling, axis=0)**2,
                    np.zeros_like(self.gf.get_virtual().energy),
                ], axis=0) for g in self.g
        )

    dyson_orbitals = qmo_coeff


    @property
    def frozen(self):
        return getattr(self, '_frozen', ((0, 0), (0, 0)))
    @frozen.setter
    def frozen(self, frozen):
        assert isinstance(frozen, tuple)
        if frozen == (None, None) or frozen == (0, 0) or frozen == ((0, 0), (0, 0)):
            self._frozen = ((0, 0), (0, 0))
        elif isinstance(frozen[0], int):
            self._frozen = ((frozen, 0), (frozen, 0))
        else:
            self._frozen = frozen

    @property
    def e_tot(self): return self.e_1b + self.e_2b
    @property
    def e_nuc(self): return self.mol.energy_nuc()
    @property
    def e_mf(self): return self.mf.e_tot
    @property
    def e_corr(self): return self.e_tot - self.e_mf

    @property
    def e_ip(self): return -max(g.get_occupied().energy.max() for g in self.gf)
    @property
    def e_ea(self): return min(g.get_virtual().energy.min() for g in self.gf)

    @property
    def nmo(self): return self._nmo or tuple(e.size for e in self.mo_energy)
    @property
    def nocc(self): return self._nocc or tuple(np.sum(o > 0) for o in self.mo_occ)
    @property
    def nvir(self): return tuple(n - o for n, o in zip(self.nmo, self.nocc))
    @property
    def nfroz(self): return tuple(f[0] + f[1] for f in self.frozen)
    @property
    def nact(self): return tuple(n - f for n, f in zip(self.nmo, self.nfroz))
