# Standard library
import dataclasses
import logging
import ctypes
import sys

# NumPy
import numpy as np

# PySCF
from pyscf import lib, ao2mo, agf2
from pyscf.agf2 import mpi_helper
from pyscf.pbc import tools, df
from pyscf.pbc.lib import kpts_helper

# Vayesta
import vayesta
from vayesta import libs
from vayesta.eagf2.ragf2 import RAGF2Options, RAGF2, DIIS
from vayesta.core.util import time_string, OptionsBase, NotSet
from vayesta.misc import gdf

# Timings
if mpi_helper.mpi == None:
    from timeit import default_timer as timer
else:
    timer = mpi_helper.mpi.Wtime

libeagf2 = getattr(libs, 'libeagf2')

#FIXME broken with mf.exxdiv = 'ewald'?


@dataclasses.dataclass
class KRAGF2Options(RAGF2Options):
    ''' Options for KRAGF2 calculations.
    '''

    # --- Additional k-space parameters
    kptlist: 'typing.Any' = None    # list of k-points for self-consistency TODO test
    direct: bool = True             # do not store full 4c ERI tensor
    keep_exxdiv: bool = False       # keep exxdiv when building Fock matrix


class kDIIS(DIIS):
    def update(self, x, xerr=None):
        shapes = [y.shape for y in x]
        sizes = [y.size for y in x]
        assert all([np.prod(y) == z for y,z in zip(shapes, sizes)])

        locs = [0,]
        for s in sizes:
            locs.append(locs[-1] + s)

        xout = DIIS.update(self, np.concatenate([y.ravel() for y in x]))
        xout = [xout[i:j].reshape(s) for i,j,s in zip(locs[:-1], locs[1:], shapes)]

        return xout


def _make_mo_eris(self, mo_coeff=None):
    ''' Get the three-center ERIs
    '''

    with_df = self.mf.with_df
    if mo_coeff is None:
        mo_coeff = self.mo_coeff
    nmo = self.nmo  #TODO support frozen?
    npair = nmo * (nmo+1) // 2

    kpts, nkpts = self.kpts, self.nkpts
    khelper = self.khelper
    kconserv = khelper.kconserv

    if kpts_helper.gamma_point(kpts):
        dtype = np.float64
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *[x.dtype for x in mo_coeff])

    eri = np.empty((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=dtype)

    for kpqr in mpi_helper.nrange(nkpts**3):
        kpq, kr = divmod(kpqr, nkpts)
        kp, kq = divmod(kpq, nkpts)
        ks = kconserv[kp,kq,kr]

        coeffs = [mo_coeff[k] for k in (kp, kq, kr, ks)]
        kijkl = kpts[[kp,kq,kr,ks]]

        eri_kpt = with_df.ao2mo(coeffs, kijkl, compact=False)
        eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)

        if dtype is np.float64:
            eri_kpt = eri_kpt.real

        eri[kp,kq,kr] = eri_kpt / nkpts

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(eri)

    return eri


def _fao2mo(eri, cp, cq, dtype, out=None):
    ''' DF ao2mo
    '''

    npq, cpq, spq = ao2mo.incore._conc_mos(cp, cq, compact=False)[1:]
    sym = dict(aosym='s1', mosym='s1')
    naux = eri.shape[0]

    if out is None:
        out = np.zeros((naux*cp.shape[1]*cq.shape[1]), dtype=dtype)
    out = out.reshape(naux, cp.shape[1]*cq.shape[1])

    if dtype == np.float64 or dtype == float:
        out = ao2mo._ao2mo.nr_e2(eri, cpq, spq, out=out, **sym)
    else:
        cpq = np.asarray(cpq, dtype=np.complex128)
        out = ao2mo._ao2mo.r_e2(eri, cpq, spq, [], None, out=out)

    out = out.reshape(naux, cp.shape[-1], cq.shape[-1])

    return out


def _make_mo_eris_direct(self, mo_coeff=None):
    ''' Get the four-center ERIs
    '''

    cell = self.cell
    with_df = self.mf.with_df
    nmo = self.nmo

    kpts, nkpts = self.kpts, self.nkpts
    kconserv = tools.get_kconserv(cell, kpts)
    ngrids = with_df.auxcell.nao_nr()

    if mo_coeff is None:
        mo_coeff = self.mo_coeff

    if kpts_helper.gamma_point(kpts):
        dtype = np.complex128
    else:
        dtype = np.float64
    dtype = np.result_type(dtype, *[x.dtype for x in mo_coeff])

    if not isinstance(with_df, (df.GDF, gdf.GDF)):
        raise NotImplementedError('AGF2 with direct=True for density '
                                  'fitting scheme which are not GDF.')

    if cell.dimension != 3:
        raise NotImplementedError('GDF for cell dimension < 3 is not '
                                  'positive definite, not supported '
                                  'in AGF2 with direct=True.')

    qij = np.zeros((nkpts, nkpts, ngrids, self.nmo, self.nmo), dtype=dtype)

    for kab in mpi_helper.nrange(nkpts**2):
        ka, kb = divmod(kab, nkpts)
        kpta_kptb = np.array((kpts[ka], kpts[kb]))
        ci = mo_coeff[ka]
        cj = mo_coeff[kb]

        p1 = 0
        for qij_r, qij_i, sign in with_df.sr_loop(kpta_kptb, compact=False):
            assert sign == 1
            p0, p1 = p1, p1 + qij_r.shape[0] 
            tmp = (qij_r + qij_i * 1j)
            qij[ka,kb,p0:p1] = _fao2mo(tmp, ci, cj, dtype, out=qij[ka,kb,p0:p1]) / np.sqrt(nkpts)

        tmp = None

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(qij)

    return qij


class KRAGF2(RAGF2):

    Options = KRAGF2Options
    DIIS = kDIIS

    def __init__(self, mf, **kwargs):
        ''' 
        Restricted k-space auxiliary second-order Green's function perturbation theory
        '''

        # k-space objects
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)

        with lib.temporary_env(mf, mol=mf.cell):
            RAGF2.__init__(self, mf, **kwargs)


    def _active_slices(self, nmo, frozen):
        ''' Get slices for frozen occupied, active, frozen virtual spaces
        '''

        slices = [
                RAGF2._active_slices(self, nmo, frozen[i]) for i in range(self.nkpts)
        ]

        focc, fvir, act = zip(*slices)

        return focc, fvir, act


    def _get_h1e(self, mo_coeffs):
        ''' Get the core Hamiltonian
        '''

        h1e_ao = self.mf.get_hcore()
        h1e = []

        for i in range(self.nkpts):
            mo_coeff = mo_coeffs[i]
            h1e.append(np.linalg.multi_dot((mo_coeff.T.conj(), h1e_ao[i], mo_coeff)))

        return h1e


    def _get_frozen_veff(self, mo_coeffs):
        ''' Get Veff due to the frozen density
        '''

        if self.veff is not None:
            self.log.info("Veff due to frozen density passed by kwarg")
        elif not all([x == (0, 0) for x in self.frozen]):
            self.log.info("Calculating Veff due to frozen density")
            self.veff = []
            for i in range(self.nkpts):
                mo_coeff = mo_coeffs[i]
                c_focc = mo_coeff[:, self.focc[i]]
                dm_froz = np.dot(c_focc, c_focc.T.conj()) * 2
                v = self.mf.get_veff(dm=dm_froz)
                v = np.linalg.multi_dot((mo_coeff.T.conj(), v, mo_coeff))
                self.veff.append(v)

        return self.veff


    def _print_sizes(self):
        ''' Print the system sizes
        '''

        nmo = self.nmo

        for i, (nocc, nvir, frozen, nact, nfroz) in enumerate(zip(
                self.nocc, self.nvir, self.frozen, self.nact, self.nfroz,
        )):
            delim = (10-len(str(i)))*" "
            self.log.info("kpt %d %s %6s %6s %6s", i, delim, 'active', 'frozen', 'total')
            self.log.info("Occupied MOs:  %6d %6d %6d", nocc-frozen[0], frozen[0], nocc)
            self.log.info("Virtual MOs:   %6d %6d %6d", nvir-frozen[1], frozen[1], nvir)
            self.log.info("General MOs:   %6d %6d %6d", nact,           nfroz,     nmo)


    def _build_moments(self, ei, ej, ea, xija, xjia, os_factor=None, ss_factor=None):
        ''' Build the occupied or virtual self-energy moments for a single kpt
        '''

        os_factor = os_factor or self.opts.os_factor
        ss_factor = ss_factor or self.opts.ss_factor
        facs = {'os_factor': os_factor, 'ss_factor': ss_factor}

        if isinstance(xija, tuple):
            xija = lib.einsum('Qij,Qkl->ijkl', xija[0], xija[1])
        if isinstance(xjia, tuple):
            xjia = lib.einsum('Qij,Qkl->ijkl', xjia[0], xjia[1])
        elif xjia is None:
            xjia = xija

        nphys = xija.shape[0]
        t = np.zeros((2*self.opts.nmom_lanczos+2, nphys, nphys), dtype=np.complex128)  #TODO can I be real?

        if self.opts.nmom_lanczos == 0 and not self.opts.diagonal_se:
            xija = np.asarray(xija, order='C', dtype=np.complex128)
            xjia = np.asarray(xjia, order='C', dtype=np.complex128)
            ei = np.asarray(ei, order='C')
            ej = np.asarray(ej, order='C')
            ea = np.asarray(ea, order='C')
            libeagf2.KAGF2ee_vv_vev_islice(
                    xija.ctypes.data_as(ctypes.c_void_p),
                    xjia.ctypes.data_as(ctypes.c_void_p),
                    ei.ctypes.data_as(ctypes.c_void_p),
                    ej.ctypes.data_as(ctypes.c_void_p),
                    ea.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_double(os_factor),
                    ctypes.c_double(ss_factor),
                    ctypes.c_int(nphys),
                    ctypes.c_int(ei.size),
                    ctypes.c_int(ej.size),
                    ctypes.c_int(ea.size),
                    ctypes.c_int(0),
                    ctypes.c_int(ei.size),
                    t[0].ctypes.data_as(ctypes.c_void_p),
                    t[1].ctypes.data_as(ctypes.c_void_p),
            )
        else:
            dtype = xija.dtype
            ija = lib.direct_sum('i+j-a->ija', ei, ej, ea)
            fpos = os_factor + ss_factor
            fneg = -ss_factor

            for n in range(2*self.opts.nmom_lanczos+2):
                xija_n = xija.reshape(nphys, -1) * np.ravel(ija**n)[None]
                if not self.opts.diagonal_se:
                    t[n] += (
                        + fpos * np.dot(xija_n, xija.reshape(nphys, -1).T.conj())
                        + fneg * np.dot(xija_n, xjia.reshape(nphys, -1).T.conj())
                    )
                else:
                    t[n][np.diag_indices_from(t[n])] += (
                        + fpos * np.sum(xija_n * xija.conj(), axis=1)
                        + fneg * np.sum(xija_n * xjia.conj(), axis=1)
                    )

        return t


    def ao2mo(self):
        ''' Get the ERIs in MO basis
        '''

        t0 = timer()

        if self.mf.with_df._cderi is None:
            self.log.info("Building DF object")
            self.mf.with_df.build()

        mo_coeff_act = [mo[:, act] for mo, act in zip(self.mo_coeff, self.act)]

        if self.opts.direct:
            self.log.info("ERIs will be direct")
            self.eri = _make_mo_eris_direct(self, mo_coeff=mo_coeff_act)
        else:
            self.log.info("ERIs will be four-centered")
            self.eri = _make_mo_eris(self, mo_coeff=mo_coeff_act)

        # --- Remove non-commutative inconsistency in hybrid parallel regimes
        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(self.eri)
        self.eri /= mpi_helper.size

        self.log.timing("Time for AO->MO:  %s", time_string(timer() - t0))

        return self.eri


    def build_self_energy(self, gf, se_prev=None, kptlist=None):
        ''' Build the self-energy using a given Green's function for all kpts
        '''

        t0 = timer()
        self.log.info("Building the self-energy")
        self.log.info("************************")

        kconserv = self.khelper.kconserv

        nmom = self.opts.nmom_lanczos
        dtype = np.result_type(self.eri.dtype, *[g.coupling.dtype for g in gf])
        t_occ = [np.zeros((2*nmom+2, nact, nact), dtype=dtype) for nact in self.nact]
        t_vir = [np.zeros((2*nmom+2, nact, nact), dtype=dtype) for nact in self.nact]

        if kptlist is None:
            kptlist = self.opts.kptlist
            if kptlist is None or se_prev is None:
                kptlist = list(range(self.nkpts))

        for kabd in mpi_helper.nrange(len(kptlist) * self.nkpts**2):
            t1 = timer()
            kab, kd = divmod(kabd, self.nkpts)
            a, kb = divmod(kab, self.nkpts)
            ka = kptlist[a]
            kc = kconserv[ka, kb, kd]

            ca_x = np.eye(self.nact[ka])
            cb_o = gf[kb].get_occupied().coupling
            cc_o = gf[kc].get_occupied().coupling
            cd_o = gf[kd].get_occupied().coupling
            cb_v = gf[kb].get_virtual().coupling
            cc_v = gf[kc].get_virtual().coupling
            cd_v = gf[kd].get_virtual().coupling

            eb_o = gf[kb].get_occupied().energy
            ec_o = gf[kc].get_occupied().energy
            ed_o = gf[kd].get_occupied().energy
            eb_v = gf[kb].get_virtual().energy
            ec_v = gf[kc].get_virtual().energy
            ed_v = gf[kd].get_virtual().energy

            if self.opts.non_dyson:
                xa_o = slice(None, self.nocc[ka]-self.frozen[ka][0])
                xa_v = slice(self.nocc[ka]-self.frozen[ka][0], None)
            else:
                xa_o = xa_v = slice(None)

            if self.eri.ndim == 7:
                #TODO xija == xjia.swapaxes(1,2) ?
                xija = lib.einsum('pqrs,pi,qj,rk,sl->ijkl',
                        self.eri[ka, kb, kc],
                        ca_x[:, xa_o].conj(), cb_o, cc_o.conj(), cd_v,
                )
                xjia = lib.einsum('pqrs,pi,qj,rk,sl->ijkl',
                        self.eri[ka, kc, kb],
                        ca_x[:, xa_o].conj(), cc_o, cb_o.conj(), cd_v,
                )
                xabi = lib.einsum('pqrs,pi,qj,rk,sl->ijkl',
                        self.eri[ka, kb, kc],
                        ca_x[:, xa_v].conj(), cb_v, cc_v.conj(), cd_o,
                )
                xbai = lib.einsum('pqrs,pi,qj,rk,sl->ijkl',
                        self.eri[ka, kc, kb],
                        ca_x[:, xa_v].conj(), cc_v, cb_v.conj(), cd_o,
                )
            else:
                qxi = _fao2mo(self.eri[ka, kb], ca_x[:, xa_o], cb_o, dtype)
                qxj = _fao2mo(self.eri[ka, kc], ca_x[:, xa_o], cc_o, dtype)
                qja = _fao2mo(self.eri[kc, kd], cc_o, cd_v, dtype)
                qia = _fao2mo(self.eri[kb, kd], cb_o, cd_v, dtype)
                qxa = _fao2mo(self.eri[ka, kb], ca_x[:, xa_v], cb_v, dtype)
                qxb = _fao2mo(self.eri[ka, kc], ca_x[:, xa_v], cc_v, dtype)
                #TODO yes?
                #qbi = qja.swapaxes(qja.ndim-1, qja.ndim-2)
                #qai = qia.swapaxes(qia.ndim-1, qia.ndim-2)
                qbi = _fao2mo(self.eri[kc, kd], cc_v, cd_o, dtype)
                qai = _fao2mo(self.eri[kb, kd], cb_v, cd_o, dtype)
                xija = (qxi, qja)
                xjia = (qxj, qia)
                xabi = (qxa, qbi)
                xbai = (qxb, qai)

            self.log.timingv(
                    "Time for MO->QMO (%d,%d,%d,%d):  %s",
                    ka, kb, kc, kd, time_string(timer() - t1),
            )
            t1 = timer()

            t_occ[ka][:, xa_o, xa_o] += self._build_moments(eb_o, ec_o, ed_v, xija, xjia)
            t_vir[ka][:, xa_v, xa_v] += self._build_moments(eb_v, ec_v, ed_o, xabi, xbai)

            del xija, xjia, xabi, xbai

        kptlist_inv = np.ones((self.nkpts,), dtype=bool)
        kptlist_inv[kptlist] = False
        for ka, do in enumerate(kptlist_inv):
            if do:
                t_occ[ka] = se_prev[ka].get_occupied().moment(range(2*nmom+2), squeeze=False)
                t_vir[ka] = se_prev[ka].get_virtual().moment(range(2*nmom+2), squeeze=False)

        for i in range(self.nkpts):
            mpi_helper.barrier()
            mpi_helper.allreduce_safe_inplace(t_occ[i])
            mpi_helper.allreduce_safe_inplace(t_vir[i])

        if not all([np.allclose(t, t.swapaxes(1, 2).conj()) for t in t_occ]):
            error = max([np.max(np.abs(t - t.swapaxes(1, 2).conj())) for t in t_occ])
            self.log.debug("Error in hermiticity of moments:  %.3g", error)
        if not all([np.allclose(t, t.swapaxes(1, 2).conj()) for t in t_vir]):
            error = max([np.max(np.abs(t - t.swapaxes(1, 2).conj())) for t in t_vir])
            self.log.debug("Error in hermiticity of moments:  %.3g", error)

        t_occ = [0.5 * (t + t.swapaxes(1, 2).conj()) for t in t_occ]
        t_vir = [0.5 * (t + t.swapaxes(1, 2).conj()) for t in t_vir]


        t0 = timer()
        nqocc = [g.get_occupied().naux for g in gf]
        nqvir = [g.get_virtual().naux for g in gf]


        # --- Occupied

        self.log.info("Occupied self-energy:")
        self.log.changeIndentLevel(1)
        self.log.info("Number of ija:  %s", np.einsum('i,j,k->', nqocc, nqocc, nqvir))

        w = np.linalg.eigvalsh([t[0] for t in t_occ])
        wmin = min(x.min() for x in w)
        if wmin < 1e-8:
            (self.log.critical if wmin<0 else self.log.warning)('Smallest eigenvalue:  %.6g', wmin)

        se_occ = []
        for i in range(self.nkpts):
            se_occ.append(self._build_se_from_moments(t_occ[i], chempot=gf[i].chempot))

        self.log.info("Build %d occupied auxiliaries", sum([x.naux for x in se_occ]))
        self.log.changeIndentLevel(-1)


        # --- Virtual

        self.log.info("Virtual self-energy:")
        self.log.changeIndentLevel(1)
        self.log.info("Number of abi:  %s", np.einsum('i,j,k->', nqvir, nqvir, nqocc))

        w = np.linalg.eigvalsh([t[0] for t in t_vir])
        wmin = min(x.min() for x in w)
        if wmin < 1e-8:
            (self.log.critical if wmin<0 else self.log.warning)('Smallest eigenvalue:  %.6g', wmin)

        se_vir = []
        for i in range(self.nkpts):
            se_vir.append(self._build_se_from_moments(t_vir[i], chempot=gf[i].chempot))

        self.log.info("Build %d virtual auxiliaries", sum([x.naux for x in se_vir]))
        self.log.changeIndentLevel(-1)


        se = [self._combine_se(o, v, gf=g) for o, v, g in zip(se_occ, se_vir, gf)]

        self.log.info("Number of auxiliaries built:  %s", sum([x.naux for x in se]))
        self.log.timing("Time for self-energy build:  %s", time_string(timer() - t0))

        return se


    def run_diis(self, se, gf, diis, se_prev=None):
        ''' Update the self-energy using DIIS and apply damping
        '''

        t = [np.array((
            s.get_occupied().moment(range(2*self.opts.nmom_lanczos+2)),
            s.get_virtual().moment(range(2*self.opts.nmom_lanczos+2)),
        )) for s in se]

        if self.opts.damping and se_prev:
            t_prev = [np.array((
                s.get_occupied().moment(range(2*self.opts.nmom_lanczos+2)),
                s.get_virtual().moment(range(2*self.opts.nmom_lanczos+2)),
            )) for s in se_prev]

            for i in range(len(t)):
                t[i] *= (1.0 - self.opts.damping)
                t[i] += self.opts.damping * t_prev[i]

        t = diis.update(t)

        se_occ = []
        se_vir = []
        se_out = []
        for i in range(self.nkpts):
            se_occ.append(self._build_se_from_moments(t[i][0], chempot=se[i].chempot))
            se_vir.append(self._build_se_from_moments(t[i][1], chempot=se[i].chempot))
            se_out.append(self._combine_se(se_occ[-1], se_vir[-1], gf=gf[i]))

        return se_out


    def build_init_greens_function(self):
        ''' Build the mean-field Green's function
        '''

        gf = []

        for i in range(self.nkpts):
            chempot = 0.5 * (
                    + self.mo_energy[i][self.mo_occ[i] > 0].max()
                    + self.mo_energy[i][self.mo_occ[i] == 0].min()
            )

            e = self.mo_energy[i][self.act[i]]
            v = np.eye(self.nact[i])
            gf.append(agf2.GreensFunction(e, v, chempot=chempot))

        self.log.info(
                "Number of active electrons in G0:  %s",
                sum([np.trace(g.make_rdm1()) for g in gf]),
        )

        return gf


    def solve_dyson(self, se=None, gf=None, fock=None):
        ''' Solve the Dyson equation
        '''

        se = se or self.se
        gf = gf or self.gf

        if fock is None:
            fock = self.get_fock(gf=gf, with_frozen=False)
        else:
            fock = [f[a, a] for f, a in zip(fock, self.act)]

        ws = []
        vs = []
        for i in range(self.nkpts):
            e = se[i].energy
            v = se[i].coupling

            f_ext = np.block([[fock[i], v], [v.T.conj(), np.diag(e)]])
            w, v = np.linalg.eigh(f_ext)

            ws.append(w)
            vs.append(v)

        return ws, vs


    def fock_loop(self, gf=None, se=None, fock=None):
        ''' Do the self-consistent Fock loop
        '''

        t0 = timer()
        gf = gf or self.gf
        se = se or self.se

        nelec = [(o - f[0]) * 2 for o,f in zip(self.nocc, self.frozen)]
        if fock is None:
            fock = self.get_fock(gf=gf, with_frozen=False)

        if not self.opts.fock_loop:
            # Just solve Dyson eqn
            w, v = self.solve_dyson(se=se, gf=gf, fock=fock)
            gf = []
            for i in range(self.nkpts):
                nact = self.nact[i]
                gf.append(agf2.GreensFunction(w[i], v[i][:nact], chempot=se[i].chempot))
                gf[i].chempot = se[i].chempot = \
                        agf2.chempot.binsearch_chempot((w[i], v[i]), nact, nelec[i])[0]
            return gf, se

        self.log.info("Fock loop")
        self.log.info("*********")

        diis = self.DIIS(space=self.opts.fock_diis_space, min_space=self.opts.fock_diis_min_space)
        rdm1_prev = [np.zeros_like(f) for f in fock]
        converged = False

        self.log.infov('%12s %9s %12s %12s', 'Iteration', 'Cycles', 'Nelec error', 'DM change')

        for niter1 in range(1, self.opts.max_cycle_outer+1):
            for i in range(self.nkpts):
                se[i], opt = agf2.chempot.minimize_chempot(
                        se[i], fock[i], nelec[i],
                        x0=se[i].chempot,
                        tol=self.opts.conv_tol_nelec*1e-2,  #FIXME - may change after rediagonalisation
                        maxiter=self.opts.max_cycle_inner,
                )

            for niter2 in range(1, self.opts.max_cycle_inner+1):
                w, v = self.solve_dyson(se=se, gf=gf, fock=fock)
                nerr = 0.0
                for i in range(self.nkpts):
                    nact = self.nact[i]
                    se[i].chempot, nerr_k = \
                            agf2.chempot.binsearch_chempot((w[i], v[i]), nact, nelec[i])
                    gf[i] = agf2.GreensFunction(w[i], v[i][:nact], chempot=se[i].chempot)
                    if abs(nerr_k) > nerr:
                        nerr = nerr_k

                fock = self.get_fock(gf=gf, with_frozen=False)
                rdm1 = self.make_rdm1(gf=gf, with_frozen=False)
                fock = diis.update(fock)

                derr = max([np.max(np.absolute(x-y)) for x,y in zip(rdm1, rdm1_prev)])
                rdm1_prev = [x.copy() for x in rdm1]

                if derr < self.opts.conv_tol_rdm1:
                    break

            self.log.infov('%12d %9d %12.4g %12.4g', niter1, niter2, nerr, derr)

            if derr < self.opts.conv_tol_rdm1 and abs(nerr) < self.opts.conv_tol_nelec:
                converged = True
                break

        (self.log.info if converged else self.log.warning)("Converged = %r", converged)
        self.log.timing('Time for fock loop:  %s', time_string(timer() - t0))

        return gf, se


    def get_fock(self, gf=None, rdm1=None, with_frozen=True):
        ''' Get the Fock matrix including all frozen contributions
        '''

        if self.opts.fock_basis.lower() == 'ao':
            return self._get_fock_via_ao(gf=gf, rdm1=rdm1, with_frozen=with_frozen)

        t0 = timer()
        self.log.debugv("Building Fock matrix")

        gf = gf or self.gf
        if rdm1 is None:
            rdm1 = self.make_rdm1(gf=gf, with_frozen=False)
        eri = self.eri
        dtype = np.result_type(eri.dtype, *[d.dtype for d in rdm1])
        kconserv = self.khelper.kconserv

        vj = [np.zeros((n, n), dtype=dtype) for n in self.nact]
        vk = [np.zeros((n, n), dtype=dtype) for n in self.nact]

        for kac in mpi_helper.nrange(self.nkpts**2):
            ka, kc = divmod(kac, self.nkpts)
            kb = ka
            kd = kconserv[ka, kb, kc]
            if self.eri.ndim == 7:
                vj[ka] += lib.einsum('ijkl,lk->ij', eri[ka, kb, kc], rdm1[kd].conj())
                vk[ka] += lib.einsum('ilkj,lk->ij', eri[ka, kd, kc], rdm1[kd].conj())
                #TODO yes?
                #vj[ka] += lib.einsum('ijkl,kl->ij', eri[ka, kb, kc], rdm1[kc])
                #vk[ka] += lib.einsum('ilkj,kl->ij', eri[ka, kd, kc], rdm1[kc])
            else:
                buf = lib.einsum('Qij,ij->Q', eri[kc, kd], rdm1[kd])
                vj[ka] += lib.einsum('Qij,Q->ij', eri[ka, kb], buf)

                buf = lib.einsum('Qij,jk->Qki', eri[ka, kd], rdm1[kd].conj())
                vk[ka] += lib.einsum('Qki,Qkj->ij', buf, eri[kc, kb]).T.conj()

                #tmp = lib.einsum('Qkl,lk->Q', eri[kc, kd], rdm1[kd].conj())
                #vj[ka] += lib.einsum('Qij,Q->ij', eri[ka, kb], tmp)
                #tmp = lib.einsum('Qil,lk->Qki', eri[ka, kd], rdm1[kd].conj())
                #vk[ka] += lib.einsum('Qki,Qkj->ij', tmp, eri[kc, kb])

                #tmp = lib.einsum('Qkl,kl->Q', eri[:, kc, kd], rdm1[kc])
                #vj[ka] += lib.einsum('Qij,Q->ij', eri[:, ka, kb], tmp)
                #tmp = lib.einsum('Qil,kl->Qki', eri[:, ka, kd], rdm1[kc])
                #vk[ka] += lib.einsum('Qki,Qkj->ij', tmp, eri[:, kb, kc])

        for ka in range(self.nkpts):
            mpi_helper.barrier()
            mpi_helper.allreduce_safe_inplace(vj[ka])
            mpi_helper.allreduce_safe_inplace(vk[ka])

        if self.opts.keep_exxdiv and self.mf.exxdiv == 'ewald':
            madelung = tools.pbc.madelung(self.mol, self.kpts)
            ewald = [madelung * x for x in rdm1]
            vk = [k+e for k, e in zip(vk, ewald)]

        fock = []
        for i in range(self.nkpts):
            act = self.act[i]
            f  = vj[i] - 0.5 * vk[i]
            if self.veff is not None:
                f += self.veff[act, act]
            f += self.h1e[i][act, act]
            fock.append(f)

        if with_frozen:
            fock_ref = [np.diag(e).astype(dtype) for e in self.mo_energy]
            for i in range(self.nkpts):
                act = self.act[i]
                fock_ref[i][act, act] = fock[i]
            fock = fock_ref

        self.log.timingv("Time for Fock matrix:  %s", time_string(timer() - t0))

        return fock


    def _get_fock_via_ao(self, gf=None, rdm1=None, with_frozen=True):
        '''
        Get the Fock matrix via AO basis integrals - result is still
        transformed into MO basis.
        '''

        t0 = timer()
        self.log.debugv("Build Fock matrix via AO integrals")

        gf = gf or self.gf
        mo_coeff = self.mo_coeff
        if rdm1 is None:
            rdm1 = self.make_rdm1(gf=gf, with_frozen=True)
        rdm1_ao = [np.linalg.multi_dot((c, d, c.T.conj())) for c,d in zip(mo_coeff, rdm1)]

        veff_ao = self.mf.get_veff(dm_kpts=rdm1_ao)
        veff = [np.linalg.multi_dot((c.T.conj(), v, c)) for c,v in zip(mo_coeff, veff_ao)]

        fock = [h+v for h,v in zip(self.h1e, veff)]

        if not with_frozen:
            fock = [f[act, act] for f,act in zip(fock, self.act)]

        self.log.timingv("Time for Fock matrix:  %s", time_string(timer() - t0))

        return fock


    def make_rdm1(self, gf=None, with_frozen=True):
        ''' Get the 1RDM
        '''

        gf = gf or self.gf
        rdm1 = [g.make_rdm1() for g in gf]

        if with_frozen:
            ovlp = self.mf.get_ovlp()
            rdm1_hf = self.mf.make_rdm1(self.mo_coeff, self.mo_occ)
            for i in range(self.nkpts):
                act = self.act[i]
                sc = np.dot(ovlp[i], self.mo_coeff[i])
                rdm1_ref = np.linalg.multi_dot((sc.T, rdm1_hf[i], sc))
                rdm1_ref[act, act] = rdm1[i]
                rdm1[i] = rdm1_ref

        return rdm1


    def make_rdm2(self, gf=None, with_frozen=True):
        ''' Get the 2RDM
        '''

        raise NotImplementedError  #TODO


    def energy_mp2(self, mo_energy=None, se=None):
        ''' Calculate the MP2 energy
        '''

        mo_energy = mo_energy or self.mo_energy
        se = se or self.se

        e_mp2 = 0.0
        for mo, s, act in zip(mo_energy, se, self.act):
            e_mp2 += agf2.ragf2.energy_mp2(None, mo[act], s).real

        e_mp2 /= self.nkpts

        return e_mp2


    def energy_1body(self, gf=None):
        ''' Calculate the one-body energy
        '''

        rdm1 = self.make_rdm1(gf=gf, with_frozen=True)
        fock = self.get_fock(gf=gf, with_frozen=True)
        h1e = self.h1e

        e_1b = 0.0
        for d, h, f in zip(rdm1, h1e, fock):
            e_1b += 0.5 * np.sum(d * (h + f)).real

        e_1b /= self.nkpts
        e_1b += self.e_nuc

        return e_1b


    def energy_2body(self, gf=None, se=None):
        ''' Calculate the two-body energy
        '''

        gf = gf or self.gf
        se = se or self.se

        e_2b = 0.0
        for g, s in zip(gf, se):
            e_2b += agf2.ragf2.energy_2body(None, g, s)

        e_2b /= self.nkpts

        return e_2b


    def population_analysis(self, method='meta-lowdin', pre_orth_method='ANO', kpt_index=0):
        ''' Population analysis, by default at the Γ-point
        '''
        #TODO does this even work? needs supercell?

        from pyscf.lo import orth
        from pyscf.scf.hf import mulliken_pop

        s = self.mol.get_ovlp()[kpt_index]
        orth_coeff = orth.orth_ao(self.mol, method, pre_orth_method, s=s)
        c_inv = np.dot(orth_coeff.T.conj(), s)

        def mulliken(dm):
            dm = np.linalg.multi_dot((c_inv, dm, c_inv.T.conj()))
            return mulliken_pop(self.mol, dm, np.eye(orth_coeff.shape[0]), verbose=0)

        rdm1_hf = self.mf.make_rdm1(self.mo_coeff, self.mo_occ)[kpt_index]
        rdm1_agf2 = np.linalg.multi_dot((
                self.mo_coeff[kpt_index],
                self.make_rdm1()[kpt_index],
                self.mo_coeff[kpt_index].T.conj()
        ))

        pop_mo, charges_mo = mulliken(rdm1_hf)
        pop_qmo, charges_qmo = mulliken(rdm1_agf2)

        self.log.info("Population analysis")
        self.log.info("*******************")
        self.log.changeIndentLevel(1)
        self.log.info("%4s  %-12s %12s %12s", "AO", "Label", "Pop. (MF)", "Pop. (AGF2)")
        for i, s in enumerate(self.mol.ao_labels()):
            self.log.info("%4d  %-12s %12.6f %12.6f", i, s, pop_mo[i], pop_qmo[i])
        self.log.changeIndentLevel(-1)

        self.log.info("Atomic charges")
        self.log.info("**************")
        self.log.changeIndentLevel(1)
        self.log.info("%4s  %-12s %12s %12s", "Atom", "Symbol", "Charge (MF)", "Charge (AGF2)")
        for i in range(self.mol.natm):
            s = self.mol.atom_symbol(i)
            self.log.info("%4d  %-12s %12.6f %12.6f", i, s, charges_mo[i], charges_qmo[i])
        self.log.changeIndentLevel(-1)

        return pop_qmo, charges_qmo


    def dip_moment(self, kpts_index=0):
        ''' Dipole moment, by default at the Γ-point
        '''
        #TODO does this even work? needs supercell?

        from pyscf.scf.hf import dip_moment

        dm_hf = self.mf.make_rdm1()[kpt_index]
        dm_agf2 = np.linalg.multi_dot((
            self.mo_coeff[kpt_index],
            self.make_rdm1()[kpt_index],
            self.mo_coeff[kpt_index].T.conj()
        ))

        dip_mo = dip_moment(self.mol, dm_hf, unit='au', verbose=0)
        tot_mo = np.linalg.norm(dip_mo)

        dip_qmo = dip_moment(self.mol, dm_agf2, unit='au', verbose=0)
        tot_qmo = np.linalg.norm(dip_qmo)

        self.log.info("Dipole moment")
        self.log.info("*************")
        self.log.changeIndentLevel(1)
        self.log.info("%6s %12s %12s", "Part", "Dip. (MF)", "Dip. (AGF2)")
        for x in range(3):
            self.log.info("%6s %12.6f %12.6f", "XYZ"[x], dip_mo[x], dip_qmo[x])
        self.log.info("%6s %12.6f %12.6f", "Total", tot_mo, tot_qmo)
        self.log.changeIndentLevel(-1)

        return dip_qmo


    def dump_chk(self, chkfile=None):
        ''' Save the calculation state
        '''

        chkfile = chkfile or self.chkfile

        if chkfile is None:
            return self

        agf2.chkfile.dump_kagf2(self, chkfile=self.mf.chkfile)

        return self


    def update_from_chk(self, chkfile=None):
        ''' Update from the calculation state
        '''

        chkfile = chkfile or self.chkfile

        if self.chkfile is None:
            return self

        mol, data = agf2.chkfile.load_kagf2(self.chkfile, key)
        self.__dict__.update(data)

        return self


    def dump_cube(self, index, kpt_index=0, cubefile='agf2.cube', ngrid=200):
        ''' Dump a QMO to a .cube file
        '''

        raise NotImplementedError  #TODO


    def print_excitations(self, gf=None, kpt_index=0):
        ''' 
        Print the excitations and some information on their character,
        by default at the Γ-point.
        '''

        gf = gf or self.gf

        if isinstance(gf, (tuple, list)):
            gf = gf[kpt_index]

        RAGF2.print_excitations(self, gf=gf)


    def _convergence_checks(self, se=None, se_prev=None, e_prev=None):
        ''' 
        Return a list of [energy, 0th moment, 1st moment] changes between
        iterations to check convergence progress.
        '''

        se = se or self.se
        e_prev = e_prev or self.mf.e_tot

        t0, t1 = sum([s.moment(range(2), squeeze=False) for s in se])
        if se_prev is None:
            t0_prev = t1_prev = np.zeros_like(t0)
        else:
            t0_prev, t1_prev = sum([s.moment(range(2), squeeze=False) for s in se_prev])

        deltas = [
                np.abs(self.e_tot - e_prev),
                np.linalg.norm(t0 - t0_prev),
                np.linalg.norm(t1 - t1_prev),
        ]

        return deltas


    @property
    def qmo_energy(self):
        return [g.energy for g in self.gf]

    @property
    def qmo_coeff(self):
        return [np.dot(mo, g.coupling) for mo, g in zip(self.mo_coeff, self.gf)]

    @property
    def qmo_occ(self):
        qmo_occ = []
        for i in range(self.nkpts):
            occ = np.linalg.norm(self.gf[i].get_occupied().coupling, axis=0)**2
            vir = np.zeros_like(self.gf[i].get_virtual().energy)
            qmo_occ.append(np.concatenate([occ, vir], axis=0))
        return qmo_occ

    dyson_orbitals = qmo_coeff


    @property
    def cell(self):
        return self.mol
    @cell.setter
    def cell(self, mol):
        self.mol = mol

    @property
    def frozen(self):
        return getattr(self, '_frozen', [(0, 0),]*self.nkpts)
    @frozen.setter
    def frozen(self, frozen):
        if frozen == None or frozen == 0 or frozen == (0, 0):
            self._frozen = (0, 0)
        elif isinstance(frozen, int):
            self._frozen = (frozen,  0)
        else:
            self._frozen = frozen
        if isinstance(self._frozen, tuple):
            self._frozen = [self._frozen,] * self.nkpts

    @property
    def e_ip(self): return -self.gf[0].get_occupied().energy.max()
    @property
    def e_ea(self): return self.gf[0].get_virtual().energy.min()

    @property
    def nmo(self): return self._nmo or self.mo_occ[0].size
    @property
    def nocc(self): return self._nocc or [np.sum(x > 0) for x in self.mo_occ]
    @property
    def nvir(self): return [self.nmo-x for x in self.nocc]
    @property
    def nfroz(self): return [f[0]+f[1] for f in self.frozen]
    @property
    def nact(self): return [self.nmo-x for x in self.nfroz]
    @property
    def nkpts(self): return len(self.kpts)



if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from vayesta import log
    log.setLevel(25)

    cell = gto.Cell()
    cell.atom = 'He 1 0 1; He 0 0 1'
    cell.basis = '6-31g'
    cell.a = np.eye(3) * 3
    cell.precision = 1e-8
    cell.verbose = 0
    cell.exp_to_discard = 0.1
    cell.build()

    #cell = gto.Cell()
    #cell.atom = 'Si 0 0 0; Si 1.37 1.37 1.37'
    #cell.basis = 'gth-szv'
    #cell.pseudo = 'gth-pade'
    #cell.a = (np.ones((3, 3)) - np.eye(3)) * 1.37 * 2
    #cell.precision = 1e-8
    #cell.verbose = 0
    #cell.exp_to_discard = 0.1
    #cell.build()

    mf = scf.KRHF(cell, cell.make_kpts([2,2,2])).density_fit().run()

    gf2 = KRAGF2(
            mf,
            log=log,
            direct=True,
            conv_tol=1e-5,
            max_cycle=40,
            damping=0.5,
            keep_exxdiv=False,
            #fock_basis='ao',
    )

    #print(np.max(np.abs(np.array(gf2.get_fock(gf=gf2.build_init_greens_function())) - np.array([np.diag(x) for x in mf.mo_energy]))))

    gf2.kernel()
    print(gf2.converged)
    print()

    from pyscf.pbc.agf2 import KRAGF2 as KRAGF2_
    gf2_ = KRAGF2_(mf)
    gf2_.conv_tol = 1e-5
    gf2_.max_cycle = 40
    gf2_.damping = 0.5
    gf2_.keep_exxdiv = False
    gf2_.direct = True
    gf2_.kernel()

    gf2.gf = gf2_.gf
    gf2.se = gf2_.se
    gf2.e_1b = gf2_.e_1b
    gf2.e_2b = gf2_.e_2b
    gf2.print_energies(output=True)
    print(gf2.converged)
