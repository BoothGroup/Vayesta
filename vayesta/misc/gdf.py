'''
Functionally equivalent to PySCF GDF, with all storage incore and
MPI parallelism.

Adapted from pyscf.pbc.df.df

Ref:
J. Chem. Phys. 147, 164119 (2017)
'''

import time
import copy
import ctypes
import logging
import collections
import numpy as np
import scipy.linalg

from pyscf import gto, lib, ao2mo, __config__
from pyscf.pbc import tools
from pyscf.df import addons
from pyscf.agf2 import mpi_helper
from pyscf.ao2mo.outcore import balance_partition
from pyscf.ao2mo.incore import iden_coeffs, _conc_mos
from pyscf.pbc.gto.cell import _estimate_rcut
from pyscf.pbc.df import df, incore, ft_ao, aft
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0, _format_dms
from pyscf.pbc.df.fft_ao2mo import _iskconserv, _format_kpts
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique, KPT_DIFF_TOL, get_kconserv

from vayesta.core.util import time_string

try:
    from mpi4py import MPI
    timer = MPI.Wtime
except ImportError:
    from timeit import default_timer as timer


COMPACT = getattr(__config__, 'pbc_df_ao2mo_get_eri_compact', True)


def make_auxcell(cell, auxbasis=None, drop_eta=None, log=None):
    '''
    Build the cell corresponding to the auxiliary functions.

    Note: almost identical to pyscf.pyscf.pbc.df.df.make_modrho_basis
    '''

    log = log or logging.getLogger(__name__)
    log.info("Auxiliary cell")
    log.info("**************")
    with log.withIndentLevel(1):

        auxcell = addons.make_auxmol(cell, auxbasis)

        steep_shls = []
        ndrop = 0
        rcut = []
        _env = auxcell._env.copy()

        for ib in range(len(auxcell._bas)):
            l = auxcell.bas_angular(ib)
            nprim = auxcell.bas_nprim(ib)
            nctr = auxcell.bas_nctr(ib)
            exps = auxcell.bas_exp(ib)
            ptr_coeffs = auxcell._bas[ib, gto.PTR_COEFF]
            coeffs = auxcell._env[ptr_coeffs:ptr_coeffs+nprim*nctr].reshape(nctr, nprim).T

            mask = exps >= (drop_eta or -np.inf)
            log.debug(
                    "Auxiliary function %4d (L = %2d):"
                    "dropped %d functions.", ib, l, np.sum(~mask),
            )

            if drop_eta is not None and np.any(exps < drop_eta):
                for k in np.where(mask == 0)[0]:
                    log.debug(" > %3d : coeff = %12.6f    exp = %12.6g", k, coeffs[k], exps[k])
                coeffs = coeffs[mask]
                exps = exps[mask]
                nprim, ndrop = len(exps), ndrop+nprim-len(exps)

            if nprim > 0:
                ptr_exps = auxcell._bas[ib, gto.PTR_EXP]
                auxcell._bas[ib, gto.NPRIM_OF] = nprim
                _env[ptr_exps:ptr_exps+nprim] = exps

                int1 = gto.gaussian_int(l*2+2, exps)
                s = np.einsum('pi,p->i', coeffs, int1)

                coeffs *= np.sqrt(0.25 / np.pi)
                coeffs /= s[None]
                _env[ptr_coeffs:ptr_coeffs+nprim*nctr] = coeffs.T.ravel()

                steep_shls.append(ib)

                r = _estimate_rcut(exps, l, np.abs(coeffs).max(axis=1), cell.precision)
                rcut.append(r.max())

        auxcell._env = _env
        auxcell._bas = np.asarray(auxcell._bas[steep_shls], order='C')
        auxcell.rcut = max(rcut)

        log.info("Dropped %d primitive functions.", ndrop)
        log.info("Auxiliary basis:  shells = %d  cGTOs = %d", auxcell.nbas, auxcell.nao_nr())
        log.info("rcut = %.6g", auxcell.rcut)

    return auxcell


def make_chgcell(auxcell, smooth_eta, rcut=15.0, log=None):
    '''
    Build the cell corresponding to the smooth Gaussian functions.

    Note: almost identical to pyscf.pyscf.pbc.df.df.make_modchg_basis
    '''

    log = log or logging.getLogger(__name__)
    log.info("Compensating cell")
    log.info("*****************")
    with log.withIndentLevel(1):

        chgcell = copy.copy(auxcell)
        chg_bas = []
        chg_env = [smooth_eta]
        ptr_eta = auxcell._env.size
        ptr = ptr_eta + 1
        l_max = auxcell._bas[:, gto.ANG_OF].max()
        norms = [0.5 / np.sqrt(np.pi) / gto.gaussian_int(l*2+2, smooth_eta) for l in range(l_max+1)]

        for ia in range(auxcell.natm):
            for l in set(auxcell._bas[auxcell._bas[:, gto.ATOM_OF] == ia, gto.ANG_OF]):
                chg_bas.append([ia, l, 1, 1, 0, ptr_eta, ptr, 0])
                chg_env.append(norms[l])
                ptr += 1

        chgcell._atm = auxcell._atm
        chgcell._bas = np.asarray(chg_bas, dtype=np.int32).reshape(-1, gto.BAS_SLOTS)
        chgcell._env = np.hstack((auxcell._env, chg_env))

        # _estimate_rcut is too tight for the model charge
        chgcell.rcut = np.sqrt(np.log(4 * np.pi * rcut**2 / auxcell.precision) / smooth_eta)

        log.info("Compensating basis:  shells = %d  cGTOs = %d", chgcell.nbas, chgcell.nao_nr())
        log.info("rcut = %.6g", chgcell.rcut)

    return chgcell


def fuse_auxcell(auxcell, chgcell, log=None):
    '''
    Fuse the cells responsible for the auxiliary functions and smooth
    Gaussian functions used to carry the charge. Returns the fused cell
    along with a function which incorporates contributions from the
    charge carrying functions into the auxiliary functions according to
    their angular momenta and spherical harmonics.

    Note: almost identical to pyscf.pyscf.pbc.df.df.fuse_auxcell
    '''

    log = log or logging.getLogger(__name__)
    log.info("Fused cell")
    log.info("**********")
    with log.withIndentLevel(1):

        fused_cell = copy.copy(auxcell)
        fused_cell._atm, fused_cell._bas, fused_cell._env = gto.conc_env(
                auxcell._atm, auxcell._bas, auxcell._env,
                chgcell._atm, chgcell._bas, chgcell._env,
        )
        fused_cell.rcut = max(auxcell.rcut, chgcell.rcut)

        log.info("Fused basis:  shells = %d  cGTOs = %d", fused_cell.nbas, fused_cell.nao_nr())
        log.info("rcut = %.6g", fused_cell.rcut)

        aux_loc = auxcell.ao_loc_nr()
        naux = aux_loc[-1]
        modchg_offset = -np.ones((chgcell.natm, 8), dtype=int)
        smooth_loc = chgcell.ao_loc_nr()

        for i in range(chgcell.nbas):
            ia = chgcell.bas_atom(i)
            l = chgcell.bas_angular(i)
            modchg_offset[ia, l] = smooth_loc[i]

        if auxcell.cart:
            # See pyscf.pyscf.pbc.df.df.fuse_auxcell

            c2s_fn = gto.moleintor.libcgto.CINTc2s_ket_sph
            aux_loc_sph = auxcell.ao_loc_nr(cart=False)
            naux_sph = aux_loc_sph[-1]

            log.debug("Building fusion function via Cartesian primitives.")

            def fuse(Lpq):
                Lpq, Lpq_chg = Lpq[:naux], Lpq[naux:]
                if Lpq.ndim == 1:
                    npq = 1
                    Lpq_sph = np.empty((naux_sph), dtype=Lpq.dtype)
                else:
                    npq = Lpq.shape[1]
                    Lpq_sph = np.empty((naux_sph, npq), dtype=Lpq.dtype)

                if Lpq.dtype == np.complex128:
                    npq *= 2

                for i in range(auxcell.nbas):
                    ia = auxcell.bas_atom(i)
                    l = auxcell.bas_angular(i)
                    p0 = modchg_offset[ia, l]

                    if p0 >= 0:
                        nd = (l+1) * (l+2) // 2
                        c0, c1 = aux_loc[i], aux_loc[i+1]
                        s0, s1 = aux_loc_sph[i], aux_loc_sph[i+1]

                        for i0, i1 in lib.prange(c0, c1, nd):
                            Lpq[i0:i1] -= Lpq_chg[p0:p0+nd]

                        if l < 2:
                            Lpq_sph[s0:s1] = Lpq[c0:c1]
                        else:
                            Lpq_cart = np.asarray(Lpq[c0:c1], order='C')
                            c2s_fn(
                                Lpq_sph[s0:s1].ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(npq * auxcell.bas_nctr(i)),
                                Lpq_cart.ctypes.data_as(ctypes.c_void_p),
                                ctpyes.c_int(l),
                            )

                return Lpq_sph

        else:
            log.debug("Building fusion function via spherical primitives.")

            def fuse(Lpq):
                Lpq, Lpq_chg = Lpq[:naux], Lpq[naux:]

                for i in range(auxcell.nbas):
                    ia = auxcell.bas_atom(i)
                    l = auxcell.bas_angular(i)
                    p0 = modchg_offset[ia, l]

                    if p0 >= 0:
                        nd = l * 2 + 1

                        for i0, i1 in lib.prange(aux_loc[i], aux_loc[i+1], nd):
                            Lpq[i0:i1] -= Lpq_chg[p0:p0+nd]

                return Lpq

    return fused_cell, fuse


def _get_kpt_hash(kpt, tol=KPT_DIFF_TOL):
    '''
    Get a hashable representation of the k-point up to a given tol to
    prevent the O(N_k) access cost.
    '''

    kpt_round = np.rint(np.asarray(kpt) / tol).astype(int)
    return tuple(kpt_round.ravel())


def _kconserve_indices(cell, uniq_kpts, kpt):
    '''
    Search which (kpts+kpt) satisfies momentum conservation.
    '''

    a = cell.lattice_vectors() / (2*np.pi)

    kdif = np.dot(a, (uniq_kpts + kpt).T)
    kdif_int = np.rint(kdif)

    mask = np.sum(np.abs(kdif - kdif_int), axis=0) < KPT_DIFF_TOL
    uniq_kptji_ids = np.where(mask)[0]

    return uniq_kptji_ids


def _get_2c2e(with_df, uniq_kpts, log=None):
    '''
    Get the bare two-center two-electron interaction, first term
    of Eq. 32.
    '''

    t0 = timer()
    log = log or logging.getLogger(__name__)

    int2c2e = with_df.fused_cell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    log.timing("Time for 2c2e:  %s", time_string(timer()-t0))

    return int2c2e


def _get_3c2e(with_df, kptij_lst, log=None):
    '''
    Get the bare three-center two-electron interaction, first term
    of Eq. 31.
    '''

    t0 = timer()
    log = log or logging.getLogger(__name__)

    cell = with_df.cell
    fused_cell = with_df.fused_cell

    nkij = len(kptij_lst)
    nao = cell.nao_nr()
    ngrids = fused_cell.nao_nr()
    aux_loc = fused_cell.ao_loc_nr(fused_cell.cart)

    int3c2e = np.zeros((nkij, ngrids, nao*nao), dtype=np.complex128)

    for p0, p1 in mpi_helper.prange(0, fused_cell.nbas, fused_cell.nbas):
        t1 = timer()

        shls_slice = (0, cell.nbas, 0, cell.nbas, p0, p1)
        q0, q1 = aux_loc[p0], aux_loc[p1]

        int3c2e_part = incore.aux_e2(cell, fused_cell, 'int3c2e', aosym='s2',
                                     kptij_lst=kptij_lst, shls_slice=shls_slice)
        int3c2e_part = lib.transpose(int3c2e_part, axes=(0, 2, 1))

        if int3c2e_part.shape[-1] != nao*nao:
            assert int3c2e_part.shape[-1] == nao*(nao+1)//2
            int3c2e_part = lib.unpack_tril(int3c2e_part, lib.HERMITIAN, axis=-1)

        int3c2e_part = int3c2e_part.reshape((nkij, q1-q0, nao*nao))
        int3c2e[:, q0:q1] = int3c2e_part

        log.timingv("Time for 3c2e [%d -> %d] part:  %s", p0, p1, time_string(timer()-t1))

    mpi_helper.allreduce_safe_inplace(int3c2e)
    mpi_helper.barrier()

    log.timing("Time for 3c2e:  %s", time_string(timer()-t0))

    return int3c2e


def _get_j2c(with_df, int2c2e, uniq_kpts, log=None):
    '''
    Build j2c using the 2c2e interaction, int2c2e, Eq. 32.
    '''

    log = log or logging.getLogger(__name__)
    t0 = timer()

    naux = with_df.auxcell.nao_nr()
    Gv, Gvbase, kws = with_df.cell.get_Gv_weights(with_df.mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    b = with_df.cell.reciprocal_vectors()

    j2c = int2c2e

    for k, kpt in enumerate(uniq_kpts):
        t1 = timer()

        G_chg = ft_ao.ft_ao(with_df.fused_cell, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase, kpt=kpt).T
        G_aux = G_chg[naux:] * with_df.weighted_coulG(kpt)

        # Eq. 32 final three terms:
        j2c_comp = np.dot(G_aux.conj(), G_chg.T)
        if is_zero(kpt):
            j2c_comp = j2c_comp.real
        j2c[k][naux:] -= j2c_comp
        j2c[k][:naux, naux:] = j2c[k][naux:, :naux].T.conj()

        j2c[k] = with_df.fuse(with_df.fuse(j2c[k]).T).T

        del G_chg, G_aux

        log.timingv("Time for j2c [kpt %d] part:  %s", k, time_string(timer()-t1))

    log.timing("Time for j2c:  %s", time_string(timer()-t0))

    return j2c


def _cholesky_decomposed_metric(with_df, j2c, uniq_kptji_id, log=None):
    '''
    Get a function which applies the Cholesky decomposed j2c.

    NOTE: matches Max's changes to pyscf DF rather than vanilla DF.
    '''

    log = log or logging.getLogger(__name__)

    j2c_chol = None

    if not with_df.linear_dep_always:
        try:
            j2c_kpt = scipy.linalg.cholesky(j2c[uniq_kptji_id], lower=True)
            def j2c_chol(x):
                return scipy.linalg.solve_triangular(j2c_kpt, x, lower=True, overwrite_b=True)
        except scipy.linalg.LinAlgError:
            log.warning("Cholesky decomposition failed for j2c [kpt %d]", uniq_kptji_id)
            pass

    if j2c_chol is None and with_df.linear_dep_method == 'regularize':
        try:
            eps = 1e-14 * np.eye(j2c[uniq_kptji_id].shape[-1])
            j2c_kpt = scipy.linalg.cholesky(j2c[uniq_kptji_id] + eps, lower=True)
            def j2c_chol(x):
                return scipy.linalg.solve_triangular(j2c_kpt, x, lower=True, overwrite_b=True)
        except scipy.linalg.LinAlgError:
            log.warning("Regularised Cholesky decomposition failed for j2c [kpt %d]", uniq_kptji_id)
            pass

    if j2c_chol is None:
        raise NotImplementedError("This may be broken")
        w, v = scipy.linalg.eigh(j2c[uniq_kptji_id])
        cond = w.max() / w.min()
        mask = w > with_df.linear_dep_threshold

        log.info("DF metric linear dependency detected [kpt %d]", uniq_kptji_id)
        log.info("Condition number = %.4g", cond)
        log.info("Dropped %d auxiliary functions.", np.sum(mask))

        j2c_kpt = v[:, mask].conj().T
        j2c_kpt /= np.sqrt(w[mask])[:, None]
        def j2c_chol(x):
            return np.dot(j2c_kpt, x)

    return j2c_chol


def _get_j3c(with_df, j2c, int3c2e, uniq_kpts, uniq_inverse_dict, kptij_lst, log=None, out=None):
    '''
    Use j2c and int3c2e to construct j3c.
    '''

    log = log or logging.getLogger(__name__)
    t0 = timer()

    cell = with_df.cell
    auxcell = with_df.auxcell

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    Gv, Gvbase, kws = cell.get_Gv_weights(with_df.mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]
    b = cell.reciprocal_vectors()
    j2c_chol = lambda v: v
    kpts = with_df.kpts
    nkpts = len(kpts)

    if out is None:
        out = np.zeros((nkpts, nkpts, naux, nao, nao), dtype=np.complex128)

    for uniq_kpt_ji_id in mpi_helper.nrange(len(uniq_kpts)):
        t1 = timer()
        log.debug("Constructing j3c [kpt %d].", uniq_kpt_ji_id)
        with log.withIndentLevel(1):

            kpt = uniq_kpts[uniq_kpt_ji_id]
            adapted_ji_idx = uniq_inverse_dict[uniq_kpt_ji_id]
            adapted_kptjs = kptij_lst[:, 1][adapted_ji_idx]

            log.debug("kpt = %s", kpt)
            log.debug("adapted_ji_idx = %s", adapted_ji_idx)

            # Prepare cholesky decomposition of j2c
            if j2c is not None:
                kptji_id = _kconserve_indices(cell, uniq_kpts, -kpt)[0]
                j2c_chol = _cholesky_decomposed_metric(with_df, j2c, kptji_id, log)
                log.debug("kptji_id = %s", kptji_id)

            # Eq. 33
            shls_slice = (auxcell.nbas, with_df.fused_cell.nbas)
            G_chg  = ft_ao.ft_ao(with_df.fused_cell, Gv, shls_slice=shls_slice,
                                 b=b, gxyz=gxyz, Gvbase=Gvbase, kpt=kpt)
            G_chg *= with_df.weighted_coulG(kpt).ravel()[:, None]
            log.debugv("Norm of FT for fused cell:  %.12g", np.linalg.norm(G_chg))

            # Eq. 26
            if is_zero(kpt):
                log.debug("Including net charge of AO products")
                vbar = with_df.fuse(with_df.auxbar(with_df.fused_cell))
                ovlp = cell.pbc_intor('int1e_ovlp', hermi=0, kpts=adapted_kptjs)
                ovlp = [np.ravel(s) for s in ovlp]

            # Eq. 24
            bstart, bend, ncol = balance_partition(cell.ao_loc_nr()*nao, nao*nao)[0]
            shls_slice = (bstart, bend, 0, cell.nbas)
            G_ao = ft_ao.ft_aopair_kpts(cell, Gv, shls_slice=shls_slice, aosym='s1', b=b,
                                        gxyz=gxyz, Gvbase=Gvbase, q=kpt, kptjs=adapted_kptjs)
            G_ao = G_ao.reshape(-1, ngrids, ncol)
            log.debugv("Norm of FT for AO cell:  %.12g", np.linalg.norm(G_ao))

            for kji, ji in enumerate(adapted_ji_idx):
                # Eq. 31 first term:
                v = int3c2e[ji]

                # Eq. 31 second term
                if is_zero(kpt):
                    for i in np.where(vbar != 0)[0]:
                        v[i] -= vbar[i] * ovlp[kji]

                # Eq. 31 third term
                v[naux:] -= np.dot(G_chg.T.conj(), G_ao[kji])

                # Fused charge-carrying part with auxiliary part
                v = with_df.fuse(v)

                # Cholesky decompose Eq. 29
                v = j2c_chol(v)
                v = v.reshape(-1, nao, nao)

                # Sum into all symmetry-related k-points
                for ki in with_df.kpt_hash[_get_kpt_hash(kptij_lst[ji][0])]:
                    for kj in with_df.kpt_hash[_get_kpt_hash(kptij_lst[ji][1])]:
                        out[ki, kj, :v.shape[0]] += v
                        log.debug("Filled j3c for kpt [%d, %d]", ki, kj)
                        if ki != kj:
                            out[kj, ki, :v.shape[0]] += lib.transpose(v, axes=(0, 2, 1)).conj()
                            log.debug("Filled j3c for kpt [%d, %d]", kj, ki)

            log.timingv("Time for j3c [kpt %d]:  %s", uniq_kpt_ji_id, time_string(timer()-t1))

    mpi_helper.allreduce_safe_inplace(out)
    mpi_helper.barrier()

    log.timing("Time for j3c:  %s", time_string(timer()-t0))

    return out


def _make_j3c(with_df, kptij_lst):
    '''
    Build the j3c array.

    cell: the unit cell for the calculation
    auxcell: the unit cell for the auxiliary functions
    chgcell: the unit cell for the smooth Gaussians
    fused_cell: auxcell and chgcell combined
    '''

    t0 = timer()
    log = with_df.log

    if with_df.cell.dimension < 3:
        raise ValueError('GDF does not support low-dimension cells')

    kptis = kptij_lst[:, 0]
    kptjs = kptij_lst[:, 1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    uniq_inverse_dict = { k: np.where(uniq_inverse == k)[0] for k in range(len(uniq_kpts)) }

    log.info("Number of unique kpts (kj - ki): %d", len(uniq_kpts))
    log.debugv("uniq_kpts:\n%s", uniq_kpts)

    # Get the 3c2e interaction:
    int3c2e = _get_3c2e(with_df, kptij_lst, log=with_df.log)

    # Get the 2c2e interaction:
    int2c2e = _get_2c2e(with_df, uniq_kpts, log=with_df.log)

    # Get j2c:
    j2c = _get_j2c(with_df, int2c2e, uniq_kpts, log=with_df.log)

    # Get j3c:
    j3c = _get_j3c(with_df, j2c, int3c2e, uniq_kpts, uniq_inverse_dict, kptij_lst, log=with_df.log)

    log.timing("Total time for j3c construction:  %s", time_string(timer()-t0))

    return j3c


def _fao2mo(eri, cp, cq):
    ''' AO2MO for 3c integrals '''
    npq, cpq, spq = _conc_mos(cp, cq, compact=False)[1:]
    naux = eri.shape[0]
    cpq = np.asarray(cpq, dtype=np.complex128)
    out = ao2mo._ao2mo.r_e2(eri, cpq, spq, [], None)
    return out.reshape(naux, cp.shape[1], cq.shape[1])


class GDF(df.GDF):
    ''' Incore Gaussian density fitting.
    '''

    def __init__(self, cell, kpts=np.zeros((1, 3)), log=None):
        if not cell.dimension == 3:
            raise ValueError('%s does not support low dimension systems' % self.__class__)

        self.log = log or logging.getLogger(__name__)
        self.log.info("Initializing %s", self.__class__.__name__)
        self.log.info("=============%s", len(str(self.__class__.__name__))*'=')

        self.cell = cell
        self.kpts = kpts
        self._auxbasis = None

        self.eta, self.mesh = self.build_mesh()
        self.exp_to_discard = cell.exp_to_discard
        self.rcut_smooth = 15.0
        self.linear_dep_threshold = 1e-9
        self.linear_dep_method = 'regularize'
        self.linear_dep_always = False

        # The follow attributes are not input options.
        self.exxdiv = None
        self.auxcell = None
        self.blockdim = None
        self.kpts_band = None
        self._j_only = False
        self._cderi = None
        self._rsh_df = {}
        self._keys = set(self.__dict__.keys())

    def build_mesh(self):
        '''
        Search for optimised eta and mesh.
        '''

        cell = self.cell

        ke_cutoff = tools.mesh_to_cutoff(cell.lattice_vectors(), cell.mesh)
        ke_cutoff = ke_cutoff[:cell.dimension].min()

        eta_cell = aft.estimate_eta_for_ke_cutoff(cell, ke_cutoff, cell.precision)
        eta_guess = aft.estimate_eta(cell, cell.precision)

        self.log.debug("ke_cutoff = %.6g", ke_cutoff)
        self.log.debug("eta_cell  = %.6g", eta_cell)
        self.log.debug("eta_guess = %.6g", eta_guess)

        if eta_cell < eta_guess:
            eta, mesh = eta_cell, cell.mesh
        else:
            eta = eta_guess
            ke_cutoff = aft.estimate_ke_cutoff_for_eta(cell, eta, cell.precision)
            mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_cutoff)

        mesh = df._round_off_to_odd_mesh(mesh)

        self.log.info("Built mesh: shape = %s  eta = %.6g", mesh, eta)

        return eta, mesh

    def reset(self, cell=None):
        if cell is not None:
            self.cell = cell
        self.auxcell = None
        self._cderi = None
        self._rsh_df = {}
        return self

    def dump_flags(self):
        self.log.info("GDF parameters:")
        for key in [
                'auxbasis', 'eta', 'exp_to_discard', 'rcut_smooth',
                'linear_dep_threshold', 'linear_dep_method', 'linear_dep_always'
        ]:
            self.log.info("  > %-24s %r", key + ':', getattr(self, key))
        return self

    def build(self, j_only=None, with_j3c=True):
        j_only = j_only or self._j_only
        if j_only:
            self.log.warn('j_only=True has not effect on overhead in %s', self.__class__)
        if not with_j3c:
            raise ValueError('%s does not support with_j3c' % self.__class__)
        if self.kpts_band is not None:
            raise ValueError('%s does not support kwarg kpts_band' % self.__class__)

        self.check_sanity()
        self.dump_flags()

        self.auxcell = make_auxcell(self.cell, auxbasis=self.auxbasis, drop_eta=self.exp_to_discard)
        self.chgcell = make_chgcell(self.auxcell, self.eta, rcut=self.rcut_smooth)
        self.fused_cell, self.fuse = fuse_auxcell(self.auxcell, self.chgcell)

        self._get_nuc = None
        self._get_pp = None

        kpts = np.asarray(self.kpts)[unique(self.kpts)[1]]
        kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
        kptij_lst = np.asarray(kptij_lst)

        self.kpt_hash = collections.defaultdict(list)
        for k, kpt in enumerate(self.kpts):
            val = _get_kpt_hash(kpt)
            self.kpt_hash[val].append(k)

        self.kptij_hash = collections.defaultdict(list)
        for k, kpt in enumerate(kptij_lst):
            val = _get_kpt_hash(kpt)
            self.kptij_hash[val].append(k)

        self._cderi = self._make_j3c(kptij_lst)

        return self

    _make_j3c = _make_j3c

    def get_naoaux(self):
        ''' Get the number of auxiliaries.
        '''

        if self._cderi is None:
            self.build()
        return self._cderi.shape[2]

    def sr_loop(self, kpti_kptj=np.zeros((2, 3)), max_memory=2000, compact=True, blksize=None):
        '''
        Short-range part.
        '''

        if self._cderi is None:
            self.build()
        kpti, kptj = kpti_kptj
        ki = self.kpt_hash[_get_kpt_hash(kpti)][0]
        kj = self.kpt_hash[_get_kpt_hash(kptj)][0]
        naux = self.get_naoaux()
        Lpq = self._cderi
        if blksize is None:
            blksize = naux

        for p0, p1 in lib.prange(0, naux, blksize):
            LpqR = Lpq[ki, kj, p0:p1].real
            LpqI = Lpq[ki, kj, p0:p1].imag
            if compact:
                LpqR = lib.pack_tril(LpqR)
                LpqI = lib.pack_tril(LpqI)
            yield LpqR, LpqI, 1

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        '''
        Build the J (direct) and K (exchange) contributions to the Fock
        matrix due to a given density matrix.
        '''

        if hermi != 1 or kpts_band is not None or omega is not None:
            raise ValueError('%s.get_jk only supports the default keyword arguments:\n'
                             '\thermi=1\n\tkpts_band=None\n\tomega=None' % self.__class__)
        if kpts is not None and not np.allclose(kpts, self.kpts):
            raise ValueError('%s.get_jk only supports kpts=None or kpts=GDF.kpts' % self.__class__)

        kpts = self.kpts
        nkpts = len(kpts)
        nao = self.cell.nao_nr()
        kconserv = get_kconserv(self.cell, kpts)
        Lpq = self._cderi

        vj = vk = None
        if with_j:
            vj = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        if with_k:
            vk = np.zeros((nkpts, nao, nao), dtype=np.complex128)

        for kik in mpi_helper.nrange(nkpts**2):
            ki, kk = divmod(kik, nkpts)
            kj = ki
            kl = kconserv[ki, kj, kk]

            # ijkl,lk->ij
            if with_j:
                buf = np.dot(Lpq[kk, kl].reshape(-1, nao*nao), dm[kl].ravel())
                vj[ki] += np.dot(Lpq[ki, kj].T, buf).reshape(nao, nao)

            # ilkj,lk->ij
            if with_k:
                buf = np.dot(Lpq[ki, kl].reshape(-1, nao), dm[kl])
                buf = buf.reshape(-1, nao, nao).swapaxes(1, 2).reshape(-1, nao)
                vk[ki] += np.dot(buf.T, Lpq[kk, kj].reshape(-1, nao)).conj()

        mpi_helper.barrier()
        if with_j:
            mpi_helper.allreduce_safe_inplace(vj[ki])
            vj /= nkpts
        if with_k:
            mpi_helper.allreduce_safe_inplace(vk[ki])
            vk /= nkpts

        if with_k and exxdiv == 'ewald':
            vk = vk[None]
            _ewald_exxdiv_for_G0(self.cell, kpts, _format_dms(dm, kpts), vk)
            vk = vk[0]

        return vj, vk

    def get_eri(self, kpts=None, compact=COMPACT):
        '''
        Get the four-center AO electronic repulsion integrals at a
        given k-point.
        '''

        if self._cderi is None:
            self.build()

        nao = self.cell.nao_nr()
        naux = self.get_naoaux()
        if kpts is None:
            kpts = self.kpts
        kptijkl = _format_kpts(kpts)
        if not _iskconserv(self.cell, kptijkl):
            self.log.warning(self.cell, 'Momentum conservation not found in '
                                        'the given k-points %s', kptijkl)
            return np.zeros((nao, nao, nao, nao))
        ki, kj, kk, kl = (self.kpt_hash[_get_kpt_hash(kpt)][0] for kpt in kptijkl)

        Lpq = self._cderi[ki, kj]
        Lrs = self._cderi[kk, kl]
        if gamma_point(kptijkl):
            Lpq = Lpq.real
            Lrs = Lrs.real
            if compact:
                Lpq = lib.pack_tril(Lpq)
                Lrs = lib.pack_tril(Lrs)
        Lpq = Lpq.reshape(naux, -1)
        Lrs = Lrs.reshape(naux, -1)

        eri = np.dot(Lpq.T, Lrs)

        return eri

    get_ao_eri = get_eri

    def ao2mo(self, mo_coeffs, kpts=None, compact=COMPACT):
        '''
        Get the four-center MO electronic repulsion integrals at a
        given k-point.
        '''

        if self._cderi is None:
            self.build()

        nao = self.cell.nao_nr()
        naux = self.get_naoaux()
        if kpts is None:
            kpts = self.kpts
        kptijkl = _format_kpts(kpts)
        if not _iskconserv(self.cell, kptijkl):
            self.log.warning(self.cell, 'Momentum conservation not found in '
                                        'the given k-points %s', kptijkl)
            return np.zeros((nao, nao, nao, nao))
        ki, kj, kk, kl = (self.kpt_hash[_get_kpt_hash(kpt)][0] for kpt in kptijkl)

        if isinstance(mo_coeffs, np.ndarray) and mo_coeffs.ndim == 2:
            mo_coeffs = (mo_coeffs,) * 4
        all_real = not any(np.iscomplexobj(mo) for mo in mo_coeffs)

        Lij = _fao2mo(self._cderi[ki, kj], mo_coeffs[0], mo_coeffs[1])
        Lkl = _fao2mo(self._cderi[kk, kl], mo_coeffs[2], mo_coeffs[3])
        if gamma_point(kptijkl) and all_real:
            Lij = Lij.real
            Lkl = Lkl.real
            if compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1]):
                Lij = lib.pack_tril(Lij)
            if compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3]):
                Lkl = lib.pack_tril(Lkl)
        Lij = Lij.reshape(naux, -1)
        Lkl = Lkl.reshape(naux, -1)

        eri = np.dot(Lij.T, Lkl)

        return eri

    get_mo_eri = ao2mo

    def get_3c_eri(self, kpts=None, compact=COMPACT):
        '''
        Get the three-center AO  electronic repulsion integrals at a
        given k-point.
        '''

        if self._cderi is None:
            self.build()

        naux = self.get_naoaux()
        if kpts is None:
            kpts = self.nkpts
        kptij = _format_kpts(kpts)
        if not _iskconserv(self.cell, kptij):
            self.log.warning(self.cell, 'Momentum conservation not found in '
                                        'the given k-points %s', kptij)
            return np.zeros_like(self._cderi)
        ki, kj = (self.kpts_hash[_get_kpt_hash(kpt)][0] for kpt in kptij)

        Lpq = self._cderi[ki, kj]
        if gamma_point(kptij):
            Lpq = Lpq.real
            if compact:
                Lpq = lib.pack_tril(Lpq)
        Lpq = Lpq.reshape(naux, -1)

        return Lpq

    get_ao_3c_eri = get_3c_eri

    def ao2mo_3c(self, mo_coeffs, kpts=None, compact=COMPACT):
        '''
        Get the three-center MO electronic repulsion integrals at a
        given k-point.
        '''

        if self._cderi is None:
            self.build()

        naux = self.get_naoaux()
        if kpts is None:
            kpts = self.nkpts
        kptij = _format_kpts(kpts)
        if not _iskconserv(self.cell, kptij):
            self.log.warning(self.cell, 'Momentum conservation not found in '
                                        'the given k-points %s', kptij)
            return np.zeros_like(self._cderi)
        ki, kj = (self.kpts_hash[_get_kpt_hash(kpt)][0] for kpt in kptij)

        if isinstance(mo_coeffs, np.ndarray) and mo_coeffs.ndim == 2:
            mo_coeffs = (mo_coeffs,) * 2
        all_real = not any(np.iscomplexobj(mo) for mo in mo_coeffs)

        Lij = _fao2mo(self._cderi[ki, kj], *mo_coeffs)
        if gamma_point(kptij) and all_real:
            Lij = Lij.real
            if compact and iden_coeffs(*mo_coeffs):
                Lij = lib.pack_tril(Lij)
        Lij = Lij.reshape(naux, -1)

        return Lij

    get_mo_3c_eri = ao2mo_3c

    def get_nuc(self, kpts=None):
        if not (kpts is None or kpts is self.kpts or np.allclose(kpts, self.kpts)):
            return super().get_nuc(kpts)
        if self._get_nuc is None:
            self._get_nuc = super().get_nuc(kpts)
        return self._get_nuc

    def get_pp(self, kpts=None):
        if not (kpts is None or kpts is self.kpts or np.allclose(kpts, self.kpts)):
            return super().get_pp(kpts)
        if self._get_pp is None:
            self._get_pp = super().get_pp(kpts)
        return self._get_pp


DF = GDF

del COMPACT


if __name__ == '__main__':
    #TODO: remove after making unit tests

    from pyscf.pbc import gto

    cell = gto.Cell()
    cell.atom = 'He 0 0 0; He 1 0 1'
    cell.a = np.eye(3) * 2
    cell.basis = '6-31g'
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts([3, 2, 1])

    t0 = time.time()

    df1 = df.GDF(cell, kpts)
    df1.build()
    eri1 = df1.ao2mo_7d([np.array([np.eye(cell.nao)]*len(kpts))]*4)

    t1 = time.time()

    df2 = GDF(cell, kpts)
    df2.build()
    eri2 = df2.ao2mo_7d([np.array([np.eye(cell.nao)]*len(kpts))]*4)

    t2 = time.time()

    print(lib.finger(eri1), lib.finger(eri2))
    print(t1-t0, t2-t1)
    assert np.allclose(eri1, eri2)
