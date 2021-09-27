'''
Functionally equivalent to PySCF GDF, with all storage incore and
MPI parallelism.

Adapted from pyscf.pbc.df.df

Ref:
J. Chem. Phys. 147, 164119 (2017)
'''

import time
import collections
import numpy as np
import scipy.linalg

from pyscf import lib, ao2mo, __config__
from pyscf.lib import logger
from pyscf.agf2 import mpi_helper
from pyscf.ao2mo.outcore import balance_partition
from pyscf.ao2mo.incore import iden_coeffs, _conc_mos
from pyscf.pbc.df import df, incore, ft_ao
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0, _format_dms
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique, KPT_DIFF_TOL, get_kconserv
from pyscf.pbc.df.fft_ao2mo import _iskconserv, _format_kpts


COMPACT = getattr(__config__, 'pbc_df_ao2mo_get_eri_compact', True)


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

    kdif = np.einsum('wx,ix->wi', a, uniq_kpts + kpt)
    kdif_int = np.rint(kdif)

    mask = np.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
    uniq_kptji_ids = np.where(mask)[0]

    return uniq_kptji_ids


def _get_2c2e(fused_cell, uniq_kpts, log):
    '''
    Get the bare two-center two-electron interaction, first term
    of Eq. 32.
    '''

    int2c2e = fused_cell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)

    return int2c2e


def _get_3c2e(cell, fused_cell, kptij_lst, log):
    '''
    Get the bare three-center two-electron interaction, first term
    of Eq. 31.
    '''

    t1 = (logger.process_clock(), logger.perf_counter())

    nkij = len(kptij_lst)
    nao = cell.nao_nr()
    ngrids = fused_cell.nao_nr()
    aux_loc = fused_cell.ao_loc_nr(fused_cell.cart)

    int3c2e = np.zeros((nkij, ngrids, nao*nao), dtype=np.complex128)

    for p0, p1 in mpi_helper.prange(0, fused_cell.nbas, fused_cell.nbas):
        log.debug2('3c2e part [%d -> %d] of %d' % (p0, p1, fused_cell.nbas))

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

        log.timer_debug1('3c2e part', *t1)

    mpi_helper.allreduce_safe_inplace(int3c2e)
    mpi_helper.barrier()

    return int3c2e


def _get_j2c(with_df, cell, auxcell, fused_cell, fuse, int2c2e, uniq_kpts, log):
    '''
    Build j2c using the 2c2e interaction, int2c2e, Eq. 32.
    '''

    naux = auxcell.nao_nr()
    mesh = with_df.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    b = cell.reciprocal_vectors()

    j2c = int2c2e.copy()

    for k, kpt in enumerate(uniq_kpts):
        coulG = with_df.weighted_coulG(kpt, False, mesh)
        aoaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        LkR = np.asarray(aoaux.real, order='C')
        LkI = np.asarray(aoaux.imag, order='C')
        aoaux = None

        # eq. 31 final three terms:
        if is_zero(kpt):  # kpti == kptj
            j2c[k][naux:] = int2c2e[k][naux:] + (
                    - lib.ddot(LkR[naux:]*coulG, LkR.T)
                    - lib.ddot(LkI[naux:]*coulG, LkI.T)
            )
            j2c[k][:naux, naux:] = j2c[k][naux:, :naux].T
        else:
            j2cR, j2cI = df.df_jk.zdotCN(LkR[naux:]*coulG, LkI[naux:]*coulG, LkR.T, LkI.T)
            j2c[k][naux:] = int2c2e[k][naux:] - (j2cR + j2cI * 1j)
            j2c[k][:naux, naux:] = j2c[k][naux:, :naux].T.conj()

        LkR = LkI = None
        coulG = None
        j2c[k] = fuse(fuse(j2c[k]).T).T

    return j2c


def _cholesky_decomposed_metric(with_df, cell, j2c, uniq_kptji_id, log):
    '''
    Get the Cholesky decomposed j2c.
    '''

    j2c_kpt = j2c[uniq_kptji_id]

    try:
        j2c_kpt = scipy.linalg.cholesky(j2c_kpt, lower=True)
        j2ctag = 'CD'
    except scipy.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(j2c_kpt)
        cond = w[-1] / w[0]
        mask = w > with_df.linear_dep_threshold
        log.debug('DF metric linear dependency for kpt %s', uniq_kptji_id)
        log.debug('cond = %.4g, drop %d bfns', cond, np.sum(mask))
        v1 = v[:, mask].conj().T
        v1 /= np.sqrt(w[mask])[:, None]
        j2c_kpt = v1
        w = v = None
        j2ctag = 'eig'

    return j2c_kpt, j2ctag


def _get_j3c(with_df, cell, auxcell, fused_cell, fuse, j2c_k, int3c2e,
             uniq_kpts, uniq_inverse_dict, kptij_lst, log, out=None):
    '''
    Build j2c using the 2c2e interaction, int2c2e, Eq. 31, and then
    contract with the Cholesky decomposed j2c.
    '''

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    Gv, Gvbase, kws = cell.get_Gv_weights(with_df.mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]
    b = cell.reciprocal_vectors()
    j2c = j2ctag = None
    kpts = with_df.kpts
    nkpts = len(kpts)

    if out is None:
        out = np.zeros((nkpts, nkpts, naux, nao, nao), dtype=np.complex128)

    for uniq_kpt_ji_id in mpi_helper.nrange(len(uniq_kpts)):
        kpt = uniq_kpts[uniq_kpt_ji_id]
        log.debug1("Constructing j3c for kpt %s", uniq_kpt_ji_id)
        log.debug1('kpt = %s', kpt)

        if j2c_k is not None:
            log.debug1('Cholesky decomposition for j2c at kpt %s', uniq_kpt_ji_id)
            kptji_id = _kconserve_indices(cell, uniq_kpts, -kpt)[0]
            j2c, j2ctag = _cholesky_decomposed_metric(with_df, cell, j2c_k, kptji_id, log)

        adapted_ji_idx = uniq_inverse_dict[uniq_kpt_ji_id]
        adapted_kptjs = kptij_lst[:, 1][adapted_ji_idx]
        log.debug1('adapted_ji_idx = %s', adapted_ji_idx)

        shls_slice = (auxcell.nbas, fused_cell.nbas)
        Gaux = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        Gaux *= with_df.weighted_coulG(kpt, False, with_df.mesh).ravel()[:, None]
        kLR = Gaux.real.copy('C')
        kLI = Gaux.imag.copy('C')
        del Gaux

        if is_zero(kpt):
            vbar = fuse(with_df.auxbar(fused_cell))
            ovlp = cell.pbc_intor('int1e_ovlp', hermi=0, kpts=adapted_kptjs)
            ovlp = [np.ravel(s) for s in ovlp]

        shranges = balance_partition(cell.ao_loc_nr()*nao, nao*nao)
        pqkRbuf = np.empty(nao*nao*ngrids)
        pqkIbuf = np.empty(nao*nao*ngrids)
        buf = np.empty(len(adapted_kptjs)*nao*nao*ngrids, dtype=np.complex128)

        bstart, bend, ncol = shranges[0]
        shls_slice = (bstart, bend, 0, cell.nbas)
        dat = ft_ao.ft_aopair_kpts(cell, Gv, shls_slice, 's1', b, gxyz,
                                   Gvbase, kpt, adapted_kptjs, out=buf)

        for kji, ji in enumerate(adapted_ji_idx):
            # eq. 31 second term:
            v = int3c2e[ji]
            if is_zero(kpt):
                for i in np.where(vbar != 0)[0]:
                    v[i] -= vbar[i] * ovlp[kji]

            j3cR = np.asarray(v.real, order='C')
            if not (is_zero(kpt) and gamma_point(adapted_kptjs[kji])):
                j3cI = np.asarray(v.imag, order='C')

            pqkR = np.ndarray((ncol, ngrids), buffer=pqkRbuf)
            pqkI = np.ndarray((ncol, ngrids), buffer=pqkIbuf)
            pqkR[:] = dat[kji].reshape(ngrids, ncol).T.real
            pqkI[:] = dat[kji].reshape(ngrids, ncol).T.imag

            # eq. 31 final term:
            lib.dot(kLR.T, pqkR.T, -1, j3cR[naux:], 1)
            lib.dot(kLI.T, pqkI.T, -1, j3cR[naux:], 1)
            if not (is_zero(kpt) and gamma_point(adapted_kptjs[kji])):
                lib.dot(kLR.T, pqkI.T, -1, j3cI[naux:], 1)
                lib.dot(kLI.T, pqkR.T,  1, j3cI[naux:], 1)

            if is_zero(kpt) and gamma_point(adapted_kptjs[kji]):
                v = fuse(j3cR)
            else:
                v = fuse(j3cR + j3cI * 1j)

            if j2ctag == 'CD':
                v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
            elif j2ctag == 'eig':
                v = lib.dot(j2c, v)
            v = v.reshape(-1, nao, nao)

            for ki in with_df.kpt_hash[_get_kpt_hash(kptij_lst[ji][0])]:
                for kj in with_df.kpt_hash[_get_kpt_hash(kptij_lst[ji][1])]:
                    out[ki, kj, :v.shape[0]] += v
                    if ki != kj:
                        out[kj, ki, :v.shape[0]] += lib.transpose(v, axes=(0, 2, 1)).conj()

        del j3cR, j3cI

    mpi_helper.allreduce_safe_inplace(out)
    mpi_helper.barrier()

    return out


def _make_j3c(with_df, cell, auxcell, kptij_lst):
    '''
    Build the j3c array.

    cell: the unit cell for the calculation
    auxcell: the unit cell for the auxiliary functions
    chgcell: the unit cell for the smooth Gaussians
    fused_cell: auxcell and chgcell combined
    '''

    log = logger.Logger(with_df.stdout, with_df.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    fused_cell, fuse = df.fuse_auxcell(with_df, auxcell)

    if cell.dimension < 3:
        raise ValueError('GDF does not support low-dimension cells')

    kptis = kptij_lst[:, 0]
    kptjs = kptij_lst[:, 1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    uniq_inverse_dict = { k: np.where(uniq_inverse == k)[0] for k in range(len(uniq_kpts)) }

    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)

    # Get the 3c2e interaction:
    int3c2e = _get_3c2e(cell, fused_cell, kptij_lst, log)
    t1 = log.timer_debug1('_get_3c2e', *t1)

    # Get the 2c2e interaction:
    int2c2e = _get_2c2e(fused_cell, uniq_kpts, log)
    t1 = log.timer_debug1('_get_2c2e', *t1)

    # Get j2c:
    j2c = _get_j2c(with_df, cell, auxcell, fused_cell,
                   fuse, int2c2e, uniq_kpts, log)
    t1 = log.timer_debug1('_get_j2c', *t1)

    # Get j3c:
    j3c = _get_j3c(with_df, cell, auxcell, fused_cell, fuse, j2c, int3c2e,
                   uniq_kpts, uniq_inverse_dict, kptij_lst, log)
    t1 = log.timer_debug1('_get_j3c', *t1)

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

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        j_only = j_only or self._j_only
        if j_only:
            raise ValueError('%s does not support j_only=%s' % (self.__class__, j_only))
        if not with_j3c:
            raise ValueError('%s does not support with_j3c=%s' % (self.__class__, with_j3c))
        if kpts_band is not None:
            raise ValueError('%s does not support kpts_band=%s' % (self.__class__, kpts_band))
        if self.cell.dimension < 3 and self.cell.low_dim_ft_type != 'inf_vacuum':
            raise NotImplementedError('%s for low dimensionality systems' % self.__class__)

        self.check_sanity()
        self.dump_flags()

        self.auxcell = df.make_modrho_basis(self.cell, self.auxbasis, self.exp_to_discard)

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

        t0 = (logger.process_clock(), logger.perf_counter())

        self._cderi = self._make_j3c(self.cell, self.auxcell, kptij_lst)

        logger.timer_debug1(self, 'j3c', *t0)

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
            logger.warn(self.cell, 'Momentum conservation not found in '
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
            logger.warn(self.cell, 'Momentum conservation not found in '
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
            logger.warn(self.cell, 'Momentum conservation not found in '
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
            logger.warn(self.cell, 'Momentum conservation not found in '
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

    kpts = cell.make_kpts([2, 2, 2])

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
