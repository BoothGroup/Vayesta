import numpy as np
import scipy.linalg
import ctypes

from pyscf import ao2mo, lib
from pyscf.agf2 import mpi_helper
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools.k2gamma import get_phase

from vayesta import libs
from vayesta.eagf2 import ragf2
from vayesta.core import linalg, foldscf
from vayesta.core.ao2mo import kao2gmo

libeagf2 = libs.load_library('eagf2')


def make_dmet_bath(frag, c_frag=None, c_env=None, tol=1e-5):
    ''' Make DMET bath orbitals.
    '''

    if c_env.shape[-1] == 0:
        nmo = c_env.shape[0]
        return np.zeros((nmo, 0)), np.zeros((nmo, 0)), np.zeros((nmo, 0))

    dm = frag.mf.make_rdm1(frag.base.qmo_coeff, frag.base.qmo_occ)
    dm_env = np.linalg.multi_dot((c_env.T.conj(), dm, c_env)) / 2

    try:
        eig, r = np.linalg.eigh(dm_env)
    except np.linalg.LinAlgError:
        eig, r = scipy.linalg.eigh(dm_env)
    eig, r = eig[::-1], r[:, ::-1]

    if eig.min() < -1e-9:
        frag.log.warning("Min eigenvalue of env. DM = %.6e", eig.min())
    if (eig.max()-1) > 1e-9:
        frag.log.warning("Max eigenvalue of env. DM = %.6e", eig.max())

    c_env = np.dot(c_env, r)

    mask_bath = np.logical_and(eig >= tol, eig <= 1-tol)
    mask_env_occ = (eig > 1-tol)
    mask_env_vir = (eig < tol)
    nbath = sum(mask_bath)

    nocc_env = sum(mask_env_occ)
    nvir_env = sum(mask_env_vir)
    frag.log.info(
            "DMET bath:  n(Bath)= %4d  n(occ-Env)= %4d  n(vir-Env)= %4d",
            nbath, nocc_env, nvir_env,
    )
    assert (nbath + nocc_env + nvir_env == c_env.shape[-1])

    c_bath = c_env[:, mask_bath].copy()
    c_env_occ = c_env[:, mask_env_occ].copy()
    c_env_vir = c_env[:, mask_env_vir].copy()

    print_tol = 1e-10
    strong_tol = 0.1
    limits = [print_tol, tol, strong_tol, 1-strong_tol, 1-tol, 1-print_tol]
    if np.any(np.logical_and(eig > limits[0], eig <= limits[-1])):
        names = [
                "Unentangled vir. env. orbital",
                "Weakly-entangled vir. bath orbital",
                "Strongly-entangled bath orbital",
                "Weakly-entangled occ. bath orbital",
                "Unentangled occ. env. orbital",
                ]
        frag.log.info("Non-(0 or 1) eigenvalues (n) of environment DM:")
        for i, e in enumerate(eig):
            name = None
            for j, llim in enumerate(limits[:-1]):
                ulim = limits[j+1]
                if (llim < e and e <= ulim):
                    name = names[j]
                    break
            if name:
                frag.log.info("  > %-34s  n= %12.6g  1-n= %12.6g", name, e, 1-e)

    entropy = np.sum(eig * (1-eig))
    entropy_bath = np.sum(eig[mask_bath] * (1-eig[mask_bath]))
    frag.log.info(
            "Entanglement entropy: total= %.6e  bath= %.6e  captured=  %.2f %%",
            entropy, entropy_bath, 100.0*entropy_bath/entropy,
    )

    return c_bath, c_env_occ, c_env_vir


def make_power_bath(frag, max_order=0, svd_tol=1e-16, c_frag=None, c_env=None, tol=1e-5):
    ''' Make power bath orbitals up to a maximum order.
    '''

    if c_frag is None: c_frag = frag.c_frag
    if c_env is None: c_env = frag.c_env

    qmo_energy = frag.base.qmo_energy
    qmo_coeff = frag.base.qmo_coeff
    fock = np.dot(qmo_coeff * qmo_energy[None], qmo_coeff.T.conj())

    c_dmet, c_env_occ, c_env_vir = make_dmet_bath(frag, c_frag=c_frag, c_env=c_env, tol=tol)

    if max_order == 0:
        return c_dmet, c_env_occ, c_env_vir

    c_power = []
    for c_env in (c_env_occ, c_env_vir):
        if c_env.shape[1] > 0:
            c = np.hstack((c_frag, c_env))
            f = np.linalg.multi_dot((c.T.conj(), fock, c))
            b, sv, orders = linalg.recursive_block_svd(f, n=c_frag.shape[1], maxblock=max_order)
            c_power.append(np.dot(c_env, b[:, sv >= svd_tol]))

    c_bath = np.hstack((c_dmet, *c_power))
    p_bath = np.linalg.multi_dot((qmo_coeff.T.conj(), c_bath, c_bath.T.conj(), qmo_coeff))
    c_bath = np.dot(qmo_coeff, scipy.linalg.orth(p_bath))

    c_cls = np.hstack((c_frag, c_bath))
    p_cls = np.linalg.multi_dot((qmo_coeff.T.conj(), c_cls, c_cls.T.conj(), qmo_coeff))
    c_env = np.dot(qmo_coeff, scipy.linalg.null_space(p_cls))
    c_env_occ, c_env_vir = frag.diagonalize_cluster_dm(c_env, tol=0.0)

    return c_bath, c_env_occ, c_env_vir


class QMOIntegrals:
    def __init__(self, frag, c_occ, c_vir, c_full=None, which='xija', keep_3c=True, make_real=True, fourier_transform=True):
        self.frag = frag
        self.which = which

        if c_full is None:
            self.c_full = np.hstack((c_occ, c_vir))
        else:
            self.c_full = c_full

        if which == 'xija':
            self.c_occ, self.c_vir = c_occ, c_vir
        elif which == 'xabi':
            self.c_occ, self.c_vir = c_vir, c_occ

        has_kpts = getattr(self.frag.base, 'kpts', None) is not None
        if not has_kpts:
            has_df = getattr(self.frag.mf, 'with_df', None) is not None

        if not has_kpts:
            if has_df:
                self.build_3c(keep_3c)
            else:
                self.build_4c()
        else:
            self.build_pbc(keep_3c, make_real, fourier_transform)

    def build_4c(self):
        coeffs = (self.c_full, self.c_occ, self.c_occ, self.c_vir)
        self.eri = ao2mo.incore.general(self.frag.mf._eri, coeffs, compact=False)
        self.eri = self.eri.reshape([c.shape[1] for c in coeffs])

    def build_3c(self, keep_3c=True):
        if self.frag.mf.with_df._cderi is None:
            with lib.temporary_env(self.frag.mf.with_df, max_memory=1e9):
                self.frag.mf.with_df.build()
        Lpq = np.concatenate([block for block in self.frag.mf.with_df.loop()])
        Lpq = np.asarray(lib.unpack_tril(Lpq, axis=-1))
        Lxo = ragf2._ao2mo_3c(Lpq, self.c_full, self.c_occ, mpi=False)
        Lov = ragf2._ao2mo_3c(Lpq, self.c_occ, self.c_vir, mpi=False)
        if keep_3c:
            self.eri = (Lxo, Lov)
        else:
            self.eri = lib.einsum('Lxi,Lja->xija', Lxo, Lov)

    def build_pbc(self, keep_3c=True, make_real=True, fourier_transform=True):
        ints3c = kao2gmo.ThreeCenterInts.init_from_gdf(self.frag.base.kdf)

        cx = self.c_full.reshape(ints3c.nk, ints3c.nao, -1)
        ci = self.c_occ.reshape(ints3c.nk, ints3c.nao, -1)
        ca = self.c_vir.reshape(ints3c.nk, ints3c.nao, -1)

        if fourier_transform:
            phase = get_phase(ints3c.cell, ints3c.kpts)[1]
            cx = lib.einsum('rk,rai->kai', phase.conj(), cx)
            ci = lib.einsum('rk,rai->kai', phase.conj(), ci)
            ca = lib.einsum('rk,rai->kai', phase.conj(), ca)

        Lxi = kao2gmo.j3c_kao2gmo(ints3c, cx, ci, only_ov=True, make_real=make_real)['ov']
        Lxi = Lxi.reshape(-1, cx.shape[-1], ci.shape[-1])
        Lja = kao2gmo.j3c_kao2gmo(ints3c, ci, ca, only_ov=True, make_real=make_real)['ov']
        Lja = Lja.reshape(-1, ci.shape[-1], ca.shape[-1])

        Lxi /= np.sqrt(ints3c.nk)
        Lja /= np.sqrt(ints3c.nk)

        if keep_3c:
            self.eri = (Lxi, Lja)
        else:
            self.eri = lib.einsum('Lxi,Lja->xija', Lxi, Lja)

    def __enter__(self):
        return self.eri

    def __exit__(self, *args):
        del self.eri


def block_lanczos(moments, tol=None):
    '''
    Perform the block Lanczos algorithm for an auxiliary space using
    the moments of the spectral function. Scales as O(N^3) where the
    moments have shape (N, N).
    '''

    dtype = np.result_type(*[m.dtype for m in moments])

    nmo = moments[0].shape[0]
    nmom = (len(moments) - 2) // 2
    nblock = nmom + 1

    if tol is None:
        tol = np.finfo(dtype).eps * nmo

    def sqrt_and_inv(x):
        try:
            w, v = np.linalg.eigh(x)
        except np.linalg.LinAlgError:
            w, v = scipy.linalg.eigh(x)
        w, v = w[w > tol], v[:, w > tol]
        bi = np.dot(v * w[None]**0.5, v.T.conj())
        binv = np.dot(v * w[None]**-0.5, v.T.conj())
        return bi, binv

    class C(dict):
        def __getitem__(self, key):
            i, n, j = key
            if i == 0 or j == 0:
                return np.zeros((nmo, nmo), dtype=dtype)
            elif i < j:
                return super().__getitem__((j, n, i)).T.conj()
            else:
                return super().__getitem__((i, n, j))

        def __setitem__(self, key, val):
            i, n, j = key
            if i < j:
                super().__setitem__((j, n, i), val.T.conj())
            else:
                super().__setitem__((i, n, j), val)

    class CB(dict):
        def __getitem__(self, key):
            i, n, j = key
            if (i, n, j) not in self:
                self[i, n, j] = np.dot(c[i, n, j], b[j].conj())
            return super().__getitem__((i, n, j))

    class MC(dict):
        def __getitem__(self, key):
            i, n, j = key
            if (i, n, j) not in mc:
                self[i, n, j] = np.dot(c[i, 1, i], c[i, n, j])
            return super().__getitem__((i, n, j))

    m = np.zeros((nblock+1, nmo, nmo), dtype=dtype)
    b = np.zeros((nblock,   nmo, nmo), dtype=dtype)
    c = C()
    cb = CB()
    mc = MC()

    def cIi(i, n):
        tmp  = c[i, n+1, i].copy()
        tmp -= cb[i, n, i-1].T.conj()
        tmp -= mc[i, n, i]
        c[i+1, n, i] = np.dot(binv.conj(), tmp)

    def cII(i, n):
        tmp  = c[i, n+2, i].copy()
        tmp -= lib.hermi_sum(cb[i, n+1, i-1])
        tmp -= lib.hermi_sum(mc[i, n+1, i])
        tmp += lib.hermi_sum(np.dot(c[i, 1, i], cb[i, n, i-1]))
        tmp += np.dot(b[i-1], cb[i-1, n, i-1])
        tmp += np.dot(mc[i, n, i], c[i, 1, i].T.conj()) #TODO yes?
        c[i+1, n, i+1] = np.linalg.multi_dot((binv.conj(), tmp, binv))

    def b2(i):
        b2  = c[i, 2, i].copy()
        b2 -= lib.hermi_sum(cb[i, 1, i-1])
        b2 -= np.dot(c[i, 1, i], c[i, 1, i].T.conj())
        if i > 1: b2 += np.dot(b[i-1], b[i-1].T.conj())
        return b2

    b[0], binv = sqrt_and_inv(moments[0])

    # Orthogonalise the moments:
    for n in range(2*nmom+2):
        c[1, n, 1] = np.linalg.multi_dot((binv, moments[n], binv))

    for i in range(1, nblock):
        # Build b^1/2 and b^-1/2:
        b[i], binv = sqrt_and_inv(b2(i))

        # Force orthogonality in n == 0:
        n = 0
        c[i+1, n, i] = np.zeros((nmo, nmo))
        c[i+1, n, i+1] = np.eye(nmo)

        # Begin recursions:
        for n in range(1, 2*(nblock-i)-1):
            # Compute C_{i+1, i, n}:
            cIi(i, n)

            # Compute C_{i+1, i+1, n}:
            cII(i, n)

        # Compute C_{i+1, i+1, 2*(nblock-i)-1}:
        cII(i, n+1)

    # Exact M blocks, M_{i} = C_{i, i, 1}
    for i in range(nblock+1):
        m[i] = c[i, 1, i]

    return m, b


def block_tridiagonal(m, b):
    ''' Build a block tridiagonal matrix from a list of on- (m) and off-
        diagonal (b) blocks of matrices.
    '''

    nphys = m[0].shape[0]
    dtype = np.result_type(*([x.dtype for x in m] + [x.dtype for x in b]))

    assert all([x.shape == (nphys, nphys) for x in m])
    assert all([x.shape == (nphys, nphys) for x in b])
    assert len(m) == len(b)+1

    zero = np.zeros((nphys, nphys), dtype=dtype)

    h = np.block([[m[i]          if i == j   else
                   b[j]          if j == i-1 else
                   b[i].T.conj() if i == j-1 else
                   zero
                   for j in range(len(m))]
                   for i in range(len(m))])

    return h


def _orth(p, tol=None, nvecs=None):
    w, v = np.linalg.eigh(p)

    if tol is None:
        if nvecs is not None:
            arg = np.argsort(np.abs(w))
            mask = arg[(w.size-nvecs):]
        else:
            tol = np.max(p.shape) * np.max(np.abs(w)) * np.finfo(v.dtype).eps
            mask = np.abs(w) > tol

    return v[:, mask]


def orth(v, tol=None, nvecs=None):
    ''' Orthonormalise vectors.
    '''

    return _orth(np.dot(v, v.T.conj()), tol=tol, nvecs=nvecs)


def null_space(v, tol=None, nvecs=None):
    ''' Construct orthonormal vectors for null space of subspace formed of v. 
    '''

    return _orth(np.eye(v.shape[0]) - np.dot(v, v.T.conj()), tol=tol, nvecs=nvecs)


class CGreensFunction(ctypes.Structure):
    '''
    C structure for the pyscf.agf2.aux.GreensFunction object.
    '''

    _fields_ = [
            ('nocc', ctypes.c_int32),
            ('nvir', ctypes.c_int32),
            ('ei', ctypes.POINTER(ctypes.c_double)),
            ('ea', ctypes.POINTER(ctypes.c_double)),
            ('ci', ctypes.POINTER(ctypes.c_double)),
            ('ca', ctypes.POINTER(ctypes.c_double)),
    ]

    @classmethod
    def build(struct, gf):
        gf_occ = gf.get_occupied()
        gf_vir = gf.get_virtual()

        c_occ = np.array(gf_occ.coupling.ravel(), order='C', dtype=np.complex128)
        c_vir = np.array(gf_vir.coupling.ravel(), order='C', dtype=np.complex128)

        cgf = struct(
                ctypes.c_int32(gf_occ.naux),
                ctypes.c_int32(gf_vir.naux),
                np.ctypeslib.as_ctypes(gf_occ.energy),
                np.ctypeslib.as_ctypes(gf_vir.energy),
                np.ctypeslib.as_ctypes(c_occ.view(np.float64)),
                np.ctypeslib.as_ctypes(c_vir.view(np.float64)),
        )

        return cgf


def build_moments_kagf2(gf, eri, kconserv, nmom, kptlist=None):
    '''
    Construct the moments via compiled code for KAGF2.
    '''

    nkpts, _, naux, nmo, _ = eri.shape

    if kptlist is None:
        kptlist = list(range(nkpts))

    t_occ = np.zeros((nkpts, nmom, nmo, nmo), dtype=np.complex128)
    t_vir = np.zeros((nkpts, nmom, nmo, nmo), dtype=np.complex128)

    eri = np.asarray(eri, dtype=np.complex128, order='C')
    kconserv = np.asarray(kconserv, dtype=np.int32, order='C')
    kptlist = np.asarray(kptlist, dtype=np.int32, order='C')
    nk3 = kptlist.size * nkpts**2
    krange = np.array(list(mpi_helper.prange(0, nk3, nk3)), dtype=np.int32)
    cgf = [CGreensFunction.build(g) for g in gf]

    pointer = lambda x: x.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

    libeagf2.construct_moments_kagf2(
            ctypes.c_int32(nmo),
            ctypes.c_int32(naux),
            ctypes.c_int32(nkpts),
            ctypes.c_int32(kptlist.size),
            ctypes.c_int32(nmom),
            (CGreensFunction * len(gf))(*cgf),
            pointer(eri),
            pointer(kconserv),
            pointer(kptlist),
            pointer(krange),
            pointer(t_occ),
            pointer(t_vir),
    )

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(t_occ)
    mpi_helper.allreduce_safe_inplace(t_vir)

    return t_occ, t_vir


def build_moments(ei, ea, ej, xiaj, yiaj=None, yjai=None, nmom=2, os_factor=1.0, ss_factor=1.0):
    '''
    Construct the moments via compiled code. Generalised for asymmetry
    in i/j and x/y.

     (0)    
    T    = Σ    (xi|ja) [ 2 (yi|ja) - (yj|ia) ]
     x,y    ija

     (1)    
    T    = Σ    (xi|ja) [ 2 (yi|ja) - (yj|ia) ] (ε_i + ε_j - ε_a)
     x,y    ija
    '''

    #TODO MPI

    ni, na, nj = ei.size, ej.size, ea.size
    nmo_p = xiaj.shape[0]
    nmo_q = nmo_p if yiaj is None else yiaj.shape[0]

    def pointer(x):
        if x is None:
            return ctypes.POINTER(ctypes.c_void_p)()
        else:
            return x.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

    if xiaj.dtype in [np.float64, float]:
        dtype = np.float64
        fdrv = libeagf2.construct_moments_real_4c
    else:
        dtype = np.complex128
        fdrv = libeagf2.construct_moments_cplx_4c

    xiaj = np.asarray(xiaj, dtype=dtype, order='C')
    assert xiaj.size == (nmo_p * ni * na * nj)

    if yiaj is not None:
        yiaj = np.asarray(yiaj, dtype=dtype, order='C')
        assert yiaj.size == (nmo_q * ni * na * nj)

    if yjai is not None:
        yjai = np.asarray(yjai, dtype=dtype, order='C')
        assert yjai.size == (nmo_q * nj * na * ni)

    t = np.zeros((nmom, nmo_p, nmo_q), dtype=dtype)

    fdrv(
            ctypes.c_int32(nmo_p),
            ctypes.c_int32(nmo_q),
            ctypes.c_int32(ni),
            ctypes.c_int32(na),
            ctypes.c_int32(nj),
            ctypes.c_int32(nmom),
            ctypes.c_int32(nmo_p),
            ctypes.c_int32(nmo_q),
            pointer(xiaj),
            pointer(yiaj),
            pointer(yjai),
            pointer(ei),
            pointer(ea),
            pointer(ej),
            pointer(t),
    )

    return t
