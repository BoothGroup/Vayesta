import numpy as np
import scipy.linalg

from pyscf import ao2mo, lib

from vayesta.eagf2 import ragf2
from vayesta.core import linalg
from vayesta.core.ao2mo import kao2gmo


def make_dmet_bath(frag, c_frag=None, c_env=None, tol=1e-5):
    ''' Make DMET bath orbitals.
    '''

    if c_env.shape[-1] == 0:
        nmo = c_env.shape[0]
        return np.zeros((nmo, 0)), np.zeros((nmo, 0)), np.zeros((nmo, 0))

    dm = frag.mf.make_rdm1(frag.qmo_coeff, frag.qmo_occ)
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


def make_power_bath(frag, max_order=0, svd_tol=0.0, c_frag=None, c_env=None):
    ''' Make power bath orbitals up to a maximum order.
    '''

    if c_frag is None: c_frag = frag.c_frag
    if c_env is None: c_env = frag.c_env

    fock = frag.se.get_array(frag.fock)
    qmo_coeff = frag.qmo_coeff

    c_dmet, c_env_occ, c_env_vir = make_dmet_bath(frag, c_frag=c_frag, c_env=c_env)

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
    def __init__(self, frag, c_occ, c_vir, which='xija', keep_3c=False):
        self.frag = frag
        self.which = which
        self.c_full = np.hstack((c_occ, c_vir))

        if which == 'xija':
            self.c_occ, self.c_vir = c_occ, c_vir
        elif which == 'xabi':
            self.c_occ, self.c_vir = c_vir, c_occ

        has_kpts = getattr(self.frag.base, 'kpts', None) is not None
        if has_kpts:
            has_df = getattr(self.frag.base, 'kdf', None) is not None
        else:
            has_df = getattr(self.frag.mf, 'with_df', None) is not None

        if not has_df and not has_kpts:
            self.build_4c()
        elif has_df and not has_kpts:
            self.build_3c(keep_3c)
        elif has_df and has_kpts and keep_3c:
            self.build_3c_pbc()
        elif has_df and has_kpts and not keep_3c:
            self.build_4c_pbc()
        else:
            raise NotImplementedError

    def build_4c(self):
        coeffs = (self.c_full, self.c_occ, self.c_occ, self.c_vir)
        self.eri = ao2mo.incore.general(self.frag.mf._eri, coeffs, compact=False)
        self.eri = self.eri.reshape([c.shape[1] for c in coeffs])

    def build_3c(self, keep_3c=False):
        if self.frag.mf.with_df._cderi is None:
            with lib.temporary_env(self.frag.mf.with_df, max_memory=1e9):
                self.frag.mf.with_df.build()
        Lpq = np.concatenate([block for block in self.frag.mf.with_df.loop()])
        Lpq = np.asarray(lib.unpack_tril(Lpq, axis=-1))
        Lxo = ragf2._ao2mo_3c(Lpq, self.c_full, self.c_occ)
        Lov = ragf2._ao2mo_3c(Lpq, self.c_occ, self.c_vir)
        if keep_3c:
            self.eri = (Lxo, Lov)
        else:
            self.eri = lib.einsum('Lxi,Lja->xija', Lxo, Lov)

    def build_4c_pbc(self):
        #TODO test conjugation
        eri = kao2gmo.gdf_to_eris(self.frag.base.kdf, self.c_full, self.c_occ.shape[-1])
        if self.which == 'xija':
            ooov = eri['ovoo'].transpose(2, 3, 0, 1)
            voov = eri['ovvo'].transpose(1, 0, 3, 2).conj()
            self.eri = np.concatenate((ooov, voov), axis=0)
        else:
            ovvo = eri['ovvo']
            vvvo = eri['ovvv'].transpose(3, 2, 1, 0).conj()
            self.eri = np.concatenate((ovvo, vvvo), axis=0)

    def build_3c_pbc(self):
        raise NotImplementedError

    def __enter__(self):
        return self.eri

    def __exit__(self, *args):
        del self.eri

