import numpy as np

import pyscf

from vayesta.eagf2 import util
from vayesta.ewf import helper

LIN_DEP_THRESHOLD = 1e-12
SVD_THRESHOLD = 1e-9
ENV_THRESHOLD = 1e-9

zmax = lambda x: np.max(x) if x.size != 0 else 0.0


def rotate_ov(c, s, rdm1, return_eigvals=False):
    ''' Rotate orbitals into an occupied & virtual representation
    '''

    sc = np.dot(s, c)
    dm = np.linalg.multi_dot((sc.T.conj(), rdm1, sc)) / 2

    w, v = np.linalg.eigh(dm)
    w, v = w[::-1], v[:, ::-1]
    nocc = np.sum(w > 0.5)
    c_rot = np.dot(c, v)

    if return_eigvals:
        return c_rot[:, :nocc], c_rot[:, nocc:], w[:nocc], w[nocc:]
    return c_rot[:, :nocc], c_rot[:, nocc:]


def orthogonality(c, s):
    ''' Return the error in the orthogonality of a set of orbitals
    '''

    p = np.linalg.multi_dot((c.T.conj(), s, c))
    i = np.eye(c.shape[1])

    return zmax(np.abs(p-i))


def idempotency(c, s, rdm1):
    ''' Return the error in the idempotency of the 1RDM in a basis
    '''

    _, _, eo, ev = rotate_ov(c, s, rdm1, return_eigvals=True)

    error_occ = zmax(1-eo) if eo.size else 0.0
    error_vir = zmax(ev) if ev.size else 0.0

    return max(error_occ, error_vir)


def nelec(c, s, rdm1):
    ''' Return the number of electrons in the 1RDM in a basis
    '''

    sc = np.dot(s, c)
    dm = np.linalg.multi_dot((sc.T.conj(), rdm1, sc))

    return np.trace(dm)


def orthonormalise(c, s, mo_coeff, remove_lindep=False, tol=LIN_DEP_THRESHOLD):
    ''' Orthonormalise a set of orbitals
    '''

    nvec = c.shape[1]
    
    c_mo = np.linalg.multi_dot((mo_coeff.T.conj(), s, c))

    norm = np.linalg.norm(c_mo, axis=0, keepdims=True)
    norm[norm == 0] = 1e-18
    c_mo /= norm

    w, v = np.linalg.eigh(np.dot(c_mo, c_mo.T.conj()))
    mask = np.argsort(w)[-nvec:]
    if remove_lindep:
        mask = [x for x in mask if w[x] > tol]
    v = v[:, mask]

    return np.dot(mo_coeff, v)


def make_ewdmet_bath(frag, c_env, nmom=0, svd_full=False, svd_tol=SVD_THRESHOLD, env_tol=ENV_THRESHOLD):
    ''' Calculate EwDMET bath orbitals

    Parameters
    ----------
    frag : QEmbeddingFragment
        Fragment object.
    c_env : ndarray
        Environment (non-fragment) orbital coefficients.
    nmom : int
        Number of bath states to calculate (default value is 0 which is
        equal to a DMET bath).
    svd_full : bool
        If True, SVD the entire occupied & virtual block together
        (default value is False).
    svd_tol : float, optional
        Threshold in singular values to be considered zero (default
        value is `SVD_THRESHOLD`).
        Warning: algorithm may be sensitive to this parameter.
    env_tol : float, optional
        Threshold in eigenvalues of the complement to the cluster space
        to be considered zero (default value is `ENV_THRESHOLD`).
        Warning: algorithm may be sensitive to this parameter.

    Returns
    -------
    c_bath : ndarray
        EwDMET bath orbitals.
    c_env_occ : ndarray
        Occupied environment orbitals.
    c_env_vir : ndarray
        Virtual environment orbitals.
    '''

    log = frag.log

    ovlp = frag.base.get_ovlp()
    rdm1 = frag.mf.make_rdm1()
    fock = frag.base.get_fock()
    mo_coeff = frag.mf.mo_coeff

    c_frag = frag.c_frag
    c_ewdmet_occ = []
    c_ewdmet_vir = []

    # n = 0
    c_dmet, c_env_occ, c_env_vir = frag.make_dmet_bath(frag.c_env)
    c_cls_occ, c_cls_vir = frag.diagonalize_cluster_dm(c_frag, c_dmet)
    c_bath = c_dmet.copy()

    c_cls = np.hstack((c_cls_occ, c_cls_vir))
    c_env = np.hstack((c_env_occ, c_env_vir))

    for n in range(1, nmom+1):
        log.info("EwDMET bath m = %d:", n)
        log.changeIndentLevel(1)

        if svd_full:
            fov = np.linalg.multi_dot((c_cls.T.conj(), fock, c_env))
            u, s, v = np.linalg.svd(fov, full_matrices=False)
            mask = s > svd_tol
            log.info("%d non-zero singular values", np.sum(mask))

            c = np.dot(c_env, v[mask].T.conj())
            c_occ, c_vir = rotate_ov(c)

            c_ewdmet_occ.append(c_occ)
            c_ewdmet_vir.append(c_vir)

        else:
            fo = np.linalg.multi_dot((c_cls_occ.T.conj(), fock, c_env_occ))
            u, s, v = np.linalg.svd(fo, full_matrices=False)
            mask = s > svd_tol
            log.info("%d non-zero occupied singular values", np.sum(mask))
            c_ewdmet_occ.append(np.dot(c_env_occ, v[mask].T.conj()))

            fv = np.linalg.multi_dot((c_cls_vir.T.conj(), fock, c_env_vir))
            u, s, v = np.linalg.svd(fv, full_matrices=False)
            mask = s > svd_tol
            log.info("%d non-zero virtual singular values", np.sum(mask))
            c_ewdmet_vir.append(np.dot(c_env_vir, v[mask].T.conj()))

        c_bath = np.hstack((c_dmet, *c_ewdmet_occ, *c_ewdmet_vir))
        #c_bath = orthonormalise(c_bath, ovlp, mo_coeff)
        c_bath_occ, c_bath_vir = rotate_ov(c_bath, ovlp, rdm1)
        log.info("Total number of bath states:  %d  (nocc=%d, nvir=%d)",
                 c_bath.shape[1], c_bath_occ.shape[1], c_bath_vir.shape[1])
        
        c_cls_occ, c_cls_vir = frag.diagonalize_cluster_dm(c_frag, c_bath)
        c_cls = np.hstack((c_cls_occ, c_cls_vir))
        #c_cls = orthonormalise(c_cls, ovlp, mo_coeff)

        csc = np.linalg.multi_dot((mo_coeff.T.conj(), ovlp, c_cls))
        p = np.dot(csc, csc.T.conj())
        w, v = np.linalg.eigh(np.eye(p.shape[0]) - p)
        c_env = np.dot(mo_coeff, v[:, np.abs(w) >= env_tol])
        c_env_occ, c_env_vir = rotate_ov(c_env, ovlp, rdm1)


        orth_cls = orthogonality(c_cls, ovlp)
        orth_env = orthogonality(c_env, ovlp)
        idem_cls = idempotency(c_cls, ovlp, rdm1)
        idem_env = idempotency(c_env, ovlp, rdm1)
        nelec_cls = nelec(c_cls, ovlp, rdm1)
        nelec_env = nelec(c_env, ovlp, rdm1)

        for key, name in [('cls', 'cluster'), ('env', 'environment')]:
            c = locals()['c_' + key]
            c_occ = locals()['c_' + key + '_occ']
            c_vir = locals()['c_' + key + '_vir']

            orth = orthogonality(c, ovlp)
            idem = idempotency(c, ovlp, rdm1)
            ne = nelec(c, ovlp, rdm1)
            occ = ne - np.rint(ne)

            log.info("Updated %s space:", name)
            (log.info if orth < 1e-8 else log.warning)("  > %-18s: %9.3g", "Orthogonal error", orth)
            (log.info if orth < 1e-8 else log.warning)("  > %-18s: %9.3g", "Idempotent error", idem)
            (log.info if orth < 1e-8 else log.warning)("  > %-18s: %9.3g", "Occupancy error", occ)
            log.info("  > %-18s: %9.6f", "n(elec)", ne)
            log.info("  > %-18s: %9d", "n(occ)", c_occ.shape[1])
            log.info("  > %-18s: %9d", "n(vir)", c_vir.shape[1])

        log.changeIndentLevel(-1)

    return c_bath, c_env_occ, c_env_vir



del (LIN_DEP_THRESHOLD, SVD_THRESHOLD, ENV_THRESHOLD)
