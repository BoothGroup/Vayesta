import copy
import logging

import numpy as np
import scipy
import scipy.linalg

log = logging.getLogger(__name__)


def update_mo_coeff(mo_coeff, t1, ovlp=None, damping=0.0, diis=None):
    nocc, nvir = t1.shape
    nmo = mo_coeff.shape[-1]
    assert nocc+nvir == nmo
    occ = np.s_[:nocc]
    vir = np.s_[nocc:]
    delta_occ = (1-damping)*np.dot(mo_coeff[:,vir], t1.T)   # qa,ai->qi
    log.debug("Change of occupied Brueckner orbitals= %.3e", np.linalg.norm(delta_occ))
    bmo_occ = mo_coeff[:,occ] + delta_occ

    # Orthogonalize occupied orbitals
    if ovlp is None:
        bmo_occ = np.linalg.qr(bmo_occ)[0]
    else:
        dm_occ = np.dot(bmo_occ, bmo_occ.T)
        e, v = scipy.linalg.eigh(dm_occ, b=ovlp, type=2)
        bmo_occ = v[:,-nocc:]
    # PCDIIS of occupied density
    if diis:
        dm_occ = np.dot(bmo_occ, bmo_occ.T)
        dm_occ = diis.update(dm_occ)
        e, v = scipy.linalg.eigh(dm_occ, b=ovlp, type=2)
        bmo_occ = v[:,-nocc:]

    # Virtual space
    if ovlp is None:
        dm_vir = (np.eye(nmo) - np.dot(bmo_occ, bmo_occ.T))
    else:
        dm_vir = np.linalg.inv(ovlp) - np.dot(bmo_occ, bmo_occ.T)
    e, v = scipy.linalg.eigh(dm_vir, b=ovlp, type=2)
    bmo_vir = v[:,-nvir:]

    assert bmo_occ.shape[-1] == nocc
    assert bmo_vir.shape[-1] == nvir
    if ovlp is None:
        ovlp = np.eye(nmo)
    bmo = np.hstack((bmo_occ, bmo_vir))
    assert np.allclose(np.linalg.multi_dot((bmo.T, ovlp, bmo))-np.eye(nmo), 0)
    return bmo_occ, bmo_vir


def update_mf(mf, t1, mo_coeff=None, inplace=False, canonicalize=True, damping=0.0, diis=None):
    """Update occupied MOs based on T1 amplitudes, to converge to Brueckner MOs.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        Mean-field object.
    t1 : (n(Occ), n(Vir)) array
        T1 amplitudes (i.e. C1 in intermediate normalization).
    mo_coeff : (n(AO), n(MO)) array, optional
        Molecular orbital coefficients. If None, `mf.mo_coeff` is used. Default: None.
    inplace : bool, optional
        If True, the mf object is updated inplace and the previous MO coefficients are overwritten. Default: False.
    canonicalize : bool or str, optional
        Diagonalize the Fock matrix within the new occupied and virtual space, to obtain quasi-canonical orbitals.
        Default: False.

    Returns
    -------
    mf : pyscf.scf.SCF
        Mean-field object with updated mf.mo_coeff and mf.e_tot
    """

    if not inplace:
        mf = copy.copy(mf)

    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[-1]
    nocc = np.count_nonzero(mf.mo_occ > 0)
    nvir = (nmo-nocc)
    assert t1.shape == (nocc, nvir)

    ovlp = mf.get_ovlp()
    if np.allclose(ovlp, np.eye(ovlp.shape[-1])):
        ovlp = None
    bmo_occ, bmo_vir = update_mo_coeff(mo_coeff, t1, ovlp, damping=damping, diis=diis)
    # Diagonalize one-electron Hamiltonian or Fock matrix within occupied and virtual space:
    if canonicalize:
        if canonicalize == 'hcore':
            h1e = mf.get_hcore()
        else:
            h1e = mf.get_fock()
        e, r = np.linalg.eigh(np.linalg.multi_dot((bmo_occ.T, h1e, bmo_occ)))
        log.debugv("Occupied BMO energies:\n%r", e.tolist())
        bmo_occ = np.dot(bmo_occ, r)
        e, r = np.linalg.eigh(np.linalg.multi_dot((bmo_vir.T, h1e, bmo_vir)))
        log.debugv("Virtual BMO energies:\n%r", e.tolist())
        bmo_vir = np.dot(bmo_vir, r)

    bmo = np.hstack((bmo_occ, bmo_vir))
    if ovlp is None:
        assert np.allclose(np.dot(bmo.T, bmo), np.eye(nmo))
    else:
        assert np.allclose(np.linalg.multi_dot((bmo.T, ovlp, bmo)), np.eye(nmo))

    mf.mo_coeff = bmo
    mf.e_tot = mf.energy_tot()
    return mf
