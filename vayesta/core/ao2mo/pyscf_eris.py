import numpy as np
import pyscf
import pyscf.cc
import pyscf.lib
import pyscf.ao2mo


def _pack_ovvv(ovvv):
    no, nva, nvb = ovvv.shape[:3]
    nvbp = nvb * (nvb + 1) // 2
    ovvv = pyscf.lib.pack_tril(ovvv.reshape(no * nva, nvb, nvb)).reshape(no, nva, nvbp)
    return ovvv


def _pack_vvvv(vvvv):
    nva = vvvv.shape[0]
    if np.all(np.asarray(vvvv.shape) == nva):
        return pyscf.ao2mo.restore(4, vvvv, nva)
    nvb = vvvv.shape[2]
    nvap = nva * (nva + 1) // 2
    nvbp = nvb * (nvb + 1) // 2
    vvvv = pyscf.lib.pack_tril(vvvv.reshape(nva * nva, nvb, nvb))
    vvvv = pyscf.lib.pack_tril(vvvv.reshape(nva, nva, nvbp), axis=0)
    vvvv = vvvv.reshape(nvap, nvbp)
    return vvvv


def make_ccsd_eris(fock, eris, nocc, mo_energy=None):
    """Generate _ChemistERIs object for pyscf.cc.ccsd.CCSD.

    Parameters
    ----------
    fock: array
        Fock matrix in MO basis.
    eris: array
        Electron-repulsion integrals in MO basis.
    nocc: int
        Number of occupied orbitals.
    mo_energy: array, optional
        MO energies. Used for the initial guess and preconditioning.
        Default: fock.diagonal().

    Returns
    -------
    eris: pyscf.cc.ccsd._ChemistERIs
        ERI object as used by pyscf.cc.ccsd.CCSD.
    """
    eris_out = pyscf.cc.ccsd._ChemistsERIs()
    nmo = fock.shape[-1]
    nvir = nmo - nocc
    o, v = np.s_[:nocc], np.s_[nocc:]
    eris_out.oooo = eris[o, o, o, o]
    eris_out.ovoo = eris[o, v, o, o]
    eris_out.oovv = eris[o, o, v, v]
    eris_out.ovvo = eris[o, v, v, o]
    eris_out.ovov = eris[o, v, o, v]
    eris_out.ovvv = _pack_ovvv(eris[o, v, v, v])
    eris_out.vvvv = _pack_vvvv(eris[v, v, v, v])
    eris_out.fock = fock
    eris_out.nocc = nocc
    if mo_energy is None:
        mo_energy = fock.diagonal()
    eris_out.mo_energy = mo_energy
    eris_out.mo_coeff = np.eye(nmo)
    return eris_out


def make_uccsd_eris(fock, eris, nocc, mo_energy=None):
    """Generate _ChemistERIs object for pyscf.cc.uccsd.UCCSD.

    Parameters
    ----------
    fock: tuple(2) of arrays
        Fock matrix in MO basis.
    eris: tuple(3) of arrays
        Electron-repulsion integrals in MO basis.
    nocc: tuple(2) of ints
        Number of occupied orbitals.
    mo_energy: tuple(2) of arrays, optional
        MO energies. Used for the initial guess and preconditioning.
        Default: fock.diagonal().

    Returns
    -------
    eris: pyscf.cc.uccsd._ChemistERIs
        ERI object as used by pyscf.cc.uccsd.UCCSD.
    """
    eris_out = pyscf.cc.uccsd._ChemistsERIs()
    nmo = (fock[0].shape[-1], fock[1].shape[-1])
    nvir = (nmo[0] - nocc[0], nmo[1] - nocc[1])
    oa, va = np.s_[: nocc[0]], np.s_[nocc[0] :]
    ob, vb = np.s_[: nocc[1]], np.s_[nocc[1] :]
    eris_aa, eris_ab, eris_bb = eris
    # Alpha-alpha
    eris_out.oooo = eris_aa[oa, oa, oa, oa]
    eris_out.ovoo = eris_aa[oa, va, oa, oa]
    eris_out.oovv = eris_aa[oa, oa, va, va]
    eris_out.ovvo = eris_aa[oa, va, va, oa]
    eris_out.ovov = eris_aa[oa, va, oa, va]
    eris_out.ovvv = _pack_ovvv(eris_aa[oa, va, va, va])
    eris_out.vvvv = _pack_vvvv(eris_aa[va, va, va, va])
    # Beta-beta
    eris_out.OOOO = eris_bb[ob, ob, ob, ob]
    eris_out.OVOO = eris_bb[ob, vb, ob, ob]
    eris_out.OOVV = eris_bb[ob, ob, vb, vb]
    eris_out.OVVO = eris_bb[ob, vb, vb, ob]
    eris_out.OVOV = eris_bb[ob, vb, ob, vb]
    eris_out.OVVV = _pack_ovvv(eris_bb[ob, vb, vb, vb])
    eris_out.VVVV = _pack_vvvv(eris_bb[vb, vb, vb, vb])
    # Alpha-beta
    eris_out.ooOO = eris_ab[oa, oa, ob, ob]
    eris_out.ovOO = eris_ab[oa, va, ob, ob]
    eris_out.ooVV = eris_ab[oa, oa, vb, vb]
    eris_out.ovVO = eris_ab[oa, va, vb, ob]
    eris_out.ovOV = eris_ab[oa, va, ob, vb]
    eris_out.ovVV = _pack_ovvv(eris_ab[oa, va, vb, vb])
    eris_out.vvVV = _pack_vvvv(eris_ab[va, va, vb, vb])
    # Beta-alpha
    eris_ba = eris_ab.transpose(2, 3, 0, 1)
    eris_out.OVoo = eris_ba[ob, vb, oa, oa]
    eris_out.OOvv = eris_ba[ob, ob, va, va]
    eris_out.OVvo = eris_ba[ob, vb, va, oa]
    eris_out.OVvv = _pack_ovvv(eris_ba[ob, vb, va, va])
    # Other
    eris_out.focka = fock[0]
    eris_out.fockb = fock[1]
    eris_out.fock = fock
    eris_out.nocc = nocc
    if mo_energy is None:
        mo_energy = (fock[0].diagonal(), fock[1].diagonal())
    eris_out.mo_energy = mo_energy
    eris_out.mo_coeff = (np.eye(nmo[0]), np.eye(nmo[1]))
    return eris_out
