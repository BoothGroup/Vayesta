import logging
import numpy as np
import pyscf
import pyscf.lib
import pyscf.ao2mo
from vayesta.core.util import *


log = logging.getLogger(__name__)


def get_kconserv(cell, kpts, nk=3):
    r'''Get the momentum conservation array for a set of k-points.

    Given k-point indices (k, l, m) the array kconserv[k,l,m] returns
    the index n that satifies momentum conservation,

    nk=1:
        (k(k) - k(n)) \dot a = 2n\pi
    nk=2:
        (k(k) - k(l) - k(n)) \dot a = 2n\pi
    nk=3:
        (k(k) - k(l) + k(m) - k(n)) \dot a = 2n\pi

    This is used for symmetry e.g. integrals of the form
        [\phi*[k](1) \phi[l](1) | \phi*[m](2) \phi[n](2)]
    are zero unless n satisfies the above.
    '''
    nkpts = kpts.shape[0]
    a = cell.lattice_vectors() / (2*np.pi)

    if nk == 1:
        return list(range(len(kpts)))
    kconserv = np.zeros(nk*[nkpts], dtype=int)
    if nk == 2:
        k_klm = kpts[:,None,:] - kpts                           # k(k) - k(l)
    elif nk == 3:
        k_klm = kpts[:,None,None,:] - kpts[:,None,:] + kpts     # k(k) - k(l) + k(m)
    else:
        raise ValueError

    for n, kn in enumerate(kpts):
        k_klmn = (k_klm - kn)
        k_klmn = einsum('wx,...x->w...', a, k_klmn)
        # check whether (1/(2pi) k_klmn dot a) is an integer
        mask = einsum('w...->...', abs(k_klmn - np.rint(k_klmn))) < 1e-9
        kconserv[mask] = n

    return kconserv


def get_full_array(eris, mo_coeff=None, out=None):
    """Get dense ERI array from CCSD _ChemistEris object."""
    if hasattr(eris, 'OOOO'):
        return get_full_array_uhf(eris, mo_coeff=mo_coeff, out=out)
    return get_full_array_rhf(eris, mo_coeff=mo_coeff, out=out)


def get_full_array_rhf(eris, mo_coeff=None, out=None):
    """Get dense ERI array from CCSD _ChemistEris object."""
    if mo_coeff is not None and not np.allclose(mo_coeff, eris.mo_coeff):
        raise NotImplementedError
    nocc, nvir = eris.ovoo.shape[:2]
    nmo = nocc + nvir
    o, v = np.s_[:nocc], np.s_[nocc:]
    if out is None:
        out = np.zeros(4*[nmo])

    swap = lambda x : x.transpose(2,3,0,1)  # Swap electrons
    conj = lambda x : x.transpose(1,0,3,2)  # Real orbital symmetry
    # 4-occ
    out[o,o,o,o] = eris.oooo
    # 3-occ
    out[o,v,o,o] = eris.ovoo[:]
    out[v,o,o,o] = conj(out[o,v,o,o])
    out[o,o,o,v] = swap(out[o,v,o,o])
    out[o,o,v,o] = conj(out[o,o,o,v])
    # 2-occ
    out[o,o,v,v] = eris.oovv[:]
    out[v,v,o,o] = swap(out[o,o,v,v])
    out[o,v,o,v] = eris.ovov[:]
    out[v,o,v,o] = conj(out[o,v,o,v])
    out[o,v,v,o] = eris.ovvo[:]
    out[v,o,o,v] = swap(eris.ovvo[:])
    # 1-occ
    out[o,v,v,v] = get_ovvv(eris)
    out[v,o,v,v] = conj(out[o,v,v,v])
    out[v,v,o,v] = swap(out[o,v,v,v])
    out[v,v,v,o] = conj(out[v,v,o,v])
    # 0-occ
    out[v,v,v,v] = get_vvvv(eris)
    return out


def get_full_array_uhf(eris, mo_coeff=None, out=None):
    """Get dense ERI array from CCSD _ChemistEris object."""
    if mo_coeff is not None and not (np.allclose(mo_coeff[0], eris.mo_coeff[0])
            and np.allclose(mo_coeff[1], eris.mo_coeff[1])):
        raise NotImplementedError
    nocca, noccb = eris.nocc
    nmoa, nmob = eris.fock[0].shape[-1], eris.fock[1].shape[-1]
    nvira, nvirb = nmoa - nocca, nmob - noccb

    # Alpha-alpha
    blocks_aa = ['oooo', 'ovoo', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv']
    eris_aa = Object()
    eris_aa.fock = eris.fock[0]
    eris_aa.nocc = nocca
    for block in blocks_aa:
        setattr(eris_aa, block, getattr(eris, block))
    eri_aa = get_full_array_rhf(eris_aa, mo_coeff=getif(mo_coeff, 0), out=getif(out, 0))
    # Beta-beta
    eris_bb = Object()
    eris_bb.fock = eris.fock[1]
    eris_bb.nocc = noccb
    blocks_bb = [b.upper() for b in blocks_aa]
    for i, block in enumerate(blocks_bb):
        setattr(eris_bb, blocks_aa[i], getattr(eris, block))
    eri_bb = get_full_array_rhf(eris_bb, mo_coeff=getif(mo_coeff, 1), out=getif(out, 2))
    # Alpha-beta
    eri_ab = np.zeros((nmoa,nmoa,nmob,nmob)) if out is None else out[1]
    oa, ob = np.s_[:nocca], np.s_[:noccb]
    va, vb = np.s_[nocca:], np.s_[noccb:]
    swap = lambda x : x.transpose(2,3,0,1)  # Swap electrons
    conj = lambda x : x.transpose(1,0,3,2)  # Real orbital symmetry
    # 4-occ
    eri_ab[oa,oa,ob,ob] = eris.ooOO[:]
    # 3-occ
    eri_ab[oa,va,ob,ob] = eris.ovOO[:]
    eri_ab[va,oa,ob,ob] = conj(eri_ab[oa,va,ob,ob])
    eri_ab[oa,oa,ob,vb] = swap(eris.OVoo[:])
    eri_ab[oa,oa,vb,ob] = conj(eri_ab[oa,oa,ob,vb])
    # 2-occ
    eri_ab[oa,oa,vb,vb] = eris.ooVV[:]
    eri_ab[va,va,ob,ob] = swap(eris.OOvv[:])
    eri_ab[oa,va,vb,ob] = eris.ovVO[:]
    eri_ab[va,oa,ob,vb] = conj(eri_ab[oa,va,vb,ob])
    eri_ab[oa,va,ob,vb] = eris.ovOV[:]
    eri_ab[va,oa,vb,ob] = conj(eri_ab[oa,va,ob,vb])
    # 1-occ
    eri_ab[oa,va,vb,vb] = get_ovVV(eris, block='ovVV')
    eri_ab[va,oa,vb,vb] = conj(eri_ab[oa,va,vb,vb])
    eri_ab[va,va,ob,vb] = swap(get_ovVV(eris, block='OVvv'))
    eri_ab[va,va,vb,ob] = conj(eri_ab[va,va,ob,vb])
    # 0-occ
    eri_ab[va,va,vb,vb] = get_vvVV(eris)
    return eri_aa, eri_ab, eri_bb


def get_ovvv(eris, block='ovvv'):
    if hasattr(eris,'OOOO'):
        s = (0 if block == 'ovvv' else 1)
        nmo = eris.fock[s].shape[-1]
        nocc = eris.nocc[s]
    else:
        nmo = eris.fock.shape[-1]
        nocc = eris.nocc
    nvir = nmo - nocc
    govvv = getattr(eris, block)[:]
    if govvv.ndim == 4:
        return govvv
    nvir_pair = nvir*(nvir+1)//2
    govvv = pyscf.lib.unpack_tril(govvv.reshape(nocc*nvir, nvir_pair))
    govvv = govvv.reshape(nocc,nvir,nvir,nvir)
    return govvv


def get_ovVV(eris, block='ovVV'):
    sl, sr = (0, 1) if block == 'ovVV' else (1, 0)
    nmoL = eris.fock[sl].shape[-1]
    nmoR = eris.fock[sr].shape[-1]
    noccL = eris.nocc[sl]
    noccR = eris.nocc[sr]
    nvirL = nmoL - noccL
    nvirR = nmoR - noccR
    govvv = getattr(eris, block)[:]
    if govvv.ndim == 4:
        return govvv
    nvir_pair = nvirR*(nvirR+1)//2
    govvv = pyscf.lib.unpack_tril(govvv.reshape(noccL*nvirL, nvir_pair))
    govvv = govvv.reshape(noccL,nvirL,nvirR,nvirR)
    return govvv


def get_vvvv(eris, block='vvvv'):
    if hasattr(eris, 'VVVV'):
        s = (0 if block == 'vvvv' else 1)
        nmo = eris.fock[s].shape[-1]
        nocc = eris.nocc[s]
    else:
        nmo = eris.fock.shape[-1]
        nocc = eris.nocc
    nvir = nmo - nocc
    if getattr(eris, block, None) is not None:
        gvvvv = getattr(eris, block)[:]
        if gvvvv.ndim == 4:
            return gvvvv
        else:
            return pyscf.ao2mo.restore(1, np.asarray(gvvvv[:]), nvir)
    # Note that this will not work for 2D systems:
    if eris.vvL.ndim == 2:
        naux = eris.vvL.shape[-1]
        vvl = pyscf.lib.unpack_tril(eris.vvL[:], axis=0).reshape(nvir,nvir,naux)
    else:
        vvl = eris.vvL[:]
    gvvvv = einsum('ijQ,klQ->ijkl', vvl, vvl)
    return gvvvv


def get_vvVV(eris, block='vvVV'):
    sl, sr = ((0, 1) if block == 'vvVV' else (1, 0))
    nmoL = eris.fock[sl].shape[-1]
    nmoR = eris.fock[sr].shape[-1]
    noccL = eris.nocc[sl]
    noccR = eris.nocc[sr]
    nvirL = nmoL - noccL
    nvirR = nmoR - noccR
    gvvvv = getattr(eris, block)[:]
    if getattr(eris, block, None) is not None:
        gvvvv = getattr(eris, block)[:]
        if gvvvv.ndim == 4:
            return gvvvv[:]
        else:
            nvv = (-1 if gvvvv.size else 0)
            xVV = pyscf.lib.unpack_tril(gvvvv[:], axis=0).reshape(nvirL**2, nvv)
            return pyscf.lib.unpack_tril(xVV[:], axis=1).reshape(nvirL,nvirL,nvirR,nvirR)
    raise NotImplementedError


def get_block(eris, block):
    if block in ['ovvv', 'OVVV']:
        return get_ovvv(eris, block=block)
    if block in ['ovVV', 'OVvv']:
        return get_ovVV(eris, block=block)
    if block in ['vvvv', 'VVVV']:
        return get_vvvv(eris, block=block)
    if block in ['vvVV', 'VVvv']:
        return get_vvVV(eris, block=block)
    return getattr(eris, block)


def pack_ovvv(ovvv):
    nocc, nvir = ovvv.shape[:2]
    ovvv = pyscf.lib.pack_tril(ovvv.reshape(nocc*nvir, nvir, nvir))
    return ovvv.reshape(nocc, nvir, -1)


def pack_vvvv(vvvv):
    nvir = vvvv.shape[0]
    return pyscf.ao2mo.restore(4, vvvv, nvir)


def contract_dm2_eris(dm2, eris):
    """Contracts _ChemistsERIs with the two-body density matrix.

    Parameters
    ----------
    dm2 : ndarry or (ndarray, ndarray, ndarray)
        Two-body density matrix or tuple of alpha-alpha, alpha-beta, beta-beta spin blocks for UHF.
    eris : _ChemistERIs
        PySCF ERIs object.

    Returns
    -------
    e2 : float
        Two-body energy.
    """
    ndim = np.ndim(dm2[0]) + 1
    if ndim == 4:
        return contract_dm2_eris_rhf(dm2, eris)
    if ndim == 5:
        return contract_dm2_eris_uhf(dm2, eris)
    raise ValueError("N(dim) of DM2: %d" % ndim)


def _contract_4d(a, b, transpose=None):
    if transpose is not None:
        b = b[:].transpose(transpose)
    #return einsum('pqrs,pqrs', a, b)
    return np.dot(a[:].reshape(-1), b[:].reshape(-1))


def contract_dm2_eris_rhf(dm2, eris):
    """Contracts _ChemistsERIs with the two-body density matrix.

    Parameters
    ----------
    dm2 : ndarry
        Two-body density matrix.
    eris : _ChemistERIs
        PySCF ERIs object.

    Returns
    -------
    e2 : float
        Two-body energy.
    """
    nocc = eris.oooo.shape[0]
    o, v = np.s_[:nocc], np.s_[nocc:]
    e_oooo = _contract_4d(dm2[o,o,o,o], eris.oooo)
    e_ovoo = _contract_4d(dm2[o,v,o,o], eris.ovoo) * 4
    e_oovv = _contract_4d(dm2[o,o,v,v], eris.oovv) * 2
    e_ovov = _contract_4d(dm2[o,v,o,v], eris.ovov) * 2
    e_ovvo = _contract_4d(dm2[o,v,v,o], eris.ovvo) * 2
    e_ovvv = _contract_4d(dm2[o,v,v,v], get_ovvv(eris)) * 4
    e_vvvv = _contract_4d(dm2[v,v,v,v], get_vvvv(eris))
    log.debugv("E(oooo)= %s", energy_string(e_oooo))
    log.debugv("E(ovoo)= %s", energy_string(e_ovoo))
    log.debugv("E(oovv)= %s", energy_string(e_oovv))
    log.debugv("E(ovov)= %s", energy_string(e_ovov))
    log.debugv("E(ovvo)= %s", energy_string(e_ovvo))
    log.debugv("E(ovvv)= %s", energy_string(e_ovvv))
    log.debugv("E(vvvv)= %s", energy_string(e_vvvv))
    e2 = e_oooo + e_ovoo + e_oovv + e_ovov + e_ovvo + e_ovvv + e_vvvv
    return e2


def contract_dm2_eris_uhf(dm2, eris):
    """Contracts _ChemistsERIs with the two-body density matrix.

    Parameters
    ----------
    dm2 : tuple(ndarray, ndarray, ndarray)
        Two-body density matrix as a tuple of alpha-alpha, alpha-beta, beta-beta spin blocks.
    eris : _ChemistERIs
        PySCF ERIs object.

    Returns
    -------
    e2 : float
        Two-body energy.
    """
    nocca = eris.oooo.shape[0]
    noccb = eris.OOOO.shape[0]
    dm2aa, dm2ab, dm2bb = dm2
    e2 = 0
    # Alpha-alpha
    o, v = np.s_[:nocca], np.s_[nocca:]
    e2 += _contract_4d(dm2aa[o,o,o,o], eris.oooo)
    e2 += _contract_4d(dm2aa[o,v,o,o], eris.ovoo) * 4
    e2 += _contract_4d(dm2aa[o,o,v,v], eris.oovv) * 2
    e2 += _contract_4d(dm2aa[o,v,o,v], eris.ovov) * 2
    e2 += _contract_4d(dm2aa[o,v,v,o], eris.ovvo) * 2
    e2 += _contract_4d(dm2aa[o,v,v,v], get_ovvv(eris)) * 4
    e2 += _contract_4d(dm2aa[v,v,v,v], get_vvvv(eris))
    # Beta-beta
    o, v = np.s_[:noccb], np.s_[noccb:]
    e2 += _contract_4d(dm2bb[o,o,o,o], eris.OOOO)
    e2 += _contract_4d(dm2bb[o,v,o,o], eris.OVOO) * 4
    e2 += _contract_4d(dm2bb[o,o,v,v], eris.OOVV) * 2
    e2 += _contract_4d(dm2bb[o,v,o,v], eris.OVOV) * 2
    e2 += _contract_4d(dm2bb[o,v,v,o], eris.OVVO) * 2
    e2 += _contract_4d(dm2bb[o,v,v,v], get_ovvv(eris, block='OVVV')) * 4
    e2 += _contract_4d(dm2bb[v,v,v,v], get_vvvv(eris, block='VVVV'))
    # Alpha-beta
    oa, va = np.s_[:nocca], np.s_[nocca:]
    ob, vb = np.s_[:noccb], np.s_[noccb:]
    e2 += _contract_4d(dm2ab[oa,oa,ob,ob], eris.ooOO) * 2
    e2 += _contract_4d(dm2ab[oa,va,ob,ob], eris.ovOO) * 4
    e2 += _contract_4d(dm2ab[oa,oa,ob,vb], eris.OVoo, transpose=(2,3,0,1)) * 4
    e2 += _contract_4d(dm2ab[oa,oa,vb,vb], eris.ooVV) * 2
    e2 += _contract_4d(dm2ab[va,va,ob,ob], eris.OOvv, transpose=(2,3,0,1)) * 2
    e2 += _contract_4d(dm2ab[oa,va,vb,ob], eris.ovVO) * 4
    e2 += _contract_4d(dm2ab[oa,va,ob,vb], eris.ovOV) * 4
    #e2 += einsum('pqrs,rspq', dm2ab[va,oa,ob,vb], eris.OVvo) * 4
    e2 += _contract_4d(dm2ab[oa,va,vb,vb], get_ovVV(eris, block='ovVV')) * 4
    e2 += _contract_4d(dm2ab[va,va,ob,vb], get_ovVV(eris, block='OVvv'), transpose=(2,3,0,1)) * 4
    e2 += _contract_4d(dm2ab[va,va,vb,vb], get_vvVV(eris)) * 2
    return e2


# Order used in PySCF for 2-DM intermediates:
dm2intermeds = ['ovov', 'vvvv', 'oooo', 'oovv', 'ovvo', 'vvov', 'ovvv', 'ooov',]


def _dm2intermeds_to_dict_rhf(dm2):
    dm2dict = {block: dm2[idx] for (idx, block) in enumerate(dm2intermeds)}
    return dm2dict


def _dm2intermeds_to_dict_uhf(dm2):
    dm2dict = {}

    def _add_spinblocks(block, idx):
        b0, b1 = block[:2], block[2:]
        dm2i = dm2[idx]
        dm2dict[block.lower()] = np.asarray(dm2i[0])
        dm2dict[b0.lower() + b1.upper()] = np.asarray(dm2i[1])
        dm2dict[b0.upper() + b1.lower()] = np.asarray(dm2i[2])
        dm2dict[block.upper()] = np.asarray(dm2i[3])

    for idx, block in enumerate(dm2intermeds):
        _add_spinblocks(block, idx)
    return dm2dict


def contract_dm2intermeds_eris_rhf(dm2, eris, destroy_dm2=True):
    """Contracts _ChemistsERIs with the two-body density matrix.

    Parameters
    ----------
    dm2 : tuple
        Intermediates of spin-restricted two-body density matrix.
    eris : _ChemistERIs
        PySCF ERIs object.

    Returns
    -------
    e2 : float
        Two-body energy.
    """
    dm2 = _dm2intermeds_to_dict_rhf(dm2)

    def _get_block(block, keep=False):
        if destroy_dm2 and not keep:
            return dm2.pop(block)
        return dm2[block]

    e_oooo = _contract_4d(_get_block('oooo'), eris.oooo) * 4
    e_ovoo = _contract_4d(_get_block('ooov'), eris.ovoo, transpose=(2,3,0,1)) * 4
    e_oovv = _contract_4d(_get_block('oovv'), eris.oovv) * 4
    e_ovov = _contract_4d(_get_block('ovov'), eris.ovov) * 4
    e_ovvo = _contract_4d(_get_block('ovvo'), eris.ovvo) * 4
    e_ovvv = _contract_4d(_get_block('ovvv'), get_ovvv(eris)) * 4
    e_vvvv = _contract_4d(_get_block('vvvv'), get_vvvv(eris)) * 4
    log.debugv("E(oooo)= %s", energy_string(e_oooo))
    log.debugv("E(ovoo)= %s", energy_string(e_ovoo))
    log.debugv("E(oovv)= %s", energy_string(e_oovv))
    log.debugv("E(ovov)= %s", energy_string(e_ovov))
    log.debugv("E(ovvo)= %s", energy_string(e_ovvo))
    log.debugv("E(ovvv)= %s", energy_string(e_ovvv))
    log.debugv("E(vvvv)= %s", energy_string(e_vvvv))
    e2 = e_oooo + e_ovoo + e_oovv + e_ovov + e_ovvo + e_ovvv + e_vvvv
    return e2


def contract_dm2intermeds_eris_uhf(dm2, eris, destroy_dm2=True):
    """Contracts _ChemistsERIs with the two-body density matrix.

    Parameters
    ----------
    dm2 : tuple
        Intermediates of spin-unrestricted two-body density matrix.
    eris : _ChemistERIs
        PySCF ERIs object.

    Returns
    -------
    e2 : float
        Two-body energy.
    """
    dm2 = _dm2intermeds_to_dict_uhf(dm2)

    def _get_block(block, keep=False):
        if destroy_dm2 and not keep:
            return dm2.pop(block)
        return dm2[block]

    e2 = 0
    # Alpha-alpha
    e2 += _contract_4d(_get_block('oooo'), eris.oooo)
    e2 += _contract_4d(_get_block('ooov'), eris.ovoo, transpose=(2,3,0,1)) * 4
    #e2 += _contract_4d(_get_block('ovvo', keep=True), eris.oovv, transpose=(0,3,2,1)) * -2
    e2 += _contract_4d(_get_block('oovv'), eris.oovv) * 2
    e2 += _contract_4d(_get_block('ovov'), eris.ovov) * 2
    e2 += _contract_4d(_get_block('ovvo'), eris.ovvo) * 2
    e2 += _contract_4d(_get_block('ovvv'), get_ovvv(eris)) * 4
    e2 += _contract_4d(_get_block('vvvv'), get_vvvv(eris))
    # Beta-beta
    e2 += _contract_4d(_get_block('OOOO'), eris.OOOO)
    e2 += _contract_4d(_get_block('OOOV'), eris.OVOO, transpose=(2,3,0,1)) * 4
    #e2 += _contract_4d(_get_block('OVVO', keep=True), eris.OOVV, transpose=(0,3,2,1)) * -2
    e2 += _contract_4d(_get_block('OOVV'), eris.OOVV) * 2
    e2 += _contract_4d(_get_block('OVOV'), eris.OVOV) * 2
    e2 += _contract_4d(_get_block('OVVO'), eris.OVVO) * 2
    e2 += _contract_4d(_get_block('OVVV'), get_ovvv(eris, block='OVVV')) * 4
    e2 += _contract_4d(_get_block('VVVV'), get_vvvv(eris, block='VVVV'))
    # Alpha-beta
    e2 += _contract_4d(_get_block('ooOO'), eris.ooOO) * 2
    e2 += _contract_4d(_get_block('OOov'), eris.ovOO, transpose=(2,3,0,1)) * 4
    e2 += _contract_4d(_get_block('ooOV'), eris.OVoo, transpose=(2,3,0,1)) * 4
    e2 += _contract_4d(_get_block('ooVV'), eris.ooVV) * 2
    e2 += _contract_4d(_get_block('OOvv'), eris.OOvv) * 2
    e2 += _contract_4d(_get_block('ovVO'), eris.ovVO) * 4
    e2 += _contract_4d(_get_block('ovOV'), eris.ovOV) * 4
    e2 += _contract_4d(_get_block('ovVV'), get_ovVV(eris, block='ovVV')) * 4
    e2 += _contract_4d(_get_block('OVvv'), get_ovVV(eris, block='OVvv')) * 4
    e2 += _contract_4d(_get_block('vvVV'), get_vvVV(eris)) * 2
    return e2


def project_ccsd_eris(eris, mo_coeff, nocc, ovlp, check_subspace=True):
    """Project CCSD ERIs to a subset of orbital coefficients.

    Parameters
    ----------
    eris : _ChemistERIs
        PySCF ERIs object
    mo_coeff : (n(AO), n(MO)) array
        New subspace MO coefficients.
    nocc: int
        Number of occupied orbitals.
    ovlp : (n(AO), n(AO)) array
        AO overlap matrix.
    check_subspace : bool, optional
        Check if c_occ and c_vir span a subspace of eris.mo_coeff.
        Return None if Not. Default: True.

    Returns
    -------
    eris : _ChemistERIs or None
        ERIs with transformed integral values, as well as transformed attributes
        `mo_coeff`, `fock`, and `mo_energy`.

    """
    # New subspace MO coefficients:
    c_occ, c_vir = np.hsplit(mo_coeff, [nocc])
    # Old MO coefficients:
    c_occ0, c_vir0 = np.hsplit(eris.mo_coeff, [eris.nocc])
    nocc0, nvir0 = c_occ0.shape[-1], c_vir0.shape[-1]
    nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
    log.debug("Projecting ERIs: N(occ)= %3d -> %3d  N(vir)= %3d -> %3d", nocc0, nocc, nvir0, nvir)

    transform_occ = (nocc != nocc0 or not np.allclose(c_occ, c_occ0))
    if transform_occ:
        r_occ = dot(c_occ.T, ovlp, c_occ0)
    else:
        r_occ = np.eye(nocc)
    transform_vir = (nvir != nvir0 or not np.allclose(c_vir, c_vir0))
    if transform_vir:
        r_vir = dot(c_vir.T, ovlp, c_vir0)
    else:
        r_vir = np.eye(nvir)

    # Do nothing
    if not (transform_occ or transform_vir):
        return eris

    # Check that c_occ and c_vir form a subspace of eris.mo_coeff
    # If not return None
    if check_subspace:
        if nocc0 < nocc:
            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
        if nvir0 < nvir:
            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
        p_occ = np.dot(r_occ.T, r_occ)
        e, v = np.linalg.eigh(p_occ)
        n = np.count_nonzero(abs(e)>1e-8)
        if n < nocc:
            log.debug("e(occ)= %d\n%r", n, e)
            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")
        p_vir = np.dot(r_vir.T, r_vir)
        e, v = np.linalg.eigh(p_vir)
        n = np.count_nonzero(abs(e)>1e-8)
        if n < nvir:
            log.debug("e(vir)= %d\n%r", n, e)
            raise RuntimeError("MO coefficients do not span subspace of eris.mo_coeff.")

    r_all = np.block([
        [r_occ, np.zeros((nocc, nvir0))],
        [np.zeros((nvir, nocc0)), r_vir]])

    transform = lambda g, t0, t1, t2, t3 : einsum("abcd,ia,jb,kc,ld -> ijkl", g, t0, t1, t2, t3)

    if hasattr(eris, 'vvL'):
        raise  NotImplementedError()

    for block in ['oooo', 'ovoo', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv']:
        log.debugv("Projecting integrals (%2s|%2s)", block[:2], block[2:])
        g = get_block(eris, block)
        shape0 = [(nocc0 if (pos == 'o') else nvir0) for pos in block]
        t0123 = [(r_occ if (pos == 'o') else r_vir) for pos in block]
        pg = transform(g[:].reshape(shape0), *t0123)
        if block == 'ovvv' and getattr(eris, block).ndim == 3:
            pg = pack_ovvv(pg)
        if block == 'vvvv' and getattr(eris, block).ndim == 2:
            pg = pack_vvvv(pg)
        setattr(eris, block, pg)

    eris.mo_coeff = np.hstack((c_occ, c_vir))
    eris.nocc = nocc
    eris.fock = dot(r_all, eris.fock, r_all.T)
    eris.mo_energy = np.diag(eris.fock)
    return eris


if __name__ == '__main__':
    def test1():
        import pyscf.gto
        import pyscf.scf
        import pyscf.cc

        import vayesta
        from vayesta.misc.molecules import water

        mol = pyscf.gto.Mole()
        mol.atom = water()
        mol.basis = 'cc-pVDZ'
        mol.build()

        hf = pyscf.scf.RHF(mol)
        #hf.density_fit(auxbasis='cc-pVDZ-ri')
        hf.kernel()

        ccsd = pyscf.cc.CCSD(hf)
        eris = ccsd.ao2mo()
        eris2 = get_full_array(eris)

        norb = hf.mo_coeff.shape[-1]
        eris_test = pyscf.ao2mo.kernel(hf._eri, hf.mo_coeff, compact=False).reshape(4*[norb])
        err = np.linalg.norm(eris2 - eris_test)
        print(err)
        assert np.allclose(eris2, eris_test)

    def test2():
        import pyscf
        import pyscf.pbc
        import pyscf.pbc.gto
        from vayesta.misc import solids
        from timeit import default_timer as timer

        cell = pyscf.pbc.gto.Cell()
        cell.a, cell.atom = solids.diamond()
        cell.build()

        #kmesh = [3,2,1]
        kmesh = [4,5,2]
        kpts = cell.get_kpts(kmesh)

        nk = 3

        t0 = timer()
        kconserv = get_kconserv(cell, kpts, nk=nk)
        t1 = timer()
        kconserv_pyscf = pyscf.pbc.lib.kpts_helper.get_kconserv(cell, kpts, n=nk)
        t2 = timer()
        assert np.all(kconserv == kconserv_pyscf)
        print(t1-t0, t2-t1)

    test2()
