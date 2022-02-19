import logging

import numpy as np

import pyscf
import pyscf.mp
import pyscf.cc
import pyscf.lib
from pyscf.mp.mp2 import _mo_without_core
from pyscf.mp.mp2 import _ChemistsERIs as MP2_ChemistsERIs
from pyscf.cc.rccsd import _ChemistsERIs as RCCSD_ChemistsERIs
from pyscf.cc.uccsd import _ChemistsERIs as UCCSD_ChemistsERIs
from pyscf.cc.dfccsd import _ChemistsERIs as DFCCSD_ChemistsERIs
from pyscf.cc.ccsd import _ChemistsERIs as CCSD_ChemistsERIs

from vayesta.core.util import *
from .kao2gmo_new import kao2gmo_cderi


log = logging.getLogger(__name__)


def postscf_ao2mo(postscf, mo_coeff=None, fock=None, mo_energy=None, e_hf=None):
    """AO to MO transformation of ERIs for post-SCF calculations.

    Use this as postscf_ao2mo(cc,...) instead of cc.ao2mo(...) to allow control
    of eris.fock, eris.mo_energy, and eris.e_hf.

    Parameters
    ----------
    postscf: PySCF Post-SCF method
        Instance of MP2, DFMP2, CCSD, RCCSD, or DFCCSD, with attribute mo_coeff set.
    mo_coeff: array, optional
        MO coefficients for the AO to MO transformation. If None, PySCF uses postscf.mo_coeff.
        Default: None.
    fock: array, optional
        Fock matrix in AO representation. If None, PySCF uses postscf._scf.get_fock(). Default: None.
    mo_energy: array, optional
        Active MO energies. If None, PySCF uses fock.diagonal(). Default: None.
    e_hf: float, optional
        Mean-field energy. If None, PySCF calculates this as postscf._scf.energy_tot(). Default: None.

    Returns
    -------
    eris: _ChemistsERIs
        PySCF ERIs object which can be used for the respective post-SCF calculation.
    """
    replace = {}
    if fock is not None:
        replace['get_fock'] = (fock if callable(fock) else lambda *args, **kwargs: fock)
    if e_hf is not None:
        replace['energy_tot'] = (e_hf if callable(e_hf) else lambda *args, **kwargs: e_hf)
    if (fock is not None and e_hf is not None):
        # make_rdm1 and get_veff are called within postscf.ao2mo, to generate
        # the Fock matrix and SCF energy - since we set these directly,
        # we can avoid performing any computation in these functions:
        do_nothing = lambda *args, **kwargs: None
        replace['make_rdm1'] = do_nothing
        replace['get_veff'] = do_nothing

    # Replace attributes in `replace` temporarily for the potscf.ao2mo call;
    # after the with statement, the attributes are reset to their intial values.
    with replace_attr(postscf._scf, **replace):
        eris = postscf.ao2mo(mo_coeff)

    if mo_energy is not None:
        eris.mo_energy = mo_energy

    return eris

def postscf_kao2gmo(postscf, gdf, fock, mo_energy, e_hf, mo_coeff=None):
    """k-AO to Gamma-MO transformation of ERIs for post-SCF calculations of supercells.

    This can be used to avoid performing the expensive density-fitting in the supercell,
    if a smaller primitive unit cell exists.

    Parameters
    ----------
    postscf: PySCF Post-SCF method
        Instance of MP2, DFMP2, CCSD, RCCSD, or DFCCSD, with attribute mo_coeff set.
    gdf: PySCF Gaussian density-fitting object
        Density-fitting object of the primitive unit cell.
    fock: array
        Fock matrix in AO representation.
    mo_energy: array
        Active MO energies.
    e_hf: float
        Mean-field energy.
    mo_coeff: array, optional
        MO coefficients for the AO to MO transformation. If None, PySCF uses postscf.mo_coeff.
        Default: None.

    Returns
    -------
    eris: _ChemistsERIs
        PySCF ERIs object which can be used for the respective supercell post-SCF calculation.
    """
    if mo_coeff is None:
        mo_coeff = postscf.mo_coeff
    # Remove frozen orbitals:
    mo_coeff = _mo_without_core(postscf, mo_coeff)
    nocc = postscf.nocc
    occ, vir = np.s_[:nocc], np.s_[nocc:]

    if isinstance(postscf, pyscf.mp.mp2.MP2):
        eris = MP2_ChemistsERIs()
        # Only occ-vir block needed for MP2
        mo_coeffs = (mo_coeff[:,occ], mo_coeff[:,vir])
    elif isinstance(postscf, pyscf.cc.rccsd.RCCSD):
        eris = RCCSD_ChemistsERIs()
        mo_coeffs = (mo_coeff, mo_coeff)
        pack = False
        store_vvl = False
    elif isinstance(postscf, pyscf.cc.dfccsd.RCCSD):
        eris = DFCCSD_ChemistsERIs()
        mo_coeffs = (mo_coeff, mo_coeff)
        pack = True
        store_vvl = True
    elif isinstance(postscf, pyscf.cc.ccsd.CCSD):
        eris = CCSD_ChemistsERIs()
        mo_coeffs = (mo_coeff, mo_coeff)
        pack = True
        store_vvl = False
    else:
        raise NotImplementedError("Unknown post-SCF method= %s" % type(postscf))

    eris.mo_coeff = mo_coeff
    eris.nocc = nocc
    eris.e_hf = e_hf
    eris.fock = dot(mo_coeff.T, fock, mo_coeff)
    eris.mo_energy = mo_energy

    cderi, cderi_neg = kao2gmo_cderi(gdf, mo_coeffs)

    # MP2
    if isinstance(postscf, pyscf.mp.mp2.MP2):
        eris.ovov = _contract_cderi(cderi, cderi_neg)
    # CCSD
    else:
        eris.oooo = _contract_cderi(cderi, cderi_neg, block='oooo', nocc=nocc)
        eris.ovoo = _contract_cderi(cderi, cderi_neg, block='ovoo', nocc=nocc)
        eris.oovv = _contract_cderi(cderi, cderi_neg, block='oovv', nocc=nocc)
        eris.ovvo = _contract_cderi(cderi, cderi_neg, block='ovvo', nocc=nocc)
        eris.ovov = _contract_cderi(cderi, cderi_neg, block='ovov', nocc=nocc)
        eris.ovvv = _contract_cderi(cderi, cderi_neg, block='ovvv', nocc=nocc, pack_right=pack)
        if store_vvl:
            if cderi_neg is not None:
                raise ValueError("Cannot use DFCCSD for 2D systems!")
            eris.vvL = pyscf.lib.pack_tril(cderi[:,vir,vir].copy()).T
        else:
            eris.vvvv = _contract_cderi(cderi, cderi_neg, block='vvvv', nocc=nocc, pack_left=pack, pack_right=pack)

    return eris

def postscf_kao2gmo_uhf(postscf, gdf, fock, mo_energy, e_hf, mo_coeff=None):
    """k-AO to Gamma-MO transformation of ERIs for unrestricted post-SCF calculations of supercells.

    This can be used to avoid performing the expensive density-fitting in the supercell,
    if a smaller primitive unit cell exists.

    Parameters
    ----------
    postscf: PySCF Post-SCF method
        Instance of UMP2 or UCCSD, with attribute mo_coeff set.
    gdf: PySCF Gaussian density-fitting object
        Density-fitting object of the primitive unit cell.
    fock: tuple(2) of arrays
        Fock matrix in AO representation (alpha, beta).
    mo_energy: tuple(2) or arrays
        Active MO energies (alpha, beta).
    e_hf: float
        Mean-field energy.
    mo_coeff: tuple(2) of arrays, optional
        MO coefficients for the AO to MO transformation (alpha, beta).
        If None, PySCF uses postscf.mo_coeff. Default: None.

    Returns
    -------
    eris: _ChemistsERIs
        PySCF ERIs object which can be used for the respective supercell post-SCF calculation.
    """
    if mo_coeff is None:
        mo_coeff = postscf.mo_coeff
    # Remove frozen orbitals:
    act = postscf.get_frozen_mask()
    mo_coeff = (moa, mob) = (mo_coeff[0][:,act[0]], mo_coeff[1][:,act[1]])
    nocc = (nocca, noccb) = postscf.nocc
    occa, vira = np.s_[:nocca], np.s_[nocca:]
    occb, virb = np.s_[:noccb], np.s_[noccb:]

    if isinstance(postscf, pyscf.mp.ump2.UMP2):
        eris = MP2_ChemistsERIs()
        # Only occ-vir block needed for MP2
        mo_coeffs_a = (moa[:,occa], moa[:,vira])
        mo_coeffs_b = (mob[:,occb], mob[:,virb])
    elif isinstance(postscf, pyscf.cc.uccsd.UCCSD):
        eris = UCCSD_ChemistsERIs()
        mo_coeffs_a = (moa, moa)
        mo_coeffs_b = (mob, mob)
        pack = True
    else:
        raise NotImplementedError("Unknown post-SCF method= %s" % type(postscf))

    eris.mo_coeff = mo_coeff
    eris.nocc = nocc
    eris.e_hf = e_hf
    eris.focka = dot(moa.T, fock[0], moa)
    eris.fockb = dot(mob.T, fock[1], mob)
    eris.fock = (eris.focka, eris.fockb)
    eris.mo_energy = mo_energy

    cderia, cderia_neg = kao2gmo_cderi(gdf, mo_coeffs_a)
    cderib, cderib_neg = kao2gmo_cderi(gdf, mo_coeffs_b)
    cderi = (cderia, cderib)
    cderi_neg = (cderia_neg, cderib_neg)

    # MP2
    if isinstance(postscf, pyscf.mp.mp2.MP2):
        eris.ovov = _contract_cderi(cderia, cderia_neg)
        eris.OVOV = _contract_cderi(cderib, cderib_neg)
        eris.ovOV = _contract_cderi_mixed(cderi, cderi_neg)
    # CCSD
    else:
        # Alpha-alpha:
        eris.oooo = _contract_cderi(cderia, cderia_neg, block='oooo', nocc=nocca)
        eris.ovoo = _contract_cderi(cderia, cderia_neg, block='ovoo', nocc=nocca)
        eris.ovvo = _contract_cderi(cderia, cderia_neg, block='ovvo', nocc=nocca)
        eris.oovv = _contract_cderi(cderia, cderia_neg, block='oovv', nocc=nocca)
        eris.ovov = _contract_cderi(cderia, cderia_neg, block='ovov', nocc=nocca)
        eris.ovvv = _contract_cderi(cderia, cderia_neg, block='ovvv', nocc=nocca, pack_right=pack)
        eris.vvvv = _contract_cderi(cderia, cderia_neg, block='vvvv', nocc=nocca, pack_left=pack, pack_right=pack)
        # Beta-beta:
        eris.OOOO = _contract_cderi(cderib, cderib_neg, block='oooo', nocc=noccb)
        eris.OVOO = _contract_cderi(cderib, cderib_neg, block='ovoo', nocc=noccb)
        eris.OVVO = _contract_cderi(cderib, cderib_neg, block='ovvo', nocc=noccb)
        eris.OOVV = _contract_cderi(cderib, cderib_neg, block='oovv', nocc=noccb)
        eris.OVOV = _contract_cderi(cderib, cderib_neg, block='ovov', nocc=noccb)
        eris.OVVV = _contract_cderi(cderib, cderib_neg, block='ovvv', nocc=noccb, pack_right=pack)
        eris.VVVV = _contract_cderi(cderib, cderib_neg, block='vvvv', nocc=noccb, pack_left=pack, pack_right=pack)
        # Alpha-beta:
        eris.ooOO = _contract_cderi_mixed(cderi, cderi_neg, block='ooOO', nocc=nocc)
        eris.ovOO = _contract_cderi_mixed(cderi, cderi_neg, block='ovOO', nocc=nocc)
        eris.ovVO = _contract_cderi_mixed(cderi, cderi_neg, block='ovVO', nocc=nocc)
        eris.ooVV = _contract_cderi_mixed(cderi, cderi_neg, block='ooVV', nocc=nocc)
        eris.ovOV = _contract_cderi_mixed(cderi, cderi_neg, block='ovOV', nocc=nocc)
        eris.ovVV = _contract_cderi_mixed(cderi, cderi_neg, block='ovVV', nocc=nocc, pack_right=pack)
        eris.vvVV = _contract_cderi_mixed(cderi, cderi_neg, block='vvVV', nocc=nocc, pack_left=pack, pack_right=pack)
        # Beta-Alpha:
        eris.OVoo = _contract_cderi_mixed(cderi[::-1], cderi_neg[::-1], block='OVoo', nocc=nocc)
        eris.OOvv = _contract_cderi_mixed(cderi[::-1], cderi_neg[::-1], block='OOvv', nocc=nocc)
        eris.OVvo = _contract_cderi_mixed(cderi[::-1], cderi_neg[::-1], block='OVvo', nocc=nocc)
        eris.OVvv = _contract_cderi_mixed(cderi[::-1], cderi_neg[::-1], block='OVvv', nocc=nocc, pack_right=pack)

    return eris

def _contract_cderi(cderi, cderi_neg, block=None, nocc=None, pack_left=False, pack_right=False, imag_tol=1e-8):
    if block is not None:
        slices = {'o': np.s_[:nocc], 'v': np.s_[nocc:]}
        get_slices = (lambda block : (slices[block[0]], slices[block[1]]))
        si, sj = get_slices(block[:2])
        sk, sl = get_slices(block[2:])
    else:
        si = sj = sk = sl = np.s_[:]

    # Positive part
    cderi_left = cderi[:,si,sj].copy()
    cderi_right = cderi[:,sk,sl].copy()
    if pack_left:
        cderi_left = pyscf.lib.pack_tril(cderi_left)
    if pack_right:
        cderi_right = pyscf.lib.pack_tril(cderi_right)
    eri = np.tensordot(cderi_left.conj(), cderi_right, axes=(0, 0))
    #log.debugv("Final shape of (%s|%s)= %r", block[:2], block[2:], list(eri.shape))
    if cderi_neg is None:
        return eri

    # Negative part (for 2D systems)
    cderi_left = cderi_neg[:,si,sj]
    cderi_right = cderi_neg[:,sk,sl]
    if pack_left:
        cderi_left = pyscf.lib.pack_tril(cderi_left)
    if pack_right:
        cderi_right = pyscf.lib.pack_tril(cderi_right)
    # Avoid allocating another N^4 object:
    max_memory = int(3e8) # 300 MB
    nblks = int((eri.size * 8 * (1+np.iscomplexobj(eri)))/max_memory)
    size = cderi_left.shape[1]
    blksize = int(size/max(nblks, 1))
    log.debugv("max_memory= %d MB  nblks= %d  size= %d  blksize= %d", max_memory/1e6, nblks, size, blksize)
    for blk in brange(0, size, blksize):
        eri[blk] -= np.tensordot(cderi_left[:,blk].conj(), cderi_right, axes=(0, 0))
    #eri -= np.tensordot(cderi_left.conj(), cderi_right, axes=(0, 0))
    assert (eri.size == 0) or (abs(eri.imag).max() < imag_tol)
    return eri

def _contract_cderi_mixed(cderi, cderi_neg, block=None, nocc=None, pack_left=False, pack_right=False, imag_tol=1e-8):
    if block is not None:
        noccl, noccr = nocc
        slices = {'o': np.s_[:noccl], 'v': np.s_[noccl:], 'O': np.s_[:noccr], 'V': np.s_[noccr:]}
        get_slices = (lambda block : (slices[block[0]], slices[block[1]]))
        si, sj = get_slices(block[:2])
        sk, sl = get_slices(block[2:])
    else:
        si = sj = sk = sl = np.s_[:]

    # Positive part
    cderi_left = cderi[0][:,si,sj].copy()
    cderi_right = cderi[1][:,sk,sl].copy()
    if pack_left:
        cderi_left = pyscf.lib.pack_tril(cderi_left)
    if pack_right:
        cderi_right = pyscf.lib.pack_tril(cderi_right)
    eri = np.tensordot(cderi_left.conj(), cderi_right, axes=(0, 0))
    #log.debugv("Final shape of (%s|%s)= %r", block[:2], block[2:], list(eri.shape))
    if cderi_neg[0] is None:
        return eri

    # Negative part (for 2D systems)
    cderi_left = cderi_neg[0][:,si,sj]
    cderi_right = cderi_neg[1][:,sk,sl]
    if pack_left:
        cderi_left = pyscf.lib.pack_tril(cderi_left)
    if pack_right:
        cderi_right = pyscf.lib.pack_tril(cderi_right)
    # Avoid allocating another N^4 object:
    max_memory = int(3e8) # 300 MB
    nblks = int((eri.size * 8 * (1+np.iscomplexobj(eri)))/max_memory)
    size = cderi_left.shape[1]
    blksize = int(size/max(nblks, 1))
    log.debugv("max_memory= %d MB  nblks= %d  size= %d  blksize= %d", max_memory/1e6, nblks, size, blksize)
    for blk in brange(0, size, blksize):
        eri[blk] -= np.tensordot(cderi_left[:,blk].conj(), cderi_right, axes=(0, 0))
    #eri -= np.tensordot(cderi_left.conj(), cderi_right, axes=(0, 0))
    assert (eri.size == 0) or (abs(eri.imag).max() < imag_tol)
    return eri
