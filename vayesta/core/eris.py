from pyscf.mp.mp2 import _mo_without_core
from vayesta.core.ao2mo import kao2gmo_cderi
from vayesta.core.ao2mo import postscf_ao2mo
from vayesta.core.ao2mo import postscf_kao2gmo

import numpy as np
from vayesta.core.util import *
import pyscf.lib


def get_cderi(emb, mo_coeff, compact=False, blksize=None):
    """Get density-fitted three-center integrals in MO basis."""
    if compact:
        raise NotImplementedError()
    if emb.kdf is not None:
        return kao2gmo_cderi(emb.kdf, mo_coeff)

    if np.ndim(mo_coeff[0]) == 1:
        mo_coeff = (mo_coeff, mo_coeff)

    nao = emb.mol.nao
    naux = (emb.df.auxcell.nao if hasattr(emb.df, 'auxcell') else emb.df.auxmol.nao)
    cderi = np.zeros((naux, mo_coeff[0].shape[-1], mo_coeff[1].shape[-1]))
    cderi_neg = None
    if blksize is None:
        blksize = int(1e9 / naux * nao * nao * 8)
    # PBC:
    if hasattr(emb.df, 'sr_loop'):
        blk0 = 0
        for labr, labi, sign in emb.df.sr_loop(compact=False, blksize=blksize):
            assert np.allclose(labi, 0)
            assert (cderi_neg is None)  # There should be only one block with sign -1
            labr = labr.reshape(-1, nao, nao)
            if (sign == 1):
                blk1 = (blk0 + labr.shape[0])
                blk = np.s_[blk0:blk1]
                blk0 = blk1
                cderi[blk] = einsum('Lab,ai,bj->Lij', labr, mo_coeff[0], mo_coeff[1])
            elif (sign == -1):
                cderi_neg = einsum('Lab,ai,bj->Lij', labr, mo_coeff[0], mo_coeff[1])
        return cderi, cderi_neg
    # No PBC:
    blk0 = 0
    for lab in emb.df.loop(blksize=blksize):
        blk1 = (blk0 + lab.shape[0])
        blk = np.s_[blk0:blk1]
        blk0 = blk1
        lab = pyscf.lib.unpack_tril(lab)
        cderi[blk] = einsum('Lab,ai,bj->Lij', lab, mo_coeff[0], mo_coeff[1])
    return cderi, None


@log_method()
def get_eris_array(emb, mo_coeff, compact=False):
    """Get electron-repulsion integrals in MO basis as a NumPy array.

    Parameters
    ----------
    mo_coeff: [list(4) of] (n(AO), n(MO)) array
        MO coefficients.

    Returns
    -------
    eris: (n(MO), n(MO), n(MO), n(MO)) array
        Electron-repulsion integrals in MO basis.
    """
    # PBC with k-points:
    if emb.kdf is not None:
        if compact:
            raise NotImplementedError
        if np.ndim(mo_coeff[0]) == 1:
            mo_coeff = 4 * [mo_coeff]
        cderi1, cderi1_neg = kao2gmo_cderi(emb.kdf, mo_coeff[:2])
        if (mo_coeff[0] is mo_coeff[2]) and (mo_coeff[1] is mo_coeff[3]):
            cderi2, cderi2_neg = cderi1, cderi1_neg
        else:
            cderi2, cderi2_neg = kao2gmo_cderi(emb.kdf, mo_coeff[2:])
        eris = einsum('Lij,Lkl->ijkl', cderi1.conj(), cderi2)
        if cderi1_neg is not None:
            eris -= einsum('Lij,Lkl->ijkl', cderi1_neg.conj(), cderi2_neg)
        return eris
    # Molecules and Gamma-point PBC:
    if hasattr(emb.mf, 'with_df') and emb.mf.with_df is not None:
        eris = emb.mf.with_df.ao2mo(mo_coeff, compact=compact)
    elif emb.mf._eri is not None:
        eris = pyscf.ao2mo.kernel(emb.mf._eri, mo_coeff, compact=compact)
    else:
        eris = emb.mol.ao2mo(mo_coeff, compact=compact)
    if not compact:
        if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
            shape = 4 * [mo_coeff.shape[-1]]
        else:
            shape = [mo.shape[-1] for mo in mo_coeff]
        eris = eris.reshape(shape)
    return eris


@log_method()
def get_eris_object(emb, postscf, fock=None):
    """Get ERIs for post-SCF methods.

    For folded PBC calculations, this folds the MO back into k-space
    and contracts with the k-space three-center integrals..

    Parameters
    ----------
    postscf: one of the following PySCF methods: MP2, CCSD, RCCSD, DFCCSD
        Post-SCF method with attribute mo_coeff set.

    Returns
    -------
    eris: _ChemistsERIs
        ERIs which can be used for the respective post-scf method.
    """
    if fock is None:
        if isinstance(postscf, pyscf.mp.mp2.MP2):
            fock = emb.get_fock()
        elif isinstance(postscf, (pyscf.ci.cisd.CISD, pyscf.cc.ccsd.CCSD)):
            fock = emb.get_fock(with_exxdiv=False)
        else:
            raise ValueError("Unknown post-SCF method: %r", type(postscf))
    # For MO energies, always use get_fock():
    mo_act = _mo_without_core(postscf, postscf.mo_coeff)
    mo_energy = einsum('ai,ab,bi->i', mo_act, emb.get_fock(), mo_act)
    e_hf = emb.mf.e_tot

    # Fold MOs into k-point sampled primitive cell, to perform efficient AO->MO transformation:
    if emb.kdf is not None:
        return postscf_kao2gmo(postscf, emb.kdf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
    # Regular AO->MO transformation
    eris = postscf_ao2mo(postscf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
    return eris
