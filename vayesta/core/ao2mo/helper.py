import logging

import numpy as np

import pyscf
import pyscf.lib
import pyscf.ao2mo

from vayesta.core.util import *

log = logging.getLogger(__name__)


def get_full_array(eris, mo_coeff=None, out=None):
    """Get dense ERI array from CCSD _ChemistEris object."""
    if mo_coeff is not None and not np.allclose(mo_coeff, eris.mo_coeff):
        raise NotImplementedError()
    nmo = eris.fock.shape[-1]
    nocc = eris.nocc
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

def get_ovvv(eris):
    nmo = eris.fock.shape[-1]
    nocc = eris.nocc
    nvir = nmo - nocc
    if eris.ovvv.ndim == 4:
        return eris.ovvv[:]
    nvir_pair = nvir*(nvir+1)//2
    ovvv = pyscf.lib.unpack_tril(eris.ovvv.reshape(nocc*nvir, nvir_pair))
    return ovvv.reshape(nocc,nvir,nvir,nvir)

def get_vvvv(eris):
    nmo = eris.fock.shape[-1]
    nocc = eris.nocc
    nvir = nmo - nocc
    if hasattr(eris, 'vvvv') and eris.vvvv is not None:
        if eris.vvvv.ndim == 4:
            return eris.vvvv[:]
        else:
            return pyscf.ao2mo.restore(1, np.asarray(eris.vvvv), nvir)
    # Note that this will not work for 2D systems!:
    if eris.vvL.ndim == 2:
        naux = eris.vvL.shape[-1]
        vvl = pyscf.lib.unpack_tril(eris.vvL, axis=0).reshape(nvir,nvir,naux)
    else:
        vvl = eris.vvL
    return einsum('ijQ,klQ->ijkl', vvl, vvl)

def pack_ovvv(ovvv):
    nocc, nvir = ovvv.shape[:2]
    ovvv = pyscf.lib.pack_tril(ovvv.reshape(nocc*nvir, nvir, nvir))
    return ovvv.reshape(nocc, nvir, -1)

def pack_vvvv(vvvv):
    nvir = vvvv.shape[0]
    return pyscf.ao2mo.restore(4, vvvv, nvir)

def get_block(eris, block):
    if block == 'ovvv':
        return get_ovvv(eris)
    if block == 'vvvv':
        return get_vvvv(eris)
    return getattr(eris, block)

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
    import pyscf.gto
    import pyscf.scf
    import pyscf.cc

    import vayesta
    from vayesta.misc.molstructs import water

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
