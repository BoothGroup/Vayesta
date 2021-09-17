import numpy as np

import pyscf
import pyscf.lib
import pyscf.ao2mo

from vayesta.core.util import *

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
