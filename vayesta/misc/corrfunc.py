"""Correlation function"""

import numpy as np
from vayesta.core.util import dot, einsum


def _get_proj_per_spin(p):
    if np.ndim(p[0]) == 2:
        return p
    if np.ndim(p[0]) == 1:
        return p, p
    raise ValueError()

# --- Correlated:

def spin_z(dm1, proj=None):
    return 0.0

def spin_z_unrestricted(dm1, proj=None):
    dm1a, dm1b = dm1
    if proj is None:
        sz = (np.trace(dm1a) - np.trace(dm1b))/2
        return sz

    pa, pb = _get_proj_per_spin(proj)
    sz = (einsum('ij,ij->', dm1a, pa)
        - einsum('ij,ij->', dm1b, pb))/2
    return sz

def spinspin_z(dm1, dm2, proj1=None, proj2=None):
    if dm2 is None:
        return spinspin_z_mf(dm1, proj1=proj1, proj2=proj2)

    dm1a = dm1/2
    dm2aa = (dm2 - dm2.transpose(0,3,2,1)) / 6
    dm2ab = (dm2/2 - dm2aa)
    if proj2 is None:
        proj2 = proj1
    if proj1 is None:
        ssz = (einsum('iijj->', dm2aa) - einsum('iijj->', dm2ab))/2
        ssz += np.trace(dm1a)/2
        return ssz
    ssz = (einsum('ijkl,ij,kl->', dm2aa, proj1, proj2)
         - einsum('ijkl,ij,kl->', dm2ab, proj1, proj2))/2
    ssz += einsum('ij,ik,jk->', dm1a, proj1, proj2)/2
    return ssz

def spinspin_z_unrestricted(dm1, dm2, proj1=None, proj2=None):
    if dm2 is None:
        return spinspin_z_mf_unrestricted(dm1, proj1=proj1, proj2=proj2)

    dm1a, dm1b = dm1
    dm2aa, dm2ab, dm2bb = dm2
    if proj2 is None:
        proj2 = proj1
    if proj1 is None:
        ssz = (einsum('iijj->', dm2aa)/4 + einsum('iijj->', dm2bb)/4
             - einsum('iijj->', dm2ab)/2)
        ssz += (np.trace(dma) + np.trace(dmb))/4
        return ssz
    p1a, p1b = _get_proj_per_spin(proj1)
    p2a, p2b = _get_proj_per_spin(proj2)
    ssz = (einsum('ijkl,ij,kl->', dm2aa, p1a, p2a)/4
         + einsum('ijkl,ij,kl->', dm2bb, p1b, p2b)/4
         - einsum('ijkl,ij,kl->', dm2ab, p1a, p2b)/4
         - einsum('ijkl,ij,kl->', dm2ab, p2a, p1b)/4)
    ssz += (einsum('ij,ik,jk->', dm1a, p1a, p2a)
          + einsum('ij,ik,jk->', dm1b, p1b, p2b))/4
    return ssz

# --- Mean-field:

def spinspin_z_mf(dm1, proj1=None, proj2=None):
    # TEMP:
    dm1 = (dm1/2, dm1/2)
    return spinspin_z_uhf(dm1=dm1, proj1=proj1, proj2=proj2)

    #if proj2 is None:
    #    proj2 = proj1
    #if proj1 is None:
    #    ssz = np.trace(dm1)/4 - einsum('ij,ij->', dm1, dm1)/8
    #    return ssz

    # TODO:
    #ssz = (einsum('ij,kl,ij,kl->', dma, dma, p1a, p2a)
    #     - einsum('il,jk,ij,kl->', dma, dma, p1a, p2a)
    #     + einsum('ij,kl,ij,kl->', dmb, dmb, p1b, p2b)
    #     - einsum('il,jk,ij,kl->', dmb, dmb, p1b, p2b)
    #     - einsum('ij,kl,ij,kl->', dma, dmb, p1a, p2b)
    #     - einsum('ij,kl,ij,kl->', dmb, dma, p1b, p2a))/4
    #ssz += (einsum('ij,ik,jk->', dma, p1a, p2a)
    #      + einsum('ij,ik,jk->', dmb, p1b, p2b))/4
    #return ssz



def spinspin_z_mf_unrestricted(dm1, proj1=None, proj2=None):
    dma, dmb = dm1
    if proj2 is None:
        proj2 = proj1
    if proj1 is None:
        ssz = (einsum('ii,jj->', dma, dma)/4
            -  einsum('ij,ij->', dma, dma)/4
            +  einsum('ii,jj->', dmb, dmb)/4
            -  einsum('ij,ij->', dmb, dmb)/4
            -  einsum('ii,jj->', dma, dmb)/2)
        ssz += (np.trace(dma) + np.trace(dmb))/4
        return ssz

    p1a, p1b = (proj1, proj1) if np.ndim(proj1[0]) == 1 else proj1
    p2a, p2b = (proj2, proj2) if np.ndim(proj2[0]) == 1 else proj2

    ssz = (einsum('ij,kl,ij,kl->', dma, dma, p1a, p2a)
         - einsum('il,jk,ij,kl->', dma, dma, p1a, p2a)
         + einsum('ij,kl,ij,kl->', dmb, dmb, p1b, p2b)
         - einsum('il,jk,ij,kl->', dmb, dmb, p1b, p2b)
         - einsum('ij,kl,ij,kl->', dma, dmb, p1a, p2b)
         - einsum('ij,kl,ij,kl->', dmb, dma, p1b, p2a))/4
    ssz += (einsum('ij,ik,jk->', dma, p1a, p2a)
          + einsum('ij,ik,jk->', dmb, p1b, p2b))/4
    return ssz


if __name__ == '__main__':
    import pyscf
    import pyscf.gto
    import pyscf.scf

    mol = pyscf.gto.Mole()
    mol.atom = """
    O  0.0000   0.0000   0.1173
    H  0.0000   0.7572  -0.4692
    H  0.0000  -0.7572  -0.4692
    """
    mol.basis = 'cc-pVDZ'
    mol.build()
    nmo = mol.nao
    nocc = mol.nelectron//2

    # RHF
    rhf = pyscf.scf.RHF(mol)
    rhf.kernel()
    dm1 = np.zeros((nmo, nmo))
    dm1[np.diag_indices(nocc)] = 2
    sz = spin_z(dm1)
    print(sz)


    ssz = spinspin_z_rhf(dm1)
    print(ssz)
    1/0
    #print('RHF: <S_z>= %.8f  <S_z S_z>= %.8f' % (sz, ssz))


    # UHF
    mol.charge = mol.spin = 1
    mol.build()
    nmo = mol.nao
    nocca, noccb = mol.nelec
    print(mol.nelec)

    uhf = pyscf.scf.UHF(mol)
    uhf.kernel()

    dm1a = np.zeros((nmo, nmo))
    dm1b = np.zeros((nmo, nmo))
    dm1a[np.diag_indices(nocca)] = 1
    dm1b[np.diag_indices(noccb)] = 1
    dm1 = (dm1a, dm1b)

    sz = spin_z_uhf(dm1)
    print(sz)
    sz = spin_z_unrestricted(dm1)
    print(sz)
    #ssz = spinspin_z_uhf(uhf)
    #print('UHF: <S_z>= %.8f  <S_z S_z>= %.8f' % (sz, ssz))
    #ssz = spinspin_z_unrestricted(uhf)
    #print('UHF: <S_z>= %.8f  <S_z S_z>= %.8f' % (sz, ssz))
