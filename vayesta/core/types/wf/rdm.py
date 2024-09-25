import numpy as np

from vayesta.core import spinalg
from vayesta.core.util import dot, einsum, callif
from vayesta.core.types import wf as wf_types
from vayesta.core.types.wf.project import (
    project_c1,
    project_c2,
    project_uc1,
    project_uc2,
    symmetrize_c2,
    symmetrize_uc2,
    transform_c1,
    transform_c2,
    transform_uc1,
    transform_uc2,
)

def RDM_WaveFunction(mo, dm1, dm2, **kwargs):
    if mo.nspin == 1:
        cls = RRDM_WaveFunction
    elif mo.nspin == 2:
        cls = URDM_WaveFunction
    return cls(mo, dm1, dm2, **kwargs)

class RRDM_WaveFunction(wf_types.WaveFunction):
    """
    Spin-restricted dummy wavefunction type that stores the 1- and 2-RDMs.
    Allows interoperability with user-defined callback solvers which only 
    return the 1- and 2-RDMs.
    
    """
    def __init__(self, mo, dm1, dm2, projector=None):
        super().__init__(mo, projector=projector)
        self.dm1 = dm1
        self.dm2 = dm2

    def make_rdm1(self, ao_basis=False, with_mf=True):
        dm1 = self.dm1.copy()
        if not with_mf:
            dm1[np.diag_indices(self.nocc)] -= 2
        if not ao_basis:
            return dm1
        return dot(self.mo.coeff, dm1, self.mo.coeff.T)
    
    def make_rdm2(self, ao_basis=False, with_dm1=True, approx_cumulant=True):
        dm1, dm2 = self.dm1.copy(), self.dm2.copy()
        if not with_dm1:
            if not approx_cumulant:
                dm2 -= self.make_rdm2_non_cumulant(ao_basis=False)  
            elif approx_cumulant in (1, True):
                dm1[np.diag_indices(self.nocc)] -= 1
                for i in range(self.nocc):
                    dm2[i, i, :, :] -= 2 * dm1
                    dm2[:, :, i, i] -= 2 * dm1
                    dm2[:, i, i, :] += dm1
                    dm2[i, :, :, i] += dm1
            elif approx_cumulant == 2:
                raise NotImplementedError
            else:
                raise ValueError
        if not ao_basis:
            return dm2
        return einsum("ijkl,ai,bj,ck,dl->abcd", dm2, *(4 * [self.mo.coeff]))

    def make_rdm2_non_cumulant(self, ao_basis=False):
        dm1 = self.dm1.copy()
        dm2 = einsum("ij,kl->ijkl", dm1, dm1) - einsum("ij,kl->iklj", dm1, dm1) / 2
        if not ao_basis:
            return dm2
        return einsum("ijkl,ai,bj,ck,dl->abcd", dm2, *(4 * [self.mo.coeff]))
    
    def copy(self):
        dm1 = spinalg.copy(self.dm1)
        dm2 = spinalg.copy(self.dm2)
        proj = callif(spinalg.copy, self.projector)
        return type(self)(self.mo.copy(), dm1, dm2, projector=proj)

    def project(self, projector, inplace):
        wf = self if inplace else self.copy()
        wf.dm1 = project_c1(wf.dm1, projector)
        wf.dm2 = project_c2(wf.dm2, projector)
        wf.projector = projector
        return wf
    
    def pack(self, dtype=float):
        """Pack into a single array of data type `dtype`.

        Useful for communication via MPI."""
        mo = self.mo.pack(dtype=dtype)
        data = (mo, dm1, dm2, self.projector)
        pack = pack_arrays(*data, dtype=dtype)
        return pack

    @classmethod
    def unpack(cls, packed):
        """Unpack from a single array of data type `dtype`.

        Useful for communication via MPI."""
        mo, dm1, dm2, projector = unpack_arrays(packed)
        mo = SpatialOrbitals.unpack(mo)
        return cls(mo, dm1, dm2, projector=projector)
    
    def restore(self):
        raise NotImplementedError()
    
    def as_unrestricted(self):
        raise NotImplementedError()

class URDM_WaveFunction(RRDM_WaveFunction):
    """
    Spin-unrestricted dummy wavefunction type that stores the 1- and 2-RDMs.
    Allows interoperability with user-defined callback solvers which only 
    return the 1- and 2-RDMs.
    """

    @property
    def dm1a(self):
        return self.dm1[0]
    
    @property
    def dm1b(self):
        return self.dm1[1]
    
    @property
    def dm2aa(self):
        return self.dm2[0]
    
    @property
    def dm2ab(self):
        return self.dm2[1]
    
    @property
    def dm2ba(self):
        if len(self.dm2) == 4:
            return self.dm2[2]
        return self.dm2[2].transpose(1, 0, 3, 2)
    
    @property
    def dm2bb(self):
        return self.dm2[-1]
    
    def make_rdm1(self, ao_basis=False, with_mf=True):
        dm1 = self.dm1a.copy(), self.dm1b.copy()
        if not with_mf:
            dm1[0][np.diag_indices(self.nocc[0])] -= 1
            dm1[1][np.diag_indices(self.nocc[1])] -= 1
        if not ao_basis:
            return dm1
        return (dot(self.mo.coeff[0], dm1[0], self.mo.coeff[0].T),
                dot(self.mo.coeff[1], dm1[1], self.mo.coeff[1].T))
    
    def make_rdm2(self, ao_basis=False, with_dm1=True, approx_cumulant=True):
        nocca, noccb = self.nocc
        dm1a, dm1b = self.dm1a.copy(), self.dm1b.copy() 
        dm2aa, dm2ab, dm2bb = self.dm2aa.copy(), self.dm2ab.copy(), self.dm2bb.copy()
        if not with_dm1:
            if not approx_cumulant:
                ncum2aa, ncum2ab, ncum2bb = self.make_rdm2_non_cumulant(ao_basis=False)   
                dm2aa -= ncum2aa
                dm2ab -= ncum2ab
                dm2bb -= ncum2bb 
            elif approx_cumulant in (1, True):
                dm1a[np.diag_indices(self.nocc[0])] -= 1
                dm1b[np.diag_indices(self.nocc[1])] -= 1
                for i in range(nocca):
                    dm2aa[i, i, :, :] -= dm1a
                    dm2aa[:, :, i, i] -= dm1a
                    dm2aa[:, i, i, :] += dm1a
                    dm2aa[i, :, :, i] += dm1a.T
                    dm2ab[i, i, :, :] -= dm1b
                for i in range(noccb):
                    dm2bb[i, i, :, :] -= dm1b
                    dm2bb[:, :, i, i] -= dm1b
                    dm2bb[:, i, i, :] += dm1b
                    dm2bb[i, :, :, i] += dm1b.T
                    dm2ab[:, :, i, i] -= dm1a

                for i in range(nocca):
                    for j in range(nocca):
                        dm2aa[i, i, j, j] -= 1
                        dm2aa[i, j, j, i] += 1
                for i in range(noccb):
                    for j in range(noccb):
                        dm2bb[i, i, j, j] -= 1
                        dm2bb[i, j, j, i] += 1
                for i in range(nocca):
                    for j in range(noccb):
                        dm2ab[i, i, j, j] -= 1

            elif approx_cumulant == 2:
                raise NotImplementedError()
            else:
                raise ValueError
        if not ao_basis:
            return (dm2aa, dm2ab, dm2bb)
        return (einsum("ijkl,ai,bj,ck,dl->abcd", dm2aa, *(4 * [self.mo.coeff[0]])),
                einsum("ijkl,ai,bj,ck,dl->abcd", dm2ab, *(2 * [self.mo.coeff[0]] + 2 * [self.mo.coeff[1]])),
                einsum("ijkl,ai,bj,ck,dl->abcd", dm2bb, *(4 * [self.mo.coeff[1]]))
               )
    
    def make_rdm2_non_cumulant(self, ao_basis=False):
        dm1a, dm1b = self.dm1a.copy(), self.dm1b.copy()
        dm2aa = einsum("ij,kl->ijkl", dm1a, dm1a) - einsum("ij,kl->iklj", dm1a, dm1a)
        dm2bb = einsum("ij,kl->ijkl", dm1b, dm1b) - einsum("ij,kl->iklj", dm1b, dm1b)
        dm2ab = einsum("ij,kl->ijkl", dm1a, dm1b)
        if not ao_basis:
            return (dm2aa, dm2ab, dm2bb)
        return (einsum("ijkl,ai,bj,ck,dl->abcd", dm2aa, *(4 * [self.mo.coeff[0]])),
                einsum("ijkl,ai,bj,ck,dl->abcd", dm2ab, *(2 * [self.mo.coeff[0]] + 2 * [self.mo.coeff[1]])),
                einsum("ijkl,ai,bj,ck,dl->abcd", dm2bb, *(4 * [self.mo.coeff[1]]))
               )

    def copy(self):
        dm1 = [spinalg.copy(d) for d in self.dm1]
        dm2 = [spinalg.copy(d) for d in self.dm2]
        proj = callif(spinalg.copy, self.projector)
        return type(self)(self.mo.copy(), dm1, dm2, projector=proj)

    def project(self, projector, inplace):
        wf = self if inplace else self.copy()
        wf.dm1 = project_uc1(wf.dm1, projector)
        wf.dm2 = project_uc2(wf.dm2, projector)
        wf.projector = projector
        return wf

    # def pack(self, dtype=float):
    #     """Pack into a single array of data type `dtype`.

    #     Useful for communication via MPI."""
    #     mo = self.mo.pack(dtype=dtype)
    #     data = (mo, *self.dm1, *self.dm2, self.projector)
    #     pack = pack_arrays(*data, dtype=dtype)
    #     return pack

    # @classmethod
    # def unpack(cls, packed):
    #     """Unpack from a single array of data type `dtype`.

    #     Useful for communication via MPI."""
    #     mo, *dm1, *dm2, projector = unpack_arrays(packed)
    #     mo = SpatialOrbitals.unpack(mo)
    #     return cls(mo, dm1, dm2, projector=projector)

    # def restore(self):
    #     raise NotImplementedError()

    
    