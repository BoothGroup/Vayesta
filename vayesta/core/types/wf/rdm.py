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
        cls = URDM_Wavefunction
    return cls(mo, dm1, dm2, **kwargs)

class RRDM_WaveFunction(wf_types.WaveFunction):
    """
    Dummy wavefunction type that stores the 1- and 2-RDMs.
    Allows interoperability with user-defined callback solvers 
    which can only return the 1- and 2-RDMs.
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
                dm2 -= einsum("ij,kl->ijkl", dm1, dm1) - einsum("ij,kl->iklj", dm1, dm1) / 2
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
    pass