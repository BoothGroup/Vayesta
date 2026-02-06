import numpy as np
from abc import ABC, abstractmethod
from dyson.util.linalg import matrix_power
from vayesta.core.util import einsum


class Dynamical(ABC):

    @abstractmethod
    def hermitian(self):
        """Whether the Moment representation is Hermitian."""
        pass

    @abstractmethod
    def nsectors(self):
        """Number of sectors."""
        pass

    @abstractmethod
    def hermitize(self):
        """Return a new Hermitian version of the Moment representation."""
        pass

    @abstractmethod
    def rotate(self, rotation):
        """Change the basis of the moments using the given unitary matrix."""
        pass

    @abstractmethod
    def project(self, projector, nproj):
        """Project the moments using the given projector."""
        pass


class GreensFunction(Dynamical):
    pass

class SelfEnergy(Dynamical):
    
    def _init_static_overlap(self, static, overlap, hermitian):

        """
        
        Function to initialize static and overlap matrices.

        If passed a single static self-energy, this will be associated with all sectors.
        If passed a list/array of statics, each will be associated with the corresponding sector.

        If None is passed for overlap, identity matrices will be assumed.

        
        Parameters
        ----------
        
        static : ndarray (nphys, nphys) or (nsectors, nphys, nphys)
            Static part of the self-energy.

        overlap : ndarray (nphys, nphys) or (nsectors, nphys, nphys) or None
            Overlap matrices for each sector.

        """

        static = np.array(static).copy()
        if overlap is not None:
            overlap = np.array(overlap).copy()
            if static.shape[-2:] != static.shape[-2:]:
                raise ValueError("Static and overlap matrices should have the same physical dimension.")
        
        if hermitian is None:
            hermitian = np.allclose(static, static.conj().swapaxes(-2,-1))
            if overlap is not None:
                hermitian = hermitian and np.allclose(overlap, overlap.conj().swapaxes(-2,-1))

        elif hermitian:
            static = 0.5 * (static + static.conj().swapaxes(-2,-1))
            if overlap is not None:
                overlap = 0.5 * (overlap + overlap.conj().swapaxes(-2,-1))

        self._statics = static
        self._overlaps = overlap

    def _hermitize_static_overlap(self):
        """Return Hermitian versions of the static and overlap matrices."""
        herm_static = 0.5 * (self.statics + self.statics.conj().swapaxes(-2,-1))
        if self.overlaps is not None:
            herm_overlap = 0.5 * (self.overlaps + self.overlaps.conj().swapaxes(-2,-1))
        else:
            herm_overlap = None
        return herm_static, herm_overlap

    def _project_static_overlap(self, projector, nproj):
        """Project the static and overlap matrices using the given projector."""

        if nproj == 1:
            proj_static = 0.5 * (einsum('pP,...Pq->...pq', projector, self.statics) + einsum('qQ,...pQ->...pq', projector, self.statics))
            proj_overlap = 0.5 * (einsum('pP,...Pq->...pq', projector, self.overlaps) + einsum('qQ,...pQ->...pq', projector, self.overlaps)) if self.overlaps is not None else None

        elif nproj == 2:
            proj_static = einsum('pP,qQ,...PQ->...pq', projector, projector, self.statics)
            proj_overlap = einsum('pP,qQ,...PQ->...pq', projector, projector, self.overlaps) if self.overlaps is not None else None

        return proj_static, proj_overlap

    def _rotate_static_overlap(self, rotmat):
        """Rotate the static and overlap matrices using the given rotation matrix."""

        rotated_static = einsum('pP,qQ,...PQ->...pq', rotmat, rotmat.conj(), self.statics)
        if self.overlaps is not None:
            rotated_overlap = einsum('pP,qQ,...PQ->...pq', rotmat, rotmat.conj(), self.overlaps)
        else:
            rotated_overlap = None
        return rotated_static, rotated_overlap
    
    def _combine_static_overlap(self):

        if self.single_static:
            combined_static = self.statics
        else:
            combined_static = np.zeros_like(self.statics[0])
            for s in range(self.nsectors):
                combined_static += self.statics[s]

        if self.single_overlap:
            combined_overlap = self.overlaps
        else:
            combined_overlap = np.zeros_like(self.overlaps[0])
            for s in range(self.nsectors):
                combined_overlap += self.overlaps[s]

        return combined_static, combined_overlap

        
    @property
    def statics(self):
        """Static part of the self-energy."""
        return self._statics

    @property
    def single_static(self):
        """Whether the static part is the same for all sectors."""
        return self.statics.ndim == 2


    @property
    def overlaps(self):
        """Overlap matrices for each sector."""
        return self._overlaps

    @property
    def single_overlap(self):
        """Whether the overlap is the same for all sectors."""
        return self.overlaps is None or self.overlaps.ndim == 2

    @property
    def _hermitian_static_overlap(self):
        """Check if the static and overlap matrices are Hermitian."""
        herm_static = np.allclose(self.statics, self.statics.conj().swapaxes(-2,-1))
        if self.overlaps is not None:
            herm_overlap = np.allclose(self.overlaps, self.overlaps.conj().swapaxes(-2,-1))
        else:
            herm_overlap = True
        return herm_static and herm_overlap
    

