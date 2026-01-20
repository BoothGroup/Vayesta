import numpy as np

from abc import ABC, abstractmethod

from dyson import Lehmann, Spectral, MBLGF, MBLSE
from dyson.util.moments import gf_moments_to_se_moments, se_moments_to_gf_moments

from vayesta.core.util import einsum
import vayesta.core.types.dynamical as dynamical 


class MomentRep(ABC):

    @property
    def nmom(self):
        """Number of moments in the Moment representation."""
        return self.moments.shape[1]
    
    @property
    def nsectors(self):
        """Number of sectors in the Moment representation."""
        if self.moments.ndim == 4:
            return self.moments.shape[0]
        return 1
    
    def shift(self, shift):
        """Shift the moments by a constant value."""
        raise NotImplementedError("Moment shifting not implemented yet.")
    
    @abstractmethod
    def hermitian(self):
        """Check if the moments are Hermitian."""
        pass

    @abstractmethod
    def rotate(self, rotation):
        """Change the basis of the moments using the given unitary matrix."""
        pass

    @abstractmethod
    def project(self, projector, nproj):
        """Project the moments using the given projector."""
        pass

    
    @abstractmethod
    def to_spectral(self):
        """Convert moments to Spectral representation using Block Lanczos."""
        pass
    


class GF_MomentRep(MomentRep):

    def __init__(self, moments, hermitian=None):

        """Initialize Green's function moment representation.

        Parameters
        ----------
        moments : ndarray (nmom, nphys, nphys) or (nsectors, nmom, nphys, nphys)
            Moments of the Green's function.
        hermitian : bool, optional
            If specified, indicates whether the moments are Hermitian. Otherwise, determined automatically.
        """

        moments = np.array(moments)
        if moments.ndim != 3 and moments.ndim != 4:
            raise ValueError("Moments should be a 3D array with shape (nmom, nphys, nphys) or (sector, nmom, nphys, nphys).")
        if moments.shape[-1] != moments.shape[-2]:
            raise ValueError("Moments should be square matrices in the last two dimensions.")
        
        if moments.ndim == 3:
            self._moments = moments.copy()[np.newaxis, :, :, :]
        else:   
            self._moments = moments.copy()

        if hermitian is None:
            hermitian = np.allclose(self.moments, self.moments.conj().transpose(0,1,3,2))
        self._hermitian = hermitian

        if hermitian:
            self._moments = 0.5 * (self._moments + self._moments.conj().transpose(0,1,3,2))

    @property
    def moments(self):
        """Moments of the Green's function."""
        return self._moments
    
    @property
    def hermitian(self):
        """Check if the moments are Hermitian."""
        return self._hermitian

    def project(self, projector, nproj):
        """Project the moments using the given projector.

        Parameters
        ----------
        projector : ndarray (nphys, nphys)
            Projector matrix from physical space to projected space.
        nproj : int
            Number of indices to project (1 or 2).

        Returns
        -------
        projected_moments : MomentRep
            Projected moment representation.
        """
        if nproj == 1:

            proj_moms = 0.5 * (einsum('pP,...Pq->...pq', projector, self.moments) + einsum('qQ,...pQ->...pq', projector, self.moments))
        elif nproj == 2:
            proj_moms = einsum('pP,qQ,...PQ->...pq', projector, projector, self.moments)
        return type(self)(proj_moms)
    
    def rotate(self, rotmat):
        """Change the basis of the moments using the rectangular matrix

        Parameters
        ----------
        rotmat : ndarray (N, nphys)
            Rotation matrix from old basis to new basis. Eg. cluster MO coefficients.

        Returns
        -------
        rotated_moments : MomentRep
            Rotated moment representation.
        """

        rotated_moms = einsum('pP,qQ,...PQ->...pq', rotmat, rotmat.conj(), self.moments)
        return type(self)(rotated_moms)


    def to_se_moments(self):
        """Convert to self-energy moments using recurrence relations."""
        se_statics = []
        se_moments = []
        overlaps = []
        for sector in range(self.nsectors):
            se_static, se_moms = gf_moments_to_se_moments(self.moments[sector])
            se_statics.append(se_static)
            se_moments.append(se_moms)
            overlaps.append(self.moments[sector][0])
        return SE_MomentRep(np.array(se_statics), np.array(se_moments), overlap=overlaps, hermitian=self.hermitian)


    def to_spectral(self, hermitian=None, combine=False):
        """Convert moments to Spectral representation using Block Lanczos.
        
        Parameters
        ----------
        hermitian : bool, optional
            If specified, passed to the Block Lanczos solver.
        combine : bool
            If True, combine all sectors into a single Spectral representation.
            
        Returns
        -------
        spectral_rep : SpectralRep
            Spectral representation of the Green's function.
        """

        spectrals = []
        hermitian = hermitian if hermitian is not None else self.hermitian
        for s in range(self.nsectors):
            
            solver = MBLGF(self.moments[s], hermitian=hermitian)
            solver.kernel()
            spectrals.append(solver.result)

        if combine:
            combined_spectral = Spectral.combine_spectrals(spectrals)
            return dynamical.SpectralRep([combined_spectral])
        else:
            return dynamical.SpectralRep(spectrals)


    def combine(self, *args):
        """Combine multiple GF_MomentRep into a single one by summing moments.

        Parameters
        ----------
        *args : list of GF_MomentRep
            Moment representations to combine.

        Returns
        -------
        combined_moments : GF_MomentRep
            Combined moment representation.
        """
        
        compatible_shapes = all(arg.moments.shape == self.moments.shape for arg in args)
        if not compatible_shapes:
            raise ValueError("All MomentRep instances must have the same shape to be combined.")
        combined_moments = np.zeros_like(self.moments)
        combined_moments += self.moments
        for arg in args:
            combined_moments += arg.moments
        return GF_MomentRep(combined_moments)
        


class SE_MomentRep(MomentRep):

    def __init__(self, static, moments, overlap=None, hermitian=None):

        """Initialize self-energy moment representation.
        
        Parameters
        ----------
        static : ndarray (nphys, nphys) or (nsectors, nphys, nphys)
            Static part of the self-energy.
        moments : ndarray (nmom, nphys, nphys) or (nsectors, nmom, nphys, nphys)
            Moments of the self-energy.
        overlap : ndarray (nphys, nphys) or (nsectors, nphys, nphys), optional
            Overlap matrices for each sector.
        hermitian : bool, optional
            If specified, indicates whether the moments are Hermitian. Otherwise, determined automatically.
            
        """

        static = np.array(static)
        moments =  np.array(moments)

        if moments.ndim != 3 and moments.ndim != 4:
            raise ValueError("Moments should be a 3D array with shape (nmom, nphys, nphys) or (sector, nmom, nphys, nphys).")
        if moments.shape[-1] != moments.shape[-2]:
            raise ValueError("Moments should be square matrices in the last two dimensions.")
        

        if moments.ndim == 3:
            self._statics = static.copy()[np.newaxis, :, :] 
            self._moments = moments.copy()[np.newaxis, :, :, :]
        else:   
            self._statics = static.copy()
            self._moments = moments.copy()

        if overlap is None:
            overlap = [None for _ in range(self._moments.shape[0])]
        elif np.array(overlap).ndim == 2:
            overlap = [overlap.copy() for _ in range(self._moments.shape[0])]
        
        self._overlaps = np.array(overlap).copy()

        if hermitian is None:
            hermitian = np.allclose(self.moments, self.moments.conj().transpose(0,1,3,2))
            hermitian = hermitian and np.allclose(self.statics, self.statics.conj().transpose(0,2,1))
            hermitian = hermitian and np.allclose(self.overlaps, self.overlaps.conj().transpose(0,2,1))

        if hermitian:
            self._statics = 0.5 * (self._statics + self._statics.conj().transpose(0,2,1))
            self._moments = 0.5 * (self._moments + self._moments.conj().transpose(0,1,3,2))
            self._overlaps = 0.5 * (self._overlaps + self._overlaps.conj().transpose(0,2,1))

        self._hermitian = hermitian



    @property
    def statics(self):
        """Static part of the self-energy."""
        return self._statics
    
    @property
    def moments(self):
        """Moments of the self-energy."""
        return self._moments
    
    @property
    def overlaps(self):
        """Overlap matrices for each sector."""
        return self._overlaps
    
    @property
    def hermitian(self):
        """Check if the moments are Hermitian."""
        return self._hermitian
    
    def project(self, projector, nproj):
        """Project the moments using the given projector.

        Parameters
        ----------
        projector : ndarray (nphys, nphys)
            Projector matrix from physical space to projected space.
        nproj : int
            Number of indices to project (1 or 2).

        Returns
        -------
        projected_moments : MomentRep
            Projected moment representation.
        """
        if nproj == 1:
            proj_moms = 0.5 * (einsum('pP,...Pq->...pq', projector, self.moments) + einsum('qQ,...pQ->...pq', projector, self.moments))
            proj_static = 0.5 * (einsum('pP,...Pq->...pq', projector, self.statics) + einsum('qQ,...pQ->...pq', projector, self.statics))
            proj_overlap = 0.5 * (einsum('pP,...Pq->...pq', projector, self.overlaps) + einsum('qQ,...pQ->...pq', projector, self.overlaps))
        elif nproj == 2:
            proj_moms = einsum('pP,qQ,...PQ->...pq', projector, projector, self.moments)
            proj_static = einsum('pP,qQ,...PQ->...pq', projector, projector, self.statics)
            proj_overlap = einsum('pP,qQ,...PQ->...pq', projector, projector, self.overlaps)
        return type(self)(proj_static, proj_moms, proj_overlap) 
    

    def rotate(self, rotmat):
        """Change the basis of the moments using the rectangular matrix

        Parameters
        ----------
        rotmat : ndarray (N, nphys)
            Rotation matrix from old basis to new basis. Eg. cluster MO coefficients.

        Returns
        -------
        rotated_moments : MomentRep
            Rotated moment representation.
        """
        rotated_static = einsum('pP,qQ,...PQ->...pq', rotmat, rotmat.conj(), self.statics)
        rotated_moms = einsum('pP,qQ,...PQ->...pq', rotmat, rotmat.conj(), self.moments)
        if np.all(self.overlaps != None):
            rotated_overlap = einsum('pP,qQ,...PQ->...pq', rotmat, rotmat.conj(), self.overlaps)
        else:
            rotated_overlap = self.overlaps
        return type(self)(rotated_static, rotated_moms, rotated_overlap)

    def to_gf_moments(self):
        """Convert to Green's function moments using recurrence relations."""
        gf_moments = []
        for s in range(self.nsectors):
            gf_moments.append(se_moments_to_gf_moments(self.statics[s], self.moments[s], overlap=self.overlaps[s]))
        return GF_MomentRep(np.array(gf_moments), hermitian=self.hermitian)
    
    def to_spectral(self, hermitian=None, combine=False):
        """Convert moments to Spectral representation using Block Lanczos.
        
        Parameters
        ----------
        hermitian : bool, optional
            If specified, use passed to the Block Lanczos solver.
        combine : bool (default: False)
            If True, combine all sectors into a single Spectral representation.
            
        Returns
        -------
        spectral_rep : SpectralRep
            Spectral representation of the self-energy.
        """

        spectrals = []
        hermitian = hermitian if hermitian is not None else self.hermitian
        for s in range(self.nsectors):
            solver = MBLSE(self.statics[s], self.moments[s], overlap=self.overlaps[s], hermitian=hermitian)
            solver.kernel()
            spectrals.append(solver.result)
        
        if combine:
            combined_spectral = Spectral.combine_spectrals(spectrals)
            return dynamical.SpectralRep([combined_spectral])
        else:
            return dynamical.SpectralRep(spectrals)

    def combine(self, *args):
        """Combine multiple SE_MomentRep into a single one by summing static and moments.

        Parameters
        ----------
        *args : list of SE_MomentRep
            Moment representations to combine.

        Returns
        -------
        combined_moments : SE_MomentRep
            Combined moment representation.
        """
        
        compatible_shapes = all(arg.moments.shape == self.moments.shape for arg in args)
        compatible_shapes = compatible_shapes and all(arg.statics.shape == self.statics.shape for arg in args)
        compatible_shapes = compatible_shapes and all(arg.overlaps.shape == self.overlaps.shape for arg in args)
        if not compatible_shapes:
            raise ValueError("All MomentRep instances must have the same shape to be combined.")
        
        combined_overlap = np.zeros_like(self.overlaps)
        combined_static = np.zeros_like(self.statics)
        combined_moments = np.zeros_like(self.moments)
        combined_overlap += self.overlaps
        combined_static += self.statics
        combined_moments += self.moments
        for arg in args:
            combined_overlap += arg.overlaps
            combined_static += arg.statics
            combined_moments += arg.moments
        return SE_MomentRep(combined_static, combined_moments, overlap=combined_overlap)


    


        
    