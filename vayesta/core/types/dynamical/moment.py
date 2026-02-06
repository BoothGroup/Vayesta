import numpy as np

from abc import ABC, abstractmethod

from dyson import Lehmann, Spectral, MBLGF, MBLSE
from dyson.util.moments import gf_moments_to_se_moments, se_moments_to_gf_moments
from dyson.util.linalg import matrix_power

from vayesta.core.util import einsum

from .dynamical import Dynamical, GreensFunction, SelfEnergy
from .spectral import SpectralRep

class MomentRep(Dynamical):

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

    @property
    def nphys(self):
        """Physical space dimension of the Moment representation."""
        return self.moments.shape[-1]
    
    def shift(self, shift):
        """Shift the moments by a constant value."""
        raise NotImplementedError("Moment shifting not implemented yet.")
    
    @abstractmethod
    def to_spectral(self):
        """Convert moments to Spectral representation using Block Lanczos."""
        pass


class GF_MomentRep(MomentRep, GreensFunction):

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

    def hermitize(self):
        """Return a new Hermitian version of the Moment representation.
        
        Returns
        -------
        hermitian_moments : GF_MomentRep
            Hermitian GF moment representation.
        """
        if self.hermitian:
            return self
        else:
            herm_moms = 0.5 * (self.moments + self.moments.conj().transpose(0,1,3,2))
            return type(self)(herm_moms, hermitian=True)

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


    def to_se_moments(self, orth_basis=False):
        """Convert to self-energy moments using recurrence relations."""
        se_statics = []
        se_moments = []
        overlaps = []
        for sector in range(self.nsectors):
            se_static, se_moms = gf_moments_to_se_moments(self.moments[sector], orth_basis=orth_basis)
            se_statics.append(se_static)
            se_moments.append(se_moms)
            if orth_basis:
                overlaps.append(np.eye(self.moments.shape[-1]))
            else:
                overlaps.append(self.moments[sector][0])
        return SE_MomentRep(np.array(se_statics), np.array(se_moments), overlap=overlaps, hermitian=self.hermitian)

    def orthogonalize(self, overlaps=None):
        """Orthogonalize the moments using their overlap."""
        if overlaps is None:
            overlaps = self.moments[:,0,:,:]
        orth_moms = []
        for s in range(self.nsectors):
            orth, error_orth = matrix_power(overlaps[s], -0.5, hermitian=self.hermitian, return_error=False)
            moms = einsum("npq,ip,qj->nij", self.moments[s], orth, orth)
            orth_moms.append(moms)
        return GF_MomentRep(np.array(orth_moms), hermitian=self.hermitian)

    def unorthogonalize(self, overlaps=None):
        """Unorthogonalize the moments using their overlap."""
        if overlaps is None:
            overlaps = self.moments[:,0,:,:]
        unorth_moms = []
        for s in range(self.nsectors):
            unorth, error_unorth = matrix_power(overlaps[s], 0.5, hermitian=self.hermitian, return_error=False)
            moms = einsum("npq,ip,qj->nij", self.moments[s], unorth, unorth)
            unorth_moms.append(moms)
        return GF_MomentRep(np.array(unorth_moms), hermitian=self.hermitian)

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
            return SpectralRep([combined_spectral])
        else:
            return SpectralRep(spectrals)


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
        





class SE_MomentRep(MomentRep, SelfEnergy):

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

        moments = np.array(moments).copy()


        self._init_static_overlap(static, overlap, hermitian)

        if moments.ndim != 3 and moments.ndim != 4:
            raise ValueError("Moments should be a 3D array with shape (nmom, nphys, nphys) or (sector, nmom, nphys, nphys).")
        if moments.shape[-1] != moments.shape[-2]:
            raise ValueError("Moments should be square matrices in the last two dimensions.")
        
        if moments.shape[-2:] != self.statics.shape[-2:]:
            raise ValueError("Static and moment matrices should have the same physical dimension.")
        if self.overlaps is not None and moments.shape[-2:] != self.overlaps.shape[-2:]:
            raise ValueError("Overlap and moment matrices should have the same physical dimension.")

        # GW has a single static part for all sectors. For CCSD/FCI it is sector-dependent.
        # if static.ndim == 2:
        #     single_static = True

        # if overlap.ndim == 2:
        #     single_overlap = True

        if hermitian is None:
            hermitian = np.allclose(moments, moments.conj().transpose(0,1,3,2))
            hermitian = hermitian and self._hermitian_static_overlap
        

        elif hermitian:
            moments = 0.5 * (moments + moments.conj().transpose(0,1,3,2))

        self._moments = moments
        self._hermitian = hermitian


    # def copy(self, overlap=None, static=None):
    #     """Create a copy of the SE_MomentRep, optionally replacing overlap or static.

    #     TODO: Remove or perform shape checks

    #     Parameters
    #     ----------
    #     overlap : ndarray (nphys, nphys) or (nsectors, nphys, nphys), optional
    #         New overlap matrices for each sector.
    #     static : ndarray (nphys, nphys) or (nsectors, nphys, nphys), optional
    #         New static part of the self-energy.

    #     Returns
    #     -------
    #     copied_moments : SE_MomentRep
    #         Copied moment representation.
    #     """
    #     new_overlap = overlap if overlap is not None else self.overlaps
    #     new_static = static if static is not None else self.statics
    #     hermitian = self.hermitian and np.allclose(new_static, new_static.conj().transpose(0,2,1)) and np.allclose(new_overlap, new_overlap.conj().transpose(0,2,1))            
    #     return SE_MomentRep(new_static, self.moments, overlap=new_overlap, hermitian=hermitian)
    

    
    @property
    def moments(self):
        """Moments of the self-energy."""
        return self._moments

    
    @property
    def hermitian(self):
        """Check if the moments are Hermitian."""
        return self._hermitian
    
    def hermitize(self):
        """Return a new Hermitian version of the Moment representation.
        
        Returns
        -------
        hermitian_moments : SE_MomentRep
            Hermitian SE moment representation.
        """
        if self.hermitian:
            return self
        else:
            herm_static, herm_overlap = self._hermitize_static_overlap()
            herm_moms = 0.5 * (self.moments + self.moments.conj().swapaxes(-2,-1))
            return type(self)(herm_static, herm_moms, overlap=herm_overlap, hermitian=True)
    
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
        proj_static, proj_overlap = self._project_static_overlap(projector, nproj)
        if nproj == 1:
            proj_moms = 0.5 * (einsum('pP,...Pq->...pq', projector, self.moments) + einsum('qQ,...pQ->...pq', projector, self.moments))
            
        elif nproj == 2:
            proj_moms = einsum('pP,qQ,...PQ->...pq', projector, projector, self.moments)
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
        rotated_static, rotated_overlap = self._rotate_static_overlap(rotmat)
        rotated_moms = einsum('pP,qQ,...PQ->...pq', rotmat, rotmat.conj(), self.moments)
        return type(self)(rotated_static, rotated_moms, rotated_overlap)


    def orthogonalize(self, overlaps=None):
        """Orthogonalize the moments using their overlap.
        
        Parameters
        ----------
        overlaps : ndarray (nphys, nphys) or (nsectors, nphys, nphys), optional
            Overlap matrices to use for orthogonalization. If None, use self.overlaps.

        Returns
        -------
        orthogonalized_moments : SE_MomentRep
            Orthogonalized moment representation
        """
        if overlaps is None:
            overlaps = self.overlaps
        orth_moms = []
        orth_static = []
        orth_overlaps = []
        for s in range(self.nsectors):
            orth, error_orth = matrix_power(overlaps[s], -0.5, hermitian=self.hermitian, return_error=False)
            moms = einsum("npq,ip,qj->nij", self.moments[s], orth, orth)
            orth_moms.append(moms)

            stat = orth @ self.statics[s] @ orth
            orth_static.append(stat)
            orth_overlaps.append(orth @ overlaps[s] @ orth) # Should old overlaps be kept?
        return SE_MomentRep(np.array(orth_static), np.array(orth_moms), overlap=np.array(orth_overlaps), hermitian=self.hermitian)

    def unorthogonalize(self, overlaps=None):
        """Unorthogonalize the moments using their overlap.
        
        Parameters
        ----------
        overlaps : ndarray (nphys, nphys) or (nsectors, nphys, nphys), optional
            Overlap matrices to use for unorthogonalization. If None, use self.overlaps.

        Returns
        -------
        unorthogonalized_moments : SE_MomentRep
            Unorthogonalized moment representation.
        """
        if overlaps is None:
            overlaps = self.overlaps
        unorth_moms = []
        unorth_static = []
        unorth_overlaps = []
        for s in range(self.nsectors):
            unorth, error_unorth = matrix_power(overlaps[s], 0.5, hermitian=self.hermitian, return_error=False)
            moms = einsum("npq,ip,qj->nij", self.moments[s], unorth, unorth)
            unorth_moms.append(moms)

            stat = unorth @ self.statics[s] @ unorth
            unorth_static.append(stat)
            unorth_overlaps.append(unorth @ self.overlaps[s] @ unorth)
        return SE_MomentRep(np.array(unorth_static), np.array(unorth_moms), overlap=np.array(overlaps), hermitian=self.hermitian)

    def to_gf_moments(self):
        """Convert to Green's function moments using recurrence relations."""
        gf_moments = []
        for s in range(self.nsectors):
            static = self.statics if self.single_static else self.statics[s]
            overlap = self.overlaps if self.single_overlap else self.overlaps[s]
            gf_moments.append(se_moments_to_gf_moments(static, self.moments[s], overlap=overlap))
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

            static = self.statics if self.single_static else self.statics[s]
            if self.overlaps is not None:
                overlap = self.overlaps if self.single_overlap else self.overlaps[s]
            else:
                overlap = None

            solver = MBLSE(static, self.moments[s], overlap=overlap, hermitian=hermitian)
            solver.kernel()
            spectrals.append(solver.result)
        
        if combine:
            combined_spectral = Spectral.combine_spectrals(spectrals)
            return SpectralRep([combined_spectral])
        else:
            return SpectralRep(spectrals)

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
        if not compatible_shapes:
            raise ValueError("All MomentRep instances must have the same shape to be combined.")
        
        combined_overlap = self.overlaps
        combined_static = self.statics
        combined_moments = self.moments
        for arg in args:
            if arg.overlaps is not None:
                combined_overlap += arg.overlaps
            combined_static += arg.statics
            combined_moments += arg.moments
        return SE_MomentRep(combined_static, combined_moments, overlap=combined_overlap)
    

    def combine_sectors(self):
        """Combine all sectors into a single SE_MomentRep by summing static and moments.

        Returns
        -------
        combined_moments : SE_MomentRep
            Combined moment representation.
        """
        
        combined_overlap = np.zeros_like(self.overlaps[0])
        combined_static = np.zeros_like(self.statics[0])
        combined_moments = np.zeros_like(self.moments[0])
        
        for s in range(self.nsectors):
            combined_moments += self.moments[s]

        if self.single_static:
            combined_static = self.statics
        else:
            for s in range(self.nsectors):
                combined_static += self.statics[s]
        

        if self.single_overlap:
            combined_overlap = self.overlaps
        elif self.overlaps is not None:
            for s in range(self.nsectors):
                combined_overlap += self.overlaps[s]

        return SE_MomentRep(combined_static, combined_moments, overlap=combined_overlap)


    


        
    