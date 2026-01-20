import numpy as np

from dyson import Spectral

import vayesta.core.types.dynamical as dynamical 

class SpectralRep(object):
    """Auxilliary space spectral representation."""

    def __init__(self, spectrals):
        """Initialize spectral representation.

        Parameters
        ----------
        spectrals : list of Spectral
            List of spectral objects for each sector.
        """
        if isinstance(spectrals, Spectral):
            spectrals = [spectrals]

        self._spectrals = spectrals

    @property
    def nsectors(self):
        """Number of sectors in the spectral representation."""
        return len(self.spectrals)

    @property
    def spectrals(self):
        """List of spectral objects for each sector."""
        return self._spectrals
    
    @property
    def hermitian(self):
        """Whether the spectral representation is Hermitian."""
        hermitian = True
        for spectral in self.spectrals:
            hermitian = hermitian and spectral.hermitian
        return hermitian
    
    def combine_sectors(self):
        """Combine all sectors objects into a single spectral object.

        Returns
        -------
        Spectral : Spectral
            Combined spectral object.
        """
        return type(self)(Spectral.combine_dyson(*self.spectrals))
    
    def to_gf_lehmann(self):
        """Convert to Lehmann Green's function representation.

        Returns
        -------
        GF_LehmannRep
            Lehmann Green's function representation.
        """
        gfs = []
        for spectral in self.spectrals:
            gf = spectral.get_greens_function()
            gfs.append(gf)
        return dynamical.GF_LehmannRep(gfs)
    
    def to_se_lehmann(self):
        """Convert to Lehmann self-energy representation.

        Returns
        -------
        SE_LehmannRep
            Lehmann self-energy representation.
        """
        statics = []
        ses = []
        overlaps = []
        for spectral in self.spectrals:
            se = spectral.get_self_energy()
            statics.append(spectral.get_static_self_energy())
            overlaps.append(spectral.get_overlap())
            ses.append(se)
        return dynamical.SE_LehmannRep(statics, ses, overlaps=overlaps)
    

    def to_gf_moments(self, nmom=None):
        """Convert to Green's function moments representation.
        
        Returns
        -------
        GF_MomentRep
            Green's function moments representation.
        """
        moms = []
        for s in range(self.nsectors):
            gf = self.spectrals[s].get_greens_function()
            if nmom is None:
                # FIXME: Determine proper nmom for both odd and even input moms
                nmom = 2 * self.spectrals[s].eigvals.shape[0] // self.spectrals[s].nphys 
            moms.append(gf.moments(range(nmom)))

        return dynamical.GF_MomentRep(np.array(moms), hermitian=self.hermitian)
    
    def to_se_moments(self, nmom=None):
        """Convert to self-energy moments representation.
        
        Returns
        -------
        SE_MomentRep
            Self-energy moments representation.
        """
        statics = []
        overlaps = []
        moms = []
        for s in range(self.nsectors):
            se = self.spectrals[s].get_self_energy()
            if nmom is None:
                # FIXME: Determine proper nmom for both odd and even input moms
                nmom = 2 * self.spectrals[s].eigvals.shape[0] // self.spectrals[s].nphys - 2
            moms.append(se.moments(range(nmom)))
            statics.append(self.spectrals[s].get_static_self_energy())
            overlaps.append(self.spectrals[s].get_overlap())
            
        return dynamical.SE_MomentRep(np.array(statics), np.array(moms), overlap=overlaps, hermitian=self.hermitian)

    def project(self, projector, nproj):
        raise NotImplementedError("Projection of Spectral representation not implemented yet.")
