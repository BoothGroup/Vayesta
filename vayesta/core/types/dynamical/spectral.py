import numpy as np
from dyson import Spectral, Lehmann
from vayesta.core.util import einsum

from .dynamical import Dynamical, GreensFunction, SelfEnergy

class SpectralRep(Dynamical):
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

        nphys = self._spectrals[0].nphys
        for spectral in spectrals:
            if spectral.nphys != nphys:
                raise ValueError("All sectors must have the same physical space dimension.")
        
    @property
    def nphys(self):
        """Physical space dimension."""
        return self.spectrals[0].nphys

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
    
    def hermitize(self):
        raise NotImplementedError("Hermitization of spectral representation is not implemented.")

    def combine_sectors(self, greens_function=False):
        """Combine all sectors objects into a single spectral object.

        Parameters
        ----------
        greens_function : bool (optional)
            If True, combine the spectral representations to reproduce the sum of their Green's functions (FCI, CCSD)
            If False, combine the spectral representations to reproduce the sum of their self-energies (AGF2, GW)

        Returns
        -------
        Spectral : Spectral
            Combined spectral object.

        """
        if greens_function:
            return type(self)(Spectral.combine_for_greens_function(*self.spectrals))
        else:
            static = self.spectrals[0].get_static_self_energy()
            overlap = self.spectrals[0].get_overlap()
            spec = Spectral.combine_for_self_energy(*self.spectrals)
            se = spec.get_self_energy()
            return type(self)([Spectral.from_self_energy(static, se, overlap=overlap)])
    
    def to_gf_lehmann(self):
        """Convert to Lehmann Green's function representation.

        Returns
        -------
        GF_LehmannRep
            Lehmann Green's function representation.
        """
        from .lehmann import GF_LehmannRep

        gfs = []
        for spectral in self.spectrals:
            gf = spectral.get_greens_function()
            gfs.append(gf)
        return GF_LehmannRep(gfs)
    
    def to_se_lehmann(self):
        """Convert to Lehmann self-energy representation.

        Returns
        -------
        SE_LehmannRep
            Lehmann self-energy representation.
        """
        from .lehmann import SE_LehmannRep

        statics = []
        ses = []
        overlaps = []
        for spectral in self.spectrals:
            se = spectral.get_self_energy()
            statics.append(spectral.get_static_self_energy())
            overlaps.append(spectral.get_overlap())
            ses.append(se)
        return SE_LehmannRep(statics, ses, overlaps=overlaps)
    

    def to_gf_moments(self, nmom=None):
        """Convert to Green's function moments representation.
        
        Returns
        -------
        GF_MomentRep
            Green's function moments representation.
        """
        from .moment import GF_MomentRep

        moms = []
        for s in range(self.nsectors):
            gf = self.spectrals[s].get_greens_function()
            if nmom is None:
                # FIXME: Determine proper nmom for both odd and even input moms
                nmom = 2 * self.spectrals[s].eigvals.shape[0] // self.spectrals[s].nphys 
            moms.append(gf.moments(range(nmom)))

        return GF_MomentRep(np.array(moms), hermitian=self.hermitian)
    
    def to_se_moments(self, nmom=None, split=True, chempot=None):
        """Convert to self-energy moments representation.
        
        Returns
        -------
        SE_MomentRep
            Self-energy moments representation.
        """
        from .moment import SE_MomentRep

        statics = []
        overlaps = []
        moms = []#

        if nmom is None:
            # FIXME: Determine proper nmom for both odd and even input moms
            nmom = 2 * self.spectrals[0].eigvals.shape[0] // self.spectrals[0].nphys - 2

        if split:

            if self.nsectors !=1:
                raise NotImplementedError("Split self-energy moments representation is only implemented for single sector spectral representations.")
            
            else:

                statics = self.spectrals[0].get_static_self_energy()
                overlaps = None

                chempot = self.spectrals[0].chempot if chempot is None else chempot

                se = self.spectrals[0].get_self_energy(chempot=chempot)
                seh = se.occupied().moments(range(nmom))
                sep = se.virtual().moments(range(nmom))
                moms = np.array([seh, sep])

        else:

            for s in range(self.nsectors):
                se = self.spectrals[s].get_self_energy()
                moms.append(se.moments(range(nmom)))
                statics.append(self.spectrals[s].get_static_self_energy())
                overlaps.append(self.spectrals[s].get_overlap())

        return SE_MomentRep(np.array(statics), np.array(moms), overlap=overlaps, hermitian=self.hermitian)


    def project_(self, projector, nproj):
        """Project the spectral representation"""
        proj_spectrals = []
        for s in range(self.nsectors):

            static = self.spectrals[s].get_static_self_energy()
            overlap = self.spectrals[s].get_overlap()
            se = self.spectrals[s].get_self_energy()

            proj_static = projector @ static @ projector.T.conj()
            proj_overlap = projector @ overlap @ projector.T.conj()
            
            if se.hermitian:
                couplings = np.array(se.couplings)
            else:
                couplings = np.array(se.unpack_couplings())
            
            proj_couplings = einsum('pP,...Pa->...pa', projector, couplings)
            proj_se = Lehmann(se.energies, proj_couplings)

            proj_spectral = Spectral.from_self_energy(
                static=proj_static,
                self_energy=proj_se,
                overlap=proj_overlap,
            )

            proj_spectrals.append(proj_spectral)
        return type(self)(proj_spectrals)
    

    def project(self, projector, nproj):
        """Project the spectral representation using arrowhead structure"""
        proj_spectrals = []
        for s in range(self.nsectors):

            arrowhead = self.spectrals[s].get_arrowhead()

            nphys = self.spectrals[s].nphys
            naux = arrowhead.shape[0] - nphys 

            proj_arrowhead = np.zeros((nproj + naux, nproj + naux), dtype=arrowhead.dtype)

            phys = slice(None, nphys)
            aux = slice(nphys, None)

            if nproj == 2:

                proj_arrowhead[phys, phys] = projector @ arrowhead[phys, phys] @ projector.T.conj()
                proj_arrowhead[phys, aux] = projector @ arrowhead[phys, aux]
                proj_arrowhead[aux, phys] = arrowhead[aux, phys] @ projector.T.conj()
                proj_arrowhead[aux, aux] = arrowhead[aux, aux]

            elif nproj == 1:

                proj_arrowhead[phys, phys] = 0.5 * (projector @ arrowhead[phys, phys] + arrowhead[phys, phys] @ projector.T.conj())
                proj_arrowhead[phys, aux] = 0.5 * (projector @ arrowhead[phys, aux] + arrowhead[phys, aux])
                proj_arrowhead[aux, phys] = 0.5 * (arrowhead[aux, phys] @ projector.T.conj() + arrowhead[aux, phys])
                proj_arrowhead[aux, aux] = arrowhead[aux, aux]

            else:
                raise ValueError("Invalid projection type nproj=%d"%nproj)

            proj_spectrals.append(Spectral.from_matrix(proj_arrowhead, chempot=self.spectrals[s].chempot, hermitian=self.spectrals[s].hermitian))
        return type(self)(proj_spectrals)

          
    def rotate(self, rotation):
        return self.project(rotation, nproj=2)
    
    def combine(self, *args, greens_function=True):
        """Combine multiple spectral representations.

        Parameters
        ----------
        *args : SpectralRep
            Spectral representations to combine.

        Returns
        -------
        SpectralRep
            Combined spectral representation.
        """
        combined_spectrals = []
        for s in range(self.nsectors):
            to_combine = [self.spectrals[s]]
            for arg in args:
                to_combine.append(arg.spectrals[s])
            if greens_function:
                combined_spectral = Spectral.combine_for_greens_function(*to_combine)
            else:
                combined_spectral = Spectral.combine_for_self_energy(*to_combine)
            combined_spectrals.append(combined_spectral)
        
        return type(self)(combined_spectrals)
