import numpy as np
import abc
import logging
import pyscf
from vayesta.core.util import einsum, dot

log = logging.getLogger(__name__)


class MeanField(abc.ABC):
    """ 
    Abstract base class for mean-field objects. 
    Currently contains an incomplete list of methods and attributes required to 
    interface with the rest of Vayesta, and to be implemented by concrete 
    mean-field classes.
    """

    @property
    def mo_coeff(self) -> np.ndarray:
        """MO coefficients. For UHF, this is a tuple of alpha and beta coefficients. """
        return self._mo_coeff

    @property
    def mo_occ(self) -> np.ndarray:
        """MO occupations. For UHF, this is a tuple of alpha and beta occupations. """
        return self._mo_occ

    @property
    def mo_energy(self) -> np.ndarray:
        """MO energies. For UHF, this is a tuple of alpha and beta energies. """
        return self._mo_energy

    @property
    def mo_energy_occ(self):
        """Energies of occupied MOs"""
        pass

    @property
    def mo_energy_vir(self):
        """Energies of virtual MOs"""
        pass

    @property
    def nao(self) -> int:
        """Number of atomic orbitals or number of computational basis functions"""
        return self._nao
    
    @property
    def pbc_dimension(self):
        """Dimension of periodicity (0 for molecules, 1 for polymers, 2 for surfaces, 3 for solids)."""
        return self._dim
    
    @property
    def e_tot(self):
        """Total mean-field energy."""
        return self.ncells * self._e_tot

    @e_tot.setter
    def e_tot(self, value):
        self._e_tot = value / self.ncells

    @property
    def ncells(self):
        """Number of primitive cells within supercell."""
        if self.kpts is None:
            return 1
        return len(self.kpts)
    
    @property
    @abc.abstractmethod
    def mo_coeff_occ(self):
        """Occupied MO coefficients"""
        pass

    @property
    @abc.abstractmethod
    def mo_coeff_vir(self):
        """Virtual MO coefficients"""
        pass
        
    @abc.abstractmethod
    def get_ovlp(self):
        """AO-overlap matrix."""
        pass

    @abc.abstractmethod
    def get_ovlp_power(self, power):
        """Get power of AO overlap matrix.

        For folded calculations, this uses the k-point sampled overlap, for better performance and accuracy.

        Parameters
        ----------
        power : float
            Matrix power.

        Returns
        -------
        spow : (n(AO), n(AO)) array
            Matrix power of AO overlap matrix
        """
        pass

    @abc.abstractmethod
    def get_hcore(self):
        """Core Hamiltonian (kinetic energy plus nuclear-electron attraction)."""
        pass 

    @abc.abstractmethod
    def get_veff(self, dm=None, with_exxdiv=True):
        """Hartree-Fock Coulomb and exchange potential in AO basis."""
        pass

    @abc.abstractmethod
    def get_exxdiv(self):
        """Get divergent exact-exchange (exxdiv) energy correction and potential."""
        pass

    @abc.abstractmethod
    def get_fock(self, dm=None, with_exxdiv=True):
        """Fock matrix in AO basis."""
        pass

    @abc.abstractmethod
    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        """Make 1-particle density matrix in AO basis.
        
        Parameters
        ----------
        mo_coeff (optional): (n(AO), n(MO)) array or tuple of such arrays
            MO coefficients. 
        mo_occ (optional): (n(MO),) array or tuple of such arrays
            MO occupations.

        Returns
        -------
        dm1 : (nspin, n(AO), n(AO)) array
            1-particle density matrix in AO basis.
        """
        pass

    @abc.abstractmethod
    def update_mf(self, mo_coeff, mo_energy=None, veff=None):
        """Update mean-field object with current MO coefficients, occupations, and energies.
        
        Parameters
        ----------
        mo_coeff: (n(AO), n(MO)) array or tuple of such arrays
            MO coefficients.
        mo_energy: (n(MO),) array or tuple of such arrays, optional
            MO energies. If not provided, the diagonal of the Fock matrix is used.
        veff: (n(AO), n(AO)) array or tuple of such arrays, optional
            Effective potential in AO basis. If not provided, it is recomputed using the new density matrix.
        """
        pass
    
class RHF_MeanField(MeanField):
    """RHF mean-field object."""
    pass

class UHF_MeanField(MeanField):
    """UHF mean-field object."""
    pass

class PySCF_MeanField(MeanField):

    """Base class for mean-field objects based on PySCF mean-field objects. 
    This forwards some methods and attributes from the underlying PySCF 
    mean-field object, and implements the rest using the underlying 
    PySCF mean-field object."""

    # Methods to be forwarded from underlying PySCF mean-field object - dependency to be removed in future refactor
    _from_pyscf = ['converged', 'energy_nuc', 'energy_tot', 'max_memory', 'with_df', '_eri', 'eig', 'conv_tol', 'conv_tol_grad', 'conv_tol_elec', 'verbose', 'stdout']

    def __getattr__(self, name):
        if name in self._from_pyscf:
            return getattr(self._mf, name)
        else:
            raise AttributeError("Attribute %r not found in %s or underlying mean-field object!" % (name, type(self).__name__))


    def __init__(self, mf):
        self._mf = mf
        self.mol = mf.mol

        self._mo_coeff = mf.mo_coeff.copy()
        self._mo_occ = mf.mo_occ.copy()
        self._mo_energy = mf.mo_energy.copy()
        self._e_tot = mf.e_tot
        self._nao = mf.mol.nao_nr()
        self._ovlp = mf.get_ovlp()
        self._hcore = mf.get_hcore()
        self._veff = mf.get_veff()

        self._dim = getattr(self.mol, "dimension", 0)
        self.kpts = None
        self.exxdiv = mf.get_exxdiv() if hasattr(mf, "get_exxdiv") else None
        self.has_exxdiv = hasattr(mf, "exxdiv") and mf.exxdiv is not None
        if self.has_exxdiv:
            self.madelung = pyscf.pbc.tools.madelung(self.mol, self.mf.kpt)

        
    @property
    def mf(self):
        """Underlying PySCF mean-field object."""
        return self._mf
    
    @property
    def nao(self):
        return self._mf.mol.nao_nr()

    def get_ovlp(self):
        return self._ovlp

    def get_hcore(self):
        return self._hcore

    def get_veff(self, dm=None, with_exxdiv=True):
        if not with_exxdiv and self.has_exxdiv:
            v_exxdiv = self.get_exxdiv()[1]
            return self.get_veff(dm=dm) - v_exxdiv
        if dm is None:
            return self._veff
        return self.mf.get_veff(dm=dm)

    def get_fock(self, dm=None, with_exxdiv=True):
        return self.get_hcore() + self.get_veff(dm=dm, with_exxdiv=with_exxdiv)

    def get_ovlp_power(self, power):
        
        if power == 0:
            return np.eye(self.nao)
        elif power == 1:
            return self.get_ovlp()
        else:
            e, v = np.linalg.eigh(self.get_ovlp())
            return np.dot(v * (e**power), v.T.conj())
    
    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        dm1 = self._mf.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)
        #dm1 = np.einsum("...ap,...o,...bp->ab", mo_coeff, mo_occ, mo_coeff.conj())
        return dm1
    



class PySCF_RHF(PySCF_MeanField, RHF_MeanField):

    @property
    def nmo(self) -> int:
        return self.mo_coeff.shape[1]

    @property
    def nocc(self) -> int:
        return np.count_nonzero(self.mo_occ > 0)

    @property
    def nvir(self) -> int:
        return np.count_nonzero(self.mo_occ == 0)

    @property
    def nelectron(self) -> int:
        return 2 * self.nocc

    @property
    def mo_energy_occ(self):
        return self.mo_energy[: self.nocc]
    
    @property
    def mo_energy_vir(self):
        return self.mo_energy[self.nocc :]

    @property
    def mo_coeff_occ(self):
        return self.mo_coeff[:, : self.nocc]

    @property
    def mo_coeff_vir(self):
        return self.mo_coeff[:, self.nocc :]

    def get_exxdiv(self):
        """Get divergent exact-exchange (exxdiv) energy correction and potential.

        Returns
        -------
        e_exxdiv: float
            Divergent exact-exchange energy correction per unit cell.
        v_exxdiv: array
            Divergent exact-exchange potential correction in AO basis.
        """
        if not self.has_exxdiv:
            return 0, None
        sc = np.dot(self.get_ovlp(), self.mo_coeff[:, : self.nocc])
        e_exxdiv = -self.madelung * self.nocc
        v_exxdiv = -self.madelung * np.dot(sc, sc.T)
        #self.log.debugv("Divergent exact-exchange (exxdiv) correction= %+16.8f Ha", e_exxdiv)
        return e_exxdiv / self.ncells, v_exxdiv


    def update_mf(self, mo_coeff, mo_energy=None, veff=None):
        """Update underlying mean-field object."""
        # Chech orthonormal MOs
        if not np.allclose(dot(mo_coeff.T, self.get_ovlp(), mo_coeff) - np.eye(mo_coeff.shape[-1]), 0):
            raise ValueError("MO coefficients not orthonormal!")
        self._mo_coeff = mo_coeff

        dm = self.mf.make_rdm1(mo_coeff=mo_coeff)
        if veff is None:
            veff = self.get_veff(dm=dm)
        self._veff = veff
        if mo_energy is None:
            # Use diagonal of Fock matrix as MO energies
            mo_energy = einsum("ai,ab,bi->i", mo_coeff, self.get_fock(), mo_coeff)
        self._mo_energy = mo_energy
        self.e_tot = self.mf.energy_tot(dm=dm, h1e=self.get_hcore(), vhf=veff)

        norm_mo = np.linalg.norm(mo_coeff)
        norm_moe = np.linalg.norm(mo_energy)
        norm_veff = np.linalg.norm(veff)

        from pyscf.lib.misc import finger
        fpocc = finger(self._mf.mo_occ)
        print("Update norms: E=%.8f, C=%.8e, e=%.8e, V=%.6e  fpo=%f" % (self.e_tot, norm_mo, norm_moe, norm_veff, fpocc))
        

class PySCF_UHF(PySCF_MeanField, UHF_MeanField):

    @property
    def nmo(self) -> int:
        return self.mo_coeff[0].shape[1], self.mo_coeff[1].shape[1]
    @property
    def nocc(self) -> int:
        return np.count_nonzero(self.mo_occ[0] > 0), np.count_nonzero(self.mo_occ[1] > 0)

    @property
    def nvir(self) -> int:
        return np.count_nonzero(self.mo_occ[0] == 0), np.count_nonzero(self.mo_occ[1] == 0)

    @property
    def nelectron(self) -> int:
        return  sum(self.nocc)
    
    @property
    def mo_coeff_occ(self):
        return self.mo_coeff[0][:, : self.nocc[0]], self.mo_coeff[1][:, : self.nocc[1]]
    
    @property
    def mo_coeff_vir(self):
        return self.mo_coeff[0][:, self.nocc[0] :], self.mo_coeff[1][:, self.nocc[1] :]

    @property
    def mo_coeff_occ(self):
        return self.mo_coeff[0][:, : self.nocc[0]], self.mo_coeff[1][:, : self.nocc[1]]

    @property
    def mo_coeff_vir(self):
        return self.mo_coeff[0][:, self.nocc[0] :], self.mo_coeff[1][:, self.nocc[1] :]
    

    def get_exxdiv(self):
        """Get divergent exact-exchange (exxdiv) energy correction and potential.

        Returns
        -------
        e_exxdiv: float
            Divergent exact-exchange energy correction per unit cell.
        v_exxdiv: array
            Divergent exact-exchange potential correction in AO basis.
        """
        if not self.has_exxdiv:
            return 0, None
        ovlp = self.get_ovlp()
        sca = np.dot(ovlp, self.mo_coeff[0][:, : self.nocc[0]])
        scb = np.dot(ovlp, self.mo_coeff[1][:, : self.nocc[1]])
        nocc = (self.nocc[0] + self.nocc[1]) / 2
        e_exxdiv = -self.madelung * nocc / self.ncells
        v_exxdiv_a = -self.madelung * np.dot(sca, sca.T)
        v_exxdiv_b = -self.madelung * np.dot(scb, scb.T)
        #self.log.debug("Divergent exact-exchange (exxdiv) correction= %+16.8f Ha", e_exxdiv)
        return e_exxdiv, (v_exxdiv_a, v_exxdiv_b)
    
    def update_mf(self, mo_coeff, mo_energy=None, veff=None):
        """Update underlying mean-field object."""
        # Chech orthonormal MOs
        if not (
            np.allclose(dot(mo_coeff[0].T, self.get_ovlp(), mo_coeff[0]) - np.eye(mo_coeff[0].shape[-1]), 0)
            and np.allclose(dot(mo_coeff[1].T, self.get_ovlp(), mo_coeff[1]) - np.eye(mo_coeff[1].shape[-1]), 0)
        ):
            raise ValueError("MO coefficients not orthonormal!")
        self._mo_coeff = mo_coeff
        dm = self.mf.make_rdm1(mo_coeff=mo_coeff)
        if veff is None:
            veff = self.mf.get_veff(dm=dm)
        self._veff = veff
        if mo_energy is None:
            # Use diagonal of Fock matrix as MO energies
            fock = self.get_fock()
            mo_energy = (
                einsum("ai,ab,bi->i", mo_coeff[0], fock[0], mo_coeff[0]),
                einsum("ai,ab,bi->i", mo_coeff[1], fock[1], mo_coeff[1]),
            )
        self._mo_energy = mo_energy
        self.e_tot = self.mf.energy_tot(dm=dm, h1e=self.get_hcore(), vhf=veff)

        






