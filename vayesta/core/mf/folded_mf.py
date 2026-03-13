import logging
import copy
import tempfile

import numpy as np
import scipy
import scipy.linalg

import pyscf
from pyscf import lib
from pyscf.pbc import tools
import pyscf.pbc.df

from vayesta.core.util import ImaginaryPartError, OrthonormalityError, dot, einsum

from .mf import PySCF_MeanField, PySCF_RHF, PySCF_UHF


log = logging.getLogger(__name__)

class Folded_PySCF_MeanField(PySCF_MeanField):
    """Fold k-point sampled SCF calculation to the BVK (Born-von Karman) supercell.

    This class automatically updates the attributes `mo_energy`, `mo_coeff`, `mo_occ`, `e_tot`, and `converged`.
    It also overwrites the methods `get_ovlp`, `get_hcore`, and `get_veff`,
    calling its more efficient k-space variant first and folding the result to the supercell.

    Since `get_hcore` and `get_veff` are implemented, `get_fock` is supported automatically,
    if the inherited base SCF class implements it.

    Attributes
    ----------
    kmf: pyscf.pbc.gto.KRHF or pyscf.pbc.gto.KRHF
        Converged k-point sampled mean-field calculation.
    kcell: pyscf.pbc.gto.Cell
        Primitive unit cell object.
    ncells: int
        Number of primitive unit cells within BVK supercell
    kphase: (ncells, ncells) array
        Transformation matrix between k-point and BVK quantities.
    """
    def __init__(self, mf):

        self._mf = mf
        self.cell = mf.cell
        self.kpts = mf.kpts

        # Only Gamma point supported for now
        self.kpt = np.array([0.0, 0.0, 0.0])

        self.subcellmesh = kpts_to_kmesh(self.mf.cell, mf.kpts)
        self.mol, self.kphase = get_phase(self.kcell, self.mf.kpts)
        
        self._dim = getattr(self.mol, "dimension", 0)
        self.exxdiv = mf.get_exxdiv() if hasattr(mf, "get_exxdiv") else None
        self.has_exxdiv = hasattr(mf, "exxdiv") and mf.exxdiv is not None
        if self.has_exxdiv:
            self.madelung = pyscf.pbc.tools.madelung(self.mol, self.kpt)


    @property
    def nao(self):
        return self.mf.mol.nao_nr() * np.prod(self.subcellmesh)

    @property
    def nkpts(self):
        return self.kpts.shape[0]

    @property
    def nkmo(self):
        return self.kmo_coeff.shape[-1]

    @property
    def nkao(self):
        return self.mf.mol.nao_nr()
        
    @property
    def e_tot(self):
        return self.ncells * self.mf.e_tot

    @e_tot.setter
    def e_tot(self, value):
        self.mf.e_tot = value / self.ncells

    @property
    def ncells(self):
        return len(self.mf.kpts)

    @property
    def kcell(self):
        return self.mf.mol

    @property
    def _eri(self):
        return None

    @property
    def kmo_coeff(self):
        return np.array(self.mf.mo_coeff)
    
    @property
    def kmo_energy(self):
        return np.array(self.mf.mo_energy)

    @property
    def kmo_occ(self):
        return np.array(self.mf.mo_occ)
    
    def get_ovlp(self, *args, **kwargs):
        sk = self.mf.get_ovlp(*args, **kwargs)
        ovlp = k2bvk_2d(sk, self.kphase)
        return ovlp

    def get_kovlp(self, *args, **kwargs):
        return self.mf.get_ovlp(*args, **kwargs)

    def get_ovlp_power(self, power):
        if power == 0:
            return np.eye(self.nao)
        elif power == 1:
            return self.get_ovlp()
        # For folded calculations, use k-point sampled overlap for better performance and accuracy
        sk = self.kcell.pbc_intor("int1e_ovlp", hermi=1, kpts=self.kpts, pbcopt=pyscf.lib.c_null_ptr())
        ek, vk = np.linalg.eigh(sk)
        spowk = einsum("kai,ki,kbi->kab", vk, ek**power, vk.conj())
        spow = pyscf.pbc.tools.k2gamma.to_supercell_ao_integrals(self.kcell, self.kpts, spowk)
        return spow

    def get_hcore(self, *args, make_real=True, **kwargs):
        hk = self.mf.get_hcore(*args, **kwargs)
        hcore = k2bvk_2d(hk, self.kphase, make_real=make_real)
        return hcore

    def get_khcore(self, *args, **kwargs):
        return self.mf.get_hcore(*args, **kwargs)

    def get_veff(self, mol=None, dm=None, *args, make_real=True, with_exxdiv=True, **kwargs):
        assert mol is None or mol is self.mol
        # Unfold DM into k-space
        if dm is not None:
            dm = bvk2k_2d(dm, self.kphase)
        vk = self.mf.get_veff(self.mf.mol, dm, *args, **kwargs)
        veff = k2bvk_2d(vk, self.kphase, make_real=make_real)

        if not with_exxdiv and self.has_exxdiv:
            v_exxdiv = self.get_exxdiv()[1]
            return veff - v_exxdiv
        else:
            return veff

    def get_kveff(self, mol=None, dm=None, *args, with_exxdiv=True, **kwargs):
        if dm is not None and dm.ndim == 2:
            dm = bvk2k_2d(dm, self.kphase)
        veff = self.mf.get_veff(self.mf.mol, dm, *args, **kwargs)

        if not with_exxdiv and self.has_exxdiv:
            raise NotImplementedError("get_kveff with with_exxdiv=False is not implemented yet")
            v_exxdiv = self.get_exxdiv()[1]
            kv_exxdiv = bvk2k_2d(v_exxdiv, self.kphase)
            return veff - kv_exxdiv
        else:
            return veff

        
    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        """Make 1-particle density matrix in AO basis."""
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        dm1 = self._dummy_mf.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ)
        #dm1 = np.einsum("...ap,...o,...bp->ab", mo_coeff, mo_occ, mo_coeff.conj())
        return dm1

    #def make_krdm1(self, mo_coeff=None, mo_occ=None):

    def energy_tot(self, *args, **kwargs):
        self._dummy_mf.mo_coeff = self.mo_coeff
        self._dummy_mf.mo_occ = self.mo_occ
        self._dummy_mf.mo_energy = self.mo_energy
        return self._dummy_mf.energy_tot(*args, **kwargs)


    def orbital_ao_to_kao(self, coeff_ao):
        """Transform supercell AO coefficients to k-AO basis.

        Parameters
        ----------
        coeff_ao : (nAO, nOrb) array
            Orbital coefficients in supercell AO basis.

        Returns
        -------
        coeff_kao : (nk, nkAO, nOrb) array
            Orbital coefficients in k-AO basis.
        """
        pass
    
    def one_body_ao_to_kao(self, obs):
        return bvk2k_2d(obs, self.kphase)

    def one_body_kao_to_ao(self, obs):
        return k2bvk_2d(obs, self.kphase)



class Folded_PySCF_RHF(Folded_PySCF_MeanField, PySCF_RHF):
    __doc__ = Folded_PySCF_MeanField.__doc__

    def __init__(self, kmf, *args, **kwargs):
        super().__init__(kmf, *args, **kwargs)
        ovlp = self.get_ovlp()
        self._mo_energy, self._mo_coeff, self._mo_occ = fold_mos(
            self.mf.mo_energy, self.mf.mo_coeff, self.mf.mo_occ, self.kphase, ovlp
        )
        self._dummy_mf = pyscf.scf.rhf.RHF(self.mol)
        assert np.all(self.mo_coeff.imag == 0)

    def orbital_ao_to_kao(self, coeff_ao):
        return unfold_orbitals(coeff_ao, self.kphase)

    def update_mf(self, mo_coeff, mo_energy=None, veff=None):
        raise NotImplementedError("update_mf is not implemented for Folded_PySCF_RHF")


class Folded_PySCF_UHF(Folded_PySCF_MeanField, PySCF_UHF):
    __doc__ = Folded_PySCF_MeanField.__doc__

    def __init__(self, kmf, *args, **kwargs):
        super().__init__(kmf, *args, **kwargs)

        ovlp = self.get_ovlp()
        self._mo_energy, self._mo_coeff, self._mo_occ = zip(
            fold_mos(self.mf.mo_energy[0], self.mf.mo_coeff[0], self.mf.mo_occ[0], self.kphase, ovlp),
            fold_mos(self.mf.mo_energy[1], self.mf.mo_coeff[1], self.mf.mo_occ[1], self.kphase, ovlp),
        )

        self._dummy_mf = pyscf.scf.uhf.UHF(self.mol)
        assert np.all(self.mo_coeff[0].imag == 0)
        assert np.all(self.mo_coeff[1].imag == 0)

    def orbital_ao_to_kao(self, coeff_ao):
        return tuple(unfold_orbitals(coeff_ao[i], self.kphase) for i in range(2))

    def update_mf(self, mo_coeff, mo_energy=None, veff=None):
        raise NotImplementedError("update_mf is not implemented for Folded_PySCF_UHF")

def unfold_orbitals(coeff_ao, kphase):
    nk = kphase.shape[0]
    nao = coeff_ao.shape[0]
    nkao = nao // nk
    norb = coeff_ao.shape[-1]
    # Reshape supercell AO -> (nk, nkAO, nOrb) via the unit cell index
    coeff_cell = coeff_ao.reshape(nk, nkao, norb)
    coeff_kao = einsum('kR,...Rmp->...kmp', kphase.conj(), coeff_cell)
    return coeff_kao


def fold_mos(kmo_energy, kmo_coeff, kmo_occ, kphase, ovlp, make_real=True, sort=True):
    # --- MO energy and occupations
    mo_energy = np.hstack(kmo_energy)
    mo_occ = np.hstack(kmo_occ)
    # --- MO coefficients
    # Number of MOs per k-point (can be k-point depedent, for example due to linear-dependency treatment)
    mo_coeff = []
    for k, ck in enumerate(kmo_coeff):
        cr = np.multiply.outer(kphase[k], ck)  # R,ai -> Rai
        mo_coeff.append(cr.reshape(cr.shape[0] * cr.shape[1], cr.shape[2]))  # Rai  -> (Ra),i
    mo_coeff = np.hstack(mo_coeff)
    # --- Sort MOs according to energy
    if sort:
        reorder = np.argsort(mo_energy)
        mo_energy = mo_energy[reorder]
        mo_coeff = mo_coeff[:, reorder]
        mo_occ = mo_occ[reorder]
    # --- Make MOs real
    if make_real:
        mo_energy, mo_coeff = make_mo_coeff_real(mo_energy, mo_coeff, ovlp)
    # Check orthonormality of folded MOs
    err = abs(dot(mo_coeff.T.conj(), ovlp, mo_coeff) - np.eye(mo_coeff.shape[-1])).max()
    if err > 1e-4:
        log.critical("Supercell MOs are not orthonormal (max error= %.3e)", err)
        raise OrthonormalityError("Supercell MOs are not orthonormal")
    else:
        if err > 1e-6:
            logf = log.error
        elif err > 1e-8:
            logf = log.warning
        else:
            logf = log.debugv
        logf("Supercell MO orthonormality error: L(inf)= %.3e", err)

    return mo_energy, mo_coeff, mo_occ


def log_error_norms(msg, err, error_tol=1e-3, warn_tol=1e-6):
    l2 = np.linalg.norm(err)
    linf = abs(err).max()
    lmax = max(l2, linf)
    if lmax > error_tol:
        log.error(msg + " !!!", l2, linf)
    elif lmax > warn_tol:
        log.warning(msg + " !", l2, linf)
    else:
        log.debug(msg, l2, linf)


def make_mo_coeff_real(mo_energy, mo_coeff, ovlp, imag_tol=1e-10):
    mo_coeff = mo_coeff.copy()
    # Check orthonormality
    ortherr = abs(dot(mo_coeff.T.conj(), ovlp, mo_coeff) - np.eye(mo_coeff.shape[-1])).max()
    log.debugv("Orthonormality error before make_mo_coeff_real: %.2e", ortherr)

    # Testing
    im = np.linalg.norm(mo_coeff.imag, axis=0) > imag_tol
    log.debugv("%d complex MOs found. L(2)= %.2e", np.count_nonzero(im), np.linalg.norm(mo_coeff.imag))
    if not np.any(im):
        return mo_energy, mo_coeff.real
    shift = 1.0 - min(mo_energy[im])
    sc = np.dot(ovlp, mo_coeff[:, im])
    fock = np.dot(sc * (mo_energy[im] + shift), sc.T.conj())
    log_error_norms("Imaginary part in folded Fock matrix: L(2)= %.2e L(inf)= %.2e", fock.imag)
    # Diagonalize subspace Fock matrix
    # TODO: eigensolver for linear dependencies...
    eigh = scipy.linalg.eigh
    # Modified PySCF:
    # eigh = cell.eigh_factory(lindep_threshold=1e-13, fallback_mode=True)
    e, v = eigh(fock.real, ovlp)
    # Extract MOs from rank-deficient Fock matrix
    mask = e > 0.5
    assert np.count_nonzero(mask) == len(mo_energy[im])
    e, v = e[mask], v[:, mask]
    log_error_norms("Error in folded MO energies: L(2)= %.2e L(inf)= %.2e", mo_energy[im] - (e - shift))
    mo_coeff[:, im] = v

    assert np.all(np.linalg.norm(mo_coeff.imag, axis=0) <= imag_tol)
    return mo_energy, mo_coeff.real


def kpts_to_kmesh(cell, kpts):
    """Guess k-mesh from k-points."""
    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    kmesh = [len(np.unique(scaled_k[:, d])) for d in range(3)]
    return kmesh


def translation_vectors_for_kmesh(cell, kmesh):
    """Translation vectors to construct super-cell of which the gamma point is
    identical to the k-point mesh of primitive cell"""
    latt_vec = cell.lattice_vectors()
    r_rel = [np.arange(kmesh[d]) for d in range(3)]
    r_vec_rel = lib.cartesian_prod(r_rel)
    r_vec_abs = np.dot(r_vec_rel, latt_vec)
    return r_vec_abs


def get_phase(cell, kpts, kmesh=None):
    """The unitary transformation that transforms the supercell basis k-mesh
    adapted basis.

    Important: This is ordered as (k,R), different to PySCF k2gamma.get_phase!
    """
    if kmesh is None:
        kmesh = kpts_to_kmesh(cell, kpts)
    r_vec_abs = translation_vectors_for_kmesh(cell, kmesh)
    nr = len(r_vec_abs)
    phase = np.exp(1j * np.dot(kpts, r_vec_abs.T)) / np.sqrt(nr)
    scell = tools.super_cell(cell, kmesh)
    return scell, phase


def k2bvk_2d(ak, phase, make_real=True, imag_tol=1e-6):
    """Transform unit-cell k-point AO integrals to the supercell gamma-point AO integrals."""
    ag = einsum("kR,...kij,kS->...RiSj", phase, ak, phase.conj())
    imag_norm = abs(ag.imag).max()
    if make_real and (imag_norm > imag_tol):
        msg = "Imaginary part of supercell integrals: %.2e (tolerance= %.2e)"
        log.fatal(msg, imag_norm, imag_tol)
        raise ImaginaryPartError(msg % (imag_norm, imag_tol))
    nr, nao = phase.shape[1], ak.shape[-1]
    shape = (*ag.shape[:-4], nr * nao, nr * nao)
    ag = ag.reshape(shape)
    if make_real:
        return ag.real
    return ag


def bvk2k_2d(ag, phase):
    """Transform supercell gamma-point AO integrals to the unit-cell k-point AO integrals."""
    nr, nao = phase.shape[1], ag.shape[-1] // phase.shape[1]
    shape = (*ag.shape[:-2], nr, nao, nr, nao)
    ag = ag.reshape(shape)
    ak = einsum("kR,...RiSj,kS->...kij", phase.conj(), ag, phase)
    return ak