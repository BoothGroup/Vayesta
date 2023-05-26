import logging
from timeit import default_timer as timer
import copy
import tempfile

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph

import pyscf
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc import scf
import pyscf.pbc.df

try:
    from .util import *
# If run as script:
except ImportError:
    from util import *

log = logging.getLogger(__name__)

def fold_scf(kmf, *args, **kwargs):
    """Fold k-point sampled mean-field object to Born-von Karman (BVK) supercell.
    See also :class:`FoldedSCF`."""
    if isinstance(kmf, pyscf.pbc.scf.khf.KRHF):
        return FoldedRHF(kmf, *args, **kwargs)
    if isinstance(kmf, pyscf.pbc.scf.kuhf.KUHF):
        return FoldedUHF(kmf, *args, **kwargs)
    raise NotImplementedError("Mean-field type= %r" % kmf)

class FoldedSCF:
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

    # Propagate the following attributes to the k-point mean-field:
    _from_kmf = ['converged', 'exxdiv', 'verbose', 'max_memory', 'conv_tol', 'conv_tol_grad',
            'stdout']

    def __init__(self, kmf, kpt=np.zeros(3), **kwargs):
        # Create a copy, so that the original mean-field object does not get modified
        kmf = copy.copy(kmf)
        # Support for k-point symmetry:
        if hasattr(kmf, 'to_khf'):
            kmf = kmf.to_khf()
        self.kmf = kmf
        self.subcellmesh = kpts_to_kmesh(self.kmf.cell, kmf.kpts)
        cell, self.kphase = get_phase(self.kcell, self.kmf.kpts)
        # We cannot call the PySCF __init__....
        #super().__init__(scell, **kwargs)
        # ... so we have to intialize a few attributes here:
        self.mol = self.cell = cell

        # From scf/hf.py:
        self.callback = None
        self.scf_summary = {}
        self._chkfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        self.chkfile = self._chkfile.name

        # from pbc/scf/hf.py:
        self.with_df = pyscf.pbc.df.FFTDF(cell)
        self.rsjk = None
        self.kpt = kpt
        if not np.allclose(kpt, 0):
            raise NotImplementedError()

    def __getattr__(self, name):
        if name in self._from_kmf:
            return getattr(self.kmf, name)
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    def __setattr__(self, name, value):
        if name in self._from_kmf:
            return setattr(self.kmf, name, value)
        return super().__setattr__(name, value)

    @property
    def e_tot(self):
        return (self.ncells * self.kmf.e_tot)

    @e_tot.setter
    def e_tot(self, value):
        self.kmf.e_tot = (value / self.ncells)

    @property
    def ncells(self):
        return len(self.kmf.kpts)

    @property
    def kcell(self):
        return self.kmf.mol

    @property
    def _eri(self):
        return None

    def get_ovlp(self, *args, **kwargs):
        sk = self.kmf.get_ovlp(*args, **kwargs)
        ovlp = k2bvk_2d(sk, self.kphase)
        return ovlp

    def get_hcore(self, *args, make_real=True, **kwargs):
        hk = self.kmf.get_hcore(*args, **kwargs)
        hcore = k2bvk_2d(hk, self.kphase, make_real=make_real)
        return hcore

    def get_veff(self, mol=None, dm=None, *args, make_real=True, **kwargs):
        assert (mol is None or mol is self.mol)
        # Unfold DM into k-space
        if dm is not None: dm = bvk2k_2d(dm, self.kphase)
        vk = self.kmf.get_veff(self.kmf.mol, dm, *args, **kwargs)
        veff = k2bvk_2d(vk, self.kphase, make_real=make_real)
        return veff

class FoldedRHF(FoldedSCF, pyscf.pbc.scf.hf.RHF):
    __doc__ = FoldedSCF.__doc__

    def __init__(self, kmf, *args, **kwargs):
        super().__init__(kmf, *args, **kwargs)
        ovlp = self.get_ovlp()
        self.mo_energy, self.mo_coeff, self.mo_occ = fold_mos(self.kmf.mo_energy, self.kmf.mo_coeff, self.kmf.mo_occ,
                                                              self.kphase, ovlp)

        assert np.all(self.mo_coeff.imag == 0)

class FoldedUHF(FoldedSCF, pyscf.pbc.scf.uhf.UHF):
    __doc__ = FoldedSCF.__doc__

    def __init__(self, kmf, *args, **kwargs):
        super().__init__(kmf, *args, **kwargs)

        ovlp = self.get_ovlp()
        self.mo_energy, self.mo_coeff, self.mo_occ = zip(
                fold_mos(self.kmf.mo_energy[0], self.kmf.mo_coeff[0], self.kmf.mo_occ[0], self.kphase, ovlp),
                fold_mos(self.kmf.mo_energy[1], self.kmf.mo_coeff[1], self.kmf.mo_occ[1], self.kphase, ovlp))
        assert np.all(self.mo_coeff[0].imag == 0)
        assert np.all(self.mo_coeff[1].imag == 0)

def fold_mos(kmo_energy, kmo_coeff, kmo_occ, kphase, ovlp, make_real=True, sort=True):
    # --- MO energy and occupations
    mo_energy = np.hstack(kmo_energy)
    mo_occ = np.hstack(kmo_occ)
    # --- MO coefficients
    # Number of MOs per k-point (can be k-point depedent, for example due to linear-dependency treatment)
    mo_coeff = []
    for k, ck in enumerate(kmo_coeff):
        cr = np.multiply.outer(kphase[k], ck)                               # R,ai -> Rai
        mo_coeff.append(cr.reshape(cr.shape[0]*cr.shape[1], cr.shape[2]))   # Rai  -> (Ra),i
    mo_coeff = np.hstack(mo_coeff)
    # --- Sort MOs according to energy
    if sort:
        reorder = np.argsort(mo_energy)
        mo_energy = mo_energy[reorder]
        mo_coeff = mo_coeff[:,reorder]
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
        log.error(msg+" !!!", l2, linf)
    elif lmax > warn_tol:
        log.warning(msg+" !", l2, linf)
    else:
        log.debug(msg, l2, linf)

def make_mo_coeff_real(mo_energy, mo_coeff, ovlp, imag_tol=1e-10):
    mo_coeff = mo_coeff.copy()
    # Check orthonormality
    ortherr = abs(dot(mo_coeff.T.conj(), ovlp, mo_coeff) - np.eye(mo_coeff.shape[-1])).max()
    log.debugv("Orthonormality error before make_mo_coeff_real: %.2e", ortherr)

    # Testing
    im = (np.linalg.norm(mo_coeff.imag, axis=0) > imag_tol)
    log.debugv("%d complex MOs found. L(2)= %.2e", np.count_nonzero(im), np.linalg.norm(mo_coeff.imag))
    if not np.any(im):
        return mo_energy, mo_coeff.real
    shift = 1.0 - min(mo_energy[im])
    sc = np.dot(ovlp, mo_coeff[:,im])
    fock = np.dot(sc*(mo_energy[im]+shift), sc.T.conj())
    log_error_norms("Imaginary part in folded Fock matrix: L(2)= %.2e L(inf)= %.2e", fock.imag)
    # Diagonalize subspace Fock matrix
    # TODO: eigensolver for linear dependencies...
    eigh = scipy.linalg.eigh
    # Modified PySCF:
    # eigh = cell.eigh_factory(lindep_threshold=1e-13, fallback_mode=True)
    e, v = eigh(fock.real, ovlp)
    # Extract MOs from rank-deficient Fock matrix
    mask = (e > 0.5)
    assert np.count_nonzero(mask) == len(mo_energy[im])
    e, v = e[mask], v[:,mask]
    log_error_norms("Error in folded MO energies: L(2)= %.2e L(inf)= %.2e", mo_energy[im]-(e-shift))
    mo_coeff[:,im] = v

    assert np.all(np.linalg.norm(mo_coeff.imag, axis=0) <= imag_tol)
    return mo_energy, mo_coeff.real

def make_mo_coeff_real_2(mo_energy, mo_coeff, mo_occ, ovlp, hcore, imag_tol=1e-8):
    mo_coeff = mo_coeff.copy()
    # Check orthonormality
    ortherr = abs(dot(mo_coeff.T.conj(), ovlp, mo_coeff) - np.eye(mo_coeff.shape[-1])).max()
    log.debugv("Orthonormality error before make_mo_coeff_real: %.2e", ortherr)

    mo_coeff_occ = mo_coeff[:,mo_occ>0]
    mo_coeff_vir = mo_coeff[:,mo_occ==0]

    e_hcore_min = scipy.linalg.eigh(hcore, b=ovlp)[0][0]
    shift = (1.0 - e_hcore_min)

    def make_subspace_real(mo_coeff_sub):
        # Diagonalize Hcore to separate symmetry sectors
        nsub = mo_coeff_sub.shape[-1]
        hsub = dot(mo_coeff_sub.T.conj(), hcore, mo_coeff_sub) + shift*np.eye(nsub)
        cs = dot(mo_coeff_sub.T.conj(), ovlp)
        hsub = dot(cs.T.conj(), hsub, cs)
        im = abs(hsub.imag).max()
        assert (im < imag_tol), ("Imaginary part of Hcore= %.3e" % im)
        e, c = scipy.linalg.eigh(hsub.real, b=ovlp)
        colspace = (e > 0.5)
        assert (np.count_nonzero(colspace) == nsub)
        mo_coeff_sub = c[:,colspace]

        # Canonicalize subspace MO coefficients
        p = dot(mo_coeff.T.conj(), ovlp, mo_coeff_sub)
        fsub = einsum('ia,i,ib->ab', p.conj(), mo_energy, p)
        im = abs(fsub.imag).max()
        assert (im < imag_tol), ("Imaginary part of Fock= %.3e" % im)
        e, r = np.linalg.eigh(fsub.real)
        mo_energy_sub = e
        mo_coeff_sub = np.dot(mo_coeff_sub, r)
        return mo_energy_sub, mo_coeff_sub

    mo_energy_occ, mo_coeff_occ = make_subspace_real(mo_coeff_occ)
    mo_energy_vir, mo_coeff_vir = make_subspace_real(mo_coeff_vir)
    mo_energy_real = np.hstack((mo_energy_occ, mo_energy_vir))
    mo_coeff_real = np.hstack((mo_coeff_occ, mo_coeff_vir))

    log_error_norms("Error in MO energies of real orbitals: L(2)= %.2e L(inf)= %.2e", (mo_energy_real-mo_energy))

    return mo_energy_real, mo_coeff_real


# ==========================
# From PySCF, modified

def kpts_to_kmesh(cell, kpts):
    """Guess k-mesh from k-points."""
    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    kmesh = [len(np.unique(scaled_k[:,d])) for d in range(3)]
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
    if kmesh is None: kmesh = kpts_to_kmesh(cell, kpts)
    r_vec_abs = translation_vectors_for_kmesh(cell, kmesh)
    nr = len(r_vec_abs)
    phase = np.exp(1j*np.dot(kpts, r_vec_abs.T)) / np.sqrt(nr)
    scell = tools.super_cell(cell, kmesh)
    return scell, phase

def k2bvk_2d(ak, phase, make_real=True, imag_tol=1e-6):
    """Transform unit-cell k-point AO integrals to the supercell gamma-point AO integrals."""
    ag = einsum('kR,...kij,kS->...RiSj', phase, ak, phase.conj())
    imag_norm = abs(ag.imag).max()
    if make_real and (imag_norm > imag_tol):
        msg = "Imaginary part of supercell integrals: %.2e (tolerance= %.2e)"
        log.fatal(msg, imag_norm, imag_tol)
        raise ImaginaryPartError(msg % (imag_norm, imag_tol))
    nr, nao = phase.shape[1], ak.shape[-1]
    shape = (*ag.shape[:-4], nr*nao, nr*nao)
    ag = ag.reshape(shape)
    if make_real:
        return ag.real
    return ag

def bvk2k_2d(ag, phase):
    """Transform supercell gamma-point AO integrals to the unit-cell k-point AO integrals."""
    nr, nao = phase.shape[1], ag.shape[-1]//phase.shape[1]
    shape = (*ag.shape[:-2], nr, nao, nr, nao)
    ag = ag.reshape(shape)
    ak = einsum('kR,...RiSj,kS->...kij', phase.conj(), ag, phase)
    return ak

# Depreciated functionality removed; rotation of mos to minimise imaginary part and conversion between kpoint and
# supercell calculations.
# Check out v1.0.0 or v1.0.1 if needed.


if __name__ == '__main__':

    import vayesta
    from pyscf.pbc import gto, scf

    log = vayesta.log

    cell = gto.Cell()
    cell.atom = '''
    H 0.0  0.0  0.0
    H 0.6  0.4  0.0
    '''

    cell.basis = 'cc-pvdz'
    cell.a = np.eye(3) * 4.0
    cell.a[2,2] = 20
    cell.unit='B'
    cell.dimension = 2
    cell.build()

    kmesh = [3, 3, 1]
    kpts = cell.make_kpts(kmesh)

    khf = scf.KRHF(cell, kpts)
    #khf = scf.KUHF(cell, kpts)
    khf.conv_tol = 1e-12
    khf = khf.density_fit(auxbasis='cc-pvdz-jkfit')
    khf.kernel()

    hf = fold_scf(khf)

    scell = pyscf.pbc.tools.super_cell(cell, kmesh)
    shf = scf.RHF(scell)
    #shf = scf.UHF(scell)
    shf.conv_tol = 1e-12
    shf = shf.density_fit(auxbasis='cc-pvdz-jkfit')
    shf.kernel()

    # Overlap matrix
    err = np.linalg.norm(hf.get_ovlp() - shf.get_ovlp())
    print("Error overlap= %.3e" % err)

    # Hcore matrix
    err = np.linalg.norm(hf.get_hcore() - shf.get_hcore())
    print("Error hcore= %.3e" % err)

    # Veff matrix
    err = np.linalg.norm(hf.get_veff() - shf.get_veff())
    print("Error veff= %.3e" % err)

    # Veff matrix for given DM
    scell, phase = get_phase(cell, kpts)
    dm = k2bvk_2d(khf.get_init_guess(), phase)
    err = np.linalg.norm(hf.get_veff(dm=dm) - shf.get_veff(dm=dm))
    print("Error veff for given DM= %.3e" % err)
