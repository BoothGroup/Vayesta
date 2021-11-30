import logging
from timeit import default_timer as timer

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph

import pyscf
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc import scf

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

    def __init__(self, kmf, *args, **kwargs):
        self.kmf = kmf
        scell, self.kphase = get_phase(self.kcell, kmf.kpts)
        super().__init__(scell, *args, **kwargs)

        # Copy attributes from k-point MF
        self.e_tot = self.ncells * self.kmf.e_tot
        self.converged = self.kmf.converged
        self.exxdiv = self.kmf.exxdiv
        self.verbose = self.kmf.verbose
        self.max_memory = self.kmf.max_memory
        self.conv_tol = self.kmf.conv_tol
        self.conv_tol_grad = self.kmf.conv_tol_grad


    @property
    def ncells(self):
        return len(self.kmf.kpts)

    @property
    def kcell(self):
        return self.kmf.mol

    def get_ovlp(self, *args, **kwargs):
        sk = self.kmf.get_ovlp(*args, **kwargs)
        ovlp = k2bvk_2d(sk, self.kphase)
        return ovlp

    def get_hcore(self, *args, **kwargs):
        hk = self.kmf.get_hcore(*args, **kwargs)
        hcore = k2bvk_2d(hk, self.kphase)
        return hcore

    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        assert (mol is None or mol is self.mol)
        # Unfold DM into k-space
        if dm is not None: dm = bvk2k_2d(dm, self.kphase)
        vk = self.kmf.get_veff(dm_kpts=dm, *args, **kwargs)
        veff = k2bvk_2d(vk, self.kphase)
        return veff

class FoldedRHF(FoldedSCF, pyscf.pbc.scf.hf.RHF):
    __doc__ = FoldedSCF.__doc__

    def __init__(self, kmf, *args, **kwargs):
        super().__init__(kmf, *args, **kwargs)
        ovlp = self.get_ovlp()
        #hcore = self.get_hcore()
        hcore = None
        self.mo_energy, self.mo_coeff, self.mo_occ = \
                fold_mos(kmf.mo_energy, kmf.mo_coeff, kmf.mo_occ, self.kphase, ovlp, hcore)

        # Test MO folding
        #nk = self.ncells
        #hk = [dot(kmf.mo_coeff[k].T, kmf.get_ovlp()[k], kmf.mo_coeff[k]) for k in range(nk)]

        #c = self.mo_coeff
        #smf = pyscf.pbc.scf.hf.RHF(self.mol)
        #h = dot(c.T.conj(), smf.get_ovlp(), c)

        #nao = self.kcell.nao
        #h2 = np.zeros_like(h)
        #for k in range(nk):
        #    s = np.s_[k*nao:(k+1)*nao]
        #    h2[s,s] = hk[k]

        #for k in range(nk):
        #    for k2 in range(nk):
        #        s1 = np.s_[k*nao:(k+1)*nao]
        #        s2 = np.s_[k2*nao:(k2+1)*nao]
        #        print(k, k2, np.linalg.norm(h[s1,s2]-h2[s1,s2]))
        #        if (k == k2):
        #            print(h[s1,s2][0,:])
        #            print(h2[s1,s2][0,:])

        ##print((h - h2)[1,:])
        #1/0

        assert np.all(self.mo_coeff.imag == 0)

class FoldedUHF(FoldedSCF, pyscf.pbc.scf.uhf.UHF):
    __doc__ = FoldedSCF.__doc__

    def __init__(self, kmf, *args, **kwargs):
        super().__init__(kmf, *args, **kwargs)
        ovlp = self.get_ovlp()
        #hcore = self.get_hcore()
        hcore = None
        self.mo_energy, self.mo_coeff, self.mo_occ = zip(
                fold_mos(kmf.mo_energy[0], kmf.mo_coeff[0], kmf.mo_occ[0], self.kphase, ovlp, hcore),
                fold_mos(kmf.mo_energy[1], kmf.mo_coeff[1], kmf.mo_occ[1], self.kphase, ovlp, hcore))
        assert np.all(self.mo_coeff[0].imag == 0)
        assert np.all(self.mo_coeff[1].imag == 0)

#def fold_mos(kmf, kmo_energy, kmo_coeff, kmo_occ, kphase, ovlp, make_real=True):
#def fold_mos(kmo_energy, kmo_coeff, kmo_occ, kphase, ovlp, make_real=False, sort=False):
def fold_mos(kmo_energy, kmo_coeff, kmo_occ, kphase, ovlp, hcore, make_real=True, sort=True):
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
        #mo_energy, mo_coeff = make_mo_coeff_real_2(mo_energy, mo_coeff, mo_occ, ovlp, hcore)
    # Check orthonormality of folded MOs
    err = abs(dot(mo_coeff.T.conj(), ovlp, mo_coeff) - np.eye(mo_coeff.shape[-1])).max()
    if err > 1e-7:
        raise RuntimeError("Folded MOs not orthonormal! L(inf)= %.3e" % err)
    else:
        log.debugv("Folded MOs orthonormality error: L(inf)= %.3e", err)

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
    sc = np.dot(ovlp, mo_coeff)
    im = (np.linalg.norm(mo_coeff.imag, axis=0) > imag_tol)
    #im = (np.linalg.norm(mo_coeff.imag, axis=0) > -1.0)
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
        raise ImaginaryPartError("Imaginary part of supercell integrals: %.2e (tolerance= %.2e)" % (imag_norm, imag_tol))
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


#def rotate_mo_to_real(cell, mo_energy, mo_coeff, degen_tol=1e-3, rotate_degen=True):
#    """Applies a phase factor to each MO, minimizing the maximum imaginary element.
#
#    Typically, this should reduce the imaginary part of a non-degenerate, Gamma point orbital to zero.
#    However, for degenerate subspaces, addition treatment is required.
#    """
#
#    # Output orbitals
#    mo_coeff_out = mo_coeff.copy()
#
#    for mo_idx, mo_e in enumerate(mo_energy):
#        # Check if MO is degnerate
#        if mo_idx == 0:
#            degen = (abs(mo_e - mo_energy[mo_idx+1]) < degen_tol)
#        elif mo_idx == (len(mo_energy)-1):
#            degen = (abs(mo_e - mo_energy[mo_idx-1]) < degen_tol)
#        else:
#            degen = (abs(mo_e - mo_energy[mo_idx-1]) < degen_tol) or (abs(mo_e - mo_energy[mo_idx+1]) < degen_tol)
#        if degen and not rotate_degen:
#            continue
#
#        mo_c = mo_coeff[:,mo_idx]
#        norm_in = np.linalg.norm(mo_c.imag)
#        # Find phase which makes the largest element of |C| real
#        maxidx = np.argmax(abs(mo_c.imag))
#        maxval = mo_c[maxidx]
#        # Determine -phase of maxval and rotate to real axis
#        phase = -np.angle(maxval)
#        mo_c2 = mo_c*np.exp(1j*phase)
#
#        # Only perform rotation if imaginary norm is decreased
#        norm_out = np.linalg.norm(mo_c2.imag)
#        if (norm_out < norm_in):
#            mo_coeff_out[:,mo_idx] = mo_c2
#        else:
#            norm_out = norm_in
#        if norm_out > 1e-8 and not degen:
#            logger.warn(cell, "Non-degenerate MO %4d at E= %+12.8f Ha: ||Im(C)||= %6.2e !", mo_idx, mo_e, norm_out)
#
#    return mo_coeff_out
#
#def mo_k2gamma(cell, mo_energy, mo_coeff, kpts, kmesh=None, degen_tol=1e-3, imag_tol=1e-9):
#    logger.debug(cell, "Starting mo_k2gamma")
#    scell, phase = get_phase(cell, kpts, kmesh)
#
#    # Supercell Gamma-point MO energies
#    e_gamma = np.hstack(mo_energy)
#    # The number of MOs may be k-point dependent (eg. due to linear dependency)
#    nmo_k = np.asarray([ck.shape[-1] for ck in mo_coeff])
#    nk = len(mo_coeff)
#    nao = mo_coeff[0].shape[0]
#    nr = phase.shape[0]
#    # Transform mo_coeff from k-points to supercell Gamma-point:
#    c_gamma = []
#    for k in range(nk):
#        c_k = np.einsum('R,um->Rum', phase[:,k], mo_coeff[k])
#        c_k = c_k.reshape(nr*nao, nmo_k[k])
#        c_gamma.append(c_k)
#    c_gamma = np.hstack(c_gamma)
#    assert c_gamma.shape == (nr*nao, sum(nmo_k))
#    # Sort according to MO energy
#    sort = np.argsort(e_gamma)
#    e_gamma, c_gamma = e_gamma[sort], c_gamma[:,sort]
#    # Determine overlap by unfolding for better accuracy
#    s_k = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts, pbcopt=lib.c_null_ptr())
#    s_gamma = to_supercell_ao_integrals(cell, kpts, s_k)
#    # Orthogonality error of unfolded MOs
#    err_orth = abs(np.linalg.multi_dot((c_gamma.conj().T, s_gamma, c_gamma)) - np.eye(c_gamma.shape[-1])).max()
#    if err_orth > 1e-4:
#        logger.error(cell, "Orthogonality error of MOs= %.2e !!!", err_orth)
#    else:
#        logger.debug(cell, "Orthogonality error of MOs= %.2e", err_orth)
#
#    # Make Gamma point MOs real:
#
#    # Try to remove imaginary parts by multiplication of simple phase factors
#    c_gamma = rotate_mo_to_real(cell, e_gamma, c_gamma, degen_tol=degen_tol)
#
#    # For degenerated MOs, the transformed orbitals in super cell may not be
#    # real. Construct a sub Fock matrix in super-cell to find a proper
#    # transformation that makes the transformed MOs real.
#    #e_k_degen = abs(e_gamma[1:] - e_gamma[:-1]) < degen_tol
#    #degen_mask = np.append(False, e_k_degen) | np.append(e_k_degen, False)
#
#    # Get eigenvalue solver with linear-dependency treatment
#    eigh = cell.eigh_factory(lindep_threshold=1e-13, fallback_mode=True)
#
#    c_gamma_out = c_gamma.copy()
#    mo_mask = (np.linalg.norm(c_gamma.imag, axis=0) > imag_tol)
#    logger.debug(cell, "Number of MOs with imaginary coefficients: %d out of %d", np.count_nonzero(mo_mask), len(mo_mask))
#    if np.any(mo_mask):
#        #mo_mask = np.s_[:]
#        #if np.any(~degen_mask):
#        #    err_imag = abs(c_gamma[:,~degen_mask].imag).max()
#        #    logger.debug(cell, "Imaginary part in non-degenerate MO coefficients= %.2e", err_imag)
#        #    # Diagonalize Fock matrix spanned by degenerate MOs only
#        #    if err_imag < 1e-8:
#        #        mo_mask = degen_mask
#
#        # F
#        #mo_mask = (np.linalg.norm(c_gamma.imag, axis=0) > imag_tol)
#
#        # Shift all MOs above the eig=0 subspace, so they can be extracted below
#        shift = 1.0 - min(e_gamma[mo_mask])
#        cs = np.dot(c_gamma[:,mo_mask].conj().T, s_gamma)
#        f_gamma = np.dot(cs.T.conj() * (e_gamma[mo_mask] + shift), cs)
#        logger.debug(cell, "Imaginary parts of Fock matrix: ||Im(F)||= %.2e  max|Im(F)|= %.2e", np.linalg.norm(f_gamma.imag), abs(f_gamma.imag).max())
#
#        e, v = eigh(f_gamma.real, s_gamma)
#
#        # Extract MOs from rank-deficient Fock matrix
#        mask = (e > 0.5)
#        assert np.count_nonzero(mask) == len(e_gamma[mo_mask])
#        e, v = e[mask], v[:,mask]
#        e_delta = e_gamma[mo_mask] - (e-shift)
#        if abs(e_delta).max() > 1e-4:
#            logger.error(cell, "Error of MO energies: ||dE||= %.2e  max|dE|= %.2e !!!", np.linalg.norm(e_delta), abs(e_delta).max())
#        else:
#            logger.debug(cell, "Error of MO energies: ||dE||= %.2e  max|dE|= %.2e", np.linalg.norm(e_delta), abs(e_delta).max())
#        c_gamma_out[:,mo_mask] = v
#
#    err_imag = abs(c_gamma_out.imag).max()
#    if err_imag > 1e-4:
#        logger.error(cell, "Imaginary part in gamma-point MOs: max|Im(C)|= %7.2e !!!", err_imag)
#    else:
#        logger.debug(cell, "Imaginary part in gamma-point MOs: max|Im(C)|= %7.2e", err_imag)
#    c_gamma_out = c_gamma_out.real
#
#    # Determine mo_phase, i.e. the unitary transformation from k-adapted orbitals to gamma-point orbitals
#    s_k_g = np.einsum('kuv,Rk->kuRv', s_k, phase.conj()).reshape(nk,nao,nr*nao)
#    mo_phase = []
#    for k in range(nk):
#        mo_phase_k = lib.einsum('um,uv,vi->mi', mo_coeff[k].conj(), s_k_g[k], c_gamma_out)
#        mo_phase.append(mo_phase_k)
#
#    return scell, e_gamma, c_gamma_out, mo_phase
#
#def k2gamma(kmf, kmesh=None):
#    r'''
#    convert the k-sampled mean-field object to the corresponding supercell
#    gamma-point mean-field object.
#
#    math:
#         C_{\nu ' n'} = C_{\vecR\mu, \veck m} = \frac{1}{\sqrt{N_{\UC}}}
#         \e^{\ii \veck\cdot\vecR} C^{\veck}_{\mu  m}
#    '''
#    def transform(mo_energy, mo_coeff, mo_occ):
#        scell, E_g, C_gamma = mo_k2gamma(kmf.cell, mo_energy, mo_coeff,
#                                         kmf.kpts, kmesh)[:3]
#        E_sort_idx = np.argsort(np.hstack(mo_energy))
#        mo_occ = np.hstack(mo_occ)[E_sort_idx]
#        return scell, E_g, C_gamma, mo_occ
#
#    if isinstance(kmf, scf.khf.KRHF):
#        scell, E_g, C_gamma, mo_occ = transform(kmf.mo_energy, kmf.mo_coeff, kmf.mo_occ)
#        mf = scf.RHF(scell)
#    elif isinstance(kmf, scf.kuhf.KUHF):
#        scell, Ea, Ca, occ_a = transform(kmf.mo_energy[0], kmf.mo_coeff[0], kmf.mo_occ[0])
#        scell, Eb, Cb, occ_b = transform(kmf.mo_energy[1], kmf.mo_coeff[1], kmf.mo_occ[1])
#        mf = scf.UHF(scell)
#        E_g = [Ea, Eb]
#        C_gamma = [Ca, Cb]
#        mo_occ = [occ_a, occ_b]
#    else:
#        raise NotImplementedError('SCF object %s not supported' % kmf)
#
#    mf.mo_coeff = C_gamma
#    mf.mo_energy = E_g
#    mf.mo_occ = mo_occ
#    mf.converged = kmf.converged
#    # Scale energy by number of primitive cells within supercell
#    mf.e_tot = len(kmf.kpts)*kmf.e_tot
#
#    # Use unfolded overlap matrix for better error cancellation
#    #s_k = kmf.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kmf.kpts, pbcopt=lib.c_null_ptr())
#    s_k = kmf.get_ovlp()
#    ovlp = to_supercell_ao_integrals(kmf.cell, kmf.kpts, s_k)
#    assert np.allclose(ovlp, ovlp.T)
#    ovlp = (ovlp + ovlp.T) / 2
#    mf.get_ovlp = lambda *args : ovlp
#
#    return mf



#def to_supercell_mo_integrals(kmf, mo_ints):
#    '''Transform from the unitcell k-point MO integrals to the supercell
#    gamma-point MO integrals.
#    '''
#    cell = kmf.cell
#    kpts = kmf.kpts
#
#    mo_k = np.array(kmf.mo_coeff)
#    Nk, nao, nmo = mo_k.shape
#    e_k = np.array(kmf.mo_energy)
#    scell, E_g, C_gamma, mo_phase = mo_k2gamma(cell, e_k, mo_k, kpts)
#
#    scell_ints = lib.einsum('xui,xuv,xvj->ij', mo_phase.conj(), mo_ints, mo_phase)
#    assert(abs(scell_ints.imag).max() < 1e-7)
#    return scell_ints.real


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
