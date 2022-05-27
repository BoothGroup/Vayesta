"""AO to MO transformation from k-space (AOs) to supercell real space (MOs)

Author: Max Nusspickel
Email:  max.nusspickel@gmail.com
"""

# Standard
from timeit import default_timer as timer
import ctypes
import logging
# External
import numpy as np
# PySCF
import pyscf
import pyscf.mp
import pyscf.cc
import pyscf.lib
import pyscf.mp
import pyscf.cc
from pyscf.mp.mp2 import _mo_without_core
import pyscf.pbc
import pyscf.pbc.df
import pyscf.pbc.tools
from pyscf.pbc.lib import kpts_helper
# Package
from vayesta.core.util import *
import vayesta.libs


log = logging.getLogger(__name__)

def gdf_to_pyscf_eris(mf, gdf, cm, fock, mo_energy, e_hf):
    """Get supercell MO eris from k-point sampled GDF.

    This folds the MO back into k-space
    and contracts with the k-space three-center integrals..

    Parameters
    ----------
    mf: pyscf.scf.hf.RHF
        Supercell mean-field object.
    gdf: pyscf.pbc.df.GDF
        Gaussian density-fit object of primitive cell (with k-points)
    cm: `pyscf.mp.mp2.MP2`, `pyscf.cc.dfccdf.RCCSD`, or `pyscf.cc.ccsd.CCSD`
        Correlated method, must have `mo_coeff` set.
    fock: (n(AO),n(AO)) array
        Fock matrix
    mo_energy: (n(MO),) array
        MO energies.

    Returns
    -------
    eris: _ChemistsERIs
        ERIs which can be used for the respective correlated method.
    """
    log.debugv("Correlated method in gdf_to_pyscf_eris= %s", type(cm))

    only_ovov = False
    store_vvl = False
    # MP2 ERIs
    if isinstance(cm, pyscf.mp.mp2.MP2):
        from pyscf.mp.mp2 import _ChemistsERIs
        sym = False
        only_ovov = True
    # Coupled-cluster ERIs
    elif isinstance(cm, pyscf.cc.rccsd.RCCSD):
        from pyscf.cc.rccsd import _ChemistsERIs
        sym = False
    elif isinstance(cm, pyscf.cc.dfccsd.RCCSD):
        from pyscf.cc.dfccsd import _ChemistsERIs
        store_vvl = True
        sym = True
    elif isinstance(cm, pyscf.cc.ccsd.CCSD):
        from pyscf.cc.ccsd import _ChemistsERIs
        sym = True
    else:
        raise NotImplementedError("Unknown correlated method= %s" % type(cm))

    eris = _ChemistsERIs()
    mo_coeff = _mo_without_core(cm, cm.mo_coeff)
    eris.mo_coeff = mo_coeff
    eris.nocc = cm.nocc
    eris.e_hf = e_hf
    fock = fock() if callable(fock) else fock
    eris.fock = dot(mo_coeff.T, fock, mo_coeff)
    eris.mo_energy = mo_energy

    # Remove EXXDIV correction from Fock matrix (necessary for CCSD)
    #if mf.exxdiv and isinstance(cm, pyscf.cc.ccsd.CCSD):
    #    madelung = pyscf.pbc.tools.madelung(mf.mol, mf.kpt)
    #    for i in range(eris.nocc):
    #        eris.fock[i,i] += madelung

    # TEST: compare real_j3c = True and False
    #g_r = gdf_to_eris(gdf, mo_coeff, cm.nocc, only_ovov=only_ovov, real_j3c=True)
    #g_i = gdf_to_eris(gdf, mo_coeff, cm.nocc, only_ovov=only_ovov, real_j3c=False)
    #for key, val in g_r.items():
    #    norm = np.linalg.norm(val - g_i[key])
    #    mx = abs(val - g_i[key]).max()
    #    log.debug("Difference in (%2s|%2s):  max= %.3e  norm= %.3e", key[:2], key[2:], mx, norm)

    g = gdf_to_eris(gdf, mo_coeff, cm.nocc, only_ovov=only_ovov, symmetry=sym, store_vvl=store_vvl)
    for key, val in g.items():
        setattr(eris, key, val)

    return eris


def gdf_to_eris(gdf, mo_coeff, nocc, only_ovov=False, real_j3c=True, symmetry=False, store_vvl=False, j3c_threshold=None):
    """Make supercell ERIs from k-point sampled, density-fitted three-center integrals.

    Parameters
    ----------
    gdf : pyscf.pbc.df.GDF
        Density fitting object of primitive cell, with k-points.
    mo_coeff : (nK*nAO, nMO) array
        MO coefficients in supercell. The AOs in the supercell must be ordered
        in the same way as the k-points in the primitive cell!
    nocc : int
        Number of occupied orbitals.
    only_ovov : bool, optional
        Only calculate (occ,vir|occ,vir)-type ERIs (for MP2). Default=False.
    real_j3c : bool, optional
        Fourier transform the auxiliary basis to the real-space domain, such that the
        resulting Gamma-point three center integrals become real. Default: True.
    symmetry : bool, optional
        If True, the 'vvvv' and 'ovvv' four-center integrals will be stored in their compact form,
        otherwise the full array will be stored. Set to True for ccsd.CCSD and False for rccsd.RCCSD.
        Default: False.
    store_vvl : bool, optional
        If True, do not perform contraction (vv|L)(L|vv)->(vv|vv) and instead store three-center
        elements (vv|L). This is needed for the pyscf.cc.dfccsd.RCCSD class. Default: False.

    Returns
    -------
    eris : dict
        Dict of supercell ERIs. Has elements 'ovov' `if only_ovov == True`
        and 'oooo', 'ovoo', 'oovv', 'ovov', 'ovvo', 'ovvv', and 'vvvv' otherwise.
    """
    # DF compatiblity layer
    ints3c = ThreeCenterInts.init_from_gdf(gdf)

    cell, kpts, nk, nao, naux = ints3c.cell, ints3c.kpts, ints3c.nk, ints3c.nao, ints3c.naux

    phase = pyscf.pbc.tools.k2gamma.get_phase(cell, kpts)[1]
    nmo = mo_coeff.shape[-1]
    nvir = nmo - nocc
    o, v = np.s_[:nocc], np.s_[nocc:]

    if only_ovov:
        mem_j3c = nk*naux*nocc*nvir * 16
        mem_eris = nocc**2*nvir**2 * 8
    else:
        mem_j3c = nk*naux*(nocc*nvir + nocc*nocc + nvir*nvir) * 16
        if symmetry:
            nvir_pair = nvir*(nvir+1)//2
            mem_eris = (nocc**4 + nocc**3*nvir + 3*nocc**2*nvir**2 + nocc*nvir*nvir_pair + nvir_pair**2) * 8
        else:
            mem_eris = (nocc**4 + nocc**3*nvir + 3*nocc**2*nvir**2 + nocc*nvir**3 + nvir**4) * 8
    log.debug("Memory needed for kAO->GMO: (L|ab)= %s (ij|kl)= %s", memory_string(mem_j3c), memory_string(mem_eris))
    log.debug("Symmetry (L|ij)=(L|ji): %r", symmetry)

    # Transform: (l|ka,qb) -> (Rl|i,j)
    mo_coeff = mo_coeff.reshape(nk, nao, nmo)
    ck_o = einsum("Rk,Rai->kai", phase.conj(), mo_coeff[:,:,o]) / np.power(nk, 0.25)
    ck_v = einsum("Rk,Rai->kai", phase.conj(), mo_coeff[:,:,v]) / np.power(nk, 0.25)
    t0 = timer()
    j3c = j3c_kao2gmo(ints3c, ck_o, ck_v, only_ov=only_ovov, make_real=real_j3c)
    t_trafo = (timer()-t0)

    # Composite auxiliary index: R,l -> L
    j3c["ov"] = j3c["ov"].reshape(nk*naux, nocc, nvir)
    if not only_ovov:
        j3c["oo"] = j3c["oo"].reshape(nk*naux, nocc, nocc)
        j3c["vv"] = j3c["vv"].reshape(nk*naux, nvir, nvir)

        # Check symmetry errors
        if False:
            err_oo = np.linalg.norm(j3c['oo'] - j3c['oo'].transpose(0, 2, 1).conj())
            err_vv = np.linalg.norm(j3c['vv'] - j3c['vv'].transpose(0, 2, 1).conj())
            log.debug("Symmetry error of (L|ij) vs (L|ji)= %.2e", err_oo)
            log.debug("Symmetry error of (L|ab) vs (L|ba)= %.2e", err_vv)
            # Symmetrize
            j3c['oo'] = 0.5*(j3c['oo'] + j3c['oo'].transpose(0,2,1).conj())
            j3c['vv'] = 0.5*(j3c['vv'] + j3c['vv'].transpose(0,2,1).conj())

            err_oo = np.linalg.norm(j3c['oo'] - j3c['oo'].transpose(0, 2, 1).conj())
            err_vv = np.linalg.norm(j3c['vv'] - j3c['vv'].transpose(0, 2, 1).conj())
            log.debug("Symmetry error of (L|ij) vs (L|ji)= %.2e", err_oo)
            log.debug("Symmetry error of (L|ab) vs (L|ba)= %.2e", err_vv)

    for key, val in j3c.items():
        log.debugv("Memory for (L|%s)= %s", key, memory_string(val.nbytes))

    # Prune?
    norm_ov = np.linalg.norm(j3c['ov'], axis=(1,2))
    log.debugv("Number of ov elements= %d - number of parts below 1E-14= %d  1E-12= %d  1E-10= %d  1E-8= %d",
            len(norm_ov), np.count_nonzero(norm_ov < 1e-14), np.count_nonzero(norm_ov < 1e-12),
            np.count_nonzero(norm_ov < 1e-10), np.count_nonzero(norm_ov < 1e-8))
    if not only_ovov:
        norm_oo = np.linalg.norm(j3c['oo'], axis=(1,2))
        log.debugv("Number of oo elements= %d - number of parts below 1E-14= %d  1E-12= %d  1E-10= %d  1E-8= %d",
                len(norm_oo), np.count_nonzero(norm_oo < 1e-14), np.count_nonzero(norm_oo < 1e-12),
                np.count_nonzero(norm_oo < 1e-10), np.count_nonzero(norm_oo < 1e-8))
        norm_vv = np.linalg.norm(j3c['vv'], axis=(1,2))
        log.debugv("Number of vv elements= %d - number of parts below 1E-14= %d  1E-12= %d  1E-10= %d  1E-8= %d",
                len(norm_vv), np.count_nonzero(norm_vv < 1e-14), np.count_nonzero(norm_vv < 1e-12),
                np.count_nonzero(norm_vv < 1e-10), np.count_nonzero(norm_vv < 1e-8))

    def prune_aux_basis(key):
        norm = np.linalg.norm(j3c[key], axis=(1,2))
        assert (len(norm) == nk*naux)
        keep = (norm > j3c_threshold)
        log.debug("(L|%s): Keeping %3d out of %3d auxiliary basis funcions (threshold= %.1e)", key, np.count_nonzero(keep), len(norm), j3c_threshold)
        j3c[key] = j3c[key][keep]

    if j3c_threshold:
        prune_aux_basis('ov')
        if not only_ovov:
            prune_aux_basis('oo')
            prune_aux_basis('vv')

    # Contract (L|ij)(L|kl)->(ij|kl)
    eris = {}
    t0 = timer()
    if not only_ovov:
    # (L|vv) dependend
        # These symmetries are used in ccsd.CCSD but not rccsd.RCCSD!
        if symmetry:
            if not store_vvl:
                eris["vvvv"] = contract_j3c(j3c, "vvvv", symmetry=4)
            else:
                eris['vvL'] = pyscf.lib.pack_tril(j3c['vv']).T
            eris["ovvv"] = contract_j3c(j3c, "ovvv", symmetry=2)
        else:
            if not store_vvl:
                eris["vvvv"] = contract_j3c(j3c, "vvvv")
            else:
                # Bugged?
                eris['vvL'] = j3c['vv']
            eris["ovvv"] = contract_j3c(j3c, "ovvv")

        eris["oovv"] = contract_j3c(j3c, "oovv")
        del j3c["vv"]
    # (L|ov) dependend
    eris["ovov"] = contract_j3c(j3c, "ovov")
    if not only_ovov:
        eris["ovvo"] = contract_j3c(j3c, "ovvo")
        eris["ovoo"] = contract_j3c(j3c, "ovoo")
        del j3c["ov"]
    # (L|oo) dependend
        eris["oooo"] = contract_j3c(j3c, "oooo")
        del j3c["oo"]
    t_contract = (timer()-t0)

    # Check that final 4c-integrals are real
    if np.iscomplexobj(eris["ovov"]):
        for key in list(eris.keys()):
            val = eris[key]
            inorm = np.linalg.norm(val.imag)
            imax = abs(val.imag).max()
            if max(inorm, imax) > 1e-5:
                log.warning("Norm of Im(%2s|%2s):  L(2)= %.2e  L(inf)= %.2e", key[:2], key[2:], inorm, imax)
            else:
                log.debugv("Norm of Im(%2s|%2s):  L(2)= %.2e  L(inf)= %.2e", key[:2], key[2:], inorm, imax)
            eris[key] = val.real

    for key, val in eris.items():
        log.debugv("Memory for (%s|%s)= %s", key[:2], key[2:], memory_string(val.nbytes))

    log.timing("Timings for kAO->GMO:  transform= %s  contract= %s", time_string(t_trafo), time_string(t_contract))

    return eris


def contract_j3c(j3c, kind, symmetry=None):
    """Contract (L|ij)(L|kl) -> (ij|kl)"""
    t0 = timer()
    left, right = kind[:2], kind[2:]
    # We do not store "vo" only "ov":
    right_t = 'ov' if right == 'vo' else right
    l, r = j3c[left], j3c[right_t]
    if right == 'vo':
        r = r.transpose(0, 2, 1)
    # Four-fold permutation symmetry
    if symmetry == 4:
        l = pyscf.lib.pack_tril(l)  # Lij->LI
        r = pyscf.lib.pack_tril(r)  # Lkl->LK
        c = np.dot(l.T.conj(), r)   # LI,LK->IK
    # Permutation symmetry only on right side
    elif symmetry == 2:
        r = pyscf.lib.pack_tril(r)
        c = einsum('Lij,LK->ijK', l.conj(), r)
    # No permutation symmetry
    else:
        c = np.tensordot(l.conj(), r, axes=(0, 0))
    log.timingv("Time to contract (%2s|%2s): %s", left, right, time_string(timer()-t0))
    del l, r

    # For 2D systems we have negative parts, otherwise we can return here
    if not (left + '-' in j3c):
        return c

    t0 = timer()
    l, r = j3c[left + "-"], j3c[right_t + "-"]
    if right == 'vo':
        r = r.T

    # We loop over blocks here, to avoid allocating another 4c-sized array
    # Allow ~1GB working memory
    mem = 1e9
    if symmetry == 4:
        l = pyscf.lib.pack_tril(l)
        r = pyscf.lib.pack_tril(r)
        size = l.shape[0]
        blksize = max(1, min(int(mem/(size*8*(1+np.iscomplexobj(l)))), size))
        log.debugv("blksize= %d out of %d (%d blocks)", blksize, size, int(np.ceil(size/blksize)))
        for p0, p1 in pyscf.lib.prange(0, size, blksize):
            blk = np.s_[p0:p1]
            c[blk] -= np.outer(l[blk].conj(), r)
    elif symmetry == 2:
        r = pyscf.lib.pack_tril(r)
        size = l.shape[0]
        blksize = max(1, min(int(mem/(size*8*(1+np.iscomplexobj(l)))), size))
        log.debugv("blksize= %d out of %d (%d blocks)", blksize, size, int(np.ceil(size/blksize)))
        for p0, p1 in pyscf.lib.prange(0, size, blksize):
            blk = np.s_[p0:p1]
            c[blk] -= einsum('ij,k->ijk', l[blk].conj(), r)
    else:
        size = l.shape[0]
        blksize = max(1, min(int(mem/(size*8*(1+np.iscomplexobj(l)))), size))
        log.debugv("blksize= %d out of %d (%d blocks)", blksize, size, int(np.ceil(size/blksize)))
        for p0, p1 in pyscf.lib.prange(0, size, blksize):
            blk = np.s_[p0:p1]
            c[blk] -= einsum("ij,kl->ijkl", l[blk].conj(), r)

    log.timingv("Time to contract (%2s|%2s)(-): %s", left, right, time_string(timer()-t0))
    return c


class ThreeCenterInts:
    """Temporary interface class for DF classes.

    This should be implemented better at some point, as a larger effort to unify integral classes
    and offer a common interface.
    """

    def __init__(self, cell, kpts, naux):
        self.cell = cell
        self.kpts = kpts
        self.naux = naux
        self.values = None
        self.df = None

    @property
    def nk(self):
        return len(self.kpts)

    @property
    def nao(self):
        return self.cell.nao_nr()

    @classmethod
    def init_from_gdf(cls, gdf):
        if gdf.auxcell is None:
            gdf.build(with_j3c=False)
        ints3c = cls(gdf.cell, gdf.kpts, gdf.auxcell.nao_nr())
        ints3c.df = gdf
        return ints3c

    def sr_loop(self, *args, **kwargs):
        return self.df.sr_loop(*args, **kwargs)

    def get_array(self, kptsym=True):

        if self.values is not None:
            return self.values, None, None

        elif isinstance(self.df._cderi, (str, type(None))):
            if kptsym:
                nkij = self.nk*(self.nk+1)//2
                j3c = np.zeros((nkij, self.naux, self.nao, self.nao), dtype=complex)
                kuniq_map = np.zeros((self.nk, self.nk), dtype=int)
            else:
                j3c = np.zeros((self.nk, self.nk, self.naux, self.nao, self.nao), dtype=complex)
                kuniq_map = None

            if self.cell.dimension < 3:
                j3c_neg = np.zeros((self.nk, self.nao, self.nao), dtype=complex)
            else:
                j3c_neg = None

            kij = 0
            for ki in range(self.nk):
                kjmax = (ki+1) if kptsym else self.nk
                for kj in range(kjmax):
                    if kptsym:
                        kuniq_map[ki,kj] = kij
                    kpts_ij = (self.kpts[ki], self.kpts[kj])
                    if kptsym:
                        j3c_kij = j3c[kij]
                    else:
                        j3c_kij = j3c[ki,kj]
                    blk0 = 0
                    for lr, li, sign in self.df.sr_loop(kpts_ij, compact=False, blksize=int(1e9)):
                        blksize = lr.shape[0]
                        blk = np.s_[blk0:blk0+blksize]
                        blk0 += blksize
                        if (sign == 1):
                            j3c_kij[blk] = (lr+1j*li).reshape(blksize, self.nao, self.nao)
                        # For 2D systems:
                        else:
                            assert (sign == -1) and (blksize == 1) and (ki == kj), ("sign= %r, blksize= %r, ki= %r, kj=%r" % (sign, blksize, ki, kj))
                            j3c_neg[ki] += (lr+1j*li)[0].reshape(self.nao, self.nao)
                    kij += 1

            if kptsym:
                # At this point, all kj <= ki are set
                # Here we loop over kj > ki
                for ki in range(self.nk):
                    for kj in range(ki+1, self.nk):
                        kuniq_map[ki,kj] = -kuniq_map[kj,ki]
                assert np.all(kuniq_map < nkij)
                assert np.all(kuniq_map > -nkij)

        # Old, deprecated code - keeping for now in case I reimplement ki/kj symmetry
        # In IncoreGDF, we can access the array directly
        #elif hasattr(self.df._cderi, '__getitem__') and 'j3c' in self.df._cderi:
        #    j3c = self.df._cderi["j3c"].reshape(-1, self.naux, self.nao, self.nao)
        #    nkuniq = j3c.shape[0]
        #    log.info("Nkuniq= %3d", nkuniq)
        #    # Check map
        #    #_get_kpt_hash = pyscf.pbc.df.df_incore._get_kpt_hash
        #    _get_kpt_hash = vayesta.misc.gdf._get_kpt_hash
        #    kuniq_map = np.zeros((self.nk, self.nk), dtype=int)
        #    # 2D systems not supported in incore version:
        #    j3c_neg = None
        #    for ki in range(self.nk):
        #        for kj in range(self.nk):
        #            kij = np.asarray((self.kpts[ki], self.kpts[kj]))
        #            kij_id = self.df._cderi['j3c-kptij-hash'].get(_get_kpt_hash(kij), [None])
        #            assert len(kij_id) == 1
        #            kij_id = kij_id[0]
        #            if kij_id is None:
        #                kij_id = self.df._cderi['j3c-kptij-hash'][_get_kpt_hash(kij[[1,0]])]
        #                assert len(kij_id) == 1
        #                # negative to indicate transpose needed
        #                kij_id = -kij_id[0]
        #            assert (abs(kij_id) < nkuniq)
        #            kuniq_map[ki,kj] = kij_id

        elif isinstance(self.df._cderi, np.ndarray):
            j3c = self.df._cderi
            j3c_neg = None
            kuniq_map = None

        else:
            raise ValueError("Unknown DF type: %r" % type(self.df))

        return j3c, j3c_neg, kuniq_map


def j3c_kao2gmo(ints3c, cocc, cvir, only_ov=False, make_real=True, driver='c'):
    """Transform three-center integrals from k-space AO to supercell MO basis.

    Returns
    -------
    j3c_ov : (Nk, Naux, Nocc, Nvir) array
        Supercell occupied-virtual three-center integrals.
    j3c_oo : (Nk, Naux, Nocc, Nocc) array or None
        Supercell occupied-occupied three-center integrals.
    j3c_vv : (Nk, Naux, Nvir, Nvir) array or None
        Supercell virtual-virtual three-center integrals.
    j3cn_ov : (Nocc, Nvir) array
        Negative supercell occupied-virtual three-center integrals.
    j3cn_oo : (Nocc, Nocc) array or None
        Negative supercell occupied-occupied three-center integrals.
    j3cn_vv : (Nvir, Nvir) array or None
        Negative supercell virtual-virtual three-center integrals.
    """
    nocc = cocc.shape[-1]
    nvir = cvir.shape[-1]
    cell, kpts, nk, naux = ints3c.cell, ints3c.kpts, ints3c.nk, ints3c.naux
    nao = cell.nao_nr()
    kconserv = kpts_helper.get_kconserv(cell, kpts, n=2)

    j3c = {}
    j3c["ov"] = np.zeros((nk, naux, nocc, nvir), dtype=complex)
    #j3c_oo = j3c_vv = j3cn_ov = j3cn_oo = j3cn_vv = None
    if not only_ov:
        j3c["oo"] = np.zeros((nk, naux, nocc, nocc), dtype=complex)
        j3c["vv"] = np.zeros((nk, naux, nvir, nvir), dtype=complex)

    if driver.lower() == 'python':  # pragma: no cover

        if cell.dimension < 3:
            j3c["ov-"] = np.zeros((nocc, nvir), dtype=complex)
            if not only_ov:
                j3c["oo-"] = np.zeros((nocc, nocc), dtype=complex)
                j3c["vv-"] = np.zeros((nvir, nvir), dtype=complex)

        for ki in range(nk):
            for kj in range(nk):
                kij = (kpts[ki], kpts[kj])
                kk = kconserv[ki,kj]
                blk0 = 0
                for lr, li, sign in ints3c.sr_loop(kij, compact=False):
                    blksize = lr.shape[0]
                    blk = np.s_[blk0:blk0+blksize]
                    blk0 += blksize
                    j3c_kij = (lr+1j*li).reshape(blksize, nao, nao)
                    if sign == 1:
                        j3c["ov"][kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, cocc[ki].conj(), cvir[kj])      # O(Nk^2 * Nocc * Nvir)
                        if only_ov: continue
                        j3c["oo"][kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, cocc[ki].conj(), cocc[kj])      # O(Nk^2 * Nocc * Nocc)
                        j3c["vv"][kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, cvir[ki].conj(), cvir[kj])      # O(Nk^2 * Nvir * Nvir)
                    # For 2D systems
                    else:
                        assert (sign == -1) and (ki == kj) and (kk == 0) and (blksize == 1)
                        j3c["ov-"] += einsum("ab,ai,bj->ij", j3c_kij[0], cocc[ki].conj(), cvir[kj])
                        if only_ov: continue
                        j3c["oo-"] += einsum("ab,ai,bj->ij", j3c_kij[0], cocc[ki].conj(), cocc[kj])
                        j3c["vv-"] += einsum("ab,ai,bj->ij", j3c_kij[0], cvir[ki].conj(), cvir[kj])

    elif driver.lower() == 'c':
        # Load j3c into memory
        t0 = timer()
        j3c_kpts, j3c_neg, kunique = ints3c.get_array()
        log.timingv("Time to load k-point sampled 3c-integrals:  %s", time_string(timer()-t0))

        cocc = cocc.copy()
        cvir = cvir.copy()
        libcore = vayesta.libs.load_library('core')
        t0 = timer()
        ierr = libcore.j3c_kao2gmo(
                ctypes.c_int64(nk),
                ctypes.c_int64(nao),
                ctypes.c_int64(nocc),
                ctypes.c_int64(nvir),
                ctypes.c_int64(naux),
                kconserv.ctypes.data_as(ctypes.c_void_p),
                #kunique_pt,
                kunique.ctypes.data_as(ctypes.c_void_p) if kunique is not None else ctypes.POINTER(ctypes.c_void_p)(),
                cocc.ctypes.data_as(ctypes.c_void_p),
                cvir.ctypes.data_as(ctypes.c_void_p),
                j3c_kpts.ctypes.data_as(ctypes.c_void_p),
                # In-out
                j3c["ov"].ctypes.data_as(ctypes.c_void_p),
                j3c["oo"].ctypes.data_as(ctypes.c_void_p) if "oo" in j3c else ctypes.POINTER(ctypes.c_void_p)(),
                j3c["vv"].ctypes.data_as(ctypes.c_void_p) if "vv" in j3c else ctypes.POINTER(ctypes.c_void_p)())
        log.timingv("Time in j3c_kao2gamo in C:  %s", time_string(timer()-t0))
        assert (ierr == 0)

        # Do the negative part for 2D systems in python (only one auxiliary function and ki==kj)
        if j3c_neg is not None:
            j3c["ov-"] = einsum("kab,kai,kbj->ij", j3c_neg, cocc.conj(), cvir)      # O(Nk * Nocc * Nvir)
            if not only_ov:
                j3c["oo-"] = einsum("kab,kai,kbj->ij", j3c_neg, cocc.conj(), cocc)      # O(Nk * Nocc * Nocc)
                j3c["vv-"] = einsum("kab,kai,kbj->ij", j3c_neg, cvir.conj(), cvir)      # O(Nk * Nvir * Nvir)

    if make_real:
        t0 = timer()
        phase = pyscf.pbc.tools.k2gamma.get_phase(cell, kpts)[1]

        def ft_auxiliary_basis(j, key):
            pj = np.tensordot(phase, j, axes=1)
            if pj.size > 0:
                inorm = np.linalg.norm(pj.imag)
                imax = abs(pj.imag).max()
                if max(inorm, imax) > 1e-5:
                    log.warning("Norm of Im(L|%2s):     L(2)= %.2e  L(inf)= %.2e", key, inorm, imax)
                else:
                    log.debug("Norm of Im(L|%2s):     L(2)= %.2e  L(inf)= %.2e", key, inorm, imax)
            return pj.real

        def check_real(j, key):
            pj = j
            if pj.size > 0:
                inorm = np.linalg.norm(pj.imag)
                imax = abs(pj.imag).max()
                if max(inorm, imax) > 1e-5:
                    log.warning("Norm of Im(L|%2s)(-):  L(2)= %.2e  L(inf)= %.2e", key[:2], inorm, imax)
                else:
                    log.debug("Norm of Im(L|%2s)(-):  L(2)= %.2e  L(inf)= %.2e", key[:2], inorm, imax)
            return pj.real

        for key in list(j3c.keys()):
            if key[-1] == "-":
                # Should already be real
                j3c[key] = check_real(j3c[key], key)
            else:
                j3c[key] = ft_auxiliary_basis(j3c[key], key)

        log.timingv("Time to rotate to real:  %s", time_string(timer()-t0))

    return j3c
