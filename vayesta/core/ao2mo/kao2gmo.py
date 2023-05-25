"""AO to MO transformation from k-space (AOs) to supercell real space (MOs)

Author: Max Nusspickel
Email:  max.nusspickel@gmail.com
"""

# Standard
import ctypes
import logging
# External
import numpy as np
# PySCF
import pyscf
import pyscf.pbc
import pyscf.pbc.tools
# Package
from vayesta.core.util import call_once, einsum, timer
from vayesta.libs import libcore
from vayesta.core.ao2mo import helper


log = logging.getLogger(__name__)

def kao2gmo_cderi(gdf, mo_coeffs, make_real=True, blksize=None, tril_kij=True, driver=None):
    """Transform density-fitted ERIs from primtive cell k-AO, to Gamma-point MOs.

    (L|ka,k'b) * C1_(Ra)i * C2_(R'b)j -> (R''L|Ri,R'j)

    Parameters
    ----------
    gdf: pyscf.pbc.df.GDF
        Gaussian density-fitting object of primitive unit cell.
    mo_coeffs: array or tuple(2) of arrays
        MO coefficients C1_(Ra)i and C2_(R'b)j for the transformation. If only one coefficient matrix is
        passed, it will be used for both C1 and C2.
    make_real: bool, optional
        Fourier-transform the auxiliary DF dimension, such that the final three-center integrals are real.
        Default: True.
    blksize: int, optional
        Blocksize for the auxiliary dimension.
    tril_kij: bool, optional
        Only load k-point pairs k >= k', and use the symmetry (L|ka,k'b) = (L|k'b,ka)*.
        Default: True.
    driver: {None, 'c', 'python'}
        Use Python or C driver for the transformation. If None, use C if compiled library is present,
        if so use C, else use python. Default: None.

    Returns
    -------
    cderi_mo: array
        Density-fitted supercell three-center integrals in MO basis.
    cderi_mo_neg: array or None
        Negative part of density-fitted supercell three-center integrals in MO basis.
        None for 3D systems.
    """

    if np.ndim(mo_coeffs[0]) == 1:
        mo_coeffs = (mo_coeffs, mo_coeffs)

    cell = gdf.cell
    nao = cell.nao
    naux = naux_pos = gdf.auxcell.nao_nr()
    if cell.dimension < 3:
        naux_pos -= 1
    kpts = gdf.kpts
    nk = len(kpts)
    kconserv = helper.get_kconserv(cell, kpts, nk=2)
    # Fourier transform MOs from supercell Gamma point to primitive cell k-points
    phase = (pyscf.pbc.tools.k2gamma.get_phase(cell, kpts)[1]).T

    if blksize is None:
        max_memory = int(1e9)   # 1 GB
        max_size = int(max_memory / (nao**2 * 16)) # how many naux fit in max_memory
        blksize = np.clip(max_size, 1, int(1e9))
        log.debugv("max_memory= %.3f MB  max_size= %d  blksize= %d", max_memory/1e6, max_size, blksize)

    if driver is None:
        if libcore is None:
            driver = 'python'
            call_once(log.warning, "Libary 'vayesta/libs/libcore.so' not found, using fallback Python driver.")
        else:
            driver = 'c'
    log.debugv("Driver for kao2gmo_cderi= %s", driver)
    if driver == 'python':
        transform = lambda cderi_kij, mo1_ki, mo2_kj : einsum('Lab,ai,bj->Lij', cderi_kij, mo1_ki.conj(), mo2_kj)
    elif driver == 'c':

        def transform(cderi_kij, mo1_ki, mo2_kj):
            naux = cderi_kij.shape[0]
            nao = cderi_kij.shape[1]
            nmo1 = mo1_ki.shape[-1]
            nmo2 = mo2_kj.shape[-1]
            buf = np.zeros((naux, nmo1, nmo2), dtype=complex)
            cderi_kij = np.asarray(cderi_kij, order='C')
            mo1_ki = np.asarray(mo1_ki.conj(), order='C')
            mo2_kj = np.asarray(mo2_kj, order='C')
            ierr = libcore.ao2mo_cderi(
                    # In
                    ctypes.c_int64(nao),
                    ctypes.c_int64(nmo1),
                    ctypes.c_int64(nmo2),
                    ctypes.c_int64(naux),
                    mo1_ki.ctypes.data_as(ctypes.c_void_p),
                    mo2_kj.ctypes.data_as(ctypes.c_void_p),
                    cderi_kij.ctypes.data_as(ctypes.c_void_p),
                    # Out
                    buf.ctypes.data_as(ctypes.c_void_p))
            assert (ierr == 0)
            return buf

    nmo1 = mo_coeffs[0].shape[-1]
    nmo2 = mo_coeffs[1].shape[-1]
    mo1 = einsum('kR,Rai->kai', phase.conj(), mo_coeffs[0].reshape(nk,nao,nmo1))
    mo2 = einsum('kR,Rai->kai', phase.conj(), mo_coeffs[1].reshape(nk,nao,nmo2))

    cderi_mo = np.zeros((nk, naux, nmo1, nmo2), dtype=complex)
    if cell.dimension < 3:
        cderi_mo_neg = np.zeros((1, nmo1, nmo2), dtype=complex)
    else:
        cderi_mo_neg = None

    # Rare case that one of the MO sets is empty - exit early
    if (nmo1 * nmo2) == 0:
        cderi_mo = cderi_mo.reshape((nk*naux, nmo1, nmo2)).real
        if cderi_mo_neg is not None:
            cderi_mo_neg = cderi_mo_neg.real
        return cderi_mo, cderi_mo_neg

    for ki in range(nk):
        kjmax = (ki+1) if tril_kij else nk
        for kj in range(kjmax):
            kk = kconserv[ki,kj]
            # Load entire 3c-integrals at k-point pair (ki, kj) into memory:
            kpts_ij = (kpts[ki], kpts[kj])
            blk0 = 0
            for labr, labi, sign in gdf.sr_loop(kpts_ij, compact=False, blksize=blksize):
                blk1 = (blk0 + labr.shape[0])
                blk = np.s_[blk0:blk1]
                blk0 = blk1
                lab = (labr + 1j*labi).reshape(-1, nao, nao)
                if (sign == 1):
                    lij = transform(lab, mo1[ki], mo2[kj])
                    cderi_mo[kk,blk] += lij
                    if tril_kij and (ki > kj):
                        kk = kconserv[kj,ki]
                        lij = transform(lab.transpose(0,2,1).conj(), mo1[kj], mo2[ki])
                        cderi_mo[kk,blk] += lij
                # For 2D systems:
                elif (sign == -1):
                    assert (ki == kj) and (sign == -1) and (lab.shape[0] == 1)
                    cderi_mo_neg += einsum('Lab,ai,bj->Lij', lab, mo1[ki].conj(), mo2[kj])
                else:
                    raise ValueError("Sign = %f" % sign)

    cderi_mo /= np.sqrt(nk)
    if cderi_mo_neg is not None:
        cderi_mo_neg /= np.sqrt(nk)

    if make_real:
        # kR,kLij->RLij
        cderi_mo = np.tensordot(phase, cderi_mo, axes=1)
        im = abs(cderi_mo.imag).max()
        if im > 1e-3:
            log.error("Imaginary part of (L|ij)= %.2e !!!", im)
        elif im > 1e-8:
            log.error("Imaginary part of (L|ij)= %.2e !", im)
        else:
            log.debugv("Imaginary part of (L|ij)= %.2e", im)
        cderi_mo = cderi_mo.real
        if cderi_mo_neg is not None:
            assert (abs(cderi_mo_neg.imag).max() < 1e-10)
            cderi_mo_neg = cderi_mo_neg.real

    cderi_mo = cderi_mo.reshape(nk*naux, nmo1, nmo2)

    return cderi_mo, cderi_mo_neg


if __name__ == '__main__':

    import pyscf
    import pyscf.pbc
    import pyscf.pbc.gto
    import pyscf.pbc.df
    import pyscf.pbc.dft
    import pyscf.pbc.tools

    import vayesta.core.ao2mo

    import kao2gmo #import gdf_to_eris

    card = 'D'

    cell = pyscf.pbc.gto.Cell()
    cell.a = 3.0*np.eye(3)
    cell.a[2,2] = 20.0
    cell.atom = "He 0 0 0"
    cell.basis = 'cc-pV%sZ' % card
    cell.dimension = 2
    cell.build()

    kmesh = [3,2,1]
    #kmesh = [3,3,3]
    kpts = cell.make_kpts(kmesh)

    gdf = pyscf.pbc.df.GDF(cell, kpts)
    gdf.auxbasis = 'cc-pV%sZ-ri' %  card
    gdf.build()

    scell = pyscf.pbc.tools.super_cell(cell, kmesh)
    sgdf = pyscf.pbc.df.GDF(scell)
    sgdf.auxbasis = 'cc-pV%sZ-ri' % card
    sgdf.build()

    print("Naux= %d" % sgdf.auxcell.nao)

    nao_sc = scell.nao
    #mf = pyscf.pbc.dft.RKS(scell)
    #mf.with_df = sgdf
    #mf.max_cycle = 1
    #mf.kernel()
    #mo_coeff = mf.mo_coeff
    mo_coeff = np.random.rand(nao_sc, nao_sc)

    # Exact
    t0 = timer()
    seri = sgdf.ao2mo(mo_coeff, compact=False).reshape(4*[nao_sc])
    print("Time SC= %.4f" % (timer()-t0))

    # Old code
    nocc = 3
    t0 = timer()
    eris_old = kao2gmo.gdf_to_eris(gdf, mo_coeff, nocc)
    print("Time old= %.4f" % (timer()-t0))

    class Object():
        pass
    eris_old_2 = Object()
    for key, val in eris_old.items():
        setattr(eris_old_2, key, val)
    eris_old_2.fock = np.zeros((nao_sc, nao_sc))
    eris_old_2.nocc = nocc
    eris_old = vayesta.core.ao2mo.helper.get_full_array(eris_old_2)

    mo_occ = mo_coeff[:,:nocc]
    mo_vir = mo_coeff[:,nocc:]

    t0 = timer()
    cderi, cderi_neg = kao2gmo_cderi(gdf, (mo_coeff, mo_coeff), driver='python', make_real=True)
    #cderi_oo, cderi_neg_oo = kao2gmo_gdf(gdf, (mo_occ, mo_occ), driver='python')
    #cderi_ov, cderi_neg_ov = kao2gmo_gdf(gdf, (mo_occ, mo_vir), driver='python')
    #cderi_vv, cderi_neg_vv = kao2gmo_gdf(gdf, (mo_vir, mo_vir), driver='python')
    print("Time k(Py)= %.4f" % (timer()-t0))
    eri = einsum('Lij,Lkl->ijkl', cderi.conj(), cderi)
    if cderi_neg is not None:
        eri -= einsum('Lij,Lkl->ijkl', cderi_neg, cderi_neg)

    t0 = timer()
    cderi, cderi_neg = kao2gmo_cderi(gdf, (mo_coeff, mo_coeff), driver='c')
    print("Time k(C)= %.4f" % (timer()-t0))
    eri2 = einsum('Lij,Lkl->ijkl', cderi ,cderi)
    if cderi_neg is not None:
        eri2 -= einsum('Lij,Lkl->ijkl', cderi_neg, cderi_neg)

    #print(seri[0,0,0,:])
    #print(eri[0,0,0,:])
    #print(eris_old[0,0,0,:])

    print(np.linalg.norm(eri - seri))
    print(np.linalg.norm(eri2 - seri))
    print(np.linalg.norm(eris_old - seri))
    print(np.linalg.norm(eris_old - eri))
