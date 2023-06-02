import numpy
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.mp.mp2 import _mo_without_core
from pyscf.cc import ccsd
from pyscf.cc.dfccsd import _ChemistsERIs

def init_eris(cc, fock, mo_energy=None):
    """Initialize ERIs directly, without calling _common_init_."""
    eris = _ChemistsERIs()
    mo_coeff = _mo_without_core(cc, cc.mo_coeff)
    eris.mo_coeff = mo_coeff
    eris.fock = fock
    if mo_energy is None: mo_energy = fock.diagonal()
    eris.mo_energy = mo_energy
    eris.nocc = cc.get_nocc()
    eris.e_hf = cc._scf.e_tot
    eris.mol = cc.mol
    return eris

def make_eris(cc, fock, mo_energy=None):
    """Make ERIs for use in DFCCSD, without recalculating the Fock matrix."""

    eris = init_eris(cc, fock, mo_energy)

    # The following code is from _make_df_eris in pyscf/cc/dfccsd.py:
    # ===============================================================
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    nvir_pair = nvir*(nvir+1)//2
    with_df = cc.with_df
    naux = eris.naux = with_df.get_naoaux()

    eris.feri = lib.H5TmpFile()
    eris.oooo = eris.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.ovoo = eris.feri.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovov = eris.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvo = eris.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.oovv = eris.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    # nrow ~ 4e9/8/blockdim to ensure hdf5 chunk < 4GB
    chunks = (min(nvir_pair,int(4e8/with_df.blockdim)), min(naux,with_df.blockdim))
    eris.vvL = eris.feri.create_dataset('vvL', (nvir_pair,naux), 'f8', chunks=chunks)

    Loo = numpy.empty((naux,nocc,nocc))
    Lov = numpy.empty((naux,nocc,nvir))
    mo = numpy.asarray(eris.mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    for k, eri1 in enumerate(with_df.loop()):
        Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1-p0,nmo,nmo)
        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
        Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])
        eris.vvL[:,p0:p1] = Lvv.T
    Lpq = Lvv = None
    Loo = Loo.reshape(naux,nocc**2)
    #Lvo = Lov.transpose(0,2,1).reshape(naux,nvir*nocc)
    Lov = Lov.reshape(naux,nocc*nvir)
    eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocc,nocc,nocc,nocc)
    eris.ovoo[:] = lib.ddot(Lov.T, Loo).reshape(nocc,nvir,nocc,nocc)
    ovov = lib.ddot(Lov.T, Lov).reshape(nocc,nvir,nocc,nvir)
    eris.ovov[:] = ovov
    eris.ovvo[:] = ovov.transpose(0,1,3,2)
    ovov = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now)
    blksize = max(ccsd.BLKMIN, int((max_memory*.9e6/8-nocc**2*nvir_pair)/(nocc**2+naux)))
    oovv_tril = numpy.empty((nocc*nocc,nvir_pair))
    for p0, p1 in lib.prange(0, nvir_pair, blksize):
        oovv_tril[:,p0:p1] = lib.ddot(Loo.T, _cp(eris.vvL[p0:p1]).T)
    eris.oovv[:] = lib.unpack_tril(oovv_tril).reshape(nocc,nocc,nvir,nvir)
    oovv_tril = Loo = None

    Lov = Lov.reshape(naux,nocc,nvir)
    vblk = max(nocc, int((max_memory*.15e6/8)/(nocc*nvir_pair)))
    vvblk = int(min(nvir_pair, 4e8/nocc, max(4, (max_memory*.8e6/8)/(vblk*nocc+naux))))
    eris.ovvv = eris.feri.create_dataset('ovvv', (nocc,nvir,nvir_pair), 'f8',
                                         chunks=(nocc,1,vvblk))
    for q0, q1 in lib.prange(0, nvir_pair, vvblk):
        vvL = _cp(eris.vvL[q0:q1])
        for p0, p1 in lib.prange(0, nvir, vblk):
            tmpLov = _cp(Lov[:,:,p0:p1]).reshape(naux,-1)
            eris.ovvv[:,p0:p1,q0:q1] = lib.ddot(tmpLov.T, vvL.T).reshape(nocc,p1-p0,q1-q0)
        vvL = None
    return eris

def _cp(a):
    return numpy.array(a, copy=False, order='C')
