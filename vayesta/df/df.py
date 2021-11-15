
import ctypes
import numpy as np
import vayesta.libs
from timeit import default_timer as timer

import pyscf
import pyscf.pbc
import pyscf.pbc.tools

# pyscf
def create_cderi(fname):
    import pyscf.pbc.gto
    import pyscf.pbc.df
    import pyscf.pbc.scf

    cell = pyscf.pbc.gto.Cell()
    cell.a = 10*np.eye(3)
    #cell.atom = ['He 0 0 0']
    #cell.atom = ['He 0 0 0', 'He 0 0 1', 'He 0 0 2', 'He 0 0 3']
    #cell.atom = ['He 0 0 %d' % i for i in range(10)]
    atom = 'Ne'
    cell.atom = []
    n = 3
    for x in range(n):
        for y in range(n):
            for z in range(n):
                cell.atom.append('%s %d %d %d' % (atom, x, y, z))
    #cell.basis = 'sto-3g'
    #cell.basis = '6-31g'
    #cell.basis = 'cc-pVDZ'
    #cell.basis = 'cc-pVTZ'
    cell.basis = 'cc-pVQZ'
    cell.build()

    #kpts = cell.make_kpts([2,2,2])

    gdf = pyscf.pbc.df.GDF(cell)
    #gdf.blockdim = 2
    gdf._cderi_to_save = fname
    #gdf.auxbasis = 'cc-pVDZ-ri'
    #gdf.auxbasis = 'cc-pVTZ-ri'
    gdf.auxbasis = 'cc-pVQZ-ri'
    #gdf.auxbasis = 'aug-cc-pV5Z-ri'
    gdf.build()

    print("n(AO)= %d" % cell.nao)
    print("n(AO)^2= %d" % ((cell.nao * (cell.nao+1))//2))
    print("n(aux)= %d" % gdf.get_naoaux())

    mf = pyscf.pbc.scf.RHF(cell)
    mf.with_df = gdf
    mf.exxdiv = None
    mf.max_cycle = 1
    mf.kernel()

    return cell, gdf, mf

def explore(group):
    for key in group.keys():
        print("Key = %s" % key)
        if hasattr(group[key], 'keys'):
            explore(group[key])
        else:
            print("Data = %r" % group[key])

def show_hdf5(fname):
    import h5py
    with h5py.File(filename, "r") as f:
    	# List all groups
        explore(f)
        #for key in f.keys():
        #    print("Key = %s" % key)

        #    print("data = %r" % f[key])

        # print values
        print(f['j3c/0/0'][:])

libdf = vayesta.libs.libdf

filename = 'bla'
cell, gdf, mf = create_cderi(filename)

#show_hdf5(filename)

def get_jk(mf, gdf, mo_coeff, filename=None, blksize=None, exxdiv=None):
    cell = gdf.cell
    if filename is None:
        filename = gdf._cderi
    nao = cell.nao
    naux = gdf.get_naoaux()
    if blksize is None:
        blksize = 32
    if exxdiv is None:
        exxdiv = mf.exxdiv
    nocc = np.count_nonzero(mf.mo_occ > 0)
    mo_occ = np.asarray(mo_coeff[:,:nocc], order='C')
    vj = np.zeros((nao, nao))
    vk = np.zeros((nao, nao))
    ierr = libdf.df_rhf_jk(
            # In
            ctypes.c_int64(nao),
            ctypes.c_int64(nocc),
            ctypes.c_int64(naux),
            ctypes.c_char_p(filename.encode('utf-8')),
            mo_occ.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(blksize),
            # Out
            vj.ctypes.data_as(ctypes.c_void_p),
            vk.ctypes.data_as(ctypes.c_void_p),
            )
    assert (ierr == 0)
    vj *= 2
    vk *= 2

    if exxdiv == 'ewald':
        t0 = timer()
        madelung = pyscf.pbc.tools.pbc.madelung(cell, mf.kpt)
        print("Time Madelung: %f" % (timer()-t0))
        s = mf.get_ovlp()
        vk += madelung * np.linalg.multi_dot((s, dm1, s))

    return vj, vk

t0 = timer()
j, k = get_jk(mf, gdf, mf.mo_coeff)
print("Time (Vayesta)= %f" % (timer()-t0))

t0 = timer()
j2, k2 = mf.get_jk()
print("Time (PySCF)= %f" % (timer()-t0))

assert np.allclose(j, j2)
assert np.allclose(k, k2)
