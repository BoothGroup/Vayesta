import numpy as np
import argparse

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.tools

import vayesta
import vayesta.ewf
from vayesta.misc import solids
from vayesta.misc import scf_with_mpi
from vayesta.core.mpi import mpi

#from vayesta.misc.plotting.plotly_mol import plot_mol

parser = argparse.ArgumentParser()
parser.add_argument('--kmesh', type=int, nargs=3, default=None)
args = parser.parse_args()

# Experimental Lattice constant
a0 = 4.4448

cell = pyscf.pbc.gto.Cell()
cell.a, cell.atom = solids.rocksalt(atoms=['Ni', 'O'], a=a0)
cell.basis = 'def2-svp'
#cell.basis = 'gth-dzvp'
#cell.pseudo = 'gth-pade'
cell.output = 'pyscf.mpi%d.out' % mpi.rank
cell.verbose = 10
cell.build()

# Make magnetic supercell [8x Ni, 8x O]
supercell = [2,2,2]
cell = pyscf.pbc.tools.super_cell(cell, supercell)

#fig = plot_mol(cell)
#fig.write_html('nio.html')

if args.kmesh is not None:
    kpts = cell.make_kpts(args.kmesh)
else:
    kpts = None

# Hartree-Fock
if kpts is not None:
    mf = pyscf.pbc.scf.KUHF(cell, kpts)
else:
    mf = pyscf.pbc.scf.UHF(cell)
#mf = mf.density_fit(auxbasis='def2-svp-ri')
mf = mf.rs_density_fit(auxbasis='def2-svp-ri')
#mf = mf.rs_density_fit()
#mf.with_df._cderi_to_save = 'cderi.h5'
mf = scf_with_mpi(mf)

# Create starting guess for AF2 solution:
ni3d_a = cell.search_ao_label(['^%d Ni 3d' % i for i in [0, 2, 4, 6]])
ni3d_b = cell.search_ao_label(['^%d Ni 3d' % i for i in [8, 10, 12, 14]])
ni3d_a = np.ix_(ni3d_a, ni3d_a)
ni3d_b = np.ix_(ni3d_b, ni3d_b)
dm_init = mf.init_guess_by_minao(breaksym=False)
assert np.allclose(dm_init[0], dm_init[1])
ddm = 0.2*dm_init[0][ni3d_a]
dm_init[0][ni3d_a] += ddm
dm_init[0][ni3d_b] -= ddm
dm_init[1][ni3d_b] += ddm
dm_init[1][ni3d_a] -= ddm
assert not np.allclose(dm_init[0], dm_init[1])
if kpts is not None:
    dm_init = np.repeat(dm_init[:,None,:,:], len(kpts), axis=1)

#dm_init = mf.get_init_guess()
#for i, (shell0, shell1, ao0, ao1) in enumerate(cell.aoslice_by_atom()):
#    aos = np.s_[ao0:ao1]
#    ni3d = np.isin(ao_labels[aos], '3d')
#    # Ni (up)
#    if i in (0, 2, 4, 6):
#        ddm = dm_init[0][aos,aos]
#        dm_init[0][aos,aos] += 0.1
#        dm_init[1][aos,aos] -= 0.1
#    # Ni (down)
#    if i in (8, 10, 12, 14):
#        dm_init[0][aos,aos] -= 0.1
#        dm_init[1][aos,aos] += 0.1
mf.kernel(dm_init)

# Embedded calculation will automatically unfold the k-point sampled mean-field
eta = 1e-6
emb = vayesta.ewf.EWF(mf, bno_threshold=eta, store_dm1=True, dmet_threshold=1e-4)
#emb.iao_fragmentation()
#ni1 = emb.add_atomic_fragment(0, sym_factor=4)
#ni2 = emb.add_atomic_fragment(8, sym_factor=4)
#o1 = emb.add_atomic_fragment(1, sym_factor=4)
#o2 = emb.add_atomic_fragment(9, sym_factor=4)

#emb.pop_analysis(dm1_init, local_orbitals='iao+pao', filename='pop-init-iao.txt')

dm1_mf = emb.mf.make_rdm1()
emb.pop_analysis(dm1_mf, local_orbitals='mulliken', filename='pop-mf-mulliken.txt')
emb.pop_analysis(dm1_mf, local_orbitals='lowdin', filename='pop-mf-lowdin.txt')
emb.pop_analysis(dm1_mf, local_orbitals='iao+pao', filename='pop-mf-iao.txt')

emb.kernel()

dm1_cc = emb.make_ccsd_rdm1()
emb.pop_analysis(dm1_cc, local_orbitals='mulliken', filename='pop-cc-mulliken.txt')
emb.pop_analysis(dm1_cc, local_orbitals='lowdin', filename='pop-cc-lowdin.txt')
emb.pop_analysis(dm1_cc, local_orbitals='iao+pao', filename='pop-cc-iao.txt')

emb.t1_diagnostic()

print("E(HF)=         %+16.8f Ha" % (mf.e_tot))
print("E(Emb. CCSD)=  %+16.8f Ha" % (emb.e_tot))
