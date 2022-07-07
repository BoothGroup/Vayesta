import numpy as np
import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.cc
import vayesta
import vayesta.ewf


cell = pyscf.pbc.gto.Cell()
a = 3.615
cell.atom = 'B 0 0 0 ; N %f %f %f' % (a/4, a/4, a/4)
cell.a = np.asarray([
    [a/2, a/2, 0],
    [0, a/2, a/2],
    [a/2, 0, a/2]])
cell.basis = 'sto-6g'
cell.output = 'pyscf.out'
cell.build()

# Hartree-Fock with k-points
kmesh = [2,2,2]
kpts = cell.make_kpts(kmesh)
mf = pyscf.pbc.scf.KRHF(cell, kpts)
mf = mf.density_fit(auxbasis='sto-6g')
mf.kernel()

# Full system CCSD
cc = pyscf.pbc.cc.KCCSD(mf)
cc.kernel()

# Embedded calculation will automatically fold the k-point sampled mean-field to the supercell
# solve_lambda=True is required, if density-matrix is needed!
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6), solver_options=dict(solve_lambda=True))
emb.kernel()

print("E(HF)=             %+16.8f Ha" % mf.e_tot)
print("E(Emb. CCSD)=      %+16.8f Ha" % emb.e_tot)
print("E(CCSD)=           %+16.8f Ha" % cc.e_tot)

# One-body density matrix in the supercell AO-basis
dm1 = emb.make_rdm1(ao_basis=True)
# Population analysis (q: charge, s: spin)
# Possible options for local_orbitals: 'mulliken', 'lowdin', 'iao+pao', or custom N(AO) x N(AO) coefficient matrix
# orbital_resolved=True is used to print orbital resolved (rather than only atom resolved) analysis
emb.pop_analysis(dm1, local_orbitals='iao+pao', orbital_resolved=True)
