# Run with: mpirun -n 3 python 90-mpi.py
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf
from vayesta.mpi import mpi

mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = "cc-pVDZ"
mol.output = "pyscf-mpi%d.out" % mpi.rank
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf = mf.density_fit()
mf = mpi.scf(mf)
mf.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6))
emb.kernel()

# Reference full system CCSD
if mpi.is_master:
    cc = pyscf.cc.CCSD(mf)
    cc.kernel()

    print("E(HF)=        %+16.8f Ha" % mf.e_tot)
    print("E(CCSD)=      %+16.8f Ha" % cc.e_tot)
    print("E(Emb. CCSD)= %+16.8f Ha" % emb.e_tot)
