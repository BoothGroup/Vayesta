import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.ewf
from vayesta.misc.molecules import ring

# User defined FCI solver - takes pyscf mf as input and returns RDMs
# The mf argment contains the hamiltonain in the orthonormal cluster basis
# Pyscf or other solvers may be used to solve the cluster problem and may return RDMs, CISD amplitudes or CCSD amplitudes
def solver(mf):
    ci = pyscf.ci.CISD(mf)
    energy, civec = ci.kernel()
    c0, c1, c2 = ci.cisdvec_to_amplitudes(civec)
    return dict(c0=c0, c1=c1, c2=c2, converged=True, energy=ci.e_corr)
    
natom = 10
mol = pyscf.gto.Mole()
mol.atom = ring("H", natom, 1.5)
mol.basis = "sto-3g"
mol.output = "pyscf.out"
mol.verbose = 5
mol.symmetry = True
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CISD
cisd = pyscf.ci.CISD(mf)
cisd.kernel()

# Vayesta options
use_sym = True
nfrag = 1 
bath_opts = dict(bathtype="dmet")   

# Run vayesta with user defined solver
emb = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='wf', bath_options=bath_opts, solver_options=dict(callback=solver))
# Set up fragments
with emb.iao_fragmentation() as f:
    if use_sym:
        # Add rotational symmetry
        with f.rotational_symmetry(order=natom//nfrag, axis=[0, 0, 1]):
            f.add_atomic_fragment(range(nfrag))
    else:
        # Add all atoms as separate fragments 
        f.add_all_atomic_fragments()
emb.kernel()

print("Hartree-Fock energy          : %s"%mf.e_tot)
print("CISD Energy                  : %s"%cisd.e_tot)
print("Emb. Projected Energy         : %s"%emb.e_tot)
