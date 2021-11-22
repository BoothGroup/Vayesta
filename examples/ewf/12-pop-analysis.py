import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['H 0 0 0', 'F 0 0 4']
mol.basis = 'aug-cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.UHF(mol)
dm0 = None
while True:
    mf.kernel(dm0=dm0)
    mo1 = mf.stability()[0]
    stable = (mo1 is mf.mo_coeff)
    if stable: break
    print("HF unstable!")
    dm0 = mf.make_rdm1(mo_coeff=mo1)
print("HF stable!")

# Activate population analysis for all fragments and write to file:
ecc = vayesta.ewf.EWF(mf, bath_type='full', make_rdm1=True)
ecc.iao_fragmentation()
frag = ecc.make_atom_fragment([0,1])
ecc.kernel()

# Mean-field
dm1 = mf.make_rdm1()
ecc.pop_analysis(dm1, local_orbitals='mulliken', filename='mulliken-mf.pop', full=True)
ecc.pop_analysis(dm1, local_orbitals='lowdin',   filename='lowdin-mf.pop',   full=True)
ecc.pop_analysis(dm1, local_orbitals='iao+pao',  filename='iaopao-mf.pop',   full=True)

# This is equivalent to frag.pop_analysis(dm1=frag.results.dm1, ...)
frag.pop_analysis(filename='pop-cluster.txt', full=True)
