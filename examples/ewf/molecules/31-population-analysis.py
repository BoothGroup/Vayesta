import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['H 0 0 0', 'F 0 0 3']
mol.basis = 'aug-cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock - note that restarts are required for stretched HF
mf = pyscf.scf.UHF(mol)
dm0 = None
while True:
    mf.kernel(dm0=dm0)
    mo1 = mf.stability()[0]
    stable = (mo1 is mf.mo_coeff)
    if stable: break
    dm0 = mf.make_rdm1(mo_coeff=mo1)

# Activate population analysis for all fragments and write to file:
emb = vayesta.ewf.EWF(mf, bath_type='full', store_dm1=True)
emb.iao_fragmentation()
frag = emb.add_atomic_fragment([0,1])
emb.kernel()

# Population analysis of mean-field density-matrix
dm1 = mf.make_rdm1()
emb.pop_analysis(dm1, local_orbitals='mulliken', filename='mulliken-mf.pop', full=True)
emb.pop_analysis(dm1, local_orbitals='lowdin',   filename='lowdin-mf.pop',   full=True)
emb.pop_analysis(dm1, local_orbitals='iao+pao',  filename='iaopao-mf.pop',   full=True)

# Population analysis if the CCSD density-matrix
# This is equivalent to frag.pop_analysis(dm1=frag.results.dm1, ...)
frag.pop_analysis(filename='cluster.pop', local_orbitals='iao+pao', full=True)
