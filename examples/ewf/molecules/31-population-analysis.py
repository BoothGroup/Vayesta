import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = 'H 0 0 0 ; F 0 0 2'
mol.basis = 'aug-cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock:
mf = pyscf.scf.UHF(mol)
mf.kernel()

# Embedded CCSD:
emb = vayesta.ewf.EWF(mf, bno_threshold=1e-4, solve_lambda=True)
emb.kernel()

# Population analysis of mean-field density-matrix:
dm1 = mf.make_rdm1()
emb.pop_analysis(dm1, local_orbitals='mulliken')
emb.pop_analysis(dm1, local_orbitals='lowdin')
emb.pop_analysis(dm1, local_orbitals='iao+pao')

# Population analysis of the correlated density-matrix:
dm1 = emb.make_rdm1(ao_basis=True)
emb.pop_analysis(dm1, local_orbitals='mulliken')
emb.pop_analysis(dm1, local_orbitals='lowdin')
emb.pop_analysis(dm1, local_orbitals='iao+pao')

# Population analysis can also be written to a file, when filename is provided,
# and include orbital resolution, if full=True:
emb.pop_analysis(dm1, local_orbitals='mulliken', filename='pop-mulliken-cc.txt', full=True)
emb.pop_analysis(dm1, local_orbitals='lowdin', filename='pop-lowdin-cc.txt', full=True)
emb.pop_analysis(dm1, local_orbitals='iao+pao', filename='pop-iaopao-cc.txt', full=True)
