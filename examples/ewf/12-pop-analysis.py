import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['H 0.0 0.0 0.0', 'F 0.0, 0.0, 2.0']
mol.basis = 'aug-cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Activate population analysis for all fragments and write to file:
ecc = vayesta.ewf.EWF(mf)
ecc.iao_fragmentation()
frag = ecc.make_atom_fragment([0,1], make_rdm1=True)
ecc.kernel(bno_threshold=-1)

# Mean-field
ecc.pop_analysis(dm1=mf.make_rdm1(), filename='pop-mf.txt')

# This is equivalent to frag.pop_analysis(dm1=frag.results.dm1, ...)
frag.pop_analysis(filename='pop-cluster.txt')
