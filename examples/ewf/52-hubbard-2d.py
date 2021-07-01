import vayesta
import vayesta.ewf
import vayesta.lattmod

#nsites = (8,8)
#nsites = (6,6)
nsites = (12,8)
nsite = nsites[0]*nsites[1]
nelectron = nsite
hubbard_u = 2.0
boundary = ('PBC', 'APBC')
mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, output='pyscf.out', verbose=10)
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()
#1/0

# Single site embedding:
ecc = vayesta.ewf.EWF(mf, bno_threshold=-1, fragment_type='Site')
ecc.make_atom_fragment(0, sym_factor=nsite)
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nsite))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot/nsite))

# Double site embedding:
#ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-8, fragment_type='Site')
#ecc.make_atom_fragment([0,1], sym_factor=nsite/2)
#ecc.kernel()
#print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nsite))
#print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot/nsite))




# Compare to PySCF:
# (see pyscf/examples/scf/40-customizing_hamiltonian.py)

#import numpy as np
#import pyscf
#import pyscf.gto
#import pyscf.scf
#import pyscf.cc
#
#mol = pyscf.gto.M()
#mol.nelectron = nelectron
##mol.verbose = 10
#
#mf = pyscf.scf.RHF(mol)
#h1 = np.zeros((nsite, nsite))
#for i in range(nsite-1):
#    h1[i,i+1] = h1[i+1,i] = -1.0
#h1[nsite-1,0] = h1[0,nsite-1] = -1.0  # PBC
#eri = np.zeros(4*[nsite])
#for i in range(nsite):
#    eri[i,i,i,i] = hubbard_u
#
#mf.get_hcore = lambda *args: h1
#mf.get_ovlp = lambda *args: np.eye(nsite)
## ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
## ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
#mf._eri = pyscf.ao2mo.restore(8, eri, nsite)
#
#mf.kernel()
#print('MF: E(tot)= % 16.8f' % (mf.e_tot/nsite))
#
## If you need to run post-HF calculations based on the customized Hamiltonian,
## setting incore_anyway=True to ensure the customized Hamiltonian (the _eri
## attribute) to be used.  Without this parameter, some post-HF method
## (particularly in the MO integral transformation) may ignore the customized
## Hamiltonian if memory is not enough.
#mol.incore_anyway = True
#
#mycc = pyscf.cc.CCSD(mf)
#mycc.kernel()
#print('CCSD: E(tot)= % 16.8f E(corr)= % 16.8f' % (mycc.e_tot/nsite, mycc.e_corr/nsite))
