import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf
import vayesta.misc.molecules
import numpy as np


mol = pyscf.gto.Mole()
# mol.atom = """
# H  0.0000   0.0000   0.0000
# H  0.0000   0.0000   0.7444
# """

mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""

mol.atom = vayesta.misc.molecules.alkane(4)
print(mol.atom)
mol.basis = 'sto-6g'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

niter = (5,0)

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=-1), solver_options=dict(solve_lambda=True))
emb.opts.nmoments= 2*niter[0] + 2

with emb.iao_fragmentation() as f:
    n = len(mol.atom)
    f.add_atomic_fragment([0,1,2,3])
    for i in range(4, n-4, 3):
        f.add_atomic_fragment([i,i+1, i+2])
    f.add_atomic_fragment([n-1,n-2,n-3,n-4])


emb.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()
cc.solve_lambda()

gfcc = pyscf.cc.gfccsd.GFCCSD(cc, niter=niter)
gfcc.kernel()
ip = gfcc.ipgfccsd(nroots=1)[0]


th = gfcc.build_hole_moments()
print(len(th))
#tp = gfcc.build_part_moments()

moms = vayesta.ewf.moments.make_ccsdgf_moms(emb)

print(th)
print(moms[0])

for i in range(len(th)):
    th[i] = (th[i] + th[i].T)/2
    moms[0][i] = (moms[0][i] + moms[0][i].T)/2

for i in range(len(th)):
    mask = np.abs(th[i]) > 1e-10
    print("%d mom: norm = %e    maxerr = %e"%(i, np.linalg.norm(th[i]-moms[0][i]), (np.abs(th[i]-moms[0][i])).max()))


mask = np.abs(th) > 1e-10
print(np.abs(th-moms[0]))
