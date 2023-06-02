import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock:
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD:
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-5),
        solver_options=dict(solve_lambda=True))
emb.kernel()

print()
print("Density-density fluctuations <dN(A)dN(B)>:")
corr_mf = emb.get_corrfunc_mf('dN,dN')
corr_cc = emb.get_corrfunc('dN,dN')
for a in range(mol.natm):
    for b in range(mol.natm):
        print('A= %d, B= %d:  HF= %+.5f  CC= %+.5f' % (a, b, corr_mf[a,b], corr_cc[a,b]))
print("Total:       HF= %+.5f  CC= %+.5f" % (corr_mf.sum(), corr_cc.sum()))

print()
print("Spin-spin correlation <Sz(A)Sz(B)>:")
corr_mf = emb.get_corrfunc_mf('Sz,Sz')
corr_cc = emb.get_corrfunc('Sz,Sz')
for a in range(mol.natm):
    for b in range(mol.natm):
        print('A= %d, B= %d:  HF= %+.5f  CC= %+.5f' % (a, b, corr_mf[a,b], corr_cc[a,b]))
print("Total:       HF= %+.5f  CC= %+.5f" % (corr_mf.sum(), corr_cc.sum()))
