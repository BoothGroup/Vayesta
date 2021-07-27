import vayesta
import vayesta.ewf
import vayesta.lattmod

nsites = (4,4)
#nsites = (6,6)
#nsites = (12,8)
nsite = nsites[0]*nsites[1]
nelectron = nsite
hubbard_u = 10.0
boundary = ('PBC', 'APBC')
mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, output='pyscf.out', verbose=10)
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Single site embedding:
ecc = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None)
ecc.make_atom_fragment(0, sym_factor=nsite)
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nsite))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot/nsite))

# Double site embedding:
ecc = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None)
ecc.make_atom_fragment([0,1], sym_factor=nsite/2)
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nsite))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot/nsite))
