import vayesta
import vayesta.ewf
import vayesta.lattmod

nsites = (4,4)
fragment = (2, 2)
hubbard_u = 6.0
boundary = ('PBC', 'APBC')

nsite = nsites[0]*nsites[1]
nfrag = fragment[0]*fragment[1]
nelectron = nsite
mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, tiles=fragment)
mol.output = "pyscf-52.out"
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

ewf = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None, make_rdm1=True, make_rdm2=True)
for i in range(0, nsite, nfrag):
    ewf.add_atomic_fragment(list(range(i, i + nfrag)), nelectron_target=nfrag)
ewf.kernel()
print("E(HF)=       %+16.8f Ha" % (mf.e_tot/nelectron))
print("E(EWF-FCI)=  %+16.8f Ha" % (ewf.e_tot/nelectron))
print("E(DMET)=     %+16.8f Ha" % (ewf.get_dmet_energy()/nelectron))
