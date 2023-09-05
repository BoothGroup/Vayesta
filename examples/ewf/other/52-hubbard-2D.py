import vayesta
import vayesta.ewf
import vayesta.lattmod


nsites = (4, 4)
fragment = (2, 2)
hubbard_u = 6.0
boundary = ("PBC", "APBC")

nsite = nsites[0] * nsites[1]
nfrag = fragment[0] * fragment[1]
nelectron = nsite
mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, tiles=fragment)
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

emb = vayesta.ewf.EWF(mf, solver="FCI", bath_options=dict(bathtype="dmet"))
with emb.site_fragmentation() as frag:
    for i in range(0, nsite, nfrag):
        frag.add_atomic_fragment(list(range(i, i + nfrag)), nelectron_target=nfrag)
emb.kernel()

print("E(HF)=       %+16.8f Ha" % (mf.e_tot / nelectron))
print("E(Emb. FCI)= %+16.8f Ha" % (emb.e_tot / nelectron))
print("E(DMET)=     %+16.8f Ha" % (emb.get_dmet_energy() / nelectron))
