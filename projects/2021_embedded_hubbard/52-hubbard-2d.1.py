import vayesta
import vayesta.ewf
import vayesta.lattmod
from vayesta.core.util import einsum

from scmf import SelfConsistentMF

nsites = (4,3)
nsite = nsites[0]*nsites[1]
nimp = 2
nelectron = nsite
boundary = ('PBC', 'PBC')
hubbard_u = 0.0

# Exact
#for hubbard_u in range(13):
#
#    mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, tiles=(2,1))
#    mf = vayesta.lattmod.LatticeMF(mol)
#    mf.kernel()
#
#    # Exact
#    #fci = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None, make_rdm1=True, make_rdm2=True)
#    #f = fci.make_atom_fragment(list(range(nsite)))
#    #fci.kernel()
#    #assert f.converged
#    #with open('fci.txt', 'a') as f:
#    #    f.write("%4.1f  %16.8f\n" % (hubbard_u, fci.e_tot/nelectron))



mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, tiles=(2,1))
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

ne_target = nelectron/nsite * nimp
scmf = SelfConsistentMF(mf, 'brueckner', nelectron_target=ne_target)
scmf.kernel(2)
print(scmf.e_dmet, scmf.e_ewf, scmf.docc)
1/0


# Double site embedding:
ewf = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None, make_rdm1=True, make_rdm2=True, nelectron_target=ne_target)
for s in range(0, nsite, nimp):
    imp = list(range(s, s+nimp))
    f = ewf.make_atom_fragment(imp)
ewf.kernel()

def get_e_dmet(f):
    """DMET (projected DM) energy of fragment f."""
    c = f.c_active
    p = f.get_fragment_projector(c)
    pdm1 = einsum('ix,xj,ai,bj->ab', p, f.results.dm1, c, c)
    pdm2 = einsum('ix,xjkl,ai,bj,ck,dl->abcd', p, f.results.dm2, c, c, c, c)
    e1 = einsum('ij,ij->', f.mf.get_hcore(), pdm1)
    e2 = einsum('iiii,iiii->', f.mf.mol.get_eri(), pdm2)/2
    return e1 + e2

e_dmet = 0.0
for f in ewf.fragments:
    e = get_e_dmet(f)
    print("Fragment %d E(DM)= %16.8f" % (f.id, e))
    e_dmet += e

print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nelectron))
print("E%-11s %+16.8f Ha" % ('(DM)=', e_dmet/nelectron))
print("E%-11s %+16.8f Ha" % ('(T)=', ewf.e_tot/nelectron))
