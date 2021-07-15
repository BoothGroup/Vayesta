import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 10
nelectron = 6
nimp = 1 
hubbard_u = 12.0
filling = nelectron/nsite
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

def get_e_dmet(dm1, dm2):
    """DMET (projected DM) energy"""
    imp = np.s_[:nimp]
    e1 = np.einsum('ij,ij->', mf.get_hcore()[imp], dm1[imp])
    e2 = hubbard_u * np.einsum('iiii->', dm2[imp,imp,imp,imp])
    return e1 + e2

# Without chemical potential optimization
ecc = vayesta.ewf.EWF(mf, bno_threshold=np.inf, fragment_type='Site', solver='FCI', make_rdm1=True, make_rdm2=True)
f = ecc.make_atom_fragment(list(range(nimp)), sym_factor=nsite/nimp)
ecc.kernel()
# c transforms to fragment site(s)
c = f.c_active
dm1 = np.einsum('ij,ai,bj->ab', f.results.dm1, c, c)
dm2 = np.einsum('ijkl,ai,bj,ck,dl->abcd', f.results.dm2, c, c, c, c)/2

# With chemical potential optimization (set nelectron_target)
ecc_cpt = vayesta.ewf.EWF(mf, bno_threshold=np.inf, fragment_type='Site', solver='FCI', make_rdm1=True, make_rdm2=True)
f = ecc_cpt.make_atom_fragment(list(range(nimp)), sym_factor=nsite/nimp, nelectron_target=nimp*filling)
ecc_cpt.kernel()
# c transforms to fragment site(s)
c = f.c_active
dm1_cpt = np.einsum('ij,ai,bj->ab', f.results.dm1, c, c)
dm2_cpt = np.einsum('ijkl,ai,bj,ck,dl->abcd', f.results.dm2, c, c, c, c)/2

# Exact energy
fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=-np.inf, fragment_type='Site')
fci.make_atom_fragment(list(range(nsite)), name='lattice')
fci.kernel()


# --- OUTPUT
print("Filling= %f" % filling)
print("1DM without/with cpt: %f %f" % (dm1[0,0], dm1_cpt[0,0]))
print("2DM without/with cpt: %f %f" % (dm2[0,0,0,0], dm2_cpt[0,0,0,0]))

print("ENERGIES")
print("--------")
print("E%-20s %+16.8f Ha" % ('(MF)=',              mf.e_tot/nelectron))
print("E%-20s %+16.8f Ha" % ('(proj-DM)[cpt=0]=', get_e_dmet(dm1, dm2)*nsite/(nimp*nelectron)))
print("E%-20s %+16.8f Ha" % ('(proj-DM)[opt. cpt]=',        get_e_dmet(dm1_cpt, dm2_cpt)*nsite/(nimp*nelectron)))
print("E%-20s %+16.8f Ha" % ('(proj-T)[cpt=0]=',  ecc.e_tot/nelectron))
print("E%-20s %+16.8f Ha" % ('(proj-T)[opt. cpt]=',         ecc_cpt.e_tot/nelectron))
print("E%-20s %+16.8f Ha" % ('(EXACT)=', fci.e_tot/nelectron))
