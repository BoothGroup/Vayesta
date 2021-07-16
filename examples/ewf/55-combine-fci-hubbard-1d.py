import numpy as np

import vayesta
import vayesta.lattmod
import vayesta.ewf

nsite = 2
nelectron = nsite
hubbard_u = 2.0
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

ecc = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site')
f1 = ecc.make_atom_fragment(0)
f2 = ecc.make_atom_fragment(1)
results1 = f1.kernel(np.inf)
results2 = f2.kernel(np.inf)

# results have attributes 'c0', 'c1', 'c2'
# 1) Get intermediately normalized c1, c2
c1_1 = results1.c1 / results1.c0        # C_i^a
c2_1 = results1.c2 / results1.c0        # C_ij^ab
c1_2 = results2.c1 / results2.c0
c2_2 = results2.c2 / results2.c0

# 2) Get fragment projector
p1 = f1.get_fragment_projector(f1.c_active_occ)
p2 = f1.get_fragment_projector(f2.c_active_occ)

# 3) Project c1 and c2 in first occupied index (apply p1 / p2)



# 4) Transform each c1, c2 to a full, common basis (HF MO basis OR site basis?)
c1_occ = f1.c_active_occ    # (site, occupied orbital)
c1_vir = f1.c_active_vir    # (site, virtual orbital)


# 5) Combine (add) to a single c1, c2
c1 = np.zeros((nsite, nsite))
c2 = np.zeros(4*[nsite])
#...


# 6) Use full c1, c2 to tailor a CCSD calculation
# TODO: Tailored CC
ecc = vayesta.ewf.EWF(mf, solver='CCSD', bno_threshold=-np.inf, fragment_type='Site')
ecc.make_atom_fragment(list(range(nsite)))
ecc.kernel()

print("E%-11s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
