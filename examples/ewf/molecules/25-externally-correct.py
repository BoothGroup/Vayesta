import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.fci
import vayesta
import vayesta.ewf
from vayesta.misc import molecules

mol = pyscf.gto.Mole()
mol.atom = """
Se	0.0000	0.0000	0.2807
O 	0.0000	1.3464	-0.5965
O 	0.0000	-1.3464	-0.5965
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# 1) Consider setup where you have a complete CCSD, externally corrected by local atomic fragment+DMET FCI clusters
emb = vayesta.ewf.EWF(mf)
with emb.iao_fragmentation() as f:
    # Add all atomic FCI fragments with DMET bath
    # Store the FCI wave functions as CCSDTQ types, so they can be used for correction later.
    fci_frags = f.add_all_atomic_fragments(solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSDTQ')
    # Add single 'complete' CCSD fragment covering all IAOs, but set as inactive
    ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'), active=False)
emb.kernel()
# Switch active <-> inactive fragments, and run the full system CCSD fragment, applying external correction
for fci_frag in fci_frags:
    fci_frag.active = False
ccsd_frag.active = True
# Setup the external correction from the FCI fragments.
# Two main options: correction_type='external-ccsdv' and 'external-fciv'. For the important T3 * V contribution to the
#   T2 amplitudes, this determines whether the V is expressed in the FCI or CCSD cluster space. The CCSD cluster is
#   larger, and hence this is likely to be better, as the correction is longer-ranged (though slightly more expensive).
#   The other option is 'projectors', which should be an integer between 0 and 2 (inclusive).
#   The larger the number, the more fragment projectors are applied to the correcting T2 contributions, and less
#   'bath' correlation from the FCI clusters is used as a constraint in the external correction of the CCSD clusters.
ccsd_frag.add_external_corrections(fci_frags, correction_type='external-ccsdv', projectors=1)
# Run kernel again
emb.kernel()
print('Total energy from full system CCSD tailored (CCSD Coulomb interaction) by atomic FCI fragments (projectors=1): {}'.format(emb.e_tot))

# 2) Now, we also fragment the CCSD spaces, and use BNOs. These CCSD fragments are individually externally corrected from the FCI clusters.
# Similar set up, but we now have multiple CCSD clusters.
emb = vayesta.ewf.EWF(mf)
with emb.iao_fragmentation() as f:
    fci_frags = f.add_all_atomic_fragments(solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSDTQ')
    ccsd_frags = f.add_all_atomic_fragments(solver='CCSD', bath_options=dict(bathtype='mp2', threshold=1.e-5), active=False)
emb.kernel()
for fci_frag in fci_frags:
    fci_frag.active = False
for ccsd_frag in ccsd_frags:
    ccsd_frag.active = True
    # Now add external corrections to all CCSD clusters, and use 'external-fciv' correction, with 2 projectors
    ccsd_frag.add_external_corrections(fci_frags, correction_type='external-fciv', projectors=2)
emb.kernel()
print('Total energy from embedded CCSD tailored (FCI Coulomb interaction) by atomic FCI fragments (projectors=2): {}'.format(emb.e_tot))
