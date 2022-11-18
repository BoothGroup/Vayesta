import pyscf
import pyscf.cc
import pyscf.fci
import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 10
nelectron = nsite + 4
hubbard_u = 4.0
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u)
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()
assert(mf.converged)

# Reference full system CCSD and FCI
cc = pyscf.cc.CCSD(mf)
cc.kernel()
fci = pyscf.fci.FCI(mf)
fci.threads = 1
fci.conv_tol = 1e-12
fci.davidson_only = True
fci.kernel()

# Perform embedded FCI with two sites
emb_simp = vayesta.ewf.EWF(mf, solver='FCI', bath_options=dict(bathtype='dmet'))
with emb_simp.site_fragmentation() as f:
    f.add_atomic_fragment([0, 1], sym_factor=nsite/2, nelectron_target=2*nelectron/nsite)
emb_simp.kernel()

# Perform full system CCSD, externally corrected by two-site+DMET bath FCI level clusters
emb = vayesta.ewf.EWF(mf)
fci_frags = []
with emb.site_fragmentation() as f:
    # Set up a two-site FCI fragmentation of full system
    # Ensure the right number of electrons on each fragment space of the FCI calculation.
    fci_frags.append(f.add_atomic_fragment([0, 1], solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSDTQ', nelectron_target=2*nelectron/nsite))
    fci_frags.append(f.add_atomic_fragment([2, 3], solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSDTQ', nelectron_target=2*nelectron/nsite))
    fci_frags.append(f.add_atomic_fragment([4, 5], solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSDTQ', nelectron_target=2*nelectron/nsite))
    fci_frags.append(f.add_atomic_fragment([6, 7], solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSDTQ', nelectron_target=2*nelectron/nsite))
    fci_frags.append(f.add_atomic_fragment([8, 9], solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSDTQ', nelectron_target=2*nelectron/nsite))
    # Add single 'complete' CCSD fragment covering all sites (currently inactive)
    ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'), solver_options=dict(solve_lambda=False), active=False)
# FCI fragment symmetry not available for external correction
#fci_frags = fci_frags.add_tsymmetric_fragments(tvecs=[5, 1, 1])
emb.kernel()

for fci_frag in fci_frags:
    fci_frag.active = False
ccsd_frag.active = True
# Setup the external correction from the CCSD fragments.
# Two main options: correction_type='external-ccsdv' and 'external-fciv'. For the important T3 * V contribution to the
#   T2 amplitudes, this determines whether the V is expressed in the FCI or CCSD cluster space. The CCSD cluster is
#   larger, and hence this is likely to be better, as the correction is longer-ranged (though slightly more expensive).
#   NOTE however that the Hubbard model only has local interactions, so these should be identical.
#   The other option is 'projectors', which should be an integer between 0 and 2 (inclusive).
#   The larger the number, the more fragment projectors are applied to the correcting T2 contributions, and less
#   'bath' correlation from the FCI clusters is used as a constraint in the external correction of the CCSD clusters.
ccsd_frag.add_external_corrections(fci_frags, correction_type='external-ccsdv', projectors=0)
# Run kernel again
emb.kernel()
e_ccsdv = []
e_ccsdv.append(emb.e_tot)

# Change number of projectors and/or Coulomb type, and re-run
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='external-ccsdv', projectors=1)
emb.kernel()
e_ccsdv.append(emb.e_tot)
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='external-ccsdv', projectors=2)
emb.kernel()
e_ccsdv.append(emb.e_tot)
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='external-ccsdv', projectors=0)
emb.kernel()
e_fciv = []
e_fciv.append(emb.e_tot)
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='external-ccsdv', projectors=1)
emb.kernel()
e_fciv.append(emb.e_tot)
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='external-ccsdv', projectors=2)
emb.kernel()
e_fciv.append(emb.e_tot)

# Compare to a simpler tailoring
e_tailor = []
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='tailor', projectors=0)
emb.kernel()
e_tailor.append(emb.e_tot)
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='tailor', projectors=1)
emb.kernel()
e_tailor.append(emb.e_tot)
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='tailor', projectors=2)
emb.kernel()
e_tailor.append(emb.e_tot)

# Compare to a delta-tailoring, where the correction is the difference between full-system
# CCSD and CCSD in the FCI cluster.
e_dtailor = []
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='delta-tailor', projectors=0)
emb.kernel()
e_dtailor.append(emb.e_tot)
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='delta-tailor', projectors=1)
emb.kernel()
e_dtailor.append(emb.e_tot)
ccsd_frag.clear_external_corrections()
ccsd_frag.add_external_corrections(fci_frags, correction_type='delta-tailor', projectors=2)
emb.kernel()
e_dtailor.append(emb.e_tot)

print("E(MF)=                                   %+16.8f Ha" % (mf.e_tot/nsite))
print("E(CCSD)=                                 %+16.8f Ha" % (cc.e_tot/nsite))
print("E(FCI)=                                  %+16.8f Ha" % (fci.e_tot/nsite))
print("E(Emb. FCI, 2-site)=                     %+16.8f Ha" % (emb_simp.e_tot/nsite))
print("E(EC-CCSD, 2-site FCI, 0 proj, ccsd V)=  %+16.8f Ha" % (e_ccsdv[0]/nsite))
print("E(EC-CCSD, 2-site FCI, 1 proj, ccsd V)=  %+16.8f Ha" % (e_ccsdv[1]/nsite))
print("E(EC-CCSD, 2-site FCI, 2 proj, ccsd V)=  %+16.8f Ha" % (e_ccsdv[2]/nsite))
print("E(EC-CCSD, 2-site FCI, 0 proj, fci V)=   %+16.8f Ha" % (e_fciv[0]/nsite))
print("E(EC-CCSD, 2-site FCI, 1 proj, fci V)=   %+16.8f Ha" % (e_fciv[1]/nsite))
print("E(EC-CCSD, 2-site FCI, 2 proj, fci V)=   %+16.8f Ha" % (e_fciv[2]/nsite))
print("E(T-CCSD, 2-site FCI, 0 proj)=           %+16.8f Ha" % (e_tailor[0]/nsite))
print("E(T-CCSD, 2-site FCI, 1 proj)=           %+16.8f Ha" % (e_tailor[1]/nsite))
print("E(T-CCSD, 2-site FCI, 2 proj)=           %+16.8f Ha" % (e_tailor[2]/nsite))
print("E(DT-CCSD, 2-site FCI, 0 proj)=          %+16.8f Ha" % (e_dtailor[0]/nsite))
print("E(DT-CCSD, 2-site FCI, 1 proj)=          %+16.8f Ha" % (e_dtailor[1]/nsite))
print("E(DT-CCSD, 2-site FCI, 2 proj)=          %+16.8f Ha" % (e_dtailor[2]/nsite))
