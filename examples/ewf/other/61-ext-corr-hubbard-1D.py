import pyscf
import pyscf.cc
import pyscf.fci
import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 10
nelectron = nsite
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
    # Add single 'complete' CCSD fragment covering all sites (currently inactive)
    ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'), solver_options=dict(solve_lambda=False, init_guess='CISD'), active=False)
for fci_sym_frag in fci_frags[0].add_tsymmetric_fragments(tvecs=[5, 1, 1]):
    # Add symmetry-derived FCI fragments to avoid multiple calculations
    fci_frags.append(fci_sym_frag)
emb.kernel()
assert(emb.converged)

def ext_corr(emb, fci_frags, ccsd_frag, mode='external-ccsdv', projectors=0):
    """Setup the external correction from the CCSD fragments.
    Two main options: correction_type='external-ccsdv' and 'external-fciv'. For the important T3 * V contribution to the
    T2 amplitudes, this determines whether the V is expressed in the FCI or CCSD cluster space. The CCSD cluster is
    larger, and hence this is likely to be better, as the correction is longer-ranged (though slightly more expensive).
    The other option is 'projectors', which should be an integer between 0 and 2 (inclusive).
    The larger the number, the more fragment projectors are applied to the correcting T2 contributions, and less
    'bath' correlation from the FCI clusters is used as a constraint in the external correction of the CCSD clusters.
    NOTE that with multiple FCI fragments providing constraints and overlapping bath spaces, proj=0 will
    overcount the correction, so do not use with multiple FCI clusters. It will not e.g. tend to the right answer as
    the FCI bath space becomes complete. Only use with a single FCI fragment.
    """

    ccsd_frag.clear_external_corrections() # Clear any previous corrections applied
    for fci_frag in fci_frags:
        fci_frag.active = False
    ccsd_frag.active = True
    ccsd_frag.add_external_corrections(fci_frags, correction_type=mode, projectors=projectors)
    # Run kernel again
    emb.kernel()
    return emb.e_tot, emb.converged

e_extcorr = []
extcorr_conv = []
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='external-ccsdv', projectors=0)
e_extcorr.append(e); extcorr_conv.append(conv)
# Change number of projectors and/or Coulomb type, and re-run to get change
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='external-ccsdv', projectors=0)
e_extcorr.append(e); extcorr_conv.append(conv)
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='external-ccsdv', projectors=1)
e_extcorr.append(e); extcorr_conv.append(conv)
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='external-ccsdv', projectors=2)
e_extcorr.append(e); extcorr_conv.append(conv)
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='external-fciv', projectors=0)
e_extcorr.append(e); extcorr_conv.append(conv)
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='external-fciv', projectors=1)
e_extcorr.append(e); extcorr_conv.append(conv)
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='external-fciv', projectors=1)
e_extcorr.append(e); extcorr_conv.append(conv)

# Compare to a simpler tailoring
e_tailor = []
tailor_conv = []
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='tailor', projectors=0)
e_tailor.append(e) ; tailor_conv.append(conv)
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='tailor', projectors=1)
e_tailor.append(e) ; tailor_conv.append(conv)
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='tailor', projectors=2)
e_tailor.append(e) ; tailor_conv.append(conv)

# Compare to a delta-tailoring, where the correction is the difference between full-system
# CCSD and CCSD in the FCI cluster.
e_dtailor = []
dtailor_conv = []
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='delta-tailor', projectors=0)
e_dtailor.append(e) ; dtailor_conv.append(conv)
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='delta-tailor', projectors=1)
e_dtailor.append(e) ; dtailor_conv.append(conv)
e, conv = ext_corr(emb, fci_frags, ccsd_frag, mode='delta-tailor', projectors=2)
e_dtailor.append(e) ; dtailor_conv.append(conv)

print("E(MF)=                                   %+16.8f Ha, conv = %s" % (mf.e_tot/nsite, mf.converged))
print("E(CCSD)=                                 %+16.8f Ha, conv = %s" % (cc.e_tot/nsite, cc.converged))
print("E(FCI)=                                  %+16.8f Ha, conv = %s" % (fci.e_tot/nsite, fci.converged))
print("E(Emb. FCI, 2-site)=                     %+16.8f Ha, conv = %s" % (emb_simp.e_tot/nsite, emb_simp.converged))
print("E(EC-CCSD, 2-site FCI, 1 proj, ccsd V)=  %+16.8f Ha, conv = %s" % ((e_extcorr[1]/nsite), extcorr_conv[1]))
print("E(EC-CCSD, 2-site FCI, 2 proj, ccsd V)=  %+16.8f Ha, conv = %s" % ((e_extcorr[2]/nsite), extcorr_conv[2]))
print("E(EC-CCSD, 2-site FCI, 1 proj, fci V)=   %+16.8f Ha, conv = %s" % ((e_extcorr[4]/nsite), extcorr_conv[4]))
print("E(EC-CCSD, 2-site FCI, 2 proj, fci V)=   %+16.8f Ha, conv = %s" % ((e_extcorr[5]/nsite), extcorr_conv[5]))
print("E(T-CCSD, 2-site FCI, 1 proj)=           %+16.8f Ha, conv = %s" % ((e_tailor[1]/nsite), tailor_conv[1]))
print("E(T-CCSD, 2-site FCI, 2 proj)=           %+16.8f Ha, conv = %s" % ((e_tailor[2]/nsite), tailor_conv[2]))
print("E(DT-CCSD, 2-site FCI, 1 proj)=          %+16.8f Ha, conv = %s" % ((e_dtailor[1]/nsite), dtailor_conv[1]))
print("E(DT-CCSD, 2-site FCI, 2 proj)=          %+16.8f Ha, conv = %s" % ((e_dtailor[2]/nsite), dtailor_conv[2]))
