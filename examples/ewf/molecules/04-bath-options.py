import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf


mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = "cc-pVDZ"
mol.output = "pyscf.txt"
mol.build()

# Hartree-Fock
mf = pyscf.scf.HF(mol)
mf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Test the exact (full system CCSD) limit is reproduced by using bathtype='full':
emb = vayesta.ewf.EWF(mf, bath_options=dict(bathtype="full"))
emb.kernel()
assert abs(cc.e_tot - emb.e_tot) < 1e-8

# By default, the EWF class will use MP2 bath natural orbitals (BNOs), to efficently truncate
# the environment. The truncation level can be controlled via the `threshold` parameter,
# which corresponds to the natural occupation or deoccupation number of the BNOs:
bath = dict(bathtype="mp2", threshold=1e-4)
emb_mp2 = vayesta.ewf.EWF(mf, bath_options=bath)
emb_mp2.kernel()

# Note that since Vayesta version 1.0.2, the MP2 bath orbitals are constructed with additional projectors,
# which weigh the coupling to the DMET bath orbitals differently to the fragment orbitals,
# leading to a more compact set of BNOs. By default both occupied dimensions of the cluster T2 amplitudes
# are projected onto the fragment space + DMET bath space, where the projection onto the latter includes
# an additional scaling (between 0 and 1), according to the DMET entanglement spectrum.
# This weights coupling to the fragment over the DMET bath orbitals in the resulting choice
# of BNOs for a given threshold, and results in a smaller bath for a given threshold.
# Other options (not shown) include projecting out all couplings to the DMET bath space, or just projecting
# one index.
# The options below are default
emb_mp2_projdmet = vayesta.ewf.EWF(
    mf, bath_options=dict(bathtype="mp2", threshold=1e-4, project_dmet_order=2, project_dmet_mode="squared-entropy")
)
emb_mp2_projdmet.kernel()
assert np.allclose(emb_mp2_projdmet.e_tot, emb_mp2.e_tot)
# To turn off this projection (which was not used in 10.1103/PhysRevX.12.011046),
# use `project_dmet_order = 0`:
bath = dict(bathtype="mp2", threshold=1e-4, project_dmet_order=0)
emb_noproj = vayesta.ewf.EWF(mf, bath_options=bath)
emb_noproj.kernel()

# Use maximally R^2-localized bath orbitals:
# rcut is the cutoff distance in Angstrom
bath = dict(bathtype="r2", rcut=1.3)
emb_r2 = vayesta.ewf.EWF(mf, bath_options=bath)
emb_r2.kernel()

# Occupied and virtual bath can be different:
bath = dict(bathtype_occ="r2", rcut_occ=1.3, bathtype_vir="mp2", threshold_vir=1e-4)
emb_mix = vayesta.ewf.EWF(mf, bath_options=bath)
emb_mix.kernel()


print("CCSD:                                                                        E(tot)= %+16.8f Ha" % cc.e_tot)
print(
    "Embedded CCSD with full bath:                     mean cluster size= %.3f  E(tot)= %+16.8f Ha"
    % (emb.get_mean_cluster_size(), emb.e_tot)
)
print(
    "Embedded CCSD with MP2 bath:                      mean cluster size= %.3f  E(tot)= %+16.8f Ha"
    % (emb_mp2.get_mean_cluster_size(), emb_mp2.e_tot)
)
print(
    "Embedded CCSD with MP2 bath (no DMET projector):  mean cluster size= %.3f  E(tot)= %+16.8f Ha"
    % (emb_noproj.get_mean_cluster_size(), emb_noproj.e_tot)
)
print(
    "Embedded CCSD with R2 bath:                       mean cluster size= %.3f  E(tot)= %+16.8f Ha"
    % (emb_r2.get_mean_cluster_size(), emb_r2.e_tot)
)
print(
    "Embedded CCSD with mixed R2/MP2 bath:             mean cluster size= %.3f  E(tot)= %+16.8f Ha"
    % (emb_mix.get_mean_cluster_size(), emb_mix.e_tot)
)
