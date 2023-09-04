from vayesta.misc import molecules
import vayesta.ewf
import pyscf.cc
from vayesta.core.types import WaveFunction


mol = pyscf.gto.Mole()
mol.atom = molecules.arene(6)
mol.basis = "6-31G"
mol.output = "pyscf.out"
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol).density_fit()
mf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

bath_opts = dict(bathtype="mp2", threshold=1e-4, project_dmet_order=1, project_dmet_mode="full")
bosonic_bath_opts = dict(bathtype="rpa", target_orbitals="full", local_projection="fragment", threshold=1e-3)

# Embedded CCSD calculation with bare interactions and no energy correction.
emb_bare = vayesta.ewf.EWF(mf, bath_options=bath_opts, solver="CCSD")
with emb_bare.iao_fragmentation() as f:
    with f.rotational_symmetry(6, "z"):
        f.add_atomic_fragment([0, 1])
emb_bare.kernel()

# Embedded CCSD with quasi-bosonic auxiliaries selected from among all cluster excitations.
emb = vayesta.ewf.EWF(mf, bath_options=bath_opts, bosonic_bath_options=bosonic_bath_opts, solver="CCSD-S-1-1")
with emb.iao_fragmentation() as f:
    with f.rotational_symmetry(6, "z"):
        f.add_atomic_fragment([0, 1])
emb.kernel()

# Energy from exact wavefunction projected into same space as other embedded calculations.
emb_exact = vayesta.ewf.EWF(mf, bath_options=bath_opts, _debug_wf=WaveFunction.from_pyscf(cc))
with emb_exact.iao_fragmentation() as f:
    with f.rotational_symmetry(6, "z"):
        f.add_atomic_fragment([0, 1])
emb_exact.kernel()

# Note that mRPA screening and external corrections often cancel with each other in the case of the energy.
print("E(CCSD)=                              %+16.8f Ha" % cc.e_tot)
print(
    "E(CCSD, projected locally)=           %+16.8f Ha  (external error= %+.8f Ha)"
    % (emb_exact.e_tot, emb_exact.e_tot - cc.e_tot)
)
print(
    "E(Emb. CCSD)=                         %+16.8f Ha  (internal error= %+.8f Ha)"
    % (emb_bare.e_tot, emb_bare.e_tot - emb_exact.e_tot)
)
print(
    "E(Emb. CCSD with bosons)=             %+16.8f Ha  (internal error= %+.8f Ha)"
    % (emb.e_tot, emb.e_tot - emb_exact.e_tot)
)
