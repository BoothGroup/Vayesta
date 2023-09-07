from vayesta.misc import molecules
import vayesta.edmet
from pyscf import gto, scf, cc


# Use ethane as a simple trial system.
mol = gto.Mole()
mol.atom = molecules.alkane(2)
mol.basis = "sto-3g"
mol.build()

mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-10
mf.kernel()

# Can generate bosons using either RPA couplings or direct projection of the Hamiltonian into the bosonic space.

rdfedmet_drpa_bos = vayesta.edmet.EDMET(
    mf,
    solver="CCSD-S-1-1",
    bath_options=dict(dmet_threshold=1e-12),
    bosonic_interaction="direct",
    oneshot=True,
    make_dd_moments=False,
)
with rdfedmet_drpa_bos.iao_fragmentation() as f:
    f.add_all_atomic_fragments()
rdfedmet_drpa_bos.kernel()

rdfedmet_qba_directbos = vayesta.edmet.EDMET(
    mf,
    solver="CCSD-S-1-1",
    bath_options=dict(dmet_threshold=1e-12),
    bosonic_interaction="qba",
    oneshot=True,
    make_dd_moments=False,
)
with rdfedmet_qba_directbos.iao_fragmentation() as f:
    f.add_all_atomic_fragments()
rdfedmet_qba_directbos.kernel()

rdfedmet_qba = vayesta.edmet.EDMET(
    mf,
    solver="CCSD-S-1-1",
    bath_options=dict(dmet_threshold=1e-12),
    bosonic_interaction="qba_bos_ex",
    oneshot=True,
    make_dd_moments=False,
)
with rdfedmet_qba.iao_fragmentation() as f:
    f.add_all_atomic_fragments()
rdfedmet_qba.kernel()

myccsd = cc.CCSD(mf)
ecc = myccsd.kernel()[0]

print("Results using dRPA-defined quasibosonic couplings:")
rdfedmet_drpa_bos.print_results()
print("Carbon Bosonic Frequencies:")
print(rdfedmet_drpa_bos.fragments[0].bos_freqs)

print("Results using projected quasibosonic couplings, excluding exchange in bosons:")
rdfedmet_qba_directbos.print_results()
print(rdfedmet_qba_directbos.fragments[0].bos_freqs)

print("Results using projected quasibosonic couplings, including exchange in bosons:")
rdfedmet_qba.print_results()
print(rdfedmet_qba.fragments[0].bos_freqs)
