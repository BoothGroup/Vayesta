import vayesta.dmet
import vayesta.edmet
import vayesta.lattmod
import vayesta.lattmod.bethe

import matplotlib.pyplot as plt

# Function to get mean-field for hubbard of given size and onsite repulsion.
def gen_hub(nsite, hubbard_u, ehubbard_v = 0.0, boundary='auto'):
    nelectron = nsite
    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, v_nn = ehubbard_v, boundary=boundary)
    mf = vayesta.lattmod.LatticeMF(mol)#, allocate_eri = False)
    mf.kernel()
    return mf

def fci_dmet_edmet(mf, nimp, max_boson_occ=1):
    nsite = len(mf.mo_occ)
    dmet = vayesta.dmet.DMET(mf, solver='FCI', charge_consistent=True, maxiter=1, bath_type=None)
    dmet.site_fragmentation()
    f = dmet.make_atom_fragment(list(range(nimp)))
    symfrags = f.make_tsymmetric_fragments(tvecs=[nsite // nimp, 1, 1])
    dmet.kernel()

    edmet = vayesta.edmet.EDMET(mf, solver="EBFCI", charge_consistent=True, maxiter=1, bath_type=None, 
            solver_options={'max_boson_occ': max_boson_occ})
    edmet.site_fragmentation()
    f = edmet.make_atom_fragment(list(range(nimp)))
    symfrags = f.make_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
    edmet.kernel()

    return sum(dmet.fragments[0].get_dmet_energy_contrib())/nimp, sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp

def fciqmc_edmet(mf, nimp, max_boson_occ=1):
    nsite = len(mf.mo_occ)
    edmet = vayesta.edmet.EDMET(mf, solver="EBFCIQMC", charge_consistent = True, maxiter=1, bath_type=None,
            solver_options={'max_boson_occ': max_boson_occ})
    edmet.site_fragmentation()
    f = edmet.make_atom_fragment(list(range(nimp)))
    symfrags = f.make_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
    edmet.kernel()
    return sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp

nimp = 3
U = 2.0
V = 1.0
L = 12
max_boson_occ = 1

mf = gen_hub(L, U, V)
res = fci_dmet_edmet(mf, nimp, max_boson_occ)
print(res)
