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

def gen_comparison(hubbard_u, nimp, ehubbard_v = 0.0, nsite = 50, bos_occ_cutoff=1):
    assert 0

    res_dmet = []
    res_edmet = []

    mf = gen_hub(nsite, hubbard_u, ehubbard_v)

    dmet = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='Site', charge_consistent=True, maxiter=1,
                             bath_type=None)
    f = dmet.make_atom_fragment(list(range(nimp)))
    # Add fragments which are translationally symmetric to f - the results of the fragment f
    # fill be automatically copied.
    dmet.kernel()
    symfrags = f.make_tsymmetric_fragments(tvecs=[nsite // nimp, 1, 1])
    res_dmet += [sum(dmet.fragments[0].get_dmet_energy_contrib())/nimp]

    edmet = vayesta.edmet.EDMET(mf, solver="EBFCI", fragment_type='Site', charge_consistent = True,maxiter=1, bath_type=None, bos_occ_cutoff=bos_occ_cutoff)
    f = edmet.make_atom_fragment(list(range(nimp)))
    # Add fragments which are translationally symmetric to f - the results of the fragment f
    # fill be automatically copied.
    symfrags = f.make_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
    edmet.kernel()
    res_edmet += [sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp]

    return sum(dmet.fragments[0].get_dmet_energy_contrib())/nimp, sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp

def gen_fciqmc_edmet(hubbard_u, nimp, ehubbard_v = 0.0, nsite = 50, bos_occ_cutoff=1):
    res = []

    mf = gen_hub(nsite, hubbard_u, ehubbard_v)

    edmet = vayesta.edmet.EDMET(mf, solver="EBFCI", fragment_type='Site', charge_consistent = True,maxiter=1, bath_type=None, bos_occ_cutoff=bos_occ_cutoff)


    f = edmet.make_atom_fragment(list(range(nimp)))
    # Add fragments which are translationally symmetric to f - the results of the fragment f
    # fill be automatically copied.
    symfrags = f.make_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
    edmet.kernel()
    res += [sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp]

    return sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp

res_dmet, res_edmet = [], []
imps_to_test = [1,2,3]
U = 2.0
V = 1.0
L = 12
res = gen_comparison(U, 3, V, L)
print(res)
assert 0
#for nimp in imps_to_test:
#    res = gen_comparison(U, nimp, V, L)
#    res_dmet += [res[0]]
#    res_edmet += [res[1]]
#res_fciqmc = gen_fciqmc_edmet(U, 3, V, L, 1)
#print(res_dmet)
#print(res_edmet)
#print(f'EDEMT energy {res_fciqmc}')
#plt.plot(imps_to_test, res_dmet, label = "DMET")
#plt.plot(imps_to_test, res_edmet, label = "EDMET")
#leg = plt.legend()
#plt.savefig('plot.pdf')

U = 2.0
V = 1.0
L = 42
mf = gen_hub(L, U, V)
from vayesta import rpa, logging
rpa = vayesta.rpa.dRPA(mf, logging)
res = rpa.kernel()
print((mf.e_tot + res)/L)



'''
# setup.py
edmet = vayesta.edmet.EDMET(mf, solver="EBFCI", fragment_type='Site', charge_consistent = True,maxiter=1, bath_type=None, bos_occ_cutoff=bos_occ_cutoff, to_archive=True, from_archive=False)
# post.py
edmet = vayesta.edmet.EDMET(solver="EBFCI", fragment_type='Site', charge_consistent = True,maxiter=1, bath_type=None, bos_occ_cutoff=bos_occ_cutoff, to_archive=False, from_archive=True)
# self-consistent
while not conv:
    edmet = vayesta.edmet.EDMET(solver="EBFCI", fragment_type='Site', charge_consistent = True,maxiter=1, bath_type=None, bos_occ_cutoff=bos_occ_cutoff, to_archive=True, from_archive=True)
'''
