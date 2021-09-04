import vayesta.dmet
import vayesta.edmet
import vayesta.lattmod
import vayesta.lattmod.bethe

import matplotlib.pyplot as plt

# Function to get mean-field for hubbard of given size and onsite repulsion.
def gen_hub(nsite, hubbard_u, ehubbard_v = 0.0):
    nelectron = nsite
    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, v_nn = ehubbard_v)
    mf = vayesta.lattmod.LatticeMF(mol)#, allocate_eri = False)
    mf.kernel()
    return mf

def gen_comparison(hubbard_u, nimp, ehubbard_v = 0.0, nsite = 50):

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

    edmet = vayesta.edmet.EDMET(mf, solver="EBFCI", fragment_type='Site', charge_consistent = True,maxiter=1, bath_type=None, bos_occ_cutoff=1)
    f = edmet.make_atom_fragment(list(range(nimp)))
    # Add fragments which are translationally symmetric to f - the results of the fragment f
    # fill be automatically copied.
    symfrags = f.make_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
    edmet.kernel()
    res_edmet += [sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp]

    return sum(dmet.fragments[0].get_dmet_energy_contrib())/nimp, sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp

if __name__ == "__main__":
    res_dmet, res_edmet = [], []
    imps_to_test = [1,2,3]
    for nimp in imps_to_test:
        res = gen_comparison(2.0, nimp, 1.0, 60)
        res_dmet += [res[0]]
        res_edmet += [res[1]]
    plt.plot(imps_to_test, res_dmet, label = "DMET")
    plt.plot(imps_to_test, res_edmet, label = "EDMET")
    leg = plt.legend()
    plt.show()