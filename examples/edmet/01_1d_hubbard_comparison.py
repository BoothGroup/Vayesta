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

sites_to_try = list(range(10, 100, 30))

def gen_comparison(hubbard_u, nimp=2, ehubbard_v = 0.0):

    res_dmet = []
    res_edmet = []

    for nsite in sites_to_try:
        mf = gen_hub(nsite, hubbard_u, ehubbard_v)

        dmet = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='Site', charge_consistent=True, maxiter=1,
                                 bath_type=None)
        f = dmet.make_atom_fragment(list(range(nimp)))
        # Add fragments which are translationally symmetric to f - the results of the fragment f
        # fill be automatically copied.
        dmet.kernel()
        symfrags = f.make_tsymmetric_fragments(tvecs=[nsite // nimp, 1, 1])
        res_dmet += [sum(dmet.fragments[0].get_dmet_energy_contrib())/nimp]

        edmet = vayesta.edmet.EDMET(mf, solver="EBFCI", fragment_type='Site', charge_consistent = True,maxiter=1, bath_type=None)
        f = edmet.make_atom_fragment(list(range(nimp)))
        # Add fragments which are translationally symmetric to f - the results of the fragment f
        # fill be automatically copied.
        symfrags = f.make_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
        edmet.kernel()
        res_edmet += [sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp]


    plt.plot(sites_to_try, res_dmet, label="DMET")
    plt.plot(sites_to_try, res_edmet, label="Original EDMET")
    ax = plt.gca()
    if abs(ehubbard_v) < 1e-6:
        ax.hlines(vayesta.lattmod.bethe.hubbard1d_bethe_energy(1.0, hubbard_u),
                  sites_to_try[0], sites_to_try[-1], label="Bethe Ansatz")
    leg = plt.legend()
    plt.show()

gen_comparison(2.0, 2, 1.0)