import vayesta
import vayesta.dmet
import vayesta.edmet
import vayesta.lattmod
import vayesta.lattmod.bethe
import matplotlib.pyplot as plt

# Function to get mean-field for hubbard of given size and onsite repulsion.
def gen_hub(nsite, hubbard_u):
    nelectron = nsite
    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u)
    mf = vayesta.lattmod.LatticeMF(mol)#, allocate_eri = False)
    mf.kernel()
    return mf

sites_to_try = list(range(10, 100, 10))

def gen_comparison(hubbard_u, nimp=2):

    res_dmet = []
    res_edmet = []

    for nsite in sites_to_try:
        mf = gen_hub(nsite, hubbard_u)

        dmet = vayesta.dmet.DMET(mf, solver='FCI', charge_consistent=True, maxiter=1,
                                 bath_options=dict(bathtype='dmet'))
        with dmet.site_fragmentation() as f:
            frag = f.add_atomic_fragment(list(range(nimp)))
        # Add fragments which are translationally symmetric to f - the results of the fragment f
        # fill be automatically copied.
        dmet.kernel()
        symfrags = frag.make_tsymmetric_fragments(tvecs=[nsite // nimp, 1, 1])
        res_dmet += [sum(dmet.fragments[0].get_dmet_energy_contrib())/nimp]

        edmet = vayesta.edmet.EDMET(mf, solver="FCI", charge_consistent = True,maxiter=1, bath_options=dict(bathtype='dmet'))
        with edmet.site_fragmentation() as f:
            frag = f.add_atomic_fragment(list(range(nimp)))
        # Add fragments which are translationally symmetric to f - the results of the fragment f
        # fill be automatically copied.
        symfrags = frag.make_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
        edmet.kernel()
        res_edmet += [sum(edmet.fragments[0].get_edmet_energy_contrib())/nimp]


    plt.plot(sites_to_try, res_dmet, label="DMET")
    plt.plot(sites_to_try, res_edmet, label="Original EDMET")
    ax = plt.gca()
    ax.hlines(vayesta.lattmod.bethe.hubbard1d_bethe_energy(1.0, hubbard_u), sites_to_try[0], sites_to_try[-1], label="Bethe Ansatz")
    leg = plt.legend()
    plt.show()

gen_comparison(10.0, 2)
