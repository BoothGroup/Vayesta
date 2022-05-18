import matplotlib.pyplot as plt

import numpy as np

from vayesta.core.util import *



def get_coulomb_interactions(bfreqs, couplings, freqdat=(0.0, 10.0, 1e-2), eta=1e-1, offset=0.0, switch_index=False):

    ind = "npq,npq" if switch_index else "npp,nqq"

    min_freq, max_freq, freqdensity = freqdat
    try:
        compressed_couplings = einsum(ind+"->n", couplings[0], couplings[1])
    except (ValueError, IndexError):
        compressed_couplings = einsum(ind+"->n", couplings, couplings)
    def get_freq_val(freqval):
        return sum(np.multiply(compressed_couplings, (bfreqs - freqval - 1.0j * eta)**(-1)))

    freqs = np.arange(min_freq, max_freq, freqdensity)
    res = [offset + get_freq_val(x) for x in freqs]
    return np.array(freqs), np.array(res)


def plot_coulomb_interaction(bfreqs, couplings, freqdat=(0.0, 10.0, 1e-2), eta=1e-1, offset=0.0):
    freqs, res = get_coulomb_interactions(bfreqs, couplings, *freqdat, eta, offset)

    fig, ax = plt.figure()
    ax.plot(freqs, res.real, label="Real")
    ax.plot(freqs, -res.imag, label="Imag")
    plt.show(block=False)
    return fig, ax

def gen_from_frag(f, freqdat=(0.0, 10.0, 1e-2), eta=1e-1, switch_index=False):

    bfreqs = f.bos_freqs
    couplings = f.couplings
    ind = "pqpq" if switch_index else "ppqq"

    if f.base.is_uhf:

        offset_aa = einsum(ind+"->", f.base.get_eris_array(f.cluster.c_active[0]))
        offset_ab = einsum(ind+"->", f.base.get_eris_array((f.cluster.c_active[0], f.cluster.c_active[0],
                                         f.cluster.c_active[1], f.cluster.c_active[1])))
        offset_bb = einsum(ind+"->", f.base.get_eris_array(f.cluster.c_active[1]))

        freqsaa, resaa = get_coulomb_interactions(bfreqs, couplings[0], freqdat, eta, offset_aa, switch_index)
        freqsbb, resbb = get_coulomb_interactions(bfreqs, couplings[1], freqdat, eta, offset_bb, switch_index)
        freqsab, resab = get_coulomb_interactions(bfreqs, (couplings[0], couplings[1]), freqdat, eta, offset_ab, switch_index)

        return freqsaa, (resaa, resab, resbb)

    else:
        offset = einsum(ind+"->", f.base.get_eris_array(f.cluster.c_active))
        freqsaa, resaa = get_coulomb_interactions(bfreqs, couplings[0], freqdat, eta, offset, switch_index)
        freqsbb, resbb = get_coulomb_interactions(bfreqs, couplings[1], freqdat, eta, offset, switch_index)
        freqsab, resab = get_coulomb_interactions(bfreqs, (couplings[0], couplings[1]), freqdat, eta, offset, switch_index)
        return freqsaa, (resaa, resab, resbb)


def run_example():
    from vayesta.misc import molecules
    import vayesta.edmet
    from pyscf import gto, scf, dft

    mol = gto.M()
    mol.atom = molecules.arene(6)
    mol.basis ="STO-3G"
    mol.build()

    rmf = scf.RHF(mol)
    rdfmf = rmf.density_fit()
    rdfmf.conv_tol=1e-10
    rdfmf.kernel()

    emb = vayesta.edmet.EDMET(rdfmf, solver="EBCCSD", dmet_threshold=1e-12, bosonic_interaction="qba_bos_ex",
                                   oneshot=True, make_dd_moments=False)
    emb.iao_fragmentation()
    # Single site fragmentation
    #emb.add_atomic_fragment([0], orbital_filter=["2pz"])
    # This would construct a fragment of all the C 2pz orbitals.
    emb.add_atomic_fragment([0,2,4,6,8,10], orbital_filter=["2pz"])
    emb.kernel()

    return emb

if __name__ == "__main__":
    emb = run_example()
    freqs, dat = gen_from_frag(emb.fragments[0], eta=5e-2, freqdat=(0.0, 4.0, 1e-2), switch_index=True)
    # dat has separate spin components, but as we're doing RHF this all are equivalent.
    plt.plot(freqs, dat[0].real, label="$\\mathcal{R}(W)$")
    plt.plot(freqs, dat[0].imag, label="$\\mathcal{I}(W)$")
    leg = plt.legend()
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_xlabel("$\\omega/E_h$")
    ax.set_xlim(0.0, 4.0)
    fig.set_size_inches(6,4)
    plt.show(block=True)

