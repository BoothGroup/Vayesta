import vayesta.ewf

from pyscf import gto, scf, lib, cc, fci
from vayesta.misc.variational_embedding import variational_params
from vayesta.misc.molecules import ring

import numpy as np


def run_ewf(natom, r, n_per_frag=1, bath_options={"bathtype": "dmet"}):
    if abs(natom / n_per_frag - natom // n_per_frag) > 1e-6:
        raise ValueError(f"Atoms per fragment ({n_per_frag}) doesn't exactly divide natoms ({natom})")

    nfrag = natom // n_per_frag

    mol = gto.M()
    mol.atom = ring("H", natom, bond_length=r)
    mol.basis = "sto-3g"
    mol.spin = 0
    mol.charge = 0
    mol.build()

    rmf = scf.RHF(mol).density_fit()
    rmf.conv_tol = 1e-12
    rmf.kernel()

    out = rmf.stability(external=True)

    rewf = vayesta.ewf.EWF(rmf, solver="FCI", bath_options=bath_options)
    with rewf.iao_fragmentation() as f:
        with f.rotational_symmetry(nfrag, axis="z"):
            f.add_atomic_fragment(atoms=list(range(n_per_frag)))
    rewf.kernel()
    return rewf, rmf


def get_wf_composite(emb, inc_mf=False):
    """Compute energy resulting from generalised eigenproblem between all local cluster wavefunctions."""
    h, s, dm = variational_params.get_wf_couplings(emb, inc_mf=inc_mf)
    w_bare, v, seig = lib.linalg_helper.safe_eigh(h, s, lindep=1e-12)
    # Return lowest eigenvalue.
    return w_bare[0]


def get_density_projected(emb, inc_mf=False):
    p_frags = [x.get_fragment_projector(x.cluster.c_active) / emb.mf.mol.nelectron for x in emb.fragments]
    barewfs = [x.results.wf for x in emb.fragments]
    wfs = [x.project(y) for x, y in zip(barewfs, p_frags)]
    h, s, dm = variational_params.get_wf_couplings(emb, emb.fragments, wfs, inc_mf=inc_mf)
    sum_energy = sum(h.reshape(-1)) / sum(s.reshape(-1))
    w, v, seig = lib.linalg_helper.safe_eigh(h, s, lindep=1e-12)
    return sum_energy, w[0]


def get_occ_projected(emb):
    p_frags = [x.get_overlap("frag|cluster-occ") for x in emb.fragments]
    p_frags = [np.dot(x.T, x) for x in p_frags]
    wfs = [x.results.wf.project_occ(y) for x, y in zip(emb.fragments, p_frags)]
    h, s, dm = variational_params.get_wf_couplings(emb, wfs=wfs, inc_mf=True)
    sum_energy = sum(h.reshape(-1)) / sum(s.reshape(-1))
    w, v, seig = lib.linalg_helper.safe_eigh(h, s, lindep=1e-12)
    return sum_energy, w[0]


def gen_comp_graph(fname):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
    fig.set_tight_layout(True)
    plot_results(fname, False, ax=ax0)
    plot_results(fname, True, ax=ax1)
    fig.set_size_inches(9, 6)
    fig.set_tight_layout(True)
    plt.draw()


def draw_comparison_error(fname1, fname2):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_tight_layout(True)
    plot_results(fname1, True, ax0)
    plot_results(fname2, True, ax1)
    fig.set_size_inches(9, 6)
    plt.show(block=False)
    plt.draw()


def plot_results(fname="results.txt", vsfci=False, ax=None, nodmenergy=True):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.subplots(1, 1)[1]
    res = np.genfromtxt(fname)
    labs = [
        "$r_{HH}/\AA$",
        "HF",
        "FCI",
        "CCSD",
        "EWF-proj-wf",
        "EWF-dm-wf",
        "NO-CAS-CI",
        "NO-dproj",
        "NO-dproj-CAS-CI",
        "NO-oproj",
        "NO-oproj-CAS-CI",
        "var-NO-FCI",
    ]

    def remove_ind(results, labels, i):
        labels = labels[:i] + labels[i + 1 :]
        results = np.concatenate([results[:, :i], results[:, i + 1 :]], axis=1)
        return results, labels

    if nodmenergy:
        res, labs = remove_ind(res, labs, 5)

    if vsfci:
        fcires = res[:, 2]
        res, labs = remove_ind(res, labs, 2)
        res[:, 1:] = (res[:, 1:].T - fcires).T
    for i in range(1, res.shape[-1]):
        ax.plot(res[:, 0], res[:, i], label=labs[i])
    ax.set_xlabel(labs[0])

    leg = ax.legend().set_draggable(True)

    if vsfci:
        ax.set_ylabel("E-$E_{FCI}/E_h$")
    else:
        ax.set_ylabel("E/$E_h$")

    plt.show(block=False)


if __name__ == "__main__":
    import sys

    nat = int(sys.argv[1])
    n_per_frag = int(sys.argv[2])
    for r in list(np.arange(0.6, 2.0, 0.1)) + list(np.arange(2.5, 10.0, 0.5)):
        emb, mf = run_ewf(nat, r, n_per_frag)
        # These calculate the standard EWF energy estimators.
        eewf_wf = emb.get_wf_energy()
        eewf_dm = emb.get_dm_energy()
        # This calculates the energy of the variationally optimal combination of the local wavefunctions in each case.
        # This uses the bare local wavefunctions...
        e_barewf = get_wf_composite(emb)
        # This uses the density projected local wavefunctions, and also returns the energy of a sum of these
        # wavefunctions.
        e_dense_proj, e_dense_opt = get_density_projected(emb)
        # This does the same, but with the local wavefunction projected via the occupied projector (as in standard EWF).
        e_occ_proj, e_occ_opt = get_occ_projected(emb)
        # This variationally optimises all coefficients in the local wavefunctions simulanteously; the value of
        # emb.fragments[x].results.wf is updated to the local portion of the global wavefunction.
        e_opt = variational_params.optimise_full_varwav(emb)

        # This computes the FCI and CCSD energies for comparison.
        myfci = fci.FCI(mf)
        efci = myfci.kernel()[0]
        mycc = cc.CCSD(mf)
        try:
            mycc.kernel()

            if mycc.converged:
                ecc = mycc.e_tot
            else:
                ecc = np.nan
        except np.linalg.LinAlgError:
            ecc = np.nan

        res = (mf.e_tot, efci, ecc, eewf_wf, eewf_dm, e_barewf, e_dense_proj, e_dense_opt, e_occ_proj, e_occ_opt, e_opt)

        with open("results.txt", "a") as f:
            f.write((f" {r:4.2f} ") + ("   {:12.8f}  " * len(res)).format(*res) + "\n")
