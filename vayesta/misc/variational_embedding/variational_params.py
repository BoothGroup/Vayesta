import numpy as np
import scipy.linalg
from pyscf import gto, scf, mcscf, ao2mo, lib
from vayesta.core.types import RFCI_WaveFunction, SpatialOrbitals
from .calc_nonorth_couplings import calc_full_couplings, calc_ci_elements, calc_full_couplings_uhf

def optimise_full_varwav(emb, lindep=1e-12):
    orig_linear = emb.get_e_corr("wf") + emb.e_mf
    h, s, nstates = gen_full_ham(emb, returnsizes=True)

    w, v, seig = lib.linalg_helper.safe_eigh(h, s, lindep=lindep)

    ci0 = v[:, 0]

    orig_ci = [f.results.wf.ci for f in emb.fragments]
    orig_ecorr = [f.results.e_corr for f in emb.fragments]
    for x in orig_ci:
        print(x)
    i = 0
    for ns, f in zip(nstates, emb.fragments):
        if f.sym_parent is None:
            print("orig:")
            c = ci0[i:i + ns].reshape(f.results.wf.ci.shape)
            print("new:")
            f.results.wf.ci = c
            ci = f.results.wf.as_cisd(c0=1.0)
            print("eris:",f._eris)
            es, ed, f.results.e_corr = f.get_fragment_energy(ci.c1, ci.c2, eris=f._eris)
            i += ns
            print("After replacement:")

    elinear = emb.get_e_corr("wf") +emb.e_mf

    for f, ci, ec in zip(emb.fragments, orig_ci, orig_ecorr):
        if f.sym_parent is None:
            f.results.wf.ci = ci
            ci = f.results.wf.as_cisd(c0=1.0)
            es, ed, f.results.e_corr = f.get_fragment_energy(ci.c1, ci.c2, eris=f._eris)

    print("Energy diff without projection: {:16.12e}".format(orig_linear - emb.get_e_corr("wf") - emb.e_mf))
    print(elinear, emb.get_e_corr("wf") + emb.e_mf)
    return w[0], elinear

def gen_full_ham(emb, fs=None, returnsizes=False):

    if fs is None:
        fs = emb.fragments

    wfs = [x.results.wf for x in fs]
    mos = [x.cluster.c_total for x in fs]

    h1e = emb.get_hcore()
    h2e = emb.get_eris_array(np.eye(emb.nao)).reshape((emb.nao**2, emb.nao**2))
    ovlp = emb.get_ovlp()
    nmo, nocc = emb.nao, emb.nocc

    nact = [x.mo.norb for x in wfs]
    ncore = [emb.nocc - x.mo.nocc for x in wfs]



    fullham = np.zeros((len(fs), len(fs)), dtype=object)
    fullovlp = np.zeros_like(fullham)
    for i in range(len(fs)):
        for j in range(i, len(fs)):
            fullham[i, j], fullovlp[i, j] = calc_full_couplings(h1e, h2e, ovlp, mos[i], mos[j], nmo, nocc, nact[i], nact[j], ncore[i], ncore[j], enuc=emb.e_nuc)
            #
            #testham, testovlp = calc_full_couplings_uhf(h1e, h2e, ovlp, mos[i], mos[j], nmo, nocc, nact[i], nact[j], ncore[i], ncore[j], enuc=emb.e_nuc)
            #print("!!!!!!",abs(fullham[i,j] - testham).max(), abs(fullovlp[i,j] - testovlp).max())
            fullham[j, i] = fullham[i, j].T
            fullovlp[j, i] = fullovlp[i, j].T

    nstates = sum([x.shape[1] for x in fullham[0]])
    h = np.zeros((nstates, nstates))
    s = np.zeros_like(h)

    ix = 0

    for (hx, sx) in zip(fullham, fullovlp):
        iy = 0
        for (hloc, sloc) in zip(hx, sx):
            dx, dy = hloc.shape
            h[ix:ix+dx, iy:iy+dy] = hloc
            s[ix:ix + dx, iy:iy + dy] = sloc
            iy += dy
        ix += dx

    if returnsizes:
        return h, s, [x.shape[1] for x in fullham[0]]
    else:
        return h, s

def get_wf_couplings(emb, fs=None, wfs=None, mos=None, inc_mf=False):
    """Calculate the hamiltonian element between multiple FCI wavefunctions in different fragments.
    This requires the CI coefficients and the basis set in which they are defined.

    If `inc_bare` is True, then the mean-field determinant of emb will be included in the calculation.
    """

    if fs is None:
        fs = emb.fragments
    if wfs is None:
        wfs = [x.results.wf for x in fs]
    if mos is None:
        mos = [x.cluster.c_total for x in fs]

    if len(fs) != len(wfs):
        raise ValueError("Number of fragments and wavefunctions provided don't match.")

    h1e = emb.get_hcore()
    h2e = emb.get_eris_array(np.eye(emb.nao)).reshape((emb.nao**2, emb.nao**2))
    ovlp = emb.get_ovlp()
    nmo, nocc = emb.nao, emb.nocc

    ci = [x.ci for x in wfs]
    nact = [x.mo.norb for x in wfs]
    ncore = [emb.nocc - x.mo.nocc for x in wfs]

    if inc_mf:
        mfwf = np.zeros_like(ci[-1])
        mfwf[0,0] = 1.0
        ci += [mfwf]

        mos += [mos[-1]]
        nact += [nact[-1]]
        ncore += [ncore[-1]]

    rdm1, h, s = calc_ci_elements(ci, h1e, h2e, ovlp, mos, nmo, nocc, nact, ncore, emb.e_nuc)
    return h, s, rdm1
