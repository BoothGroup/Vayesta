import numpy as np
import scipy.linalg
from pyscf import gto, scf, mcscf, ao2mo, lib
from vayesta.core.types import RFCI_WaveFunction, SpatialOrbitals
from .calc_nonorth_couplings import calc_full_couplings, calc_full_couplings_uhf, calc_ci_elements, calc_ci_elements_uhf

def get_emb_info(emb, wfs):
    h1e = emb.get_hcore()
    ovlp = emb.get_ovlp()

    if emb.is_rhf:
        h2e = emb.get_eris_array(np.eye(emb.nao)).reshape((emb.nao**2, emb.nao**2))
        nmo, nocc = emb.nao, emb.nocc
        nact = [x.mo.norb for x in wfs]
        ncore = [emb.nocc - x.mo.nocc for x in wfs]
    else:
        h2e = emb.get_eris_array(np.eye(emb.nao)).reshape((emb.nao**2, emb.nao**2))
        nmo, nocc = emb.nao, emb.nocc
        nact = [x.mo.norb for x in wfs]
        ncore = [[y - z for y,z in zip(emb.nocc, x.mo.nocc)] for x in wfs]
    return ovlp, h1e, h2e, nmo, nocc, nact, ncore

def optimise_full_varwav(emb, fs=None, lindep=1e-12, replace_wf=False):
    fullham, fullovlp = gen_loc_hams(emb, fs)
    h, s, nstates = to_single_ham(fullham, fullovlp, returnsizes=True)

    w, v, seig = lib.linalg_helper.safe_eigh(h, s, lindep=lindep)

    ci0 = v[:, 0]

    orig_ci = [f.results.wf.ci for f in emb.fragments]

    for x in orig_ci:
        print(x)
    i = 0
    if replace_wf:
        for ns, f in zip(nstates, emb.fragments):
            if f.sym_parent is None:
                c = ci0[i:i + ns].reshape(f.results.wf.ci.shape)
                # Copy to avoid wiping any other copies.
                f.results.wf = f.results.wf.copy()
                f.results.wf.ci = c
            i += ns

    return w[0]

def gen_loc_hams(emb, fs=None):
    if fs is None:
        fs = emb.fragments

    wfs = [x.results.wf for x in fs]
    mos = [x.cluster.c_total for x in fs]

    ovlp, h1e, h2e, nmo, nocc, nact, ncore = get_emb_info(emb, wfs)

    if emb.is_rhf:
        coupl_func = calc_full_couplings
    else:
        coupl_func = calc_full_couplings_uhf

    fullham = np.zeros((len(fs), len(fs)), dtype=object)
    fullovlp = np.zeros_like(fullham)
    for i in range(len(fs)):
        for j in range(i, len(fs)):
            fullham[i, j], fullovlp[i, j] = coupl_func(h1e, h2e, ovlp, mos[i], mos[j], nmo, nocc, nact[i],
                                                                    nact[j], ncore[i], ncore[j], enuc=emb.e_nuc)
            fullham[j, i] = fullham[i, j].T
            fullovlp[j, i] = fullovlp[i, j].T
    return fullham, fullovlp


def to_single_ham(fullham, fullovlp, returnsizes=False):

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

    ovlp, h1e, h2e, nmo, nocc, nact, ncore = get_emb_info(emb, wfs)

    ci = [x.ci for x in wfs]

    if inc_mf:
        mfwf = np.zeros_like(ci[-1])
        mfwf[0,0] = 1.0
        ci += [mfwf]

        mos += [mos[-1]]
        nact += [nact[-1]]
        ncore += [ncore[-1]]

    if emb.is_rhf:
        rdm1, h, s = calc_ci_elements(ci, h1e, h2e, ovlp, mos, nmo, nocc, nact, ncore, emb.e_nuc)
    else:
        rdm1, h, s = calc_ci_elements_uhf(ci, h1e, h2e, ovlp, mos, nmo, nocc, nact, ncore, emb.e_nuc)
    return h, s, rdm1
