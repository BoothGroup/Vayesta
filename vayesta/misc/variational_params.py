import numpy as np
import scipy.linalg
from pyscf import gto, scf, mcscf, ao2mo, lib
from pygnme import wick, utils
from vayesta.core.types import RFCI_WaveFunction, SpatialOrbitals

def owndata(x):
    # CARMA requires numpy arrays to have data ownership
    if not x.flags["OWNDATA"]:
        y = np.zeros(x.shape, order="C")
        y[:] = x
        x = y
    assert x.flags["OWNDATA"]
    return x

def optimise_full_varwav(emb, lindep=1e-12):
    h, s = gen_full_ham(emb)
    print(len(emb.fragments), h.shape, s.shape)
    w, v, seig = lib.linalg_helper.safe_eigh(h, s, lindep=lindep)
    return w[0]

def gen_full_ham(emb, fs=None):

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


def calc_ci_elements(ci, h1e, h2e, ovlp, mo, nmo, nocc, nact, ncore, enuc=0.0):

    ci = tuple([owndata(x) for x in ci])
    h1e, h2e, ovlp = owndata(h1e), owndata(h2e), owndata(ovlp)
    mo = tuple([owndata(x) for x in mo])

    # Compute coupling terms
    nstate = len(ci)
    h = np.zeros((nstate, nstate))
    s = np.zeros((nstate, nstate))
    rdm1 = np.zeros((nstate, nstate, nmo, nmo))
    for x in range(nstate):
        for w in range(x, nstate):
            # FIXME Can't have different numbers of nact, ncore here:
            orbs = wick.wick_orbitals[float, float](nmo, nmo, nocc, mo[x], mo[w], ovlp, nact[x], ncore[x])

            mb = wick.wick_rscf[float, float, float](orbs, ovlp, enuc)
            mb.add_one_body(h1e)
            mb.add_two_body(h2e)

            vx = utils.fci_bitset_list(nocc - ncore[x], nact[x])
            vw = utils.fci_bitset_list(nocc - ncore[w], nact[w])

            # Loop over FCI occupation strings
            for iwa in range(len(vw)):
                for iwb in range(len(vw)):
                    for ixa in range(len(vx)):
                        for ixb in range(len(vx)):
                            # Compute S and H contribution for this pair of determinants
                            stmp, htmp = mb.evaluate(vx[ixa], vx[ixb], vw[iwa], vw[iwb])
                            h[x, w] += htmp * ci[w][iwa, iwb] * ci[x][ixa, ixb]
                            s[x, w] += stmp * ci[w][iwa, iwb] * ci[x][ixa, ixb]

                            # Compute RDM1 contribution for this pair of determinants
                            tmpPa = np.zeros((orbs.m_nmo, orbs.m_nmo))
                            tmpPb = np.zeros((orbs.m_nmo, orbs.m_nmo))
                            mb.evaluate_rdm1(vx[ixa], vx[ixb], vw[iwa], vw[iwb], stmp, tmpPa, tmpPb)
                            rdm1[x, w] += (tmpPa + tmpPb) * ci[w][iwa, iwb] * ci[x][ixa, ixb]

            rdm1[x, w] = np.linalg.multi_dot((mo[w], rdm1[x, w], mo[x].T))

            h[w, x] = h[x, w]
            s[w, x] = s[x, w]
            rdm1[w, x] = rdm1[x, w].T

    return rdm1, h, s


def calc_full_couplings(h1e, h2e, ovlp, mo1, mo2, nmo, nocc, nact1, nact2, ncore1, ncore2, enuc=0.0):

    h1e, h2e, ovlp = owndata(h1e), owndata(h2e), owndata(ovlp)
    mo1, mo2 = owndata(mo1), owndata(mo2)

    # Can't currently support different size active spaces.
    assert(nact1 == nact2)
    assert(ncore1 == ncore2)
    orbs = wick.wick_orbitals[float, float](nmo, nmo, nocc, mo1, mo2, ovlp, nact1, ncore1)

    mb = wick.wick_rscf[float, float, float](orbs, ovlp, enuc)
    mb.add_one_body(h1e)
    mb.add_two_body(h2e)

    v1 = utils.fci_bitset_list(nocc - ncore1, nact1)
    v2 = utils.fci_bitset_list(nocc - ncore2, nact2)

    h = np.zeros((len(v1), len(v1), len(v2), len(v2)))
    s = np.zeros_like(h)
    for i1a in range(len(v1)):
        for i1b in range(len(v1)):
            for i2a in range(len(v2)):
                for i2b in range(len(v2)):
                    stmp, htmp = mb.evaluate(v1[i1a], v1[i1b], v2[i2a], v2[i2b])
                    h[i1a, i1b, i2a, i2b] = htmp
                    s[i1a, i1b, i2a, i2b] = stmp

    return h.reshape((len(v1)**2, len(v2)**2)), s.reshape((len(v1)**2, len(v2)**2))
