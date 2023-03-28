import numpy as np
from pygnme import wick, utils

def owndata(x):
    # CARMA requires numpy arrays to have data ownership
    if not x.flags["OWNDATA"]:
        y = np.zeros(x.shape, order="C")
        y[:] = x
        x = y
    assert x.flags["OWNDATA"]
    return x

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

def calc_ci_elements_uhf(ci, h1e, h2e, ovlp, mo, nmo, nocc, nact, ncore, enuc=0.0):

    h1e, h2e, ovlp = owndata(h1e), owndata(h2e), owndata(ovlp)
    mo = tuple([tuple([owndata(y) for y in x]) for x in mo])

    # Compute coupling terms
    nstate = len(ci)
    h = np.zeros((nstate, nstate))
    s = np.zeros((nstate, nstate))
    rdm1 = np.zeros((nstate, nstate, 2, nmo, nmo))
    for x in range(nstate):
        for w in range(x, nstate):

            nact1, ncore1, mo1 = nact[x], ncore[x], mo[x]
            nact2, ncore2, mo2 = nact[w], ncore[w], mo[w]

            # Can't currently support different size active spaces.
            assert(nact1[0] == nact2[0] and nact1[1] == nact2[1])
            assert(ncore1[0] == ncore2[0] and ncore1[1] == ncore2[1])
            orbs1 = wick.wick_orbitals[float, float](nmo, nmo, nocc[0], mo1[0], mo2[0], ovlp, nact1[0], ncore1[0])
            orbs2 = wick.wick_orbitals[float, float](nmo, nmo, nocc[1], mo1[1], mo2[1], ovlp, nact1[1], ncore1[1])

            mb = wick.wick_uscf[float, float, float](orbs1, orbs2, ovlp, enuc)
            mb.add_one_body(h1e)
            mb.add_two_body(h2e)

            vxa = utils.fci_bitset_list(nocc[0] - ncore1[0], nact1[0])
            vxb = utils.fci_bitset_list(nocc[1] - ncore1[1], nact1[1])

            vwa = utils.fci_bitset_list(nocc[0] - ncore2[0], nact2[0])
            vwb = utils.fci_bitset_list(nocc[1] - ncore2[1], nact2[1])
            # Loop over FCI occupation strings
            for iwa in range(len(vwa)):
                for iwb in range(len(vwb)):
                    for ixa in range(len(vxa)):
                        for ixb in range(len(vxb)):
                            # Compute S and H contribution for this pair of determinants
                            stmp, htmp = mb.evaluate(vxa[ixa], vxb[ixb], vwa[iwa], vwb[iwb])
                            h[x, w] += htmp * ci[w][iwa, iwb] * ci[x][ixa, ixb]
                            s[x, w] += stmp * ci[w][iwa, iwb] * ci[x][ixa, ixb]

                            # Compute RDM1 contribution for this pair of determinants
                            tmpPa = np.zeros((orbs1.m_nmo, orbs1.m_nmo))
                            tmpPb = np.zeros((orbs1.m_nmo, orbs1.m_nmo))
                            mb.evaluate_rdm1(vxa[ixa], vxb[ixb], vwa[iwa], vwb[iwb], stmp, tmpPa, tmpPb)
                            rdm1[x, w, 0] += tmpPa * ci[w][iwa, iwb] * ci[x][ixa, ixb]
                            rdm1[x, w, 1] += tmpPb * ci[w][iwa, iwb] * ci[x][ixa, ixb]

            rdm1[x, w, 0] = np.linalg.multi_dot((mo2[0], rdm1[x, w, 0], mo1[0].T))
            rdm1[x, w, 0] = np.linalg.multi_dot((mo2[1], rdm1[x, w, 1], mo1[1].T))

            h[w, x] = h[x, w]
            s[w, x] = s[x, w]
            rdm1[w, x, 0] = rdm1[x, w, 0].T
            rdm1[w, x, 1] = rdm1[x, w, 1].T
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


def calc_full_couplings_uhf(h1e, h2e, ovlp, mo1, mo2, nmo, nocc, nact1, nact2, ncore1, ncore2, enuc=0.0):

    h1e, h2e, ovlp = owndata(h1e), owndata(h2e), owndata(ovlp)
    mo1, mo2 = [owndata(x) for x in mo1], [owndata(x) for x in mo2]

    # Can't currently support different size active spaces.
    assert(nact1[0] == nact2[0] and nact1[1] == nact2[1])
    assert(ncore1[0] == ncore2[0] and ncore1[1] == ncore2[1])
    orbs1 = wick.wick_orbitals[float, float](nmo, nmo, nocc[0], mo1[0], mo2[0], ovlp, nact1[0], ncore1[0])
    orbs2 = wick.wick_orbitals[float, float](nmo, nmo, nocc[1], mo1[1], mo2[1], ovlp, nact1[1], ncore1[1])

    mb = wick.wick_uscf[float, float, float](orbs1, orbs2, ovlp, enuc)
    mb.add_one_body(h1e)
    mb.add_two_body(h2e)

    v1a = utils.fci_bitset_list(nocc[0] - ncore1[0], nact1[0])
    v1b = utils.fci_bitset_list(nocc[1] - ncore1[1], nact1[1])

    v2a = utils.fci_bitset_list(nocc[0] - ncore2[0], nact2[0])
    v2b = utils.fci_bitset_list(nocc[1] - ncore2[1], nact2[1])

    h = np.zeros((len(v1a), len(v1b), len(v2a), len(v2b)))
    s = np.zeros_like(h)
    for i1a in range(len(v1a)):
        for i1b in range(len(v1b)):
            for i2a in range(len(v2a)):
                for i2b in range(len(v2b)):
                    stmp, htmp = mb.evaluate(v1a[i1a], v1b[i1b], v2a[i2a], v2b[i2b])
                    h[i1a, i1b, i2a, i2b] = htmp
                    s[i1a, i1b, i2a, i2b] = stmp

    return h.reshape((len(v1a) * len(v1b), len(v2a) * len(v2b))), s.reshape((len(v1a) * len(v1b), len(v2a) * len(v2b)))
