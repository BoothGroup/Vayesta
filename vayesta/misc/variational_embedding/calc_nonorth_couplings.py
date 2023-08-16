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
    """
    Calculates the coupling terms for a given set of configuration interaction (CI) coefficients.

    Args:
        ci (tuple): The CI coefficients for each state.
        h1e: A 2D array containing the one-electron Hamiltonian matrix elements.
        h2e: A 4D array containing the two-electron Hamiltonian matrix elements.
        ovlp: A 2D array containing the overlap matrix elements.
        mo (tuple): The molecular orbital coefficients for each state.
        nmo (int): The total number of molecular orbitals.
        nocc (int): The number of occupied molecular orbitals.
        nact (list): A list of integers containing the number of active orbitals for each state.
        ncore (list): A list of integers containing the number of core orbitals for each state.
        enuc (float): The nuclear repulsion energy.

    Returns:
        h: A 2D array containing the coupling terms between each pair of states.
        s: A 2D array containing the overlap integrals between each pair of states.
        rdm1: A 4D array containing the one-particle reduced density matrices (1-RDMs) for each pair of states.
        rdm1 dimensions: (state1, state2, orbital1, orbital2)
    """

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
            # Setup biorthogonalised orbital pair
            refx = wick.reference_state[float](nmo, nmo, nocc, nact[x], ncore[x], mo[x])
            refw = wick.reference_state[float](nmo, nmo, nocc, nact[w], ncore[w], mo[w])

            # Setup paired orbitals
            orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

            mb = wick.wick_rscf[float, float, float](orbs, enuc)
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
            # Setup biorthogonalised orbital pairs
            refx1 = wick.reference_state[float](nmo, nmo, nocc[0], nact[x][0], ncore[x][0], mo[x][0])
            refx2 = wick.reference_state[float](nmo, nmo, nocc[1], nact[x][1], ncore[x][1], mo[x][1])
            refw1 = wick.reference_state[float](nmo, nmo, nocc[0], nact[w][0], ncore[w][0], mo[w][0])
            refw2 = wick.reference_state[float](nmo, nmo, nocc[1], nact[w][1], ncore[w][1], mo[w][1])

            # Setup paired orbitals
            orbs1 = wick.wick_orbitals[float, float](refx1, refw1, ovlp)
            orbs2 = wick.wick_orbitals[float, float](refx2, refw2, ovlp)

            mb = wick.wick_uscf[float, float, float](orbs1, orbs2, enuc)

            mb.add_one_body(h1e)
            mb.add_two_body(h2e)

            vxa = utils.fci_bitset_list(nocc[0] - ncore[x][0], nact[x][0])
            vxb = utils.fci_bitset_list(nocc[1] - ncore[x][1], nact[x][1])

            vwa = utils.fci_bitset_list(nocc[0] - ncore[w][0], nact[w][0])
            vwb = utils.fci_bitset_list(nocc[1] - ncore[w][1], nact[w][1])
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

    # Setup biorthogonalised orbital pair
    refx = wick.reference_state[float](nmo, nmo, nocc, nact1, ncore1, mo1)
    refw = wick.reference_state[float](nmo, nmo, nocc, nact2, ncore2, mo2)

    # Setup paired orbitals
    orbs = wick.wick_orbitals[float, float](refx, refw, ovlp)

    # Setup matrix builder object
    mb = wick.wick_rscf[float, float, float](orbs, enuc)

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

    return h.reshape((len(v1) ** 2, len(v2) ** 2)), s.reshape((len(v1) ** 2, len(v2) ** 2))


def calc_full_couplings_uhf(h1e, h2e, ovlp, mo1, mo2, nmo, nocc, nact1, nact2, ncore1, ncore2, enuc=0.0):
    h1e, h2e, ovlp = owndata(h1e), owndata(h2e), owndata(ovlp)
    mo1, mo2 = [owndata(x) for x in mo1], [owndata(x) for x in mo2]

    # Setup biorthogonalised orbital pair
    refx1 = wick.reference_state[float](nmo, nmo, nocc[0], nact1[0], ncore1[0], mo1[0])
    refx2 = wick.reference_state[float](nmo, nmo, nocc[1], nact1[1], ncore1[1], mo1[1])
    refw1 = wick.reference_state[float](nmo, nmo, nocc[0], nact2[0], ncore2[0], mo2[0])
    refw2 = wick.reference_state[float](nmo, nmo, nocc[1], nact2[1], ncore2[1], mo2[1])

    # Setup paired orbitals
    orbs1 = wick.wick_orbitals[float, float](refx1, refw1, ovlp)
    orbs2 = wick.wick_orbitals[float, float](refx2, refw2, ovlp)

    # Setup matrix builder object
    mb = wick.wick_uscf[float, float, float](orbs1, orbs2, enuc)

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
