import numpy as np
import scipy.linalg
from pyscf import gto, scf, mcscf, ao2mo
from pygnme import wick, utils


def owndata(x):
    # CARMA requires numpy arrays to have data ownership
    if not x.flags["OWNDATA"]:
        y = np.zeros(x.shape, order="C")
        y[:] = x
        x = y
    assert x.flags["OWNDATA"]
    return x


def get_wf_couplings(emb, fs=None):
    """Calculate the hamiltonian element between multiple FCI wavefunctions in different fragments.
    This requires the CI coefficients and the basis set in which they are defined.
    """
    if fs is None:
        fs = emb.fragments
    wfs = [x.results.wf for x in fs]

    h1e = emb.get_hcore()
    h2e = emb.get_eris_array(np.eye(emb.nao)).reshape((emb.nao**2, emb.nao**2))
    ovlp = emb.get_ovlp()
    nmo, nocc = emb.nao, emb.nocc

    ci = [x.ci for x in wfs]
    mo = [x.cluster.c_total for x in fs]
    nact = [x.mo.norb for x in wfs]
    ncore = [emb.nocc - x.mo.nocc for x in wfs]
    print(emb.mol.nelectron, emb.nao)
    rdm1, h, s = calc_ci_elements(ci, h1e, h2e, ovlp, mo, nmo, nocc, nact, ncore, emb.e_nuc)
    return h, s, rdm1


def calc_ci_elements(ci, h1e, h2e, ovlp, mo, nmo, nocc, nact, ncore, enuc=0.0):
    print(ci[0].shape, ci[1].shape, mo[0].shape, mo[1].shape)
    print(h1e.shape, h2e.shape, ovlp.shape)
    print(nmo, nocc, nact, ncore)

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

    for x in range(nstate):
        for w in range(nstate):
            print("\n RDM-1 for ⟨Ψ_%d| and |Ψ_%d⟩" % (x, w))
            print(rdm1[x, w])

    print("\n Hamiltonian")
    print(h)
    print("\n Overlap")
    print(s)

    #w, v = scipy.linalg.eigh(h, b=s)
    #print("\n Recoupled NO-CAS-CI eigenvalues")
    #print(w)
    #print("\n Recoupled NO-CAS-CI eigenvectors")
    #print(v)

    return rdm1, h, s
