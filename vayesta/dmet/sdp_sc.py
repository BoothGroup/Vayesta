import cvxpy as cp
import numpy as np
from pyscf.lib import diis
from .fragment import DMETFragmentExit

solver = cp.MOSEK#cp.SCS

def perform_SDP_fit(nelec, fock, impurity_projectors, target_rdms, ovlp, log):
    """Given all required information about the system, generate the correlation potential reproducing the local DM
    via a semidefinite program, as described in doi.org/10.1103/PhysRevB.102.085123. Initially use SCS solver, though
    others (eg. MOSEK) could be used if this runs into issues.

    Note that initially we will only have a single impurity in each symmetry class, until the fragmentation routines are
    updated to include symmetry.

    Parameters
    ----------
    nelec: integer
        Number of electrons in the system.
    fock: np.array
        Fock matrix of the full system, in the spatial orbital basis.
    impurity_projectors: list of list of np.array
        For each class of symmetry-equivalent impurities, projection from the full spatial orbital basis to the impurity
        set.
    target_rdms: list of np.array
        Impurity-local one-body density matrices for each class of impurities.
    ovlp: np.array
        Overlap matrix of the AOs, so we can implicitly transform into the PAO basis.
    """
    print("Target RDMs:")
    print(target_rdms)
    print([x.trace() for x in target_rdms])

    print("Nsym:")
    print([x[0].shape for x in impurity_projectors])
    # First calculate the number of different symmetry-eqivalent orbital sets for each class of impurity (this is the
    # symmetry factor, elsewhere in the code).
    nimp = [x[0].shape[1] for x in impurity_projectors]
    nsym = [len(x) for x in impurity_projectors]
    log.info("Number of site fragments: %s" % nimp)
    log.info("Number of symmetry-related fragments: %s" % nsym)

    z = cp.Variable(fock.shape, PSD=True)
    us = [cp.Variable((i, i), symmetric=True) for i in nimp]
    alpha = cp.Variable(1)

    # We have the coefficients of the impurity orbitals in the nonorthogonal AO basis, C.
    # The coefficients of the AOs in the impurity orbitals is then equal to S @ C.T
    AO_in_imps = [[aproj.T @ ovlp for aproj in curr_proj] for curr_proj in impurity_projectors]

    # Check this is correctly orthogonal.
    #print([x[0].T @ y[0].T for (x,y) in zip(impurity_projectors, AO_in_imps)])

    # Want to construct the full correlation potential, in the AO basis.
    utot = sum([cp.matmul(aproj.T, cp.matmul(us[i], aproj)) for i, curr_proj in enumerate(AO_in_imps) for aproj in curr_proj])
    #import scipy
    #c_pao = np.linalg.inv(scipy.linalg.sqrtm(ovlp))

    # First constraint in AO basis required for solution, second enforces tracelessness of Vcorr.
    constraints = [
        fock + utot + z - alpha * ovlp >> 0,
        sum([cp.trace(x) for x in us]) == 0,
    ]
    # Object function computed implicitly in PAO basis; partially use the fragment basis thanks to the invariance of the
    # trace to orthogonal rotations (if you're using nonorthogonal fragment orbitals this will need a bit more thought).
    objective = cp.Minimize(
        sum([cp.trace(rdm1 @ ux) * ni for ni, rdm1, ux in zip(nsym, target_rdms, us)])
        - alpha * nelec + cp.trace(cp.matmul(z, np.linalg.inv(ovlp)))
    )
    prob = cp.Problem(objective, constraints)

    solval = prob.solve(solver=cp.SCS, eps=1e-8)
    msg = "SDP fitting completed. Status= %s" % prob.status
    if not prob.status in [cp.OPTIMAL]:#, cp.OPTIMAL_INACCURATE]:
        log.warning(msg)
    else:
        log.info(msg)

    # Our local correlation potential values are then contained within the values of this list; use our projector to
    # map back into full space.
    #print("Local Correlation Potential:")
    #print([x.value for x in us])

    #fullv = sum([np.linalg.pinv(aproj).T @ us[i].value @ np.linalg.pinv(aproj) for i, curr_proj in enumerate(impurity_projectors) for aproj in curr_proj])
    #print(fullv)
    #print("Overall Correlation Potential:")
    #print(utot.value)
    #print("Projected local correlation potential:")
    #print(np.linalg.multi_dot([impurity_projectors[0][0].T, utot.value, impurity_projectors[0][0]]))
    #print(np.linalg.multi_dot([impurity_projectors[1][0].T, utot.value, impurity_projectors[1][0]]))
    #print(np.linalg.multi_dot([impurity_projectors[2][0].T, utot.value, impurity_projectors[2][0]]))
    # Report the result of the optimisation.
    return utot.value

# Code to check that the correlation potential does what it says on the tin, saved for later.
def WIP():
    raise NotImplementedError
    resham = fock + fullv
    # print(resham)

    ea, ca = np.linalg.eigh(resham[::2, ::2])
    eb, cb = np.linalg.eigh(resham[1::2, 1::2])
    # print("Res:")#, fock.shape, resham.shape, fullv.shape, rdms[0].shape)
    # print(e)
    n = len(impurities[0][0])

    temp = np.zeros_like(resham)
    temp[::2, ::2] = np.dot(ca[:, :nelec // 2], ca[:, :nelec // 2].T)
    temp[1::2, 1::2] = np.dot(cb[:, :nelec // 2], cb[:, :nelec // 2].T)
    # print(temp)
    maxdev = np.array(
        [
            [abs(temp[np.ix_(i, i)] - rdm[:len(i), :len(i)]).max() for i in impset
             ] for impset, rdm in zip(self.impurities, rdms)
        ]).max()
    # print(maxdev)
    if maxdev > 1e-4:
        print("Exact reproduction of 1rdm not achieved; Maximum deviation: ", maxdev, end=". ")
        print("Resultant system may have unaccounted for broken symmetry or be gapless. FMO energies are")
        print("alpha:", ea[nelec // 2 - 1: nelec // 2 + 1], "beta:",
              eb[nelec // 2 - 1: nelec // 2 + 1])
        # print("fock")
        # print(fock)
        # print("Overall ham resulting")
        # print(resham)
        # print("New correlation potential")
        # print(fullv)
        # print(ea)
        # print(eb)

    # print(fullv)
    fullv = np.array((fullv[::2, ::2], fullv[1::2, 1::2]))
    # print(fullv)
    # print(abs(fullv - mf.vcorr).max())
    # value of u is now the optimiser for the DMET problem, with fixed fock matrix.
    return fullv, prob.status != cp.OPTIMAL