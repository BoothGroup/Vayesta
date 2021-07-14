import cvxpy as cp
import numpy as np
from pyscf.lib import diis
from .fragment import DMETFragmentExit

solver = cp.SCS

def perform_SDP_fit(nelec, fock, impurity_projectors, target_rdms, log):
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
    """
    # First calculate the number of different symmetry-eqivalent orbital sets for each class of impurity (this is the
    # symmetry factor, elsewhere in the code).
    nimp = [len(x) for x in impurity_projectors]

    z = cp.Variable(fock.shape, PSD=True)
    us = [cp.Variable((i, i), symmetric=True) for i in nimp]
    alpha = cp.Variable(1)

    # print("Targets2:")
    # for i in rdms: print(i)

    # print(fock)

    # It might be neater to have U as the variable and impose a sparsity pattern, but I've coded this
    # version up now and it'll make it easier to extend with symmetry-related impurities.
    constraints = [
        (fock + sum([aproj @ us[i] @ aproj.T for i, curr_proj in enumerate(impurity_projectors) for aproj in curr_proj])
         + z - alpha * np.eye(fock.shape[0])) >> 0,
        cp.trace(sum([aproj @ us[i] @ aproj.T for i, curr_proj in enumerate(impurity_projectors) for aproj in
                      curr_proj])) == 0,
    ]

    objective = cp.Minimize(
        sum([cp.trace(rdm1 @ ux) * nimp for ni, rdm1, ux in zip(nimp, target_rdms, us)])
        - alpha * nelec + cp.trace(z)
    )
    prob = cp.Problem(objective, constraints)

    solval = prob.solve(solver=cp.SCS, eps=1e-8)
    # Our local correlation potential values are then contained within the values of this list; use our projector to
    # map back into full space.
    fullv = sum([aproj @ us[i].value @ aproj.T for i, curr_proj in enumerate(impurity_projectors) for aproj in curr_proj])

    # Report the result of the optimisation, as warning if optimal solution not found.
    msg = "SDP fitting completed. Status= %s" % prob.status
    if not prob.status in [cp.OPTIMAL]:#, cp.OPTIMAL_INACCURATE]:
        log.warning(msg)
    else:
        log.info(msg)
    return fullv

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