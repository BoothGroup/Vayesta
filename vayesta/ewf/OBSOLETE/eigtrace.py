def eigassign(e1, v1, e2, v2, b=None, cost_matrix="e^2/v"):
    """
    Parameters
    ----------
    b : ndarray
        If set, eigenvalues and eigenvectors belong to a generalized eigenvalue problem of the form Av=Bve.
    cost_matrix : str
        Defines the way to calculate the cost matrix.
    """

    if e1.shape != e2.shape:
        raise ValueError("e1=%r with shape=%r and e2=%r with shape=%r are not compatible." % (e1, e1.shape, e2, e2.shape))
    if v1.shape != v2.shape:
        raise ValueError("v1=%r with shape=%r and v2=%r with shape=%r are not compatible." % (v1, v1.shape, v2, v2.shape))
    if e1.shape[0] != v1.shape[-1]:
        raise ValueError("e1=%r with shape=%r and v1=%r with shape=%r are not compatible." % (e1, e1.shape, v1, v1.shape))
    if e2.shape[0] != v2.shape[-1]:
        raise ValueError("e2=%r with shape=%r and v2=%r with shape=%r are not compatible." % (e2, e2.shape, v2, v2.shape))

    assert np.allclose(e1.imag, 0)
    assert np.allclose(e2.imag, 0)
    assert np.allclose(v1.imag, 0)
    assert np.allclose(v2.imag, 0)

    # Define a cost matrix ("dist") which measures the difference of two eigenpairs (ei,vi), (e'j, v'j)
    # of different eigenvalue problems
    if b is None:
        vmat = np.abs(np.dot(v1.T, v2))
    else:
        vmat = np.abs(np.linalg.multi_dot((v1.T, b, v2)))
    emat = np.abs(np.subtract.outer(e1, e2))

    # relative energy difference
    ematrel = emat / np.fmax(abs(e1), 1e-14)[:,np.newaxis]

    # Original formulation
    if cost_matrix == "(1-v)*e":
        dist = (1-vmat) * emat
    elif cost_matrix == "(1-v)":
        dist = (1-vmat)
    elif cost_matrix == "1/v":
        dist = 1/np.fmax(vmat, 1e-14)
    elif cost_matrix == "v/e":
        dist = -vmat / (emat + 1e-14)
    elif cost_matrix == "e/v":
        dist = emat / np.fmax(vmat, 1e-14)
    elif cost_matrix == "er/v":
        dist = ematrel / np.fmax(vmat, 1e-14)
    elif cost_matrix == "e/v^2":
        dist = emat / np.fmax(vmat, 1e-14)**2
    # This performed best in tests
    elif cost_matrix == "e^2/v":
        dist = emat**2 / np.fmax(vmat, 1e-14)
    elif cost_matrix == "e^2/v**2":
        dist = emat**2 / (vmat + 1e-14)**2
    elif cost_matrix == "e/sqrt(v)":
        dist = emat / np.sqrt(vmat + 1e-14)
    else:
        raise ValueError("Unknown cost_matrix: %s" % cost_matrix)

    row, col = scipy.optimize.linear_sum_assignment(dist)
    # The col indices are the new sorting
    cost = dist[row,col].sum()
    sort = col
    return sort, cost

def eigreorder_logging(e, reorder, log):
    for i, j in enumerate(reorder):
        # No reordering
        if i == j:
            continue
        # Swap between two eigenvalues
        elif reorder[j] == i:
            if i < j:
                log("Reordering eigenvalues %3d <-> %3d : %+6.3g <-> %+6.3g", j, i, e[j], e[i])
        # General reordering
        else:
            log("Reordering eigenvalues %3d --> %3d : %+6.3g --> %+6.3g", j, i, e[j], e[i])


