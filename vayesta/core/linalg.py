import logging
import numpy as np


log = logging.getLogger(__name__)

def recursive_block_svd(a, n, tol=1e-10, maxblock=100):
    """Perform SVD of rectangular, offdiagonal blocks of a matrix recursively.

    Parameters
    ----------
    a : (m, m) array
        Input matrix.
    n : int
        Number of rows of the first offdiagonal block.
    tol : float, optional
        Singular values below the tolerance are considered uncoupled. Default: 1e-10.
    maxblock : int, optional
        Maximum number of recursions. Default: 100.

    Returns
    -------
    coeff : (m-n, m-n) array
        Coefficients.
    sv : (m-n) array
        Singular values.
    order : (m-n) array
        Orders.
    """
    size = a.shape[-1]
    log.debugv("Recursive block SVD of %dx%d matrix" % a.shape)
    coeff = np.eye(size)
    sv = np.full((size-n,), 0.0)
    orders = np.full((size-n,), np.inf)

    ndone = 0
    low = np.s_[:n]
    env = np.s_[n:]

    for order in range(1, maxblock+1):
        blk = np.linalg.multi_dot((coeff.T, a, coeff))[low,env]
        nmax = blk.shape[-1]
        assert blk.ndim == 2
        assert np.all(np.asarray(blk.shape) > 0)

        u, s, vh = np.linalg.svd(blk)
        rot = vh.T.conj()
        ncpl = np.count_nonzero(s >= tol)
        log.debugv("Order= %3d - found %3d bath orbitals in %3d with tol= %8.2e: SV= %r" % (order, ncpl, blk.shape[1], tol, s[:ncpl].tolist()))
        if ncpl == 0:
            log.debugv("Remaining environment orbitals are decoupled; exiting.")
            break
        # Update output
        coeff[:,env] = np.dot(coeff[:,env], rot)
        sv[ndone:(ndone+ncpl)] = s[:ncpl]
        orders[ndone:(ndone+ncpl)] = order
        # Update spaces
        low = np.s_[(n+ndone):(n+ndone+ncpl)]
        env = np.s_[(n+ndone+ncpl):]
        ndone += ncpl

        if ndone == (size - n):
            log.debugv("All bath orbitals found; exiting.")
            break
        assert (ndone < (size - n))
    else:
        log.debug("Found %d out of %d bath orbitals in %d recursions", ndone, size-n, maxblock)

    coeff = coeff[n:,n:]
    assert np.allclose(np.dot(coeff.T, coeff)-np.eye(coeff.shape[-1]), 0)
    log.debugv("SV= %r", sv)
    log.debugv("orders= %r", orders)
    return coeff, sv, orders


if __name__ == '__main__':
    import pyscf
    import pyscf.gto
    import pyscf.scf
    atom = """
    Ti 0.0 0.0 0.0
    O  -%f 0.0 0.0
    O  +%f 0.0 0.0
    O  0.0 -%f 0.0
    O  0.0 +%f 0.0
    O  0.0 0.0 -%f
    O  0.0 0.0 +%f
    """
    d = 1.85
    basis = '6-31G'
    atom = atom % tuple(6*[d])
    mol = pyscf.gto.Mole(atom=atom, basis=basis, charge=-2)
    mol.build()

    hf = pyscf.scf.RHF(mol)
    hf.kernel()

    import vayesta
    import vayesta.ewf
    log = vayesta.log
    ewf = vayesta.ewf.EWF(hf)
    #f = ewf.make_atom_fragment(0)
    f = ewf.make_ao_fragment('Ti 3d')
    c_cluster_occ, c_cluster_vir, c_env_occ, _, c_env_vir, _ = f.make_bath(bath_type=None)
    #f.kernel()
    dm_hf = hf.make_rdm1()
    fock = ewf.get_fock()
    c_frag = f.c_frag
    c_env = f.c_env

    # Construct via SVD
    dmocc1 = np.linalg.multi_dot((c_frag.T, fock, c_env_occ))
    u, s, vh = np.linalg.svd(dmocc1)
    mo_svd = np.dot(c_env_occ, vh.T)
    ncpl = len(s)
    print("Order1 SV= %r" % s)

    dmocc2 = np.linalg.multi_dot((mo_svd[:,:ncpl].T, fock, mo_svd[:,ncpl:]))
    u, s, vh = np.linalg.svd(dmocc2)
    print("Order2 SV= %r" % s)

    # Use function:
    nimp = c_frag.shape[-1]
    c = np.hstack((c_frag, c_env_occ))
    f = np.linalg.multi_dot((c.T, fock, c))
    mo_svd2, sv, orders = recursive_block_svd(f, n=nimp)
    mo_svd2 = np.dot(c_env_occ, mo_svd2)
    print(sv)
    print(orders)

    e_svd = np.linalg.eigh(np.dot(mo_svd, mo_svd.T))[0]
    e_svd2 = np.linalg.eigh(np.dot(mo_svd2, mo_svd2.T))[0]
    assert np.allclose(e_svd, e_svd2)

    # Construct directly
    #nocc = np.count_nonzero(hf.mo_occ > 0)
    #occ = np.s_[:nocc]
    #ovlp = hf.get_ovlp()
    #lhs = np.linalg.multi_dot((c_frag.T, ovlp, hf.mo_coeff[:,occ]))
    #rhs = np.linalg.multi_dot((c_env_occ.T, ovlp, hf.mo_coeff[:,occ]))
    #mo_direct = np.einsum('ai,i,xi->xa', lhs, hf.mo_energy[occ], rhs)
