"""Project and reconstruct observables across fragments"""

import numpy as np
from vayesta.core.util import dot, einsum
from vayesta.mpi import mpi

def make_global_one_body(emb, cluster_observable, symmetrize=False, use_sym=True, proj=1, fragments=None):
    """
    Construct full system static one body observable from fragment cluster observables.

    Parameters
    ----------
    emb : EWF object
        Embedding object
    cluster_observable : ndarray (nfrag, ..., nmo_cluster, nmo_cluster)
        List of cluster observables for each fragment
    symmetrize : bool
        Symmetrize the final observable.
    use_sym : bool
        Include symmetry equivalent fragments.
    proj : int
        Number of projectors to use (1 or 2)
    fragments : list of Fragment objects, optional
        List of fragments to use. If None, all fragments are used.

    Returns
    -------
    global_observable : ndarry (..., nmo, nmo)
        Full system one body observable (MO basis)
    """


    nao, nmo = emb.mo_coeff.shape
    dtype = cluster_observable[0].dtype
    nfrag = len(cluster_observable)
    shape = tuple( list(cluster_observable[0].shape[:-2]) + [nmo, nmo])
    global_observable = np.zeros(shape, dtype=dtype)

    if fragments is None:
        fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    assert len(fragments) == nfrag

    for i, f in enumerate(fragments):

        obs_clus = cluster_observable[i]
        
        mc = f.get_overlap('mo|cluster')
        mf = f.get_overlap('mo|frag')
        fc = f.get_overlap('frag|cluster')
        cfc = fc.T @ fc
        
        if proj == 0:
            # No projection onto fragment, just embed cluster observable directly
            # global_observable += np.einsum('iI,jJ,...IJ->ij', mc, mc, obs_clus)
            global_observable += np.matmul(mc, np.matmul(obs_clus, mc.T))
            if use_sym:
                for child in f.get_symmetry_children():
                    mc_child = child.get_overlap('mo|cluster')
                    global_observable += np.matmul(mc_child, np.matmul(obs_clus, mc_child.T))
        elif proj == 1:
            # Symmetrically project onto fragment 
            # obs_frag = 0.5 * (np.einsum('iI,...Ij->ij', cfc, obs_clus) + np.einsum('jJ,...iJ->ij', cfc, obs_clus))
            if symmetrize:
                obs_frag = 0.5 * (np.matmul(cfc, obs_clus) + np.matmul(obs_clus, cfc))
            else:
                obs_frag = np.matmul(cfc, obs_clus)
            # global_observable += np.einsum('iI,jJ,...IJ->ij', mc, mc, obs_frag)
            global_observable += np.matmul(mc, np.matmul(obs_frag, mc.T))

            if use_sym:
                for child in f.get_symmetry_children():
                    mc_child = child.get_overlap('mo|cluster')
                    global_observable += np.matmul(mc_child, np.matmul(obs_frag, mc_child.T))

        elif proj == 2:
            # Project onto fragment from both sides
            # obs_frag = np.einsum('iI,jJ,...IJ->ij', fc, fc, obs_clus)
            obs_frag = np.matmul(fc, np.matmul(obs_clus, fc.T))
            global_observable += np.matmul(mf, np.matmul(obs_frag, mf.T))
            if use_sym:
                for child in f.get_symmetry_children():
                    mf_child = child.get_overlap('mo|frag')
                    global_observable += np.matmul(mf_child, np.matmul(obs_frag, mf_child.T))

    if symmetrize:
        assert np.allclose(global_observable, np.moveaxis(global_observable, -1, -2))
        #global_observable = 0.5 * (global_observable + np.moveaxis(global_observable, -1, -2).conj())


    return global_observable

def make_local_one_body(emb, global_observable, fragments=None, use_sym=True):
    """
    Construct fragment cluster one body observable from full system observable.

    Parameters
    ----------
    emb : EWF object
        Embedding object
    global_observable : ndarray (..., nmo, nmo)
        Full system one body observable (MO basis)
    use_sym : bool
        Include symmetry equivalent fragments
    fragments : list of Fragment objects, optional
        List of fragments to use. If None, all fragments are used.        
    
    Returns
    -------
    cluster_observable : ndarray (nfrag, ..., nmo_cluster, nmo_cluster)
        List of cluster observables for each fragment
    """
    
    nao, nmo = emb.mo_coeff.shape
    dtype = global_observable.dtype

    if fragments is None:
        fragments = emb.get_fragments(sym_parent=None) if use_sym else emb.get_fragments()
    nfrag = len(fragments)

    cluster_observable = []

    for i, f in enumerate(fragments):

        mc = f.get_overlap('mo|cluster')

        # obs_frag = np.einsum('iI,jJ,...ij->IJ', fc.T, fc.T, np.einsum('iI,jJ,...ij->ij', mf.T, mf.T, global_observable))
        obs_clus = np.matmul(mc.T, np.matmul(global_observable, mc))
        #obs_clus = np.einsum('Ii,Jj,...ij->IJ', mc.T, mc.T, global_observable)
        cluster_observable.append(obs_clus)

    return np.array(cluster_observable, dtype=dtype)