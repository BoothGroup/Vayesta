import logging

import numpy as np

from vayesta.core.util import *

log = logging.getLogger(__name__)


def indices_to_bools(indices, n):
    bools = np.zeros(n, dtype=bool)
    bools[np.asarray(indices)] = True
    return bools

def transform_amplitudes(t1, t2, u_occ, u_vir):
    """(Old basis|new basis)"""
    if t1 is not None:
        t1 = einsum("ia,ix,ay->xy", t1, u_occ, u_vir)
    else:
        t1 = None
    if t2 is not None:
        t2 = einsum("ijab,ix,jy,az,bw->xyzw", t2, u_occ, u_occ, u_vir, u_vir)
    else:
        t2 = None
    return t1, t2

def atom_labels_to_ao_indices(mol, atom_labels):
    """Convert atom labels to AO indices of mol object."""
    atom_labels_mol = np.asarray([ao[1] for ao in mol.ao_labels(None)])
    ao_indices = np.nonzero(np.isin(atom_labels_mol, atom_labels))[0]
    return ao_indices

def atom_label_to_ids(mol, atom_label):
    """Get all atom IDs corresponding to an atom label."""
    atom_labels = np.asarray([mol.atom_symbol(atomid) for atomid in range(mol.natm)])
    atom_ids = np.where(np.in1d(atom_labels, atom_label))[0]
    return atom_ids

def get_ao_indices_at_atoms(mol, atomids):
    """Return indices of AOs centered at a given atom ID."""
    ao_indices = []
    if not hasattr(atomids, "__len__"):
        atomids = [atomids]
    for atomid in atomids:
        ao_slice = mol.aoslice_by_atom()[atomid]
        ao_indices += list(range(ao_slice[2], ao_slice[3]))
    return ao_indices

def orthogonalize_mo(c, s, tol=1e-6):
    """Orthogonalize MOs, such that C^T S C = I (identity matrix).

    Parameters
    ----------
    c : ndarray
        MO orbital coefficients.
    s : ndarray
        AO overlap matrix.
    tol : float, optional
        Tolerance.

    Returns
    -------
    c_out : ndarray
        Orthogonalized MO coefficients.
    """
    assert np.all(c.imag == 0)
    assert np.allclose(s, s.T)
    l = np.linalg.cholesky(s)
    c2 = np.dot(l.T, c)
    #chi = np.linalg.multi_dot((c.T, s, c))
    chi = np.dot(c2.T, c2)
    chi = (chi + chi.T)/2
    e, v = np.linalg.eigh(chi)
    assert np.all(e > 0)
    r = einsum("ai,i,bi->ab", v, 1/np.sqrt(e), v)
    c_out = np.dot(c, r)
    chi_out = np.linalg.multi_dot((c_out.T, s, c_out))
    # Check orthogonality within tol
    nonorth = abs(chi_out - np.eye(chi_out.shape[-1])).max()
    if tol is not None and nonorth > tol:
        log.error("Orbital non-orthogonality= %.1e", nonorth)

    return c_out

def amplitudes_c2t(c1, c2):
    t1 = c1.copy()
    t2 = c2 - einsum("ia,jb->ijab", c1, c1)
    return t1, t2

def amplitudes_t2c(t1, t2):
    c1 = t1.copy()
    c2 = t2 + einsum("ia,jb->ijab", t1, t1)
    return c1, c2
