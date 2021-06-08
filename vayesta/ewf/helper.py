import logging

import numpy as np

from vayesta.core.util import *

log = logging.getLogger(__name__)

default_minao = {
        "gth-dzv" : "gth-szv",
        "gth-dzvp" : "gth-szv",
        "gth-tzvp" : "gth-szv",
        "gth-tzv2p" : "gth-szv",
        }

def get_minimal_basis(basis):
    minao = default_minao.get(basis, "minao")
    return minao

def indices_to_bools(indices, n):
    bools = np.zeros(n, dtype=bool)
    bools[np.asarray(indices)] = True
    return bools

def transform_amplitudes(t1, t2, u_occ, u_vir):
    if t1 is not None:
        t1 = einsum("ia,ix,ay->xy", t1, u_occ, u_vir)
    else:
        t1 = None
    if t2 is not None:
        t2 = einsum("ijab,ix,jy,az,bw->xyzw", t2, u_occ, u_occ, u_vir, u_vir)
    else:
        t2 = None
    return t1, t2

def plot_histogram(values, bins=None, maxbarlength=50):
    if bins is None:
        bins = np.hstack([np.inf, np.logspace(-3, -12, 10), -np.inf])
    bins = bins[::-1]
    hist = np.histogram(values, bins)[0]
    bins, hist = bins[::-1], hist[::-1]
    cumsum = 0
    lines = ["  {:^13s}  {:^4s}   {:^51s}".format("Interval", "Sum", "Histogram").rstrip()]
    for i, hval in enumerate(hist):
        cumsum += hval
        barlength = int(maxbarlength * hval/hist.max())
        if hval == 0:
            bar = ""
        else:
            barlength = max(barlength, 1)
            bar = ((barlength-1) * "|") + "]" + ("  (%d)" % hval)
        #log.info("  %5.0e - %5.0e  %4d   |%s", bins[i+1], bins[i], cumsum, bar)
        lines.append("  %5.0e - %5.0e  %4d   |%s" % (bins[i+1], bins[i], cumsum, bar))
    return lines

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

def amplitudes_C2T(C1, C2):
    T1 = C1.copy()
    T2 = C2 - einsum("ia,jb->ijab", C1, C1)
    return T1, T2


def amplitudes_T2C(T1, T2):
    C1 = T1.copy()
    C2 = T2 + einsum("ia,jb->ijab", T1, T1)
    return C1, C2
