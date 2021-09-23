import numpy as np
import scipy
import scipy.linalg

from vayesta.core.util import *

class Fragmentation:
    """Fragmentation for a quantum embedding method class."""

    name = "<not set>"

    def __init__(self, mf, log):
        self.mf = mf
        self.log = log
        self.ovlp = self.mf.get_ovlp()
        #
        self.nfrag = 0
        self.coeff = None
        self.labels = None

    # --- For convenience:

    @property
    def mol(self):
        return self.mf.mol

    @property
    def nao(self):
        return self.mol.nao_nr()

    @property
    def nmo(self):
        return self.mf.mo_coeff.shape[-1]

    def get_ovlp(self):
        return self.ovlp

    @property
    def mo_coeff(self):
        return self.mf.mo_coeff

    @property
    def mo_occ(self):
        return self.mf.mo_occ

    def kernel(self):
        self.coeff = self.get_coeff()
        self.labels = self.get_labels()

    # --- These need to be implemented

    def get_coeff(self):
        """Abstract method."""
        raise NotImplementedError()

    def get_labels(self):
        """Abstract method."""
        raise NotImplementedError()

    def search_labels(self, labels):
        """Abstract method."""
        raise NotImplementedError()

    # ---

    def get_next_fid(self):
        """Get next free fragment ID."""
        fid = self.nfrag
        self.nfrag += 1
        return fid

    def get_atoms(self):
        """Get the base atom for each fragment orbital."""
        return [l[0] for l in self.labels]

    def symmetric_orth(self, mo_coeff, ovlp=None, tol=1e-15):
        """Use as mo_coeff = np.dot(mo_coeff, x) to get orthonormal orbitals."""
        if ovlp is None: ovlp = self.get_ovlp()
        m = dot(mo_coeff.T, ovlp, mo_coeff)
        e, v = scipy.linalg.eigh(m)
        e_min = e.min()
        keep = (e >= tol)
        e, v = e[keep], v[:,keep]
        x = dot(v/np.sqrt(e), v.T)
        return x, e_min

    def check_orth(self, mo_coeff, mo_name=None, tol=1e-7):
        """Check orthonormality of mo_coeff."""
        err = dot(mo_coeff.T, self.get_ovlp(), mo_coeff) - np.eye(mo_coeff.shape[-1])
        l2 = np.linalg.norm(err)
        linf = abs(err).max()
        if mo_name is None: mo_name = self.name
        if max(l2, linf) > tol:
            self.log.error("Orthogonality error of %ss: L(2)= %.2e  L(inf)= %.2e !", mo_name, l2, linf)
        else:
            self.log.debugv("Orthogonality error of %ss: L(2)= %.2e  L(inf)= %.2e", mo_name, l2, linf)
        return l2, linf

    def get_atom_indices_symbols(self, atoms):
        """Convert a list of integer or strings to atom indices and symbols."""
        if np.ndim(atoms) == 0: atoms = [atoms]

        if isinstance(atoms[0], (int, np.integer)):
            atom_indices = atoms
            atom_symbols = [self.mol.atom_symbol(atm) for atm in atoms]
            return atom_indices, atom_symbols
        if isinstance(atoms[0], str):
            atom_symbols = atoms
            all_atom_symbols = [self.mol.atom_symbol(atm) for atm in range(self.mol.natm)]
            for sym in atom_symbols:
                if sym not in all_atom_symbols:
                    raise ValueError("Cannot find atom with symbol %s in system." % sym)
            atom_indices = np.nonzero(np.isin(all_atom_symbols, atom_symbols))[0]
            return atom_indices, atom_symbols
        raise ValueError("A list of integers or string is required! atoms= %r" % atoms)

    def get_atomic_fragment_indices(self, atoms, orbital_filter=None, name=None):
        """Get fragment indices for one atom or a set of atoms.

        Parameters
        ----------
        atoms: list or int or str
            List of atom IDs or labels. For a single atom, a single integer or string can be passed as well.
        orbital_filter: list, optional
            Additionally restrict fragment orbitals to a specific orbital type (e.g. '2p'). Default: None.
        name: str, optional
            Name for fragment.

        Returns
        -------
        name: str
            Name of fragment.
        indices: list
            List of fragment orbitals indices, with coefficients corresponding to `self.coeff[:,indices]`.
        """
        atom_indices, atom_symbols = self.get_atom_indices_symbols(atoms)
        if name is None: name = '/'.join(atom_symbols)
        self.log.debugv("Atom indices of fragment %s: %r", name, atom_indices)
        self.log.debugv("Atom symbols of fragment %s: %r", name, atom_symbols)

        # Indices of IAOs based at atoms
        indices = np.nonzero(np.isin(self.get_atoms(), atom_indices))[0]
        # Filter orbital types
        if orbital_filter is not None:
            keep = self.search_labels(orbital_filter)
            indices = [i for i in indices if i in keep]

        # Some output
        self.log.debugv("Fragment %ss:\n%r", self.name, indices)
        self.log.debug("Fragment %ss of fragment %s:", self.name, name)
        for a, sym, nl, ml in np.asarray(self.labels)[indices]:
            if ml:
                self.log.debug("  %3s %4s %2s-%s", a, sym, nl, ml)
            else:
                self.log.debug("  %3s %4s %2s", a, sym, nl)

        return name, indices

    def get_orbital_indices_labels(self, orbitals):
        """Convert a list of integer or strings to orbital indices and labels."""
        if np.ndim(orbitals) == 0: orbitals = [orbitals]

        if isinstance(orbitals[0], (int, np.integer)):
            orbital_indices = orbitals
            orbital_labels = (np.asarray(self.labels, dtype=object)[orbitals]).tolist()
            orbital_labels = [('%d%3s %s%-s' % tuple(l)).strip() for l in orbital_labels]
            return orbital_indices, orbital_labels
        if isinstance(orbitals[0], str):
            orbital_labels = orbitals
            # Check labels
            for l in orbital_labels:
                if len(self.search_labels(l)) == 0:
                    raise ValueError("Cannot find orbital with label %s in system." % l)
            orbital_indices = self.search_labels(orbital_labels)
            return orbital_indices, orbital_labels
        raise ValueError("A list of integers or string is required! orbitals= %r" % orbitals)

    def get_orbital_fragment_indices(self, orbitals, atom_filter=None, name=None):
        if atom_filter is not None:
            raise NotImplementedError()
        indices, orbital_labels = self.get_orbital_indices_labels(orbitals)
        if name is None: name = '/'.join(orbital_labels)
        self.log.debugv("Orbital indices of fragment %s: %r", name, indices)
        self.log.debugv("Orbital labels of fragment %s: %r", name, orbital_labels)
        return name, indices

    def get_frag_coeff(self, indices):
        """Get fragment coefficients for a given set of orbital indices."""
        c_frag = self.coeff[:,indices].copy()
        return c_frag

    def get_env_coeff(self, indices):
        """Get environment coefficients for a given set of orbital indices."""
        env = np.ones((self.coeff.shape[-1]), dtype=bool)
        env[indices] = False
        c_env = self.coeff[:,env].copy()
        return c_env
