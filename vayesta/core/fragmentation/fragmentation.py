import contextlib
import numpy as np
import scipy
import scipy.linalg

from vayesta.core.util import dot, fix_orbital_sign, time_string, timer
from vayesta.core.fragmentation import helper

def check_orthonormal(log, mo_coeff, ovlp, mo_name="orbital", tol=1e-7):
    """Check orthonormality of mo_coeff.

    Supports both RHF and UHF.
    """
    # RHF
    if np.ndim(mo_coeff[0]) == 1:
        err = dot(mo_coeff.T, ovlp, mo_coeff) - np.eye(mo_coeff.shape[-1])
        l2 = np.linalg.norm(err)
        linf = abs(err).max()
        if max(l2, linf) > tol:
            log.error("Orthogonality error of %ss: L(2)= %.2e  L(inf)= %.2e !", mo_name, l2, linf)
        else:
            log.debugv("Orthogonality error of %ss: L(2)= %.2e  L(inf)= %.2e", mo_name, l2, linf)
        return l2, linf
    # UHF
    l2a, linfa = check_orthonormal(log, mo_coeff[0], ovlp, mo_name='alpha-%s' % mo_name, tol=tol)
    l2b, linfb = check_orthonormal(log, mo_coeff[1], ovlp, mo_name='beta-%s' % mo_name, tol=tol)
    return (l2a, l2b), (linfa, linfb)


class Fragmentation:
    """Fragmentation for a quantum embedding method class."""

    name = "<not set>"

    def __init__(self, emb, add_symmetric=True, log=None):
        self.emb = emb
        self.add_symmetric = add_symmetric
        self.log = log or emb.log
        self.log.info('%s Fragmentation' % self.name)
        self.log.info('%s--------------' % (len(self.name)*'-'))
        self.ovlp = self.mf.get_ovlp()
        # Secondary fragment state:
        self.secfrag_register = []
        # Rotational symmetry state:
        self.sym_register = []
        # -- Output:
        self.coeff = None
        self.labels = None
        # Generated fragments:
        self.fragments = []

    def kernel(self):
        self.coeff = self.get_coeff()
        self.labels = self.get_labels()

    # --- As contextmanager:

    def __enter__(self):
        self.log.changeIndentLevel(1)
        self._time0 = timer()
        self.kernel()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            return

        if self.add_symmetric:
            # Rotational + inversion symmetries are now done within own context
            # Translation symmetry are done in the fragmentation context (here):
            translation = self.emb.symmetry.translation
            if translation is not None:
                fragments_sym = self.emb.create_transsym_fragments(translation, fragments=self.fragments)
                self.log.info("Adding %d translationally-symmetry related fragments from %d base fragments",
                              len(fragments_sym), len(self.fragments))
                self.fragments.extend(fragments_sym)

        # Add fragments to embedding class
        self.log.debug("Adding %d fragments to embedding class", len(self.fragments))
        self.emb.fragments.extend(self.fragments)

        # Check if fragmentation is (occupied) complete and orthonormal:
        orth = self.emb.has_orthonormal_fragmentation()
        comp = self.emb.has_complete_fragmentation()
        occcomp = self.emb.has_complete_occupied_fragmentation()
        self.log.info("Fragmentation: orthogonal= %r, occupied-complete= %r, virtual-complete= %r",
                      self.emb.has_orthonormal_fragmentation(),
                      self.emb.has_complete_occupied_fragmentation(),
                      self.emb.has_complete_virtual_fragmentation())
        self.log.timing("Time for %s fragmentation: %s", self.name, time_string(timer()-self._time0))
        del self._time0
        self.log.changeIndentLevel(-1)

    # --- Adding fragments:

    def add_atomic_fragment(self, atoms, orbital_filter=None, name=None, **kwargs):
        """Create a fragment of one or multiple atoms, which will be solved by the embedding method.

        Parameters
        ----------
        atoms: int, str, list[int], or list[str]
            Atom indices or symbols which should be included in the fragment.
        name: str, optional
            Name for the fragment. If None, a name is automatically generated from the chosen atoms. Default: None.
        **kwargs:
            Additional keyword arguments are passed through to the fragment constructor.

        Returns
        -------
        Fragment:
            Fragment object.
        """
        atom_indices, atom_symbols = self.get_atom_indices_symbols(atoms)
        name, indices = self.get_atomic_fragment_indices(atoms, orbital_filter=orbital_filter, name=name)
        return self._create_fragment(indices, name, atoms=atom_indices, **kwargs)

    def add_atomshell_fragment(self, atoms, shells, **kwargs):
        if isinstance(shells, (int, np.integer)):
            shells = [shells]
        orbitals = []
        atom_indices, atom_symbols = self.get_atom_indices_symbols(atoms)
        for idx, sym in zip(atom_indices, atom_symbols):
            for shell in shells:
                orbitals.append('%d%3s %s' % (idx, sym, shell))
        return self.add_orbital_fragment(orbitals, atoms=atom_indices, **kwargs)

    def add_orbital_fragment(self, orbitals, atom_filter=None, name=None, **kwargs):
        """Create a fragment of one or multiple orbitals, which will be solved by the embedding method.

        Parameters
        ----------
        orbitals: int, str, list[int], or list[str]
            Orbital indices or labels which should be included in the fragment.
        name: str, optional
            Name for the fragment. If None, a name is automatically generated from the chosen orbitals. Default: None.
        **kwargs:
            Additional keyword arguments are passed through to the fragment constructor.

        Returns
        -------
        Fragment:
            Fragment object.
        """
        name, indices = self.get_orbital_fragment_indices(orbitals, atom_filter=atom_filter, name=name)
        return self._create_fragment(indices, name, **kwargs)

    def add_all_atomic_fragments(self, **kwargs):
        """Create a single fragment for each atom in the system.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed through to each fragment constructor.
        """
        fragments = []
        natom = self.emb.kcell.natm if self.emb.kcell is not None else self.emb.mol.natm
        for atom in range(natom):
            frag = self.add_atomic_fragment(atom, **kwargs)
            fragments.append(frag)
        return fragments

    def add_full_system(self, name='full-system', **kwargs):
        atoms = list(range(self.mol.natm))
        return self.add_atomic_fragment(atoms, name=name, **kwargs)

    def _create_fragment(self, indices, name, **kwargs):
        if len(indices) == 0:
            raise ValueError("Fragment %s is empty." % name)
        c_frag = self.get_frag_coeff(indices)
        c_env = self.get_env_coeff(indices)
        fid, mpirank = self.emb.register.get_next()
        frag = self.emb.Fragment(self.emb, fid, name, c_frag, c_env, mpi_rank=mpirank, **kwargs)
        self.fragments.append(frag)

        # Log fragment orbitals:
        self.log.debugv("Fragment %ss:\n%r", self.name, indices)
        self.log.debug("Fragment %ss of fragment %s:", self.name, name)
        labels = np.asarray(self.labels)[indices]
        helper.log_orbitals(self.log.debug, labels)
        # Secondary fragments:
        if self.secfrag_register:
            self.secfrag_register[0].append(frag)
        # Symmetric fragments:
        for sym in self.sym_register:
            sym.append(frag)

        return frag

    # --- Rotational symmetry fragments:

    @contextlib.contextmanager
    def rotational_symmetry(self, order, axis, center=(0,0,0), unit='Ang'):
        if self.secfrag_register:
            raise NotImplementedError("Rotational symmetries have to be added before adding secondary fragments")
        self.sym_register.append([])
        yield
        fragments = self.sym_register.pop()
        fragments_sym = self.emb.create_rotsym_fragments(order, axis, center, fragments=fragments, unit=unit,
                                                         symbol='R%d' % len(self.sym_register))
        self.log.info("Adding %d rotationally-symmetric fragments", len(fragments_sym))
        self.fragments.extend(fragments_sym)
        # For additional (nested) symmetries:
        for sym in self.sym_register:
            sym.extend(fragments_sym)

    @contextlib.contextmanager
    def inversion_symmetry(self, center=(0,0,0), unit='Ang'):
        if self.secfrag_register:
            raise NotImplementedError("Symmetries have to be added before adding secondary fragments")
        self.sym_register.append([])
        yield
        fragments = self.sym_register.pop()
        fragments_sym = self.emb.create_invsym_fragments(center, fragments=fragments, unit=unit, symbol='I')
        self.log.info("Adding %d inversion-symmetric fragments", len(fragments_sym))
        self.fragments.extend(fragments_sym)
        # For additional (nested) symmetries:
        for sym in self.sym_register:
            sym.extend(fragments_sym)

    @contextlib.contextmanager
    def mirror_symmetry(self, axis, center=(0,0,0), unit='Ang'):
        if self.secfrag_register:
            raise NotImplementedError("Symmetries have to be added before adding secondary fragments")
        self.sym_register.append([])
        yield
        fragments = self.sym_register.pop()
        fragments_sym = self.emb.create_mirrorsym_fragments(axis, center=center, fragments=fragments, unit=unit,
                                                            symbol='M')
        self.log.info("Adding %d mirror-symmetric fragments", len(fragments_sym))
        self.fragments.extend(fragments_sym)
        # For additional (nested) symmetries:
        for sym in self.sym_register:
            sym.extend(fragments_sym)


    # --- Secondary fragments:

    @contextlib.contextmanager
    def secondary_fragments(self, bno_threshold=None, bno_threshold_factor=0.1, solver='MP2'):
        if self.secfrag_register:
            raise NotImplementedError("Nested secondary fragments")
        self.secfrag_register.append([])
        yield
        fragments = self.secfrag_register.pop()
        fragments_sec = self._create_secondary_fragments(fragments, bno_threshold=bno_threshold,
                                                         bno_threshold_factor=bno_threshold_factor, solver=solver)
        self.log.info("Adding %d secondary fragments", len(fragments_sec))
        self.fragments.extend(fragments_sec)
        # If we are already in a symmetry context, the symmetry-related secondary fragments should also be added:
        for sym in self.sym_register:
            sym.extend(fragments_sec)

    def _create_secondary_fragments(self, fragments, bno_threshold=None, bno_threshold_factor=0.1, solver='MP2'):

        def _create_fragment(fx, flags=None, **kwargs):
            if fx.sym_parent is not None:
                raise NotImplementedError("Secondary fragments need to be added before symmetry-derived fragments")
            flags = (flags or {}).copy()
            flags['is_secfrag'] = True
            fx_copy = fx.copy(solver=solver, flags=flags, **kwargs)
            fx_copy.flags.bath_parent_fragment_id = fx.id
            self.log.debugv("Adding secondary fragment: %s", fx_copy)
            return fx_copy

        fragments_sec = []
        for fx in fragments:

            bath_opts = fx.opts.bath_options.copy()
            if bno_threshold is not None:
                bath_opts['threshold'] = bno_threshold
                bath_opts.pop('threshold_occ', None)
                bath_opts.pop('threshold_vir', None)
            else:
                if bath_opts.get('threshold', None) is not None:
                    bath_opts['threshold'] *= bno_threshold_factor
                if bath_opts.get('threshold_occ', None) is not None:
                    bath_opts['threshold_occ'] *= bno_threshold_factor
                if bath_opts.get('threshold_vir', None) is not None:
                    bath_opts['threshold_vir'] *= bno_threshold_factor
            frag = _create_fragment(fx, name='%s[x%d|+%s]' % (fx.name, fx.id, solver), bath_options=bath_opts)
            fragments_sec.append(frag)
            # Double counting
            wf_factor = -(fx.opts.wf_factor or 1)
            frag = _create_fragment(fx, name='%s[x%d|-%s]' % (fx.name, fx.id, solver), wf_factor=wf_factor, flags=dict(is_envelop=False))
            fragments_sec.append(frag)
            fx.flags.is_envelop = False
        return fragments_sec

    # --- For convenience:

    @property
    def mf(self):
        return self.emb.mf

    @property
    def mol(self):
        return self.mf.mol

    @property
    def nao(self):
        return self.mol.nao_nr()

    @property
    def nmo(self):
        return self.mo_coeff.shape[-1]

    def get_ovlp(self):
        return self.ovlp

    @property
    def mo_coeff(self):
        return self.mf.mo_coeff

    @property
    def mo_occ(self):
        return self.mf.mo_occ

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
        x = fix_orbital_sign(x)[0]
        return x, e_min

    #def check_orth(self, mo_coeff, mo_name=None, tol=1e-7):
    #    """Check orthonormality of mo_coeff."""
    #    err = dot(mo_coeff.T, self.get_ovlp(), mo_coeff) - np.eye(mo_coeff.shape[-1])
    #    l2 = np.linalg.norm(err)
    #    linf = abs(err).max()
    #    if mo_name is None: mo_name = self.name
    #    if max(l2, linf) > tol:
    #        self.log.error("Orthogonality error of %ss: L(2)= %.2e  L(inf)= %.2e !", mo_name, l2, linf)
    #    else:
    #        self.log.debugv("Orthogonality error of %ss: L(2)= %.2e  L(inf)= %.2e", mo_name, l2, linf)
    #    return l2, linf

    def check_orthonormal(self, mo_coeff, mo_name=None, tol=1e-7):
        if mo_name is None: mo_name = self.name
        return check_orthonormal(self.log, mo_coeff, self.get_ovlp(), mo_name=mo_name, tol=tol)

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
