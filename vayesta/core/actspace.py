import numpy as np

from .orbitals import OrbitalSpace


def ActiveSpace(mf, *args, **kwargs):
    if mf.mo_coeff[0].ndim == 1:
        return ActiveSpace_RHF(mf, *args, **kwargs)
    return ActiveSpace_UHF(mf, *args, **kwargs)

class ActiveSpace_RHF:

    def __init__(self, mf, c_active_occ, c_active_vir, c_frozen_occ=None, c_frozen_vir=None):
        self.mf = mf
        # Active
        self._active_occ = OrbitalSpace(c_active_occ, name="active-occupied")
        self._active_vir = OrbitalSpace(c_active_vir, name="active-virtual")
        # Frozen
        if c_frozen_occ is 0:
            c_frozen_occ = np.zeros((self.nao, 0))
            if self.is_uhf:
                c_frozen_occ = (c_frozen_occ, c_frozen_occ)
        if c_frozen_occ is not None:
            self._frozen_occ = OrbitalSpace(c_frozen_occ, name="frozen-occupied")
        else:
            self._frozen_occ = None
        if c_frozen_vir is 0:
            c_frozen_vir = np.zeros((self.nao, 0))
            if self.is_uhf:
                c_frozen_vir = (c_frozen_vir, c_frozen_vir)
        if c_frozen_vir is not None:
            self._frozen_vir = OrbitalSpace(c_frozen_vir, name="frozen-virtual")
        else:
            self._frozen_vir = None

    # --- General

    def __repr__(self):
        return ("ActiveSpace(nocc_active= %d, nvir_active= %d, nocc_frozen= %d, nvir_frozen= %d)" %
            (self.nocc_active, self.nvir_active, self.nocc_frozen, self.nvir_frozen))

    # --- Mean-field:

    @property
    def is_rhf(self):
        return True

    @property
    def is_uhf(self):
        return False

    @property
    def mol(self):
        """PySCF Mole or Cell object."""
        return self.mf.mol

    @property
    def nao(self):
        """Number of atomic orbitals."""
        return self.mol.nao_nr()

    @property
    def norb(self):
        """Total number of molecular orbitals in the system."""
        return self.mf.mo_coeff.shape[-1]

    @property
    def nocc(self):
        """Total number of occupied molecular orbitals in the system."""
        return np.count_nonzero(self.mf.mo_occ > 0)

    @property
    def nvir(self):
        """Total number of virtual molecular orbitals in the system."""
        return np.count_nonzero(self.mf.mo_occ == 0)

    # --- Sizes:

    @property
    def norb_active(self):
        """Number of active orbitals."""
        return (self.nocc_active + self.nvir_active)

    @property
    def nocc_active(self):
        """Number of active occupied orbitals."""
        return self._active_occ.size

    @property
    def nvir_active(self):
        """Number of active virtual orbitals."""
        return self._active_vir.size

    @property
    def norb_frozen(self):
        """Number of frozen orbitals."""
        return (self.norb - self.norb_active)

    @property
    def nocc_frozen(self):
        """Number of frozen occupied orbitals."""
        return (self.nocc - self.nocc_active)

    @property
    def nvir_frozen(self):
        """Number of frozen virtual orbitals."""
        return (self.nvir - self.nvir_active)

    # --- Indices and slices

    def get_active_slice(self):
        return np.s_[self.nocc_frozen:self.nocc_frozen+self.norb_active]

    def get_active_indices(self):
        return list(range(self.nocc_frozen, self.nocc_frozen+self.norb_active))

    def get_frozen_indices(self):
        return list(range(self.nocc_frozen)) + list(range(self.norb-self.nvir_frozen, self.norb))

    # --- Orbital coefficients

    # All:

    @property
    def coeff(self):
        return self.all.coeff

    @property
    def c_occ(self):
        return self.occupied.coeff

    @property
    def c_vir(self):
        return self.virtual.coeff

    # Active:

    @property
    def c_active(self):
        return self.active.coeff

    @property
    def c_active_occ(self):
        return self.active.occupied.coeff

    @property
    def c_active_vir(self):
        return self.active.virtual.coeff

    # Frozen:

    @property
    def c_frozen(self):
        return self.frozen.coeff

    @property
    def c_frozen_occ(self):
        return self.frozen.occupied.coeff

    @property
    def c_frozen_vir(self):
        return self.frozen.virtual.coeff

    # --- Electron numbers:

    @property
    def nelectron(self):
        """Total number of electrons in the system."""
        return 2*self.nocc

    @property
    def nelectron_active(self):
        """Number of active electrons in the system."""
        return 2*self.nocc_active

    @property
    def nelectron_frozen(self):
        """Number of frozen electrons in the system."""
        return 2*self.nocc_frozen

    # --- Combined spaces:

    @property
    def active(self):
        active = (self._active_occ + self._active_vir)
        active.occupied = self._active_occ
        active.virtual = self._active_vir
        return active

    @property
    def frozen(self):
        frozen = (self._frozen_occ + self._frozen_vir)
        frozen.occupied = self._frozen_occ
        frozen.virtual = self._frozen_vir
        return frozen

    @property
    def occupied(self):
        occupied = (self._frozen_occ + self._active_occ)
        occupied.active = self._active_occ
        occupied.frozen = self._frozen_occ
        return occupied

    @property
    def virtual(self):
        virtual = (self._active_vir + self._frozen_vir)
        virtual.active = self._active_occ
        virtual.frozen = self._frozen_occ
        return virtual

    @property
    def all(self):
        return (self._frozen_occ + self._active_occ + self._active_vir + self._frozen_vir)

    # --- Other:

    def add_frozen_rdm1(self, dm1_active):
        if dm1_active.shape != (self.norb_active, self.norb_active):
            raise ValueError()
        dm1 = np.zeros((self.norb, self.norb))
        dm1[np.diag_indices(self.nocc)] = 2
        act = self.get_active_slice()
        dm1[act,act] = dm1_active
        return dm1

    def get_cas_size(self):
        return (self.nelectron_active, self.norb_active)

    # --- Modify

    def copy(self):
        copy = ActiveSpace(self.mf, self.c_active_occ, self.c_active_vir, self.c_frozen_occ, self.c_frozen_vir)
        return copy

    def transform(self, trafo):
        cp = self.copy()
        cp._active_occ.transform(trafo, inplace=True)
        cp._active_vir.transform(trafo, inplace=True)
        if cp._frozen_occ is not None:
            cp._frozen_occ.transform(trafo, inplace=True)
        if cp._frozen_vir is not None:
            cp._frozen_vir.transform(trafo, inplace=True)
        return cp

    def log_sizes(self, logger, header=None):
        if header:
            logger(header)
            logger(len(header)*'-')
        logger("             Active                    Frozen")
        logger("             -----------------------   -----------------------")
        fmt = '  %-8s' + 2*'   %5d / %5d (%6.1f%%)'
        get_sizes = lambda a, f, n : (a, n, 100*a/n, f, n, 100*f/n)
        logger(fmt, "Occupied", *get_sizes(self.nocc_active, self.nocc_frozen, self.nocc))
        logger(fmt, "Virtual",  *get_sizes(self.nvir_active, self.nvir_frozen, self.nvir))
        logger(fmt, "Total",    *get_sizes(self.norb_active, self.norb_frozen, self.norb))

    #def check(self):
    #    assert self.norb_active == self.active.size == self.active.coeff.shape[-1]
    #    assert self.nocc_active == self.active.occupied.size == self.active.occupied.coeff.shape[-1] == self.c_active_occ.shape[-1]
    #    assert self.nvir_active == self.active.virtual.size == self.active.virtual.coeff.shape[-1]

class ActiveSpace_UHF(ActiveSpace_RHF):

    def __repr__(self):
        return ("ActiveSpace(nocc_active= (%d, %d), nvir_active= (%d, %d), nocc_frozen= (%d, %d), nvir_frozen= (%d, %d))" %
            (*self.nocc_active, *self.nvir_active, *self.nocc_frozen, *self.nvir_frozen))

    @property
    def is_rhf(self):
        return False

    @property
    def is_uhf(self):
        return True

    @property
    def norb(self):
        return np.asarray((self.mf.mo_coeff[0].shape[-1],
                           self.mf.mo_coeff[1].shape[-1]))

    @property
    def nocc(self):
        return np.asarray((np.count_nonzero(self.mf.mo_occ[0] > 0),
                           np.count_nonzero(self.mf.mo_occ[1] > 0)))

    @property
    def nvir(self):
        return np.asarray((np.count_nonzero(self.mf.mo_occ[0] == 0),
                           np.count_nonzero(self.mf.mo_occ[1] == 0)))

    # --- Indices and slices

    def get_active_slice(self):
        return (np.s_[self.nocc_frozen[0]:self.nocc_frozen[0]+self.norb_active[0]],
                np.s_[self.nocc_frozen[1]:self.nocc_frozen[1]+self.norb_active[1]])

    def get_active_indices(self):
        return (list(range(self.nocc_frozen[0], self.nocc_frozen[0]+self.norb_active[0])),
                list(range(self.nocc_frozen[1], self.nocc_frozen[1]+self.norb_active[1])))

    def get_frozen_indices(self):
        return (list(range(self.nocc_frozen[0])) + list(range(self.norb[0]-self.nvir_frozen[0], self.norb[0])),
                list(range(self.nocc_frozen[1])) + list(range(self.norb[1]-self.nvir_frozen[1], self.norb[1])))

    # --- Electron numbers:

    @property
    def nelectron(self):
        """Total number of electrons in the system."""
        return self.nocc

    @property
    def nelectron_active(self):
        """Number of active electrons in the system."""
        return self.nocc_active

    @property
    def nelectron_frozen(self):
        """Number of frozen electrons in the system."""
        return self.nocc_frozen

    # --- Other:

    def add_frozen_rdm1(self, dm1_active):
        if (dm1_active[0].shape != (self.norb_active[0], self.norb_active[0])
         or dm1_active[1].shape != (self.norb_active[1], self.norb_active[1])):
            raise ValueError("Invalid DM shape: %r %r. N(active)= %d %d" % (
                list(dm1_active[0].shape), list(dm1_active[1].shape), *self.norb_active))
        dm1a = np.zeros((self.norb[0], self.norb[0]))
        dm1b = np.zeros((self.norb[1], self.norb[1]))
        dm1a[np.diag_indices(self.nocc[0])] = 1
        dm1b[np.diag_indices(self.nocc[1])] = 1
        acta, actb = self.get_active_slice()
        dm1a[acta,acta] = dm1_active[0]
        dm1b[actb,actb] = dm1_active[1]
        return (dm1a, dm1b)

    def log_sizes(self, logger, header=None):
        if header:
            logger(header)
            logger(len(header)*'-')
        logger("                   Active                    Frozen")
        logger("                   -----------------------   -----------------------")
        fmt = '  %-14s' + 2*'   %5d / %5d (%6.1f%%)'
        get_sizes = lambda a, f, n : (a, n, 100*a/n, f, n, 100*f/n)
        logger(fmt, "Alpha occupied", *get_sizes(self.nocc_active[0], self.nocc_frozen[0], self.nocc[0]))
        logger(fmt, "Beta  occupied", *get_sizes(self.nocc_active[1], self.nocc_frozen[1], self.nocc[1]))
        logger(fmt, "Alpha virtual",  *get_sizes(self.nvir_active[0], self.nvir_frozen[0], self.nvir[0]))
        logger(fmt, "Beta  virtual",  *get_sizes(self.nvir_active[1], self.nvir_frozen[1], self.nvir[1]))
        logger(fmt, "Alpha total",    *get_sizes(self.norb_active[0], self.norb_frozen[0], self.norb[0]))
        logger(fmt, "Beta  total",    *get_sizes(self.norb_active[1], self.norb_frozen[1], self.norb[1]))


##class Cluster:
##
##    def __init__(self, mf, c_active_occ, c_active_vir, c_frozen_occ, c_frozen_vir, log, sym_op=None, sym_parent=None):
##        self.mf = mf
##        self.log = log
##        self._c_active_occ = c_active_occ
##        self._c_active_vir = c_active_vir
##        self._c_frozen_occ = c_frozen_occ
##        self._c_frozen_vir = c_frozen_vir
##        self.sym_op = sym_op
##        self.sym_parent = sym_parent
##
##    # --- Mean-field:
##
##    @property
##    def mol(self):
##        """PySCF Mole or Cell object."""
##        return self.mf.mol
##
##    @property
##    def nao(self):
##        """Number of atomic orbitals."""
##        return self.mol.nao_nr()
##
##    #@property
##    #def mo_coeff(self):
##    #    return self.mf.mo_coeff
##
##    # --- Sizes:
##
##    @property
##    def norb(self):
##        """Total number of occupied orbitals in the system."""
##        return self.nocc + self.nvir
##
##    @property
##    def nocc(self):
##        """Total number of occupied orbitals in the system."""
##        return self.nocc_active + self.nocc_frozen
##
##    @property
##    def nvir(self):
##        """Total number of virtual orbitals in the system."""
##        return self.nvir_active + self.nvir_frozen
##
##    @property
##    def norb_active(self):
##        """Number of active orbitals."""
##        return (self.nocc_active + self.nvir_active)
##
##    @property
##    def nocc_active(self):
##        """Number of active occupied orbitals."""
##        return self.c_active_occ.shape[-1]
##
##    @property
##    def nvir_active(self):
##        """Number of active virtual orbitals."""
##        return self.c_active_vir.shape[-1]
##
##    @property
##    def norb_frozen(self):
##        """Number of frozen orbitals."""
##        return (self.nocc_frozen + self.nvir_frozen)
##
##    @property
##    def nocc_frozen(self):
##        """Number of frozen occupied orbitals."""
##        return self.c_frozen_occ.shape[-1]
##
##    @property
##    def nvir_frozen(self):
##        """Number of frozen virtual orbitals."""
##        return self.c_frozen_vir.shape[-1]
##
##    # --- Electrons:
##
##    @property
##    def nelectron(self):
##        """Total number of electrons in the system."""
##        return 2*self.nocc
##
##    @property
##    def nelectron_active(self):
##        """Number of active electrons in the system."""
##        return 2*self.nocc_active
##
##    @property
##    def nelectron_frozen(self):
##        """Number of frozen electrons in the system."""
##        return 2*self.nocc_frozen
##
##    # --- Orbital coefficients:
##
##    @property
##    def c_active(self):
##        """Active orbital coefficients."""
##        if self.c_active_occ is None:
##            return None
##        return hstack(self.c_active_occ, self.c_active_vir)
##
##    @property
##    def c_active_occ(self):
##        """Active occupied orbital coefficients."""
##        if self.sym_parent is None:
##            return self._c_active_occ
##        else:
##            return self.sym_op(self.sym_parent.c_active_occ)
##
##    @property
##    def c_active_vir(self):
##        """Active virtual orbital coefficients."""
##        if self.sym_parent is None:
##            return self._c_active_vir
##        else:
##            return self.sym_op(self.sym_parent.c_active_vir)
##
##    @property
##    def c_frozen(self):
##        """Frozen orbital coefficients."""
##        if self.c_frozen_occ is None:
##            return None
##        return hstack(self.c_frozen_occ, self.c_frozen_vir)
##
##    @property
##    def c_frozen_occ(self):
##        """Frozen occupied orbital coefficients."""
##        if self.sym_parent is None:
##            return self._c_frozen_occ
##        else:
##            return self.sym_op(self.sym_parent.c_frozen_occ)
##
##    @property
##    def c_frozen_vir(self):
##        """Frozen virtual orbital coefficients."""
##        if self.sym_parent is None:
##            return self._c_frozen_vir
##        else:
##            return self.sym_op(self.sym_parent.c_frozen_vir)
##
##    def log_sizes(self, log, header=None):
##        if header:
##            log.info(header)
##            log.info(len(header)*'-')
##        log.info("             Active                    Frozen")
##        log.info("             -----------------------   -----------------------")
##        #log.info(13*' ' + "%-23s   %s", "Active", "Frozen")
##        #log.info(13*' ' + "%-23s   %s", 23*'-', 23*'-')
##        fmt = '  %-8s' + 2*'   %5d / %5d (%6.1f%%)'
##        get_sizes = lambda a, f, n : (a, n, 100*a/n, f, n, 100*f/n)
##        log.info(fmt, "Occupied", *get_sizes(self.nocc_active, self.nocc_frozen, self.nocc))
##        log.info(fmt, "Virtual", *get_sizes(self.nvir_active, self.nvir_frozen, self.nvir))
##        log.info(fmt, "Total", *get_sizes(self.norb_active, self.norb_frozen, self.norb))
