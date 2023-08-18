import numpy as np


class BosonicOrbitals:
    """Base class for representing bosonic rotations, similar to Orbitals.
    Note that unlike fermionic indices our final degrees of freedom can be formed as a combination of both excitations
    and deexcitations in our original bosonic basis.
    Name subject to change...
    """
    def __init__(self, coeff_ex, coeff_dex=None, energy=None, labels=None):
        self.coeff_ex = np.asarray(coeff_ex, dtype=float)
        self.coeff_dex = coeff_dex
        self.energy = np.asarray(energy, dtype=float) if energy is not None else None
        self.labels = labels

    @property
    def nbos(self):
        return self.coeff_ex.shape[0]

    @property
    def nex(self):
        return self.coeff_ex.shape[1]

    def copy(self):
        return type(self)(coeff_ex=_copy(self.coeff_ex), coeff_dex=_copy(self.coeff_dex), energy=_copy(self.energy),
                          labels=_copy(self.labels))

    def fbasis_transform(self, *args, **kwargs):
        """This class represents a true bosonic excitation, so transformations of the fermionic basis have no effect."""
        pass


class QuasiBosonOrbitals(BosonicOrbitals):
    """Class to represent quasi-bosonic excitations.
    Includes specification of orbital space
    """

    def __init__(self, forbitals, *args, **kwargs):
        # Ensure we have spin orbitals object, as will have spin-dependence here.
        if hasattr(forbitals, "alpha"):
            self.forbitals = self.forbitals
        else:
            self.forbitals = forbitals.to_spin_orbitals()
        super().__init__(*args, **kwargs)

    @property
    def has_dex(self):
        return self.coeff_dex is not None

    @property
    def ova(self):
        return self.forbitals.alpha.nocc * self.forbitals.alpha.nvir

    @property
    def ovb(self):
        return self.forbitals.beta.nocc * self.forbitals.beta.nvir

    @property
    def coeff_ex_3d(self):
        return bcoeff_ov_to_o_v(self.coeff_ex, self.forbitals.nocc, self.forbitals.nvir)

    @property
    def coeff_dex_3d(self):
        return None if self.coeff_dex is None else bcoeff_ov_to_o_v(self.coeff_dex, self.forbitals.nocc, self.forbitals.nvir)

    @property
    def coeff_3d_ao(self):
        """Get bosonic coefficient in basis of ao excitations"""
        alpha, beta = bcoeff_mo2ao(self.coeff_ex_3d, self.forbitals.coeff, self.forbitals.coeff)
        if self.has_dex:
            dexa, dexb = bcoeff_mo2ao(self.coeff_dex_3d, self.forbitals.coeff, self.forbitals.coeff, transpose=True)
            alpha += dexa
            beta += dexb
        return alpha, beta

    def copy(self):
        return type(self)(forbitals=self.forbitals.copy(), coeff_ex=_copy(self.coeff_ex),
                          coeff_dex=_copy(self.coeff_dex), energy=_copy(self.energy), labels=_copy(self.labels))

    def fbasis_transform(self, trafo, inplace=False):
        if not hasattr(trafo, '__len__'):
            trafo = (trafo, trafo)
        cp = self if inplace else self.copy()
        cp.forbitals.basis_transform(trafo[0], inplace=True)
        return cp

def bcoeff_ov_to_o_v(cbos, no, nv):
    noa, nob = no if isinstance(no, tuple) else (no, no)
    nva, nvb = nv if isinstance(nv, tuple) else (nv, nv)
    nbos = cbos.shape[0]
    ova = noa * nva
    ca, cb = cbos[:, :ova], cbos[:, ova:]
    return ca.reshape((nbos, noa, nva)), cb.reshape((nbos, nob, nvb))

def bcoeff_mo2ao(cbos, co, cv, transpose=False):
    def _spinchannel_bcoeff_mo2ao(cbos, co, cv, transpose=False):
        """Convert bosonic coefficients from MO basis to AO basis."""
        cbos = np.tensordot(np.tensordot(cbos, co, (1, 1)), cv, (1, 1))
        if transpose:
            cbos = cbos.transpose((0, 2, 1))
        return cbos

    return _spinchannel_bcoeff_mo2ao(cbos[0], co[0], cv[0], transpose=transpose), \
        _spinchannel_bcoeff_mo2ao(cbos[1], co[1], cv[1], transpose=transpose)


def _copy(x):
    if x is None:
        return None
    if np.isscalar(x):
        return x
    if isinstance(x, tuple):
        return tuple(_copy(y) for y in x)
    if isinstance(x, list):
        return [_copy(y) for y in x]
    if isinstance(x, np.ndarray):
        return x.copy()
    raise ValueError
