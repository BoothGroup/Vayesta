import numpy as np

import vayesta
from vayesta.core.util import *
from vayesta.core.helper import pack_arrays, unpack_arrays

__all__ = [
        'Orbitals', 'SpatialOrbitals', 'SpinOrbitals', 'GeneralOrbitals',
        ]

class MolecularOrbitals:
    """Abstract base class"""

    def __repr__(self):
        return "%s(norb= %r, nocc= %r, nvir= %r)" % (self.__class__.__name__, self.norb, self.nocc, self.nvir)

def Orbitals(coeff, *args, **kwargs):
    if np.ndim(coeff[0]) == 2:
        return SpinOrbitals(coeff, *args, **kwargs)
    return SpatialOrbitals(coeff, *args, **kwargs)

class SpatialOrbitals(MolecularOrbitals):

    def __init__(self, coeff, energy=None, occ=None, labels=None, maxocc=2):
        self.coeff = np.asarray(coeff, dtype=float)
        self.energy = np.asarray(energy, dtype=float)
        self.maxocc = maxocc
        if isinstance(occ, (int, np.integer)):
            occ = np.asarray(occ*[self.maxocc] + (self.norb-occ)*[0])
        self.occ = np.asarray(occ, dtype=float)
        self.labels = labels

    @property
    def nspin(self):
        return 1

    @property
    def norb(self):
        return self.coeff.shape[-1]

    @property
    def nocc(self):
        if self.occ is None:
            return None
        return np.count_nonzero(self.occ > 0)

    @property
    def nvir(self):
        if self.occ is None:
            return None
        return np.count_nonzero(self.occ == 0)

    @property
    def nelec(self):
        ne = self.occ.sum()
        if abs(np.rint(ne)-ne) < 1e-14:
            return int(np.rint(ne))
        return ne

    @property
    def coeff_occ(self):
        return self.coeff[:,:self.nocc]

    @property
    def coeff_vir(self):
        return self.coeff[:,self.nocc:]

    def to_spin_orbitals(self):
        return SpinOrbitals.from_spatial_orbitals(self)

    def to_general_orbitals(self):
        return GeneralOrbitals.from_spatial_orbitals(self)

    def pack(self, dtype=float):
        """Pack into a single array of data type `dtype`.

        Useful for communication via MPI."""
        data = (self.coeff, self.energy, self.occ)
        return pack_arrays(*data, dtype=dtype)

    @classmethod
    def unpack(cls, packed):
        """Unpack from a single array of data type `dtype`.

        Useful for communication via MPI."""
        coeff, energy, occ = unpack_arrays(packed)
        return cls(coeff, energy=energy, occ=occ)


class SpinOrbitals(MolecularOrbitals):

    def __init__(self, coeff, energy=None, occ=None, labels=None, maxocc=1):
        if energy is None: energy = (None, None)
        if occ is None: occ = (None, None)
        if labels is None: labels = (None, None)
        self.alpha = SpatialOrbitals(coeff[0], energy=energy[0], occ=occ[0], labels=labels[0], maxocc=maxocc)
        self.beta = SpatialOrbitals(coeff[1], energy=energy[1], occ=occ[1], labels=labels[1], maxocc=maxocc)

    @classmethod
    def from_spatial_orbitals(cls, orbitals):
        energy = (orbitals.energy, orbitals.energy) if orbitals.energy is not None else None
        occ = (orbitals.occ/2, orbitals.occ/2) if orbitals.occ is not None else None
        labels = (orbitals.labels, orbitals.labels) if orbitals.labels is not None else None
        return cls((orbitals.coeff, orbitals.coeff), energy=energy, occ=occ, labels=labels)

    def to_general_orbitals(self):
        return GeneralOrbitals.from_spin_orbitals(self)

    @property
    def norba(self):
        return self.alpha.norb

    @property
    def norbb(self):
        return self.beta.norb

    @property
    def nocca(self):
        return self.alpha.nocc

    @property
    def noccb(self):
        return self.beta.nocc

    @property
    def nvira(self):
        return self.alpha.nvir

    @property
    def nvirb(self):
        return self.beta.nvir

    @property
    def nspin(self):
        return 2

    def __getattr__(self, name):
        if name in ('norb', 'nocc', 'nvir', 'nelec', 'energy', 'coeff', 'occ', 'labels'):
            return (getattr(self.alpha, name), getattr(self.beta, name))
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

    def __setattr__(self, name, value):
        if name in ('energy', 'coeff', 'occ', 'labels'):
            setattr(self.alpha, name, value[0])
            setattr(self.beta, name, value[1])
            return
        super().__setattr__(name, value)

    def pack(self, dtype=float):
        """Pack into a single array of data type `dtype`.

        Useful for communication via MPI."""
        data = (*self.coeff, *self.energy, *self.occ)
        return pack_arrays(*data, dtype=dtype)

    @classmethod
    def unpack(cls, packed):
        """Unpack from a single array of data type `dtype`.

        Useful for communication via MPI."""
        unpacked = unpack_arrays(packed)
        coeff = unpacked[:2]
        energy = unpacked[2:4]
        occ = unpacked[4:6]
        return cls(coeff, energy=energy, occ=occ)


class GeneralOrbitals(SpatialOrbitals):

    @property
    def nspin(self):
        return 3

    @classmethod
    def from_spatial_orbitals(cls, orbitals):
        raise NotImplementedError

    @classmethod
    def from_spin_orbitals(cls, orbitals):
        raise NotImplementedError

if __name__ == '__main__':

    def test_spatial(nao=20, nocc=5, nvir=15):
        coeff = np.random.rand(nao, nao)
        energy = np.random.rand(nao)
        occ = np.asarray(nocc*[2] + nvir*[0])

        orbs = SpatialOrbitals(coeff, energy=energy, occ=occ)
        packed = orbs.pack()
        orbs2 = SpatialOrbitals.unpack(packed)
        assert np.all(orbs.coeff == orbs2.coeff)
        assert np.all(orbs.energy == orbs2.energy)
        assert np.all(orbs.occ == orbs2.occ)

    def test_spin(nao=20, nocc=5, nvir=15):
        coeff = []
        energy = []
        occ = []
        for s in range(2):
            coeff.append(np.random.rand(nao, nao))
            energy.append(np.random.rand(nao))
            occ.append(np.asarray(nocc*[2] + nvir*[0]))

        orbs = SpinOrbitals(coeff, energy=energy, occ=occ)
        packed = orbs.pack()
        orbs2 = SpinOrbitals.unpack(packed)
        for s in range(2):
            assert np.all(orbs.coeff[s] == orbs2.coeff[s])
            assert np.all(orbs.energy[s] == orbs2.energy[s])
            assert np.all(orbs.occ[s] == orbs2.occ[s])

    test_spatial()
    test_spin()
