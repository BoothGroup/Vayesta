import numpy as np

from vayesta.core.types.orbitals import Orbitals
from vayesta.core.types.bosonic_orbitals import BosonicOrbitals
from vayesta.core.spinalg import add_numbers, hstack_matrices

__all__ = ["Cluster", "ClusterRHF", "ClusterUHF"]


class Cluster:
    def __init__(self, active_orbitals, frozen_orbitals, bosons=None):
        self.active_orbitals = active_orbitals
        self.frozen_orbitals = frozen_orbitals
        self.bosons = bosons

    @staticmethod
    def from_coeffs(c_active_occ, c_active_vir, c_frozen_occ, c_frozen_vir):
        c_active = hstack_matrices(c_active_occ, c_active_vir)
        c_frozen = hstack_matrices(c_frozen_occ, c_frozen_vir)
        is_rhf = c_active_occ[0].ndim == 1
        if is_rhf:
            nocc_active = c_active_occ.shape[-1]
            nocc_frozen = c_frozen_occ.shape[-1]
        else:
            nocc_active = tuple(x.shape[-1] for x in c_active_occ)
            nocc_frozen = tuple(x.shape[-1] for x in c_frozen_occ)
        active = Orbitals(c_active, occ=nocc_active)
        frozen = Orbitals(c_frozen, occ=nocc_frozen)
        if is_rhf:
            return ClusterRHF(active, frozen)
        return ClusterUHF(active, frozen)

    # --- Active

    @property
    def norb_active(self):
        return self.active_orbitals.norb

    @property
    def nocc_active(self):
        return self.active_orbitals.nocc

    @property
    def nvir_active(self):
        return self.active_orbitals.nvir

    @property
    def c_active(self):
        return self.active_orbitals.coeff

    @property
    def c_active_occ(self):
        return self.active_orbitals.coeff_occ

    @property
    def c_active_vir(self):
        return self.active_orbitals.coeff_vir

    # Shorthand:
    norb = norb_active
    nocc = nocc_active
    nvir = nvir_active
    coeff = c_active
    c_occ = c_active_occ
    c_vir = c_active_vir

    # --- Frozen

    @property
    def norb_frozen(self):
        return self.frozen_orbitals.norb

    @property
    def nocc_frozen(self):
        return self.frozen_orbitals.nocc

    @property
    def nvir_frozen(self):
        return self.frozen_orbitals.nvir

    @property
    def c_frozen(self):
        return self.frozen_orbitals.coeff

    @property
    def c_frozen_occ(self):
        return self.frozen_orbitals.coeff_occ

    @property
    def c_frozen_vir(self):
        return self.frozen_orbitals.coeff_vir

    # --- Active+Frozen

    @property
    def norb_total(self):
        return add_numbers(self.norb_active, self.norb_frozen)

    @property
    def nocc_total(self):
        return add_numbers(self.nocc_active, self.nocc_frozen)

    @property
    def nvir_total(self):
        return add_numbers(self.nvir_active, self.nvir_frozen)

    @property
    def c_total(self):
        return hstack_matrices(self.c_total_occ, self.c_total_vir)

    @property
    def c_total_occ(self):
        return hstack_matrices(self.c_frozen_occ, self.c_active_occ)

    @property
    def c_total_vir(self):
        return hstack_matrices(self.c_active_vir, self.c_frozen_vir)

    @property
    def inc_bosons(self):
        return self.bosons is not None

    def copy(self):
        return type(self)(
            self.active_orbitals.copy(),
            self.frozen_orbitals.copy(),
            None if self.bosons is None else self.bosons.copy(),
        )

    def basis_transform(self, trafo, inplace=False):
        cp = self if inplace else self.copy()
        cp.active_orbitals.basis_transform(trafo, inplace=True)
        cp.frozen_orbitals.basis_transform(trafo, inplace=True)
        if self.inc_bosons:
            cp.bosons.fbasis_transform(trafo, inplace=True)
        return cp


class ClusterRHF(Cluster):
    spinsym = "restricted"

    def __repr__(self):
        return "%s(norb_active= %d, norb_frozen= %d)" % (self.__class__.__name__, self.norb_active, self.norb_frozen)

    # Indices and slices

    def get_active_slice(self):
        return np.s_[self.nocc_frozen : self.nocc_frozen + self.norb_active]

    def get_active_indices(self):
        return list(range(self.nocc_frozen, self.nocc_frozen + self.norb_active))

    def get_frozen_indices(self):
        return list(range(self.nocc_frozen)) + list(range(self.norb_total - self.nvir_frozen, self.norb_total))

    def repr_size(self):
        lines = []
        fmt = 10 * " " + 2 * "   %-15s" + "   %-5s"
        lines += [fmt % ("Active", "Frozen", "Total")]
        lines += [fmt % (15 * "-", 15 * "-", 5 * "-")]
        fmt = "  %-8s" + 2 * "   %5d (%6.1f%%)" + "   %5d"
        get_values = lambda a, f, n: (a, 100 * a / n, f, 100 * f / n, n)
        lines += [fmt % ("Occupied", *get_values(self.nocc_active, self.nocc_frozen, self.nocc_total))]
        lines += [fmt % ("Virtual", *get_values(self.nvir_active, self.nvir_frozen, self.nvir_total))]
        lines += [fmt % ("Total", *get_values(self.norb_active, self.norb_frozen, self.norb_total))]
        return "\n".join(lines)


class ClusterUHF(Cluster):
    spinsym = "unrestricted"

    def __repr__(self):
        return "%s(norb_active= (%d, %d), norb_frozen= (%d, %d))" % (
            self.__class__.__name__,
            *self.norb_active,
            *self.norb_frozen,
        )

    # Indices and slices

    def get_active_slice(self):
        return (
            np.s_[self.nocc_frozen[0] : self.nocc_frozen[0] + self.norb_active[0]],
            np.s_[self.nocc_frozen[1] : self.nocc_frozen[1] + self.norb_active[1]],
        )

    def get_active_indices(self):
        return (
            list(range(self.nocc_frozen[0], self.nocc_frozen[0] + self.norb_active[0])),
            list(range(self.nocc_frozen[1], self.nocc_frozen[1] + self.norb_active[1])),
        )

    def get_frozen_indices(self):
        return (
            list(range(self.nocc_frozen[0]))
            + list(range(self.norb_total[0] - self.nvir_frozen[0], self.norb_total[0])),
            list(range(self.nocc_frozen[1]))
            + list(range(self.norb_total[1] - self.nvir_frozen[1], self.norb_total[1])),
        )

    def repr_size(self):
        lines = []
        fmt = 10 * " " + 2 * "   %-22s" + "   %-12s"
        lines += [(fmt % ("Active", "Frozen", "Total")).rstrip()]
        lines += [fmt % (22 * "-", 22 * "-", 12 * "-")]
        fmt = "  %-8s" + 2 * "   %5d, %5d (%6.1f%%)" + "   %5d, %5d"
        get_values = lambda a, f, n: (
            a[0],
            a[1],
            100 * (a[0] + a[1]) / (n[0] + n[1]),
            f[0],
            f[1],
            100 * (f[0] + f[1]) / (n[0] + n[1]),
            n[0],
            n[1],
        )
        lines += [fmt % ("Occupied", *get_values(self.nocc_active, self.nocc_frozen, self.nocc_total))]
        lines += [fmt % ("Virtual", *get_values(self.nvir_active, self.nvir_frozen, self.nvir_total))]
        lines += [fmt % ("Total", *get_values(self.norb_active, self.norb_frozen, self.norb_total))]
        return "\n".join(lines)
