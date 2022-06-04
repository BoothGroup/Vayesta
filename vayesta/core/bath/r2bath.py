import numpy as np

from vayesta.core.util import *
from .bath import FragmentBath
from . import helper

BOHR = 0.529177210903

def _to_bohr(rmax, unit):
    unit = unit.lower()
    if unit.startswith('ang'):
        return rmax/BOHR
    if unit.startswith('b'):
        return rmax
    raise ValueError("Invalid unit: %s" % unit)

class R2_Bath(FragmentBath):

    def __init__(self, fragment, dmet_bath, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
        if len(self.fragment.atoms) != 1:
            raise NotImplementedError
        atom = self.fragment.atoms[0]
        self.center = self.mol.atom_coord(atom)
        # Output
        self.c_occ = None
        self.c_vir = None
        self.r_occ = None
        self.r_vir = None

    def diagonalize_r2(self, mo_coeff):
        with self.mol.with_common_origin(self.center):
            r2 = self.mol.intor('int1e_r2')
        r2 = dot(mo_coeff.T, r2, mo_coeff)
        eig, rot = np.linalg.eigh(r2)
        assert np.all(eig > -1e-13)
        eig = np.sqrt(np.clip(eig, 0, None))
        coeff = np.dot(mo_coeff, rot)
        return eig, coeff

    def make_occupied(self):
        self.r_occ, self.c_occ = self.diagonalize_r2(self.dmet_bath.c_env_occ)
        self.log.debug("Occupied eigenvalues (A):\n%r", self.r_occ*BOHR)
        self.log_histogram(self.r_occ, 'occupied')

    def make_virtual(self):
        self.r_vir, self.c_vir = self.diagonalize_r2(self.dmet_bath.c_env_vir)
        self.log.debug("Virtual eigenvalues (A):\n%r", self.r_vir*BOHR)
        self.log_histogram(self.r_vir, 'virtual')

    def kernel(self, occupied=True, virtual=True):
        if occupied:
            self.make_occupied()
        if virtual:
            self.make_virtual()

    def get_occupied_bath(self, rmax, unit='Ang'):
        if self.r_occ is None:
            self.make_occupied()
        rmax = _to_bohr(rmax, unit)
        nbath = np.count_nonzero(self.r_occ <= rmax)
        c_bath, c_rest = np.hsplit(self.c_occ, [nbath])
        return c_bath, c_rest

    def get_virtual_bath(self, rmax, unit='Ang'):
        if self.r_vir is None:
            self.make_virtual()
        rmax = _to_bohr(rmax, unit)
        nbath = np.count_nonzero(self.r_vir <= rmax)
        c_bath, c_rest = np.hsplit(self.c_vir, [nbath])
        return c_bath, c_rest

    def log_histogram(self, r, name):
        if len(r) == 0:
            return
        self.log.info("%s R2-bath histogram:", name.capitalize())
        bins = np.linspace(0, 20, 20)
        self.log.info(helper.make_histogram(r, bins=bins, invertx=False))
