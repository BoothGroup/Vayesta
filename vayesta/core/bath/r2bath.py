import numpy as np

from vayesta.core.util import *
from .bath import Bath
from . import helper


BOHR = 0.529177210903

def _to_bohr(rcut, unit):
    unit = unit.lower()
    if unit.startswith('ang'):
        return rcut/BOHR
    if unit.startswith('b'):
        return rcut
    raise ValueError("Invalid unit: %s" % unit)

class R2_Bath_RHF(Bath):

    def __init__(self, fragment, dmet_bath, occtype, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        if getattr(self.mol, 'dimension', 0) > 0:
            self.log.warning("R2_Bath does not work correctly for PBC systems!")
        self.dmet_bath = dmet_bath
        if occtype not in ('occupied', 'virtual'):
            raise ValueError("Invalid occtype: %s" % occtype)
        self.occtype = occtype
        if len(self.fragment.atoms) != 1:
            raise NotImplementedError
        atom = self.fragment.atoms[0]
        self.center = self.mol.atom_coord(atom)
        self.coeff, self.eig = self.kernel()

    @property
    def c_env(self):
        if self.occtype == 'occupied':
            return self.dmet_bath.c_env_occ
        if self.occtype == 'virtual':
            return self.dmet_bath.c_env_vir

    def diagonalize_r2(self, mo_coeff):
        mol = self.mol
        with mol.with_common_origin(self.center):
            # PBC: Does not yield hermitian matrices....?
            if getattr(mol, 'dimension', 0) > 0:
                r2 = mol.pbc_intor('int1e_r2')
                #r2 = mol.pbc_intor('int1e_r2', hermi=1)
            # Molecule
            else:
                r2 = mol.intor_symmetric('int1e_r2')
        r2 = dot(mo_coeff.T, r2, mo_coeff)
        hermierr = np.linalg.norm(r2 - r2.T)
        if hermierr > 1e-11:
            self.log.warning("Hermiticity error= %.3e", hermierr)
            r2 = (r2 + r2.T)/2
        else:
            self.log.debug("Hermiticity error= %.3e", hermierr)
        eig, rot = np.linalg.eigh(r2)
        if np.any(eig < -1e-13):
            raise RuntimeError("Negative eigenvalues: %r" % eig[eig<0])
        eig = np.sqrt(np.clip(eig, 0, None))
        coeff = np.dot(mo_coeff, rot)
        return coeff, eig

    def kernel(self):
        coeff, eig = self.diagonalize_r2(self.c_env)
        self.log.debug("%s eigenvalues (A):\n%r", self.occtype.capitalize(), eig*BOHR)
        self.log_histogram(eig, self.occtype)
        return coeff, eig

    def get_bath(self, rcut, unit='Ang'):
        rcut = _to_bohr(rcut, unit)
        nbath = np.count_nonzero(self.eig <= rcut)
        c_bath, c_rest = np.hsplit(self.coeff, [nbath])
        return c_bath, c_rest

    def log_histogram(self, r, name):
        if len(r) == 0:
            return
        self.log.info("%s R2-bath histogram:", name.capitalize())
        bins = np.linspace(0, 20, 20)
        self.log.info(helper.make_histogram(r, bins=bins, invertx=False))
