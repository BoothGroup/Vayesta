import logging

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
#import pyscf.gto
import pyscf.scf
import pyscf.lib
from pyscf.lib.parameters import BOHR

log = logging.getLogger(__name__)

class LatticeMole(pyscf.pbc.gto.Cell):
    """For PySCF compatibility

    Needs to implement:
    a
    copy()
    build()
    intor_cross()
    intor_symmetric()
    pbc_intor()
    basis

    ?
    atom_coord
    unit
    """

    def __init__(self, verbose=0, output=None):
        super().__init__(verbose=verbose, output=output)
        self._built = True
        self.incore_anyway = True

    def __getattr__(self, attr):
        raise AttributeError("Attribute %s" % attr)

    @property
    def natm(self):
        return self.nsite

    def nao_nr(self):
        return self.nsite

    def ao_labels(self, *args):
        return ['X%d' % i for i in range(self.nsite)]

    def atom_symbol(self, site):
        return 'X%d' % site

    def atom_pure_symbol(self, site):
        return 'X'

    #def build(self):
    #    pass

    def search_ao_label(self):
        raise NotImplementedError()


class Hubbard(LatticeMole):
    """Abstract Hubbard model class."""

    def __init__(self, nsite, nelectron=None, hubbard_t=1.0, hubbard_u=0.0, v_nn=0.0,
            verbose=0, output=None):
        super().__init__(verbose=verbose, output=output)
        self.nsite = nsite
        if nelectron is None:
            nelectron = nsite
        self.nelectron = nelectron
        self.hubbard_t = hubbard_t
        self.hubbard_u = hubbard_u
        self.v_nn = v_nn

    def aoslice_by_atom(self):
        """One basis function per site ("atom")."""
        aorange = np.stack(4*[np.arange(self.nsite)], axis=1)
        aorange[:,1] += 1
        aorange[:,3] += 1
        return aorange


class Hubbard1D(Hubbard):
    """Hubbard model in 1D."""

    def __init__(self, nsite, nelectron=None, hubbard_t=1.0, hubbard_u=0.0, v_nn=0.0, boundary='auto',
            verbose=0, output=None):
        super().__init__(nsite, nelectron, hubbard_t, hubbard_u, v_nn=v_nn, verbose=verbose, output=output)

        self.nsites = [nsite]
        self.dimension = 1
        if boundary == 'auto':
            if (nsite % 4 == 0):
                boundary = 'APBC'
            elif (nsite % 4 == 2):
                boundary = 'PBC'
            else:
                raise ValueError()
        self.boundary = boundary
        if boundary.upper() == 'PBC':
            bfac = 1
        elif boundary.upper() == 'APBC':
            bfac = -1
        h1e = np.zeros((nsite, nsite))
        for i in range(nsite-1):
            h1e[i,i+1] = h1e[i+1,i] = -hubbard_t
        h1e[nsite-1,0] = h1e[0,nsite-1] = bfac * -hubbard_t
        self.h1e = h1e

    def get_eri(self, hubbard_u=None, v_nn=None):
        if hubbard_u is None:
            hubbard_u = self.hubbard_u
        if v_nn is None:
            v_nn = self.v_nn

        eri = np.zeros(4*[self.nsite])
        np.fill_diagonal(eri, hubbard_u)
        # Nearest-neighbor interaction
        if v_nn:
            for i in range(self.nsite-1):
                eri[i,i,i+1,i+1] = eri[i+1,i+1,i,i] = v_nn
            eri[self.nsite-1,self.nsite-1,0,0] = eri[0,0,self.nsite-1,self.nsite-1] = v_nn
        return eri


    def lattice_vectors(self):
        """Lattice vectors of 1D Hubbard model.

        An arbitrary value of 1 A is assumed between sites. The lattice vectors, however, are saved in units of Bohr.
        """
        rvecs = np.eye(3)
        rvecs[0,0] = self.nsite
        return rvecs / BOHR

    def atom_coords(self):
        coords = np.zeros((self.nsite, 3))
        coords[:,0] = np.arange(self.nsite)
        return coords / BOHR


class Hubbard2D(Hubbard):

    def __init__(self, nsites, nelectron=None, hubbard_t=1.0, hubbard_u=0.0, boundary=('PBC', 'APBC'),
            verbose=0, output=None):
        nsite = nsites[0]*nsites[1]
        super().__init__(nsite, nelectron, hubbard_t, hubbard_u, verbose=verbose, output=output)

        self.nsites = nsites
        self.dimension = 2
        self.boundary = boundary

        bfac = 2*[None]
        for i in range(2):
            if boundary[i].lower() == 'open':
                bfac[i] = 0
            elif boundary[i].lower() in ('periodic', 'pbc'):
                bfac[i] = +1
            elif boundary[i].lower() in ('anti-periodic', 'apbc'):
                bfac[i] = -1
            else:
                raise ValueError('Invalid boundary: %s' % boundary[i])

        def get_index(i, j):
            fac = 1
            if i % nsites[0] != i:
                fac *= bfac[0]
            if j % nsites[1] != j:
                fac *= bfac[1]
            idx = (i%nsites[0])*nsites[1] + (j%nsites[1])
            return idx, fac

        h1e = np.zeros((nsite, nsite))
        for i in range(nsites[0]):
            for j in range(nsites[1]):
                idx, _ = get_index(i, j)
                idx_l, fac_l = get_index(i, j-1)
                idx_r, fac_r = get_index(i, j+1)
                idx_u, fac_u = get_index(i-1, j)
                idx_d, fac_d = get_index(i+1, j)
                h1e[idx,idx_l] += fac_l * -hubbard_t
                h1e[idx,idx_r] += fac_r * -hubbard_t
                h1e[idx,idx_u] += fac_u * -hubbard_t
                h1e[idx,idx_d] += fac_d * -hubbard_t
        self.h1e = h1e

    def get_eri(self, hubbard_u=None, v_nn=None):
        if hubbard_u is None:
            hubbard_u = self.hubbard_u
        if v_nn is None:
            v_nn = self.v_nn

        eri = np.zeros(4*[self.nsite])
        np.fill_diagonal(eri, hubbard_u)
        # Nearest-neighbor interaction
        if v_nn:
            raise NotImplementedError()
        return eri


class LatticeMF(pyscf.scf.hf.RHF):
#class LatticeMF(pyscf.pbc.scf.hf.RHF):

    def __init__(self, mol, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        self._eri = mol.get_eri()

    @property
    def cell(self):
        return self.mol

    def get_hcore(self, *args, **kwargs):
        return self.mol.h1e

    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self.mol.v_nn is not None and mol.v_nn != 0:
            raise NotImplementedError()
        return np.diag(np.diag(dm))*mol.hubbard_u/2

    def get_ovlp(self):
        return np.eye(self.mol.nsite)

    def kernel_hubbard(self):
        mo_energy, mo_coeff = np.linalg.eigh(self.mol.h1e)
        nocc = self.mol.nelectron//2
        nvir = self.mol.nsite - nocc
        self.mo_energy = mo_energy
        log.info("MO energies:")
        for i in range(0, len(mo_energy), 5):
            e = mo_energy[i:i+5]
            fmt = '  ' + len(e)*'  %+16.8f'
            log.info(fmt, *e)
        if nocc > 0:
            homo = self.mo_energy[nocc-1]
        else:
            homo = np.nan
        if nvir > 0:
            lumo = self.mo_energy[nocc]
        else:
            lumo = np.nan
        gap = (lumo-homo)
        log.info("HOMO= %+16.8f  LUMO= %+16.8f  gap= %+16.8f", homo, lumo, gap)
        if gap < 1e-6:
            log.critical("Zero HOMO-LUMO gap!")
            raise RuntimeError()
        elif gap < 0.1:
            log.warning("Small HOMO-LUMO gap!")

        self.mo_coeff = mo_coeff
        self.mo_occ = np.asarray((nocc*[2] + nvir*[0]))

        dm = self.make_rdm1()
        veff = self.get_veff()
        self.e_tot = np.einsum('ab,ba->', (self.get_hcore() + veff/2), dm)
        self.converged = True

        return self.e_tot

    #class with_df:
    #    """Dummy density-fitting"""

    #    @classmethod
    #    def ao2mo(cls, mo_coeff):
    #        pass

    kernel = kernel_hubbard
