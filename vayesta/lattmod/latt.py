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

    def __init__(self, nsite, order=None, verbose=0, output=None):
        """
        Parameters
        ----------
        order:
            Ordering of lattice sites.
        """
        super().__init__(verbose=verbose, output=output)
        self.nsite = nsite
        self._basis = {self.atom_symbol(i) : None for i in range(self.nsite)}
        self._built = True
        self.incore_anyway = True
        self.order = order

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
        return '%s%d' % (self.atom_pure_symbol(site), site)

    def atom_pure_symbol(self, site):
        return 'X'

    #def build(self):
    #    pass

    def search_ao_label(self):
        raise NotImplementedError()


class Hubbard(LatticeMole):
    """Abstract Hubbard model class."""

    def __init__(self, nsite, nelectron=None, hubbard_t=1.0, hubbard_u=0.0, v_nn=0.0,
            order=None, verbose=0, output=None):
        super().__init__(nsite, order=order, verbose=verbose, output=output)
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

    def __init__(self, nsite, nelectron=None, hubbard_t=1.0, hubbard_u=0.0, v_nn=0.0, boundary='auto', **kwargs):
        super().__init__(nsite, nelectron, hubbard_t, hubbard_u, v_nn=v_nn, **kwargs)

        self.nsites = [nsite]
        self.dimension = 1
        if boundary == 'auto':
            if (nsite % 4 == 2):
                boundary = 'PBC'
            elif (nsite % 4 == 0):
                boundary = 'APBC'
            else:
                raise ValueError()
            log.warning("boundary condition: %s", boundary)
        self.boundary = boundary
        if boundary.upper() == 'PBC':
            bfac = 1
        elif boundary.upper() == 'APBC':
            bfac = -1
        h1e = np.zeros((nsite, nsite))
        for i in range(nsite-1):
            h1e[i,i+1] = h1e[i+1,i] = -hubbard_t
        h1e[nsite-1,0] = h1e[0,nsite-1] = bfac * -hubbard_t
        if self.order is not None:
            h1e = h1e[self.order][:,self.order]
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
            if self.order is not None:
                # Not tested:
                order = self.order
                eri = eri[order][:,order][:,:,order][:,:,:,order]
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
        if self.order is not None:
            coords = coords[self.order]
        return coords / BOHR


class Hubbard2D(Hubbard):

    def __init__(self, nsites, nelectron=None, hubbard_t=1.0, hubbard_u=0.0, boundary='auto',
            tiles=(1, 1), order=None, **kwargs):
        nsite = nsites[0]*nsites[1]
        if order is None and tiles != (1, 1):
            order = self.get_tiles_order(nsites, tiles)
        super().__init__(nsite, nelectron, hubbard_t, hubbard_u, order=order, **kwargs)

        self.nsites = nsites
        self.dimension = 2

        if isinstance(boundary, str) and boundary.lower() == 'auto':
            if nelectron == nsite:
                if self.nsites[0] % 4 == 0 and self.nsites[1] % 4 == 0:
                    boundary = ('PBC', 'APBC')
                elif self.nsites[0] % 4 == 0 and self.nsites[1] % 4 == 2:
                    boundary = ('APBC', 'APBC')
                    # Also possible:
                    #boundary = ('APBC', 'PBC')
                elif self.nsites[0] % 4 == 2 and self.nsites[1] % 4 == 0:
                    boundary = ('APBC', 'APBC')
                    # Also possible:
                    #boundary = ('PBC', 'APBC')
                elif self.nsites[0] % 4 == 2 and self.nsites[1] % 4 == 2:
                    boundary = ('PBC', 'APBC')
            else:
                raise NotImplementedError("Please specifiy boundary conditions.")
        if np.ndim(boundary) == 0:
            boundary = (boundary, boundary)
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
        log.debugv('boundary phases= %r', bfac)

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
        if self.order is not None:
            h1e = h1e[self.order][:,self.order]
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

    def lattice_vectors(self):
        """Lattice vectors of 1D Hubbard model.

        An arbitrary value of 1 A is assumed between sites. The lattice vectors, however, are saved in units of Bohr.
        """
        rvecs = np.eye(3)
        rvecs[0,0] = self.nsites[0]
        rvecs[1,1] = self.nsites[1]
        return rvecs / BOHR

    def atom_coords(self):
        """Sites are ordered by default as:
        ...
        6 7 8
        3 4 5
        0 1 2
        """
        coords = np.zeros((self.nsite, 3))
        for row in range(self.nsites[1]):
            slc = np.s_[row*self.nsites[0]:(row+1)*self.nsites[0]]
            coords[slc,0] = np.arange(self.nsites[0])
            coords[slc,1] = row
        if self.order is not None:
            coords = coords[self.order]
        return coords / BOHR


    @staticmethod
    def get_tiles_order(nsites, tiles):
        assert(nsites[0] % tiles[0] == 0)
        assert(nsites[1] % tiles[1] == 0)
        ntiles = [nsites[0] // tiles[0], nsites[1] // tiles[1]]
        tsize = tiles[0]*tiles[1]

        def get_xy(site):
            tile, pos = divmod(site, tsize)
            ty, tx = divmod(tile, ntiles[0])
            py, px = divmod(pos, tiles[0])
            return tx*tiles[0]+px, ty*tiles[1]+py

        nsite = nsites[0]*nsites[1]
        order = []
        for site in range(nsite):
            x, y = get_xy(site)
            idx = y*nsites[0] + x
            order.append(idx)
        return order


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
