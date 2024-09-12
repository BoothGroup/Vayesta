import logging

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.scf
import pyscf.lib
from pyscf.lib.parameters import BOHR

from vayesta.core.util import einsum

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

    def __init__(self, nsite, nelectron=None, spin=0, order=None, incore_anyway=True, verbose=0, output=None):
        """
        Parameters
        ----------
        order:
            Ordering of lattice sites.
        """
        super().__init__(verbose=verbose, output=output)
        self.nsite = nsite
        if nelectron is None:
            nelectron = nsite
        self.nelectron = nelectron
        self.spin = spin
        self._basis = {self.atom_symbol(i): None for i in range(self.nsite)}
        self._built = True
        self.incore_anyway = incore_anyway
        self.order = order

    def __getattr__(self, attr):
        raise AttributeError("Attribute %s" % attr)

    @property
    def natm(self):
        return self.nsite

    def nao_nr(self):
        return self.nsite

    def ao_labels(self, fmt=True):
        if fmt:
            return ["%s%d" % (self.atom_pure_symbol(i), i) for i in range(self.nsite)]
        elif fmt is None:
            return [(i, self.atom_pure_symbol(i), "", "") for i in range(self.nsite)]

    def atom_symbol(self, site):
        return "%s%d" % (self.atom_pure_symbol(site), site)

    def atom_pure_symbol(self, site):
        return "S"

    # def build(self):
    #    pass

    def search_ao_label(self):
        raise NotImplementedError()


class Hubbard(LatticeMole):
    """Abstract Hubbard model class."""

    def __init__(self, nsite, nelectron=None, spin=0, hubbard_t=1.0, hubbard_u=0.0, v_nn=0.0, **kwargs):
        super().__init__(nsite, nelectron=nelectron, spin=spin, **kwargs)
        self.hubbard_t = hubbard_t
        self.hubbard_u = hubbard_u
        self.v_nn = v_nn

    def aoslice_by_atom(self):
        """One basis function per site ("atom")."""
        aorange = np.stack(4 * [np.arange(self.nsite)], axis=1)
        aorange[:, 1] += 1
        aorange[:, 3] += 1
        return aorange

    def ao2mo(self, mo_coeffs, compact=False):
        if compact:
            raise NotImplementedError()
        if self.v_nn:
            raise NotImplementedError()

        if isinstance(mo_coeffs, np.ndarray) and np.ndim(mo_coeffs) == 2:
            eris = self.hubbard_u * einsum("ai,aj,ak,al->ijkl", mo_coeffs, mo_coeffs, mo_coeffs, mo_coeffs)
        else:
            eris = self.hubbard_u * einsum("ai,aj,ak,al->ijkl", *mo_coeffs)
        eris = eris.reshape(eris.shape[0] * eris.shape[1], eris.shape[2] * eris.shape[3])
        return eris


class Hubbard1D(Hubbard):
    """Hubbard model in 1D."""

    def __init__(
        self, nsite, nelectron=None, spin=0, hubbard_t=1.0, hubbard_u=0.0, v_nn=0.0, boundary="auto", **kwargs
    ):
        super().__init__(nsite, nelectron, spin, hubbard_t, hubbard_u, v_nn=v_nn, **kwargs)

        self.nsites = [nsite]
        self.dimension = 1
        if boundary == "auto":
            if nsite % 4 == 2:
                boundary = "PBC"
            elif nsite % 4 == 0:
                boundary = "APBC"
            else:
                raise ValueError()
            log.debug("Automatically chosen boundary condition: %s", boundary)
        self.boundary = boundary
        if boundary.upper() == "PBC":
            bfac = 1
        elif boundary.upper() == "APBC":
            bfac = -1
        self.bfac = bfac
        h1e = np.zeros((nsite, nsite))
        for i in range(nsite - 1):
            h1e[i, i + 1] = h1e[i + 1, i] = -hubbard_t
        h1e[nsite - 1, 0] = h1e[0, nsite - 1] = bfac * -hubbard_t
        if self.order is not None:
            h1e = h1e[self.order][:, self.order]
        self.h1e = h1e

    def get_eri(self, hubbard_u=None, v_nn=None):
        if hubbard_u is None:
            hubbard_u = self.hubbard_u
        if v_nn is None:
            v_nn = self.v_nn

        eri = np.zeros(4 * [self.nsite])
        np.fill_diagonal(eri, hubbard_u)
        # Nearest-neighbor interaction
        if v_nn:
            for i in range(self.nsite - 1):
                eri[i, i, i + 1, i + 1] = eri[i + 1, i + 1, i, i] = v_nn
            eri[self.nsite - 1, self.nsite - 1, 0, 0] = eri[0, 0, self.nsite - 1, self.nsite - 1] = v_nn
            if self.order is not None:
                # Not tested:
                order = self.order
                eri = eri[order][:, order][:, :, order][:, :, :, order]
        return eri

    def lattice_vectors(self):
        """Lattice vectors of 1D Hubbard model.

        An arbitrary value of 1 A is assumed between sites. The lattice vectors, however, are saved in units of Bohr.
        """
        rvecs = np.eye(3)
        rvecs[0, 0] = self.nsite
        return rvecs / BOHR

    def atom_coords(self):
        coords = np.zeros((self.nsite, 3))
        coords[:, 0] = np.arange(self.nsite)
        if self.order is not None:
            coords = coords[self.order]
        return coords / BOHR

    def get_index(self, i):
        bfac = self.bfac
        fac = 1
        if i % self.nsites[0] != i:
            fac *= bfac[0] * (i // self.nsites[0])
        idx = i
        return idx, fac


class Hubbard2D(Hubbard):
    def __init__(
        self,
        nsites,
        nelectron=None,
        spin=0,
        hubbard_t=1.0,
        hubbard_u=0.0,
        boundary="auto",
        tiles=(1, 1),
        order=None,
        **kwargs,
    ):
        nsite = nsites[0] * nsites[1]
        if order is None and tiles != (1, 1):
            order = self.get_tiles_order(nsites, tiles)
        super().__init__(nsite, nelectron, spin, hubbard_t, hubbard_u, order=order, **kwargs)

        self.nsites = nsites
        self.dimension = 2

        if isinstance(boundary, str) and boundary.lower() == "auto":
            if nelectron == nsite:
                if self.nsites[0] % 4 == 0 and self.nsites[1] % 4 == 0:
                    boundary = ("PBC", "APBC")
                elif self.nsites[0] % 4 == 0 and self.nsites[1] % 4 == 2:
                    boundary = ("APBC", "APBC")
                    # Also possible:
                    # boundary = ('APBC', 'PBC')
                elif self.nsites[0] % 4 == 2 and self.nsites[1] % 4 == 0:
                    boundary = ("APBC", "APBC")
                    # Also possible:
                    # boundary = ('PBC', 'APBC')
                elif self.nsites[0] % 4 == 2 and self.nsites[1] % 4 == 2:
                    boundary = ("PBC", "APBC")
                else:
                    raise NotImplementedError("Please specify boundary conditions.")
            else:
                raise NotImplementedError("Please specify boundary conditions.")
        if np.ndim(boundary) == 0:
            boundary = (boundary, boundary)
        self.boundary = boundary

        bfac = 2 * [None]
        for i in range(2):
            if boundary[i].lower() == "open":
                bfac[i] = 0
            elif boundary[i].lower() in ("periodic", "pbc"):
                bfac[i] = +1
            elif boundary[i].lower() in ("anti-periodic", "apbc"):
                bfac[i] = -1
            else:
                raise ValueError("Invalid boundary: %s" % boundary[i])
        log.debugv("boundary phases= %r", bfac)
        self.bfac = bfac
        h1e = np.zeros((nsite, nsite))
        for i in range(nsites[0]):
            for j in range(nsites[1]):
                idx, _ = self.get_index(i, j)
                idx_l, fac_l = self.get_index(i, j - 1)
                idx_r, fac_r = self.get_index(i, j + 1)
                idx_u, fac_u = self.get_index(i - 1, j)
                idx_d, fac_d = self.get_index(i + 1, j)
                h1e[idx, idx_l] += fac_l * -hubbard_t
                h1e[idx, idx_r] += fac_r * -hubbard_t
                h1e[idx, idx_u] += fac_u * -hubbard_t
                h1e[idx, idx_d] += fac_d * -hubbard_t
        if self.order is not None:
            h1e = h1e[self.order][:, self.order]
        self.h1e = h1e

    def get_index(self, i, j):
        bfac = self.bfac
        fac = 1
        if i % self.nsites[0] != i:
            fac *= bfac[0]
        if j % self.nsites[1] != j:
            fac *= bfac[1]
        idx = (i % self.nsites[0]) * self.nsites[1] + (j % self.nsites[1])
        return idx, fac

    def get_eri(self, hubbard_u=None, v_nn=None):
        if hubbard_u is None:
            hubbard_u = self.hubbard_u
        if v_nn is None:
            v_nn = self.v_nn

        eri = np.zeros(4 * [self.nsite])
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
        rvecs[0, 0] = self.nsites[0]
        rvecs[1, 1] = self.nsites[1]
        return rvecs / BOHR

    def atom_coords(self):
        """Sites are ordered by default as:

        6 7 8
        3 4 5
        0 1 2
        """
        coords = np.zeros((self.nsite, 3))
        for row in range(self.nsites[1]):
            slc = np.s_[row * self.nsites[0] : (row + 1) * self.nsites[0]]
            coords[slc, 0] = np.arange(self.nsites[0])
            coords[slc, 1] = row
        if self.order is not None:
            coords = coords[self.order]
        return coords / BOHR

    @staticmethod
    def get_tiles_order(nsites, tiles):
        assert nsites[0] % tiles[0] == 0
        assert nsites[1] % tiles[1] == 0
        ntiles = [nsites[0] // tiles[0], nsites[1] // tiles[1]]
        tsize = tiles[0] * tiles[1]

        def get_xy(site):
            tile, pos = divmod(site, tsize)
            ty, tx = divmod(tile, ntiles[0])
            py, px = divmod(pos, tiles[0])
            return tx * tiles[0] + px, ty * tiles[1] + py

        nsite = nsites[0] * nsites[1]
        order = []
        for site in range(nsite):
            x, y = get_xy(site)
            idx = y * nsites[0] + x
            order.append(idx)
        return order


class HubbardDF:
    def __init__(self, mol):
        if mol.v_nn:
            raise NotImplementedError()
        self.mol = mol
        self.blockdim = self.get_naoaux()

    def ao2mo(self, *args, **kwargs):
        return self.mol.ao2mo(*args, **kwargs)

    def get_naoaux(self):
        return self.mol.nsite

    def loop(self, blksize=None):
        """Note that blksize is ignored."""
        nsite = self.mol.nsite
        j3c = np.zeros((nsite, nsite, nsite))
        np.fill_diagonal(j3c, np.sqrt(self.mol.hubbard_u))
        # Pack (Q|ab) -> (Q|A)
        j3c = pyscf.lib.pack_tril(j3c)
        yield j3c


class LatticeSCF:
    def __init__(self, mol, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        if self.mol.incore_anyway:
            self._eri = mol.get_eri()
        else:
            self.density_fit()

    @property
    def cell(self):
        return self.mol

    def get_hcore(self, *args, **kwargs):
        return self.mol.h1e

    def get_ovlp(self, mol=None):
        return np.eye(self.mol.nsite)

    def density_fit(self):
        self.with_df = HubbardDF(self.mol)
        return self


class LatticeRHF(LatticeSCF, pyscf.scf.hf.RHF):
    def get_init_guess(self, mol=None, key=None, s1e=None):
        e, c = np.linalg.eigh(self.get_hcore())
        nocc = self.mol.nelectron // 2
        dm = 2 * np.dot(c[:, :nocc], c[:, :nocc].T)
        return dm

    def get_jk(self, mol=None, dm=None, *args, **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self.mol.v_nn is not None and mol.v_nn != 0:
            raise NotImplementedError()
        j = np.diag(np.diag(dm)) * mol.hubbard_u
        k = j
        return j, k

    def check_lattice_symmetry(self, dm=None):
        if dm is None:
            dm = self.make_rdm1()
        occ = np.diag(dm)
        if not np.all(np.isclose(occ[0], occ)):
            log.warning("Mean-field not lattice symmetric! Site occupations=\n%r", occ)
        else:
            log.debugv("Mean-field site occupations=\n%r", occ)


class LatticeUHF(LatticeSCF, pyscf.scf.uhf.UHF):
    def get_init_guess(self, mol=None, key=None, s1e=None):
        e, c = np.linalg.eigh(self.get_hcore())
        nocc = self.mol.nelec
        dma = np.dot(c[:, : nocc[0]], c[:, : nocc[0]].T)
        dmb = np.dot(c[:, : nocc[1]], c[:, : nocc[1]].T)
        # Create small random offset to break symmetries.

        offset = np.full_like(dma.diagonal(), fill_value=1e-2)
        if self.mol.dimension == 1:
            for x in range(self.mol.nsites[0]):
                ind, fac = self.mol.get_index(x)
                offset[ind] *= (-1) ** (x % 2)
        elif self.mol.dimension == 2:
            for x in range(self.mol.nsites[0]):
                for y in range(self.mol.nsites[1]):
                    ind, fac = self.mol.get_index(x, y)
                    offset[ind] *= (-1) ** (x % 2 + y % 2)
        else:
            raise NotImplementedError("LatticeUHF only supports 1- and 2-D Hubbard models.")
        dma[np.diag_indices_from(dma)] += offset
        return (dma, dmb)

    def get_jk(self, mol=None, dm=None, *args, **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self.mol.v_nn is not None and mol.v_nn != 0:
            raise NotImplementedError()
        dma, dmb = dm
        ja = np.diag(np.diag(dma)) * mol.hubbard_u
        jb = np.diag(np.diag(dmb)) * mol.hubbard_u
        ka, kb = ja, jb
        return (ja, jb), (ka, kb)
