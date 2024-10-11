import logging

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.scf
import pyscf.lib

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
    

class LatticeKSCF:
    def __init__(self, mol, kpts, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        self.kpts = kpts
        self.nk = len(kpts)
        if self.mol.incore_anyway:
            self._eri = mol.get_eri()
        else:
            self.density_fit()

    def get_hcore(self, *args, **kwargs):
        nao = self.mol.nao
        h1e = np.zeros((nao, nao))
        for i in range(nao - 1):
            h1e[i, i + 1] = h1e[i + 1, i] = -self.mol.hubbard_t
        h1e[nao - 1, 0] = h1e[0, nao - 1] = -self.mol.hubbard_t

        h1es = []
        for i, kpt in enumerate(self.kpts):
            m = np.zeros((nao, nao), dtype=np.complex128)
            m += h1e.copy()
            m[0,nao-1] = -self.mol.hubbard_t * np.exp(-1j*kpt[0]*self.mol.nsite)
            m[nao-1,0] = -self.mol.hubbard_t * np.exp(1j*kpt[0]*self.mol.nsite)
            h1es.append(m)
        return np.array(h1es)

    def get_ovlp(self, mol=None):
        return np.array([np.eye(self.mol.nao)] * self.nk)

    def density_fit(self):
        self.with_df = HubbardDF(self.mol, kpts=self.kpts)
        return self
    
class LatticeKRHF(LatticeKSCF, pyscf.pbc.scf.khf.KRHF):
    def get_init_guess(self, mol=None, key=None, s1e=None):
        dms = []
        hcore = self.get_hcore()
        for h in hcore:
            e, c = np.linalg.eigh(h)
            nocc = self.mol.nelectron // 2
            dm = 2 * np.dot(c[:, :nocc], c[:, :nocc].T)
            dms.append(dm)
        return np.array(dms)

    def get_jk(self, mol=None, dm=None, *args, **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self.mol.v_nn is not None and mol.v_nn != 0:
            raise NotImplementedError()

        js, ks = [], []
        for i, kpt in enumerate(self.kpts):
            j = np.diag(np.diag(dm[i])) * mol.hubbard_u
            js.append(j)

        js = np.array(js)
        ks = np.array(js)
        return js, ks


class HubbardDF(pyscf.pbc.df.DF):
    def __init__(self, mol, kpts=None, *args, **kwargs):
        if mol.v_nn:
            raise NotImplementedError()
        
        super().__init__(mol, *args, **kwargs)
        self.mol = mol
        self.kpts = kpts
        self.blockdim = self.get_naoaux()
        self.build()

    # def ao2mo(self, *args, **kwargs):
    #     return self.mol.ao2mo(*args, **kwargs)

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        nsite = self.mol.nsite
        self.cderi = np.zeros((nsite, nsite, nsite))
        np.fill_diagonal(self.cderi, np.sqrt(self.mol.hubbard_u))
        return self

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

    def sr_loop(self, kpti_kptj=np.zeros((2, 3)), max_memory=2000, compact=True, blksize=None):    
        
        LpqR = self.cderi 
        LpqI = np.zeros_like(LpqR)
        if compact and LpqR.shape[1] == n**2:
            LpqR = lib.pack_tril(LpqR.reshape(-1, n, n))
            LpqI = lib.pack_tril(LpqI.reshape(-1, n, n))
        yield LpqR, LpqI, 1

