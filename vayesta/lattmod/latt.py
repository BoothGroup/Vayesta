import logging

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
#import pyscf.gto
import pyscf.scf

log = logging.getLogger(__name__)

class LatticeMole(pyscf.pbc.gto.Cell):
    """For PySCF compatibility

    Needs to implement:
    a
    natm                    x
    nao_nr()                x
    ao_labels()             x
    search_ao_label()
    atom_symbol()
    copy()
    build()                 x
    intor_cross()
    intor_symmetric()
    pbc_intor()
    basis

    ?
    lattice_vectors()
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
        return ['SiteOrb_%d' % i for i in range(self.nsite)]

    def atom_symbol(self, site):
        return 'Site_%d' % site

    #def build(self):
    #    pass

    def search_ao_label(self):
        raise NotImplementedError()


class Hubbard(LatticeMole):
    """Abstract Hubbard model class."""

    def __init__(self, nsite, nelectron=None, hubbard_t=1.0, hubbard_u=0.0, verbose=0,
            output=None):
        super().__init__(verbose=verbose, output=output)
        self.nsite = nsite
        if nelectron is None:
            nelectron = nsite
        self.nelectron = nelectron
        self.hubbard_t = hubbard_t
        self.hubbard_u = hubbard_u


class Hubbard1D(Hubbard):
    """Hubbard model in 1D."""

    def __init__(self, nsite, nelectron=None, hubbard_t=1.0, hubbard_u=0.0, boundary='auto',
            verbose=0, output=None):
        super().__init__(nsite, nelectron, hubbard_t, hubbard_u, verbose=verbose, output=output)

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


class LatticeMF(pyscf.scf.hf.RHF):

    def __init__(self, mol, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        self._eri = np.zeros(4*[self.mol.nsite])
        np.fill_diagonal(self._eri, self.mol.hubbard_u)


    def get_hcore(self, *args, **kwargs):
        return self.mol.h1e

    def get_ovlp(self):
        return np.eye(self.mol.nsite)

    def kernel(self):
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

    #def get_veff(self, dm=None, *args, **kwargs):
    #    if dm is None:
    #        dm = self.make_rdm1()
    #    return np.diag(np.diag(self.mol.hubbard_u * dm/2))

    #def energy_tot(self, *args, **kwargs):
    #    return self.e_tot
