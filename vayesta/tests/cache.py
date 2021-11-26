import pyscf.gto
import pyscf.scf
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.tools.ring

import numpy as np

from vayesta.misc import molstructs, gdf
from vayesta.lattmod import latt
from vayesta import log


allowed_keys_mole = [
        'h2_ccpvdz', 'h2o_ccpvdz', 'h2o_ccpvdz_df', 'n2_631g',
        'n2_ccpvdz_df', 'lih_ccpvdz', 'h6_sto6g', 'h10_sto6g',
]


allowed_keys_cell = [
        'he2_631g_222', 'he_631g_222', 'h2_sto3g_331_2d',
]


allowed_keys_latt = [
        'hubb_6_u0', 'hubb_10_u2', 'hubb_16_u4', 'hubb_6x6_u0_1x1imp',
        'hubb_6x6_u2_1x1imp', 'hubb_6x6_u6_1x1imp', 'hubb_8x8_u2_2x2imp',
        'hubb_8x8_u2_2x1imp',
]


class Cache:
    def __init__(self):
        self._cache = {}

    def register_system(self, key):
        raise NotImplementedError

    def __getitem__(self, key):
        if key not in self._cache:
            self.register_system(key)
        return self._cache[key]

    def register_all(self):
        for key in self.allowed_keys:
            self.register_system(key)


def register_system_mole(cache, key):
    """Register one of the preset molecular test systems in the cache.
    """

    mol = pyscf.gto.Mole()
    rhf = uhf = False
    df = False

    if key == 'h2_ccpvdz':
        mol.atom = 'H1 0 0 0; H2 0 0 1'
        mol.basis = 'cc-pvdz'
        rhf = uhf = True
    elif key == 'h2_ccpvdz_stretch':
        mol.atom = 'H1 0 0 0; H2 0 0 1.4'
        mol.basis = 'cc-pvdz'
        rhf = uhf = True
    elif key == 'h2_ccpvdz_diss':
        mol.atom = 'H1 0 0 0; H2 0 0 5.0'
        mol.basis = 'cc-pvdz'
        uhf = True
    elif key == 'h2o_ccpvdz':
        mol.atom = molstructs.water()
        mol.basis = 'cc-pvdz'
        rhf = uhf = True
    elif key == 'h2o_ccpvdz_df':
        mol.atom = molstructs.water()
        mol.basis = 'cc-pvdz'
        rhf = uhf = True
        df = True
    elif key == 'n2_631g':
        mol.atom = 'N1 0 0 0; N2 0 0 1.1'
        mol.basis = '6-31g'
        rhf = uhf = True
    elif key == 'n2_ccpvdz_df':
        mol.atom = 'N1 0 0 0; N2 0 0 1.1'
        mol.basis = 'cc-pvdz'
        rhf = uhf = True
        df = True
    elif key == 'lih_ccpvdz':
        mol.atom = 'Li 0 0 0; H 0 0 1.4'
        mol.basis = 'cc-pvdz'
        rhf = uhf = True
    elif key == 'h6_sto6g':
        mol.atom = ['H %f %f %f' % xyz for xyz in pyscf.tools.ring.make(6, 1.0)]
        mol.basis = 'sto-6g'
        rhf = uhf = True
    elif key == 'h10_sto6g':
        mol.atom = ['H %f %f %f' % xyz for xyz in pyscf.tools.ring.make(10, 1.0)]
        mol.basis = 'sto-6g'
        rhf = uhf = True
    else:
        log.error("No system with key '%s'", key)
        return {}

    log.info("Registering system '%s'", key)

    mol.verbose = 0
    mol.max_memory = 1e9
    mol.build()

    #TODO check stability
    if rhf:
        rhf = pyscf.scf.RHF(mol)
        if df:
            rhf = rhf.density_fit()
        rhf.conv_tol = 1e-14
        rhf.kernel()

    if uhf:
        uhf = pyscf.scf.UHF(mol)
        if df:
            uhf = uhf.density_fit()
        uhf.conv_tol = 1e-14
        uhf.kernel()

    cache._cache[key] = {
        'mol': mol,
        'rhf': rhf,
        'uhf': uhf,
    }


def register_system_cell(cache, key):
    """Register one of the preset solid test systems in the cache.
    """

    cell = pyscf.pbc.gto.Cell()
    kpts = None
    rhf = uhf = False

    if key == 'he2_631g_222':
        cell.atom = 'He 3 2 3; He 1 1 1'
        cell.basis = '6-31g'
        cell.a = np.eye(3) * 5
        cell.mesh = [11, 11, 11]
        kpts = cell.make_kpts([2, 2, 2])
        rhf = uhf = True
    elif key == 'he_631g_222':
        cell.atom = 'He 0 0 0'
        cell.basis = '6-31g'
        cell.a = np.eye(3) * 3
        kpts = cell.make_kpts([2, 2, 2])
        rhf = True
    elif key == 'h2_sto3g_331_2d':
        cell.atom = 'H 0 0 0; H 0 0 1.8'
        cell.basis = 'sto3g'
        cell.a = [[4, 0, 0], [0, 4, 0], [0, 0, 30]]
        cell.dimension = 2
        kpts = cell.make_kpts([3, 3, 1])
        rhf = True
    else:
        log.error("No system with key '%s'", key)
        return {}

    log.info("Registering system '%s'", key)

    cell.verbose = 0
    cell.max_memory = 1e9
    cell.build()

    if rhf:
        rhf = pyscf.pbc.scf.KRHF(cell, kpts)
        rhf.conv_tol = 1e-12
        if cell.dimension == 3:
            rhf.with_df = gdf.GDF(cell, kpts)
        else:
            rhf = rhf.density_fit()
        rhf.with_df.build()
        rhf.kernel()

    if uhf:
        uhf = pyscf.pbc.scf.KUHF(cell, kpts)
        uhf.conv_tol = 1e-12
        if rhf:
            uhf.with_df = rhf.with_df
        else:
            if cell.dimension == 3:
                uhf.with_df = gdf.GDF(cell, kpts)
            else:
                uhf = uhf.density_fit()
        uhf.with_df.build()
        uhf.kernel()

    cache._cache[key] = {
        'cell': cell,
        'kpts': kpts,
        'rhf': rhf,
        'uhf': uhf,
    }


def register_system_latt(cache, key):
    """Register on the preset lattice test systems in the cache.
    """

    cell = None
    rhf = uhf = False

    if key == 'hubb_6_u0':
        cell = latt.Hubbard1D(6, hubbard_u=0.0, nelectron=6)
        rhf = True
    elif key == 'hubb_10_u2':
        cell = latt.Hubbard1D(10, hubbard_u=2.0, nelectron=10)
        rhf = True
    elif key == 'hubb_16_u4':
        cell = latt.Hubbard1D(10, hubbard_u=4.0, nelectron=16, boundary='apbc')
        rhf = True
    elif key == 'hubb_6x6_u0_1x1imp':
        cell = latt.Hubbard2D((6, 6), hubbard_u=0.0, nelectron=26, tiles=(1, 1), boundary='pbc')
        rhf = True
    elif key == 'hubb_6x6_u2_1x1imp':
        cell = latt.Hubbard2D((6, 6), hubbard_u=2.0, nelectron=26, tiles=(1, 1), boundary='pbc')
        rhf = True
    elif key == 'hubb_6x6_u6_1x1imp':
        cell = latt.Hubbard2D((6, 6), hubbard_u=6.0, nelectron=26, tiles=(1, 1), boundary='pbc')
        rhf = True
    elif key == 'hubb_8x8_u2_2x2imp':
        cell = latt.Hubbard2D((8, 8), hubbard_u=2.0, nelectron=50, tiles=(2, 2), boundary='pbc')
        rhf = True
    elif key == 'hubb_8x8_u2_2x1imp':
        cell = latt.Hubbard2D((8, 8), hubbard_u=2.0, nelectron=50, tiles=(2, 1), boundary='pbc')
        rhf = True
    else:
        log.error("No system with key '%s'", key)
        return {}

    if rhf:
        rhf = latt.LatticeRHF(cell)
        rhf.conv_tol = 1e-12
        rhf.kernel()

    if uhf:
        uhf = latt.LatticeUHF(cell)
        uhf.conv_tol = 1e-12
        uhf.kernel()

    cache._cache[key] = {
        'latt': cell,
        'rhf': rhf,
        'uhf': uhf,
    }


class MoleculeCache(Cache):
    register_system = register_system_mole
    allowed_keys = allowed_keys_mole

moles = MoleculeCache()


class SolidCache(Cache):
    register_system = register_system_cell
    allowed_keys = allowed_keys_cell

cells = SolidCache()


class LatticeCache(Cache):
    register_system = register_system_latt
    allowed_keys = allowed_keys_latt

latts = LatticeCache()
