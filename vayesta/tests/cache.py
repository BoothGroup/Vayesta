import pyscf.gto
import pyscf.scf
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.tools.ring

import numpy as np

from vayesta.misc import molstructs, gdf
from vayesta import log

#TODO check stability
#TODO check spin_square for UHF


allowed_keys_mol = [
        'h2_ccpvdz', 'h2_ccpvdz_stretch', 'h2o_ccpvdz', 'h2o_augccpvdz',
        'n2_ccpvdz', 'lih_ccpvdz', 'h6_sto6g', 'h10_sto6g',
]


allowed_keys_cell = [
        'he2_631g_222', 'he2_ccpvdz_222', 'he_631g_222', 'h2_sto3g_331_2d',
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


def register_system_mol(cache, key):
    """Register one of the preset molecular test systems in the cache.
    """

    mol = pyscf.gto.Mole()
    rhf = uhf = rhf_df = uhf_df = False

    if key == 'h2_ccpvdz':
        mol.atom = 'H1 0 0 0; H2 0 0 1'
        mol.basis = 'cc-pvdz'
        rhf = uhf = True
    elif key == 'h2_ccpvdz_stretch':
        mol.atom = 'H1 0 0 0; H2 0 0 1.4'
        mol.basis = 'cc-pvdz'
        rhf = uhf = True
    elif key == 'h2o_ccpvdz':
        mol.atom = molstructs.water()
        mol.basis = 'cc-pvdz'
        rhf = uhf = rhf_df = uhf_df = True
    elif key == 'h2o_augccpvdz':
        mol.atom = molstructs.water()
        mol.basis = 'aug-cc-pvdz'
        rhf_df = uhf_df = True
    elif key == 'n2_ccpvdz':
        mol.atom = 'N1 0 0 0; N2 0 0 1.1'
        mol.basis = 'cc-pvdz'
        rhf = uhf = rhf_df = uhf_df = True
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
        rhf.conv_tol = 1e-12
        rhf.kernel()

    if uhf:
        uhf = pyscf.scf.UHF(mol)
        uhf.conv_tol = 1e-12
        uhf.kernel()

    if rhf_df:
        rhf_df = pyscf.scf.RHF(mol)
        rhf_df = rhf_df.density_fit()
        rhf_df.conv_tol = 1e-12
        rhf_df.kernel()

    if uhf_df is not None:
        uhf_df = pyscf.scf.UHF(mol)
        uhf_df = uhf_df.density_fit()
        uhf_df.conv_tol = 1e-12
        uhf_df.kernel()

    cache._cache[key] = {
        'mol': mol,
        'rhf': rhf,
        'uhf': uhf,
        'rhf_df': rhf_df,
        'uhf_df': uhf_df,
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
        rhf = True
    elif key == 'he2_ccpvdz_222':
        cell.atom = 'He 3 2 3; He 1 1 1'
        cell.basis = 'cc-pvdz'
        cell.exp_to_discard = 0.1
        cell.a = np.eye(3) * 5
        cell.mesh = [11, 11, 11]
        kpts = cell.make_kpts([2, 2, 2])
        rhf = True
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
        rhf = pyscf.pbc.scf.KRHF(cell)
        rhf.conv_tol = 1e-12
        if cell.dimension == 3:
            rhf.with_df = gdf.GDF(cell, kpts)
        else:
            rhf = rhf.density_fit()
        rhf.with_df.build()
        rhf.kernel()

    if uhf:
        uhf = pyscf.pbc.scf.KUHF(cell)
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


class MoleculeCache(Cache):
    register_system = register_system_mol
    allowed_keys = allowed_keys_mol

mols = MoleculeCache()


class SolidCache(Cache):
    register_system = register_system_cell
    allowed_keys = allowed_keys_cell

cells = SolidCache()
