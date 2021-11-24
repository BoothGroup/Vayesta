import unittest
import numpy as np
from pyscf import gto, scf
from pyscf.pbc import gto as pbc_gto, scf as pbc_scf
from vayesta.misc import molstructs


class MolstructsTests:
    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.known_values = {}

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.known_values

    HF = scf.RHF

    def test_e_nuc(self):
        if 'e_nuc' not in self.known_values:
            return
        mols = self.mol if isinstance(self.mol, list) else [self.mol]
        for mol in mols:
            e_nuc = mol.energy_nuc()
            self.assertAlmostEqual(e_nuc, self.known_values['e_nuc'], 8)

    def test_e_mf(self):
        if 'e_tot' not in self.known_values:
            return
        mols = self.mol if isinstance(self.mol, list) else [self.mol]
        for mol in mols:
            mf = scf.RHF(mol)
            mf.kernel()
            self.assertAlmostEqual(mf.e_tot, self.known_values['e_tot'], 8)


class WaterTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=molstructs.molecules.water(), verbose=0)
        cls.known_values = {
                'e_nuc': 9.189533762934902,
                'e_tot': -74.96302313846098,
        }


class AlkaneTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = [
                gto.M(atom=molstructs.molecules.alkane(3), verbose=0),
                gto.M(atom=molstructs.molecules.alkane(3, numbering='atom'), verbose=0),
                gto.M(atom=molstructs.molecules.alkane(3, numbering='unit'), verbose=0),
        ]
        cls.known_values = {
                'e_nuc': 82.6933759181699,
                'e_tot': -116.88512481584637,
        }


class AreneTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=molstructs.molecules.arene(4), verbose=0)
        cls.known_values = {
                'e_nuc': 102.06071721540009,
                'e_tot': -151.66864765196442,
        }


class NeopentaneTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=molstructs.molecules.neopentane(), verbose=0)
        cls.known_values = {
                'e_nuc': 198.347989543373,
                'e_tot': -194.04575839870574,
        }


class BoroneneTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=molstructs.molecules.boronene(), verbose=0)
        cls.known_values = {
                'e_nuc': 1788.3645271898426,
        }


class CoroneneTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=molstructs.molecules.coronene(), verbose=0)
        cls.known_values = {
                'e_nuc': 1837.229262707072,
        }


class NO2Tests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom=molstructs.molecules.no2(), spin=1, verbose=0)
        cls.known_values = {
                'e_nuc': 65.07473745355408,
                #'e_tot': -201.27201791520167,  # not very stable
        }

    HF = scf.UHF


class DiamondTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        a, atom = molstructs.solids.diamond()
        cls.mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        cls.mol.exp_to_discard = 0.1
        cls.mol.build()
        cls.known_values = {
                'e_nuc': -12.775667300117394,
        }


class GrapheneTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        a, atom = molstructs.solids.graphene()
        cls.mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        cls.mol.exp_to_discard = 0.1
        cls.mol.build()
        cls.known_values = {
                'e_nuc': 47.86219216629114,
        }


class GraphiteTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        a, atom = molstructs.solids.graphite()
        cls.mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        cls.mol.exp_to_discard = 0.1
        cls.mol.build()
        cls.known_values = {
                'e_nuc': -16.925780021798708,
        }


class RocksaltTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        a, atom = molstructs.solids.rocksalt(primitive=False)
        cls.mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        cls.mol.exp_to_discard = 0.1
        cls.mol.build()
        cls.known_values = {
                'e_nuc': -137.60716858207616,
        }


class RocksaltPrimitiveTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        a, atom = molstructs.solids.rocksalt(primitive=True)
        cls.mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        cls.mol.exp_to_discard = 0.1
        cls.mol.build()
        cls.known_values = {
                'e_nuc': -34.4017921624733,
        }


class PerovskiteTests(unittest.TestCase, MolstructsTests):
    @classmethod
    def setUpClass(cls):
        a, atom = molstructs.solids.perovskite()
        cls.mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        cls.mol.exp_to_discard = 0.1
        cls.mol.build()
        cls.known_values = {
                'e_nuc': -106.25339755801713,
        }


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
