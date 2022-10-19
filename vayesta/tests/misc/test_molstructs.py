import pytest
import unittest
import numpy as np

from pyscf import gto, scf
from pyscf.pbc import gto as pbc_gto, scf as pbc_scf

from vayesta.misc import molecules
from vayesta.misc import solids
from vayesta.tests.common import TestCase


@pytest.mark.fast
class MolstructsTests(TestCase):
    PLACES_ENERGY = 7

    def _test_nuclear_energy(self, mol, known_values):
        """Test the nuclear energy.
        """

        self.assertAlmostEqual(mol.energy_nuc(), known_values['e_nuc'], self.PLACES_ENERGY)

    def test_water(self):
        """Tests for water molecule.
        """

        mol = gto.M(atom=molecules.water(), verbose=0)
        known_values = {
                'e_nuc': 9.189533762934902,
                'e_tot': -74.96302313846098,
        }

        self._test_nuclear_energy(mol, known_values)

    def test_alkane(self):
        """Tests for alkane molecules.
        """

        mols = [
                gto.M(atom=molecules.alkane(3), verbose=0),
                gto.M(atom=molecules.alkane(3, numbering='atom'), verbose=0),
                gto.M(atom=molecules.alkane(3, numbering='unit'), verbose=0),
        ]
        known_values = {
                'e_nuc': 82.6933759181699,
                'e_tot': -116.88512481584637,
        }

        for mol in mols:
            self._test_nuclear_energy(mol, known_values)

    def test_arene(self):
        """Tests for arene molecules.
        """

        mol = gto.M(atom=molecules.arene(4), verbose=0)
        known_values = {
                'e_nuc': 102.06071721540009,
                'e_tot': -151.66864765196442,
        }

        self._test_nuclear_energy(mol, known_values)

    def test_neopentane(self):
        """Tests for neopentane molecule.
        """

        mol = gto.M(atom=molecules.neopentane(), verbose=0)
        known_values = {
                'e_nuc': 198.347989543373,
                'e_tot': -194.04575839870574,
        }

        self._test_nuclear_energy(mol, known_values)

    def test_boronene(self):
        """Tests for boronene molecule.
        """

        mol = gto.M(atom=molecules.boronene(), verbose=0)
        known_values = {
                'e_nuc': 1788.3645271898426,
        }

        self._test_nuclear_energy(mol, known_values)

    def test_coronene(self):
        """Tests for coronene molecule.
        """

        mol = gto.M(atom=molecules.coronene(), verbose=0)
        known_values = {
                'e_nuc': 1837.229262707072,
        }

        self._test_nuclear_energy(mol, known_values)

    def test_no2(self):
        """Tests for NO2 molecule.
        """

        mol = gto.M(atom=molecules.no2(), spin=1, verbose=0)
        known_values = {
                'e_nuc': 65.07473745355408,
                #'e_tot': -201.27201791520167,  # UHF not very stable
        }

        self._test_nuclear_energy(mol, known_values)

    def test_diamond(self):
        """Tests for diamond cell.
        """

        a, atom = solids.diamond()
        mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        mol.exp_to_discard = 0.1
        mol.build()
        known_values = {
                'e_nuc': -12.775667300117394,
        }

        self._test_nuclear_energy(mol, known_values)

    def test_graphene(self):
        """Tests for graphene cell.
        """

        a, atom = solids.graphene()
        mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        mol.exp_to_discard = 0.1
        mol.build()
        known_values = {
                'e_nuc': 47.86219216629114,
        }

        self._test_nuclear_energy(mol, known_values)

    def test_graphite(self):
        """Tests for graphite cell.
        """

        a, atom = solids.graphite()
        mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        mol.exp_to_discard = 0.1
        mol.build()
        known_values = {
                'e_nuc': -16.925780021798708,
        }

        self._test_nuclear_energy(mol, known_values)

    def test_rocksalt(self):
        """Tests for rocksalt cell.
        """

        a, atom = solids.rocksalt(unitcell='cubic')
        mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        mol.exp_to_discard = 0.1
        mol.build()
        known_values = {
                'e_nuc': -137.60716858207616,
        }

        self._test_nuclear_energy(mol, known_values)

        a, atom = solids.rocksalt(unitcell='primitive')
        mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        mol.exp_to_discard = 0.1
        mol.build()
        known_values = {
                'e_nuc': -34.4017921624733,
        }

        self._test_nuclear_energy(mol, known_values)

    def test_perovskite(self):
        """Tests for perovskite cell.
        """

        a, atom = solids.perovskite()
        mol = pbc_gto.Cell(atom=atom, a=a, verbose=0, basis='gth-szv-molopt-sr', pseudo='gth-pade')
        mol.exp_to_discard = 0.1
        mol.build()
        known_values = {
                'e_nuc': -106.25339755801713,
        }

        self._test_nuclear_energy(mol, known_values)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
