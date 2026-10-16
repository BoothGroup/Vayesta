import os
import tempfile

import pytest
import numpy as np
import pyscf.dft.numint

from vayesta.misc.cubefile import CubeFile
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


def read_cube(filename):
    """Read header information and voxel data of a cube file."""
    with open(filename) as f:
        lines = f.readlines()
    natm, *origin = lines[2].split()[:4]
    natm = int(natm)
    nfields = 1
    if natm < 0:
        nfields = int(lines[2].split()[4])
        natm = -natm
    grid = [int(lines[3 + i].split()[0]) for i in range(3)]
    ndata = 6 + natm + (1 if nfields > 1 else 0)
    data = np.asarray(" ".join(lines[ndata:]).split(), dtype=float)
    data = data.reshape(np.prod(grid), nfields)
    return dict(natm=natm, nfields=nfields, grid=grid, origin=np.asarray(origin, dtype=float), data=data)


@pytest.mark.fast
class TestCubeFileMolecule(TestCase):
    gridsize = (6, 7, 8)

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_sto3g.rhf()

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.filename = os.path.join(self.tmpdir.name, "test.cube")

    def tearDown(self):
        self.tmpdir.cleanup()

    def get_ao(self, cube):
        return self.mf.mol.eval_gto("GTOval", cube.coords)

    def test_single_orbital(self):
        cube = CubeFile(self.mf.mol, self.filename, gridsize=self.gridsize)
        cube.add_orbital(self.mf.mo_coeff[:, 0])
        cube.write()
        res = read_cube(self.filename)
        self.assertEqual(res["natm"], self.mf.mol.natm)
        self.assertEqual(res["nfields"], 1)
        self.assertEqual(res["grid"], list(self.gridsize))
        self.assertAllclose(res["origin"], cube.origin, atol=1e-6)
        expected = np.dot(self.get_ao(cube), self.mf.mo_coeff[:, 0])
        self.assertAllclose(res["data"][:, 0], expected, atol=1e-4, rtol=1e-4)

    def test_multiple_fields(self):
        dm = self.mf.make_rdm1()
        with CubeFile(self.mf.mol, self.filename, gridsize=self.gridsize) as cube:
            cube.add_orbital(self.mf.mo_coeff[:, :2])
            cube.add_density(dm)
        self.assertEqual(cube.nfields, 3)
        res = read_cube(self.filename)
        self.assertEqual(res["nfields"], 3)
        ao = self.get_ao(cube)
        self.assertAllclose(res["data"][:, :2], np.dot(ao, self.mf.mo_coeff[:, :2]), atol=1e-4, rtol=1e-4)
        rho = pyscf.dft.numint.eval_rho(self.mf.mol, ao, dm)
        self.assertAllclose(res["data"][:, 2], rho, atol=1e-4, rtol=1e-4)

    def test_save_load_state(self):
        cube = CubeFile(self.mf.mol, self.filename, gridsize=self.gridsize)
        cube.add_orbital(self.mf.mo_coeff[:, 0])
        statefile = os.path.join(self.tmpdir.name, "state.pkl")
        cube.save_state(statefile)
        self.assertIs(cube.cell, self.mf.mol)
        cube2 = CubeFile.load_state(statefile, cell=self.mf.mol)
        self.assertEqual(cube2.nfields, 1)
        self.assertAllclose(cube2.coords, cube.coords)

    def test_resolution(self):
        cube = CubeFile(self.mf.mol, self.filename, resolution=2.0)
        expected = [min(int(np.ceil(abs(cube.a[i, i]) * 2.0)), 192) for i in range(3)]
        self.assertEqual([cube.nx, cube.ny, cube.nz], expected)
        # Low resolution only logs a warning:
        CubeFile(self.mf.mol, self.filename, resolution=0.5)


@pytest.mark.fast
class TestCubeFileSolid(TestCase):
    def test_orbital(self):
        cell = testsystems.h2_sto3g_s311.mol
        mf = testsystems.h2_sto3g_s311.rhf()
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "subdir", "test.cube")
            with CubeFile(cell, filename, gridsize=(4, 4, 6)) as cube:
                cube.add_orbital(mf.mo_coeff[:, 0])
            self.assertTrue(cube.has_pbc)
            self.assertAllclose(cube.origin, 0)
            res = read_cube(filename)
        ao = cell.pbc_eval_gto("PBCGTOval", cube.coords)
        self.assertAllclose(res["data"][:, 0], np.dot(ao, mf.mo_coeff[:, 0]), atol=1e-4, rtol=1e-4)
