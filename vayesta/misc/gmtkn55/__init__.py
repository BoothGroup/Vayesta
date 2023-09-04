import numpy as np
import os.path
from pathlib import Path
import glob
import pyscf
import pyscf.gto


moddir = Path(__file__).parent


class TestSet:
    def __init__(self, path):
        path = moddir / path
        dtype = [("key", object), ("charge", int), ("multiplicity", int)]
        systems = np.loadtxt(path / "list.txt", dtype=dtype)

        self.systems = {}
        dtype = [("atom", object), ("x", float), ("y", float), ("z", float)]
        for key, charge, multiplicity in systems:
            # Get atomic structure
            data = np.loadtxt(path / key / "struc.xyz", skiprows=2, dtype=dtype)
            if data.ndim == 1:
                structure = [(data["atom"][i], (data["x"][i], data["y"][i], data["z"][i])) for i in range(len(data))]
            else:
                structure = [(str(data["atom"]), (float(data["x"]), float(data["y"]), float(data["z"])))]
            # Check .CHRG file
            try:
                chrg = int(np.loadtxt(path / key / ".CHRG", dtype=int))
                assert chrg == charge
            except OSError:
                print("File '.CHRG' not found for system %s." % key)
            # Check .UHF file
            try:
                uhf = bool(np.loadtxt(path / key / ".UHF", dtype=bool))
                assert uhf == (multiplicity > 1)
            except OSError:
                print("File '.UHF' not found for system %s." % key)

            self.systems[key] = (structure, charge, multiplicity)

    def loop(self, min_atoms=0, max_atoms=np.inf, include_uhf=True, systems=None, **kwargs):
        for key in systems or self.systems:
            mol = self.get_mol(key, **kwargs)
            if mol.natm < min_atoms or mol.natm > max_atoms:
                continue
            if not include_uhf and mol.spin > 0:
                continue
            yield key, mol

    def get_mol(self, key, **kwargs):
        structure, charge, multiplicity = self.systems[key]
        mol = pyscf.gto.Mole()
        mol.atom = structure
        if charge != 0:
            mol.charge = charge
        if multiplicity != 1:
            mol.spin = multiplicity - 1
        for attr, val in kwargs.items():
            setattr(mol, attr, val)
        mol.build()
        return mol


W411 = TestSet("W4-11")


if __name__ == "__main__":
    for key, mol in W411.loop(min_atoms=3, max_atoms=4, include_uhf=False):
        print(mol.natm, mol.spin)
