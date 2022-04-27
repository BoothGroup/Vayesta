import numpy as np
import os.path
import glob

import pyscf
import pyscf.gto


testset_w4_11 = [x.replace('W4-11/', '') for x in glob.glob('W4-11/*')]

def get_system(testset, key, build=True, **kwargs):

    mol = pyscf.gto.Mole()
    path = os.path.join(os.path.dirname(__file__), testset, key)
    dtype = [('atom', object), ('x', float), ('y', float), ('z', float)]
    data = np.loadtxt(os.path.join(path, 'struc.xyz'), skiprows=2, dtype=dtype)
    if data.ndim == 1:
        mol.atom = [(data['atom'][i], (data['x'][i],  data['y'][i], data['z'][i])) for i in range(len(data))]
    else:
        mol.atom = [(str(data['atom']), (float(data['x']),  float(data['y']), float(data['z'])))]

    try:
        charge = int(np.loadtxt(os.path.join(path, '.CHRG'), dtype=int))
    except OSError:
        print("File '.CHRG' not found for %s system %s - returning 0." % (testset, key))
        charge = 0
    try:
        uhf = bool(np.loadtxt(os.path.join(path, '.UHF'), dtype=bool))
    except OSError:
        print("File '.UHF' not found for %s system %s - returning False." % (testset, key))
        uhf = False
    mol.charge = charge

    if mol.nelectron % 2 == 1:
        mol.spin = 1

    for attr, val in kwargs.items():
        setattr(mol, attr, val)
    if build:
        mol.build()
    return mol, uhf


if __name__ == '__main__':

    for key in testset_w4_11:
        mol, uhf = get_system('W4-11', key)
        print('%s : %s' % (key, mol.atom))
