import os.path
import numpy as np

def _load_datafile(filename):
    datafile = os.path.join(os.path.dirname(__file__), os.path.join("data", filename))
    data = np.loadtxt(datafile, dtype=[("atoms", object), ("coords", np.float64, (3,))])
    #return data["atoms"], data["coords"]
    atoms = data["atoms"]
    coords = data["coords"]
    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]
    return atom

def water(atoms=['O', 'H']):
    atom = [[atoms[0], [0.0000,  0.0000,  0.1173]],
            [atoms[1], [0.0000,  0.7572, -0.4692]],
            [atoms[1], [0.0000, -0.7572, -0.4692]]]
    return atom

def alkane(n, atoms=['C', 'H'], cc_bond=1.54, ch_bond=1.09, numbering=False):

    assert numbering in (False, 'atom', 'unit')

    index = 0
    def get_symbol(symbol):
        nonlocal index
        if not numbering:
            return symbol
        if numbering == 'atom':
            out = '%s%d' % (symbol, index)
            index += 1
            return out
        return '%s%d' % (symbol, index)

    dk = 1 if (numbering == 'atom') else 0

    phi = np.arccos(-1.0/3)
    cph = 1/np.sqrt(3.0)
    sph = np.sin(phi/2.0)
    dcy = cc_bond * cph
    dcz = cc_bond * sph
    dchs = ch_bond * sph
    dchc = ch_bond * cph
    x = 0.0

    atom = []
    for i in range(n):
        # Carbon atoms
        sign = (-1)**i
        y = sign * dcy/2
        z = i*dcz
        atom.append([get_symbol(atoms[0]), [x, y, z]])
        # Hydrogen atoms on side
        dy = sign * dchc
        atom.append([get_symbol(atoms[1]), [x+dchs, y+dy, z]])
        atom.append([get_symbol(atoms[1]), [x-dchs, y+dy, z]])
        # Terminal hydrogen atom 1
        if (i == 0):
            atom.append([get_symbol(atoms[1]), [0.0, y-dchc, z-dchs]])
        # Terminal hydrogen atom 2
        # Do not use elif, since for n=1 (Methane), we need to add both terminal hydrogen atoms:
        if (i == n-1):
            atom.append([get_symbol(atoms[1]), [0.0, y-sign*dchc, z+dchs]])
        if numbering == 'unit':
            index += 1

    return atom

def arene(n, atoms=['C', 'H'], cc_bond=1.39, ch_bond=1.09, unit=1.0):
    """Bond length for benzene."""
    r1 = unit*cc_bond/(2*np.sin(np.pi/n))
    r2 = unit*(r1 + ch_bond)
    atom = []
    for i in range(n):
        phi = 2*i*np.pi/n
        atomidx = (2*i) % len(atoms)
        atom.append((atoms[atomidx], (r1*np.cos(phi), r1*np.sin(phi), 0.0)))
        atom.append((atoms[atomidx+1], (r2*np.cos(phi), r2*np.sin(phi), 0.0)))
    return atom

def neopentane():
    """Structure from B3LYP//aug-cc-pVTZ."""
    atom = _load_datafile('neopentane.dat')
    return atom

def boronene():
    atom = _load_datafile('boronene.dat')
    return atom

def coronene():
    atom = _load_datafile('coronene.dat')
    return atom

def no2():
    atom = [
	('N', (0.0000,  0.0000, 0.0000)),
	('O', (0.0000,  1.0989, 0.4653)),
	('O', (0.0000, -1.0989, 0.4653)),
	]
    return atom

def ethanol(oh_bond=None):
    atom = _load_datafile('ethanol.dat')
    if oh_bond is not None:
        pos_o = atom[2][1]
        pos_h = atom[3][1]
        voh = (pos_h - pos_o)
        voh = voh / np.linalg.norm(voh)
        pos_h = pos_o + oh_bond*voh
        atom[3][1] = pos_h
    return atom

def propanol():
    atom = _load_datafile('propanol.dat')
    return atom

def chloroethanol():
    atom = _load_datafile('chloroethanol.dat')
    return atom

def ketene(cc_bond=None):
    atom = _load_datafile('ketene.dat')
    if cc_bond is not None:
        pos_c1 = atom[0][1]
        pos_c2 = atom[1][1]
        vcc = (pos_c2 - pos_c1)
        vcc = vcc / np.linalg.norm(vcc)
        new_c2 = pos_c1 + cc_bond * vcc
        new_o = atom[2][1] + (new_c2 - pos_c2)
        atom[1][1] = new_c2
        atom[2][1] = new_o
    return atom
