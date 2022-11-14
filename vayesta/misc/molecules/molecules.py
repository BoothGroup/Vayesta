import os.path
import numpy as np

def _load_datafile(filename, scale=1):
    datafile = os.path.join(os.path.dirname(__file__), os.path.join("data", filename))
    data = np.loadtxt(datafile, dtype=[("atoms", object), ("coords", np.float64, (3,))])
    atoms = data["atoms"]
    coords = scale*data["coords"]
    atom = [[atoms[i], coords[i]] for i in range(len(atoms))]
    return atom

def water(atoms=['O', 'H'], origin=(0, 0, 0), scale=1):
    origin = np.asarray(origin)
    atom = [[atoms[0], scale*np.asarray([0.0000,  0.0000,  0.1173]) - origin],
            [atoms[1], scale*np.asarray([0.0000,  0.7572, -0.4692]) - origin],
            [atoms[1], scale*np.asarray([0.0000, -0.7572, -0.4692]) - origin]]
    return atom

def alkane(n, atoms=['C', 'H'], cc_bond=1.54, ch_bond=1.09, scale=1.0, numbering=False):
    """Alkane with idealized tetrahedral (sp3) coordination."""

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
    dcy = scale*cc_bond * cph
    dcz = scale*cc_bond * sph
    dchs = scale*ch_bond * sph
    dchc = scale*ch_bond * cph
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

def alkene(ncarbon, cc_bond=1.33, ch_bond=1.09):
    """Alkene with idealized trigonal planar (sp2) coordination."""

    if (ncarbon < 2):
        raise ValueError
    if (ncarbon % 2 == 1):
        raise NotImplementedError

    cos30 = np.sqrt(3)/2
    sin30 = 1/2

    r0 = x0, y0, z0 = (0, 0, 0)
    atom = [('H', (x0-ch_bond*cos30, y0+ch_bond*sin30, z0))]
    for nc in range(ncarbon):
        sign = (-1)**nc
        atom += [('C', r0)]
        atom += [('H', (x0, y0-sign*ch_bond, z0))]
        if nc == (ncarbon-1):
            break
        r0 = x0, y0, z0 = (x0+cc_bond*cos30, y0+sign*cc_bond*sin30, 0)
    atom += [('H', (x0+ch_bond*cos30, y0+sign*ch_bond*sin30, z0))]

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

def no2():
    atom = [
	('N', (0.0000,  0.0000, 0.0000)),
	('O', (0.0000,  1.0989, 0.4653)),
	('O', (0.0000, -1.0989, 0.4653)),
	]
    return atom

def ethanol(oh_bond=None, scale=1):
    atom = _load_datafile('ethanol.dat', scale=scale)
    if oh_bond is not None:
        pos_o = atom[2][1]
        pos_h = atom[3][1]
        voh = (pos_h - pos_o)
        voh = voh / np.linalg.norm(voh)
        pos_h = pos_o + oh_bond*voh
        atom[3][1] = pos_h
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

def ring(atom, natom, bond_length=None, radius=None, z=0.0, numbering=None):
    if radius is None:
        r = bond_length/(2*np.sin(np.pi/natom))
    else:
        r = radius
    atoms = []
    if isinstance(atom, str):
        atom = [atom]
    for i in range(natom):
        theta = i * (2*np.pi/natom)
        atom_i = atom[i%len(atom)]
        if numbering is not None:
            atom_i += str(int(numbering) + i)
        atoms.append([atom_i, np.asarray([r*np.cos(theta), r*np.sin(theta), z])])
    return atoms

# --- From datafiles:

def acetic_acid():
    atom = _load_datafile('acetic.dat')
    return atom

def ferrocene_b3lyp():
    atom = _load_datafile('ferrocene.dat')
    return atom

def ferrocene(atoms=['Fe', 'C', 'H'], conformation='eclipsed', dFeCp=1.648, dCC=1.427, dCH=1.079, aCpH=0.52, numbering=None):
    """From https://pubs.acs.org/doi/pdf/10.1021/ct700152c"""
    if conformation != 'eclipsed':
        raise NotImplementedError

    rHH = dCC/(2*np.sin(np.pi/5)) + dCH*np.cos(aCpH*np.pi/180)
    zH = dCH*np.sin(aCpH*np.pi/180)

    atom = [(atoms[0] + ('1' if numbering else ''), np.asarray((0, 0, 0)))]
    atom += ring(atoms[1], 5, dCC, z=dFeCp, numbering=2 if numbering else None)
    atom += ring(atoms[2], 5, radius=rHH, z=dFeCp-zH, numbering=7 if numbering else None)
    atom += ring(atoms[1], 5, dCC, z=-dFeCp, numbering=12 if numbering else None)
    atom += ring(atoms[2], 5, radius=rHH, z=-dFeCp+zH, numbering=17 if numbering else None)

    return atom

def propyl():
    atom = _load_datafile('propyl.dat')
    return atom

def phenyl():
    atom = _load_datafile('phenyl.dat')
    return atom

def propanol():
    atom = _load_datafile('propanol.dat')
    return atom

def chloroethanol():
    atom = _load_datafile('chloroethanol.dat')
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

def glycine():
    atom = _load_datafile('glycine.dat')
    return atom
