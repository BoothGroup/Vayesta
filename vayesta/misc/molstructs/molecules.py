import numpy as np

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
        # Terminal Hydrogen atoms
        if (i == 0):
            atom.append([get_symbol(atoms[1]), [0.0, y-dchc, z-dchs]])
        # Not elif, if n == 1 (Methane)
        if (i == n-1):
            atom.append([get_symbol(atoms[1]), [0.0, y-sign*dchc, z+dchs]])
        if numbering == 'unit':
            index += 1

    return atom


def arene(n, atoms=['C', 'H'], cc_bond=1.39, ch_bond=1.09):
    """Bond length for benzene."""

    r1 = cc_bond/(2*np.sin(np.pi/n))
    r2 = r1 + ch_bond
    z = 0.0
    atom = []
    for i in range(n):
        phi = 2*i*np.pi/n
        atom.append((atoms[0], (r1*np.cos(phi), r1*np.sin(phi), z)))
        atom.append((atoms[1], (r2*np.cos(phi), r2*np.sin(phi), z)))
    return atom
