import logging

import numpy as np

import pyscf
import pyscf.lib

log = logging.getLogger(__name__)

def make_mol(mol, atom, rmax, nimages=1, unit='A', **kwargs):
    """Make molecule object for counterposise calculation.

    WARNING: This has only been tested for periodic systems so far!

    Parameters
    ----------
    mol : pyscf.gto.Mole or pyscf.pbc.gto.Cell object
        PySCF molecule or cell object.
    atom : int
        Atom index for which the counterpoise correction should be calculated.
        TODO: allow list of atoms.
    rmax : float
        All atom centers within range `rmax` are added as ghost-atoms in the counterpoise correction.
    nimages : int, optional
        Number of neighboring unit cell in each spatial direction. Has no effect in open boundary
        calculations. Default: 1.
    unit : ['A', 'B']
        Unit for `rmax`, either Angstrom (`A`) or Bohr (`B`).
    **kwargs :
        Additional keyword arguments for returned PySCF Mole/Cell object.

    Returns
    -------
    mol_cp : pyscf.gto.Mole or pyscf.pbc.gto.Cell
        Mole or Cell object with periodic boundary conditions removed
        and with ghost atoms added depending on `rmax` and `nimages`.
    """
    unit_pyscf = 'ANG' if (unit.upper()[0] == 'A') else unit
    # Add physical atom
    atoms = [(mol.atom_symbol(atom), mol.atom_coord(atom, unit=unit_pyscf))]
    log.debugv("Counterpoise: adding atom %6s at %8.5f %8.5f %8.5f %s", atoms[0][0], *(atoms[0][1]), unit)

    # If rmax > 0: Atomic calculation with additional basis functions
    # else: Atom only
    if rmax:
        images = np.zeros(3, dtype=int)
        if (not hasattr(mol, 'a')) or (mol.a is None):
            pass    # Open boundary conditions - do not create images
        elif mol.dimension == 1:
            images[0] = nimages
        elif mol.dimension == 2:
            images[:2] = nimages
        elif mol.dimension == 3:
            images[:] = nimages
        log.debugv('Counterpoise images= %r', images)
        # Add ghost atoms. Note that rx = ry = rz = 0 for open boundary conditions
        center = mol.atom_coord(atom, unit=unit_pyscf)
        log.debugv('Counterpoise center= %r %s', center, unit)
        amat = mol.lattice_vectors().copy()    # In Bohr
        log.debugv('Latt. vec.= %r', amat)
        log.debugv('mol.unit= %r', mol.unit)
        if unit.upper()[0] == 'A':
            amat *= pyscf.lib.param.BOHR
        log.debugv('amat= %r', amat)
        for rx in range(-images[0], images[0]+1):
            for ry in range(-images[1], images[1]+1):
                for rz in range(-images[2], images[2]+1):
                    for atm in range(mol.natm):
                        symb = mol.atom_symbol(atm)
                        coord = mol.atom_coord(atm, unit=unit_pyscf)
                        # This is a fragment atom - already included above as real atom
                        if (abs(rx)+abs(ry)+abs(rz) == 0) and (atm == atom):
                            assert (symb == atoms[0][0])
                            assert np.allclose(coord, atoms[0][1])
                            continue
                        # This is either a non-fragment atom in the unit cell (rx = ry = rz = 0) or in a neighbor cell
                        if abs(rx)+abs(ry)+abs(rz) > 0:
                            coord += (rx*amat[0] + ry*amat[1] + rz*amat[2])
                        if not symb.lower().startswith('ghost'):
                            symb = 'Ghost-' + symb
                        distance = np.linalg.norm(coord - center)
                        if (not rmax) or (distance <= rmax):
                            log.debugv("Counterpoise: adding atom %6s at %8.5f %8.5f %8.5f with distance %8.5f %s", symb, *coord, distance, unit)
                            atoms.append((symb, coord))
    mol_cp = mol.copy()
    mol_cp.atom = atoms
    mol_cp.unit = unit_pyscf
    mol_cp.a = None
    for key, val in kwargs.items():
        log.debugv("Counterpoise: setting attribute %s to %r", key, val)
        setattr(mol_cp, key, val)
    mol_cp.build(False, False)
    return mol_cp
