import os.path
import numpy as np

def load_from_file(filename):
    path = os.path.dirname(__file__)
    data = np.loadtxt(os.path.join(path, filename))
    amat = data[:3]
    coords = data[3:]
    return amat, coords

class NO2_Graphene:

    def __init__(self, supercell, structure=-1, distance=None, vacuum_size=None, z_graphene=None):
        supercell = tuple(supercell)
        if supercell not in ((3, 3), (4,4), (5,5), (10, 10)):
            raise ValueError()
        self.supercell = supercell
        self.structure = structure
        self.distance = distance
        self.vacuum_size = vacuum_size
        self.z_graphene = z_graphene

    def get_amat_coords(self):
        amat, coords = load_from_file('no2_graphene_q%d_%dx%d.dat' % (self.structure, *self.supercell))
        if self.distance is not None:
            coords[:3,2] += (self.distance - 3.0)
        if self.vacuum_size is not None:
            amat[2,2] = self.vacuum_size
            if self.z_graphene is None:
                self.z_graphene = 17.0 + (self.vacuum_size-30.0)/2
        if self.z_graphene is not None:
            coords[:,2] += (self.z_graphene - 17.0)
        return amat, coords

    def get_amat_atom(self):
        amat, coords = self.get_amat_coords()
        atoms = ['N', 'O', 'O'] + (len(coords)-3)*['C']
        atoms = [(s, coords[i]) for i, s in enumerate(atoms)]
        return amat, atoms


if __name__ == '__main__':
    supercell = (5, 5)

    cell = NO2_Graphene(supercell)
    print(cell.get_amat_atom()[0])
    print(cell.get_amat_atom()[1][:5])

    cell = NO2_Graphene(supercell, 0)
    print(cell.get_amat_atom()[0])
    print(cell.get_amat_atom()[1][:5])
