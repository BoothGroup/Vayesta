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
        self.supercell = supercell
        self.structure = structure
        self.distance = distance
        self.vacuum_size = vacuum_size
        self.z_graphene = z_graphene

        # Output
        self.amat, self.coords = self.get_amat_coords()
        self.atom = self.get_atom()

    #@classmethod
    #def load_from_file(self, supercell, structure=-1, **kwargs):
    #    supercell = tuple(supercell)
    #    if supercell not in ((3, 3), (4,4), (5,5), (10, 10)):
    #        raise ValueError()
    #    cell = NO2_Graphene(supercell, structure=structure, 
    #    self.amat, self.coords = self.get_amat_coords()
    #    self.atom = self.get_atom()


    #    self.supercell = supercell
    #    self.structure = structure
    #    self.distance = distance
    #    self.vacuum_size = vacuum_size
    #    self.z_graphene = z_graphene


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

    def get_atom(self):
        atom = ['N', 'O', 'O'] + (len(self.coords)-3)*['C']
        atom = [(s, self.coords[i]) for i, s in enumerate(atom)]
        return atom

    def get_amat_atom(self):
        amat, coords = self.get_amat_coords()
        atoms = ['N', 'O', 'O'] + (len(coords)-3)*['C']
        atoms = [(s, coords[i]) for i, s in enumerate(atoms)]
        return amat, atoms

    def get_subcell_indices(self, subcell):
        bmat = np.linalg.inv(self.amat)
        internal = np.dot(self.coords, bmat)
        internal = np.dot(internal, np.diag([*self.supercell, 1]))
        indices = []
        for idx, atom in enumerate(internal):
            if np.any(atom[:2] > (np.asarray(subcell)-1e-10)):
                print("Skipping %r (%r)" % (atom, self.coords[idx]))
                continue
            print("Adding   %r (%r)" % (atom, self.coords[idx]))
            indices.append(idx)
        return indices

    def make_subcell(self, subcell):
        indices = self.get_subcell_indices(subcell)
        scaling = np.asarray([*subcell, 1]) / np.asarray([*self.supercell, 1])
        amat = self.amat * scaling[:,None]
        atom = (np.asarray(self.atom, dtype=object)[indices]).tolist()

        return amat, atom

if __name__ == '__main__':
    #supercell = (10, 10)
    supercell = (5, 5)

    cell = NO2_Graphene(supercell, -1)
    print(cell.amat)
    print(cell.atom[:5])

    subcell = (4,4)
    amat, atom = cell.make_subcell(subcell)

    cell = NO2_Graphene(subcell, -1)

    print(amat)
    print(cell.amat)

    for idx, at in enumerate(atom):
        try:
            print("%s %-40s - %s %-40s  ---- %r" % (*at, *cell.atom[idx], np.allclose(at[1], cell.atom[idx][1])))
        except:
            print("%s %-40s" % (at[0], at[1]))

    #assert np.allclose(amat, cell.amat)
    #for idx, at in enumerate(atom):
    #    assert at[0] == cell.atom[idx][0]
    #    assert np.allclose(at[1], cell.atom[idx][1]), "%r vs %r" % (at[1], cell.atom[idx][1])
