# TODO

class Latt:
    """Fake pyscf.gto.Mole object, for lattice models.

    Needs to implement:
    a
    dimension
    nelectron
    natm
    nao_nr()
    ao_labels()
    search_ao_label()
    atom_symbol()
    copy()
    build()
    intor_cross()
    intor_symmetric()
    pbc_intor()
    basis

    ?
    lattice_vectors()
    atom_coord
    unit

    """

    def __init__(self, nsite, nelectron):
        self.nsite = nsite
        self.nelectron
        self.hubbard_u


    def make_mol(self):
        # TODO
        #mol = 


    # PySCF compatibility:
    @property
    def natm(self):
        return self.nsite

