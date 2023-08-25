import numpy as np

from vayesta.core.util import dot, einsum, time_string, timer
from vayesta.core.bath.bath import Bath
from vayesta.core.bath import helper


BOHR = 0.529177210903


def _to_bohr(rcut, unit):
    unit = unit.lower()
    if unit.startswith("ang"):
        return rcut / BOHR
    if unit.startswith("b"):
        return rcut
    raise ValueError("Invalid unit: %s" % unit)


def _get_r2(mol, center, mesh=None):
    if getattr(mol, "dimension", 0) == 0:
        # TODO: instead of evaluating for each center R,
        # use <r-R|r-R> = <r|r> - 2*<r|R> + <R|R>^2
        with mol.with_common_origin(center):
            return mol.intor_symmetric("int1e_r2")

    # For PBC:
    # Numerical integration over unit cell
    if mesh is None:
        mesh = 3 * [100]
    dx, dy, dz = 1 / (2 * np.asarray(mesh))
    x = np.linspace(-0.5 + dx, 0.5 - dx, mesh[0])
    y = np.linspace(-0.5 + dy, 0.5 - dy, mesh[1])
    z = np.linspace(-0.5 + dz, 0.5 - dz, mesh[2])
    mx, my, mz = np.meshgrid(x, y, z, indexing="ij")
    grid = np.stack((mx.flatten(), my.flatten(), mz.flatten()), axis=1)
    coords = np.dot(grid, mol.lattice_vectors())
    # Instead of:
    # coords = mol.get_uniform_grids(mesh))

    assert not mol.cart
    # We shift the coords around the center:
    gtoval = mol.pbc_eval_gto("GTOval_sph", coords + center)
    r2norm = np.linalg.norm(coords, axis=1) ** 2
    dvol = mol.vol / len(coords)
    r2 = dvol * einsum("xa,x,xb->ab", gtoval, r2norm, gtoval)
    return r2


class R2_Bath_RHF(Bath):
    def __init__(self, fragment, dmet_bath, occtype, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
        if occtype not in ("occupied", "virtual"):
            raise ValueError("Invalid occtype: %s" % occtype)
        self.occtype = occtype
        if len(self.fragment.atoms) != 1:
            raise NotImplementedError
        atom = self.fragment.atoms[0]
        self.center = self.mol.atom_coord(atom)  # In Bohr!
        self.coeff, self.eig = self.kernel()

    @property
    def c_env(self):
        if self.occtype == "occupied":
            return self.dmet_bath.c_env_occ
        if self.occtype == "virtual":
            return self.dmet_bath.c_env_vir

    def get_r2(self):
        r2 = _get_r2(self.mol, self.center)
        r2 = dot(self.c_env.T, r2, self.c_env)
        hermierr = np.linalg.norm(r2 - r2.T)
        if hermierr > 1e-11:
            self.log.warning("Hermiticity error= %.3e", hermierr)
            r2 = (r2 + r2.T) / 2
        else:
            self.log.debug("Hermiticity error= %.3e", hermierr)
        return r2

    def kernel(self):
        t_init = t0 = timer()
        r2 = self.get_r2()
        t_r2 = timer() - t0
        t0 = timer()
        eig, rot = np.linalg.eigh(r2)
        t_diag = timer() - t0
        if np.any(eig < -1e-13):
            raise RuntimeError("Negative eigenvalues: %r" % eig[eig < 0])
        eig = np.sqrt(np.clip(eig, 0, None))
        coeff = np.dot(self.c_env, rot)
        self.log.debug("%s eigenvalues (A):\n%r", self.occtype.capitalize(), eig * BOHR)
        self.log_histogram(eig, self.occtype)
        self.log.timing(
            "Time R2 bath:  R2= %s  diagonal.= %s  total= %s", *map(time_string, (t_r2, t_diag, (timer() - t_init)))
        )
        return coeff, eig

    def get_bath(self, rcut, unit="Ang"):
        rcut = _to_bohr(rcut, unit)
        nbath = np.count_nonzero(self.eig <= rcut)
        c_bath, c_rest = np.hsplit(self.coeff, [nbath])
        return c_bath, c_rest

    def log_histogram(self, r, name):
        if len(r) == 0:
            return
        self.log.info("%s R2-bath histogram:", name.capitalize())
        bins = np.linspace(0, 20, 20)
        self.log.info(helper.make_histogram(r, bins=bins, invertx=False))
