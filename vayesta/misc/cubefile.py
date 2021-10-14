"""Modified from PySCF - at the moment only for PBC systems"""
import logging
import os
import os.path

import numpy as np
import pickle

import pyscf
import pyscf.lib
import pyscf.gto
from pyscf.dft import numint

import vayesta

log = logging.getLogger(__name__)


class CubeFile:

    def __init__(self, cell, filename=None, gridsize=(100, 100, 100), resolution=None,
            title=None, comment=None, origin=(0.0, 0.0, 0.0), fmt="%13.5E",
            crop=None):
        """Initialize a cube file object. Data can be added using the `add_orbital` and `add_density` methods.

        This class can also be used as a context manager:

        >>> with CubeFile(cell, "mycubefile.cube") as f:
        >>>     f.add_orbital(hf.mo_coeff[:,0])
        >>>     f.add_orbital(hf.mo_coeff[:,5:10])
        >>>     f.add_density(hf.make_rdm1())

        Parameters
        ----------
            cell : pbc.gto.Cell object
            filename : str
                Filename for cube file. Should include ".cube" extension.
            nx, ny, nz : int, optional
                Number of grid points in x, y, and z direction. If specified,
                they take precedence over `resolution` parameter. Default: None.
            resolution : float, optional
                Resolution in units of 1/Bohr for automatic choice of `nx`,`ny`, and `nz`. Default: 15.0.
            title : str, optional
                First comment line in cube-file.
            comment : str, optional
                Second comment line in cube-file.
            origin : array(3), optional
                Origin in x, y, z coordinates.
            fmt : str, optional
                Float formatter for voxel data. According to the cube-file standard this is required to
                be "%13.5E", but some applications may support a different format. Default: "%13.5E"
            crop : dict, optional
                By default, the coordinate grid will span the entire unit cell. `crop` can be set
                to crop the unit cell. `crop` should be a dictionary with possible keys
                ["a0", "a1", "b0", "b1", "c0", "c1"], where "a0" crops the first lattice vector at the
                start, "a1" crops the first lattice vector at the end, "b0" crops the second lattice vector
                at the start etc. The corresponding values is the distance which should be cropped in
                units of Bohr.
                EXPERIMENTAL FEATURE - NOT FULLY TESTED.
        """

        self.cell = cell
        self.filename = filename
        # Make user aware of different behavior of resolution, compared to pyscf.tools.cubegen
        if resolution is not None and resolution < 1:
            log.warning(cell, "Warning: resolution is below 1/Bohr. Recommended values are 5/Bohr or higher.")
        self.a = self.cell.lattice_vectors().copy()
        self.origin = np.asarray(origin)
        if crop is not None:
            a = self.a.copy()
            norm = np.linalg.norm(self.a, axis=1)
            a[0] -= (crop.get('a0', 0) + crop.get('a1', 0)) * self.a[0]/norm[0]
            a[1] -= (crop.get('b0', 0) + crop.get('b1', 0)) * self.a[1]/norm[1]
            a[2] -= (crop.get('c0', 0) + crop.get('c1', 0)) * self.a[2]/norm[2]
            self.origin += crop.get('a0', 0)*self.a[0]/norm[0]
            self.origin += crop.get('b0', 0)*self.a[1]/norm[1]
            self.origin += crop.get('c0', 0)*self.a[2]/norm[2]
            self.a = a
        # Use resolution if provided, else gridsize
        if resolution is not None:
            self.nx = min(np.ceil(abs(self.a[0,0]) * resolution).astype(int), 192)
            self.ny = min(np.ceil(abs(self.a[1,1]) * resolution).astype(int), 192)
            self.nz = min(np.ceil(abs(self.a[2,2]) * resolution).astype(int), 192)
        else:
            self.nx, self.ny, self.nz = gridsize
        self.title = title or "<title>"
        self.comment = comment or ("Generated with Vayesta v%s" % vayesta.__version__)
        self.fmt = fmt
        self.coords = self.get_coords()

        self.fields = []

    def get_coords(self):
        xs = np.arange(self.nx) / (self.nx-1)
        ys = np.arange(self.ny) / (self.ny-1)
        zs = np.arange(self.nz) / (self.nz-1)
        coords = pyscf.lib.cartesian_prod([xs, ys, zs])
        coords = np.dot(coords, self.a)
        coords = np.asarray(coords, order='C') + self.origin
        return coords

    @property
    def ncoords(self):
        """Number of grod points."""
        return self.nx*self.ny*self.nz

    @property
    def nfields(self):
        """Number of datasets (orbitals + density matrices)."""
        return len(self.fields)

    def save_state(self, filename):
        cell, self.cell = self.cell, None       # Do not pickle cell
        pickle.dump(self, open(filename, 'wb'))
        self.cell = cell                        # Restore self.cell

    @classmethod
    def load_state(cls, filename, cell=None):
        self = pickle.load(open(filename, 'rb'))
        self.cell = cell
        return self

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.write()

    def add_orbital(self, coeff, dset_idx=None):
        """Add one or more orbitals to the cube file.

        Arguments
        ---------
        coeff : (N) or (N,M) array
            AO coefficients of orbitals. Supports adding a single orbitals,
            where `coeff` is a one-dimensional array, or multiple orbitals,
            in which case the second dimension of `coeff` labels the orbitals.
        dset_idx : int, optional
            Dataset index of orbital(s). In the application, the orbitals will
            be labelled as 'Orbital <dset_idx>' or similar. If set to `None`,
            the smallest unused, positive integer will be used. Default: None.
        """
        coeff = np.array(coeff) # Force copy
        if coeff.ndim == 1: coeff = coeff[:,np.newaxis]
        assert (coeff.ndim == 2)
        for i, c in enumerate(coeff.T):
            idx = dset_idx+i if dset_idx is not None else None
            self.fields.append((c, 'orbital', idx))

    def add_density(self, dm, dset_idx=None):
        """Add one or more densities to the cube file.

        Arguments
        ---------
        dm : (N,N) or (M,N,N) array
            Density-matrix in AO-representation. Supports adding a single density-matrix,
            where `dm` is a two-dimensional array, or multiple density-matrices,
            in which case the first dimension of `dm` labels the matrices.
        dset_idx : int, optional
            Dataset index of density-matrix. In the application, the density-matrix will
            be labelled as 'Orbital <dset_idx>' or similar. If set to `None`,
            the smallest unused, positive integer will be used. Default: None.
        """
        dm = np.array(dm)   # Force copy
        if dm.ndim == 2: dm = dm[np.newaxis]
        assert (dm.ndim == 3)
        for i, d in enumerate(dm):
            idx = dset_idx+i if dset_idx is not None else None
            self.fields.append((d, 'density', idx))

    def add_mep(self, dm, dset_idx=None):
        # TODO
        raise NotImplementedError()

    def write(self, filename=None):
        filename = (filename or self.filename)
        # Create directories if necessary
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Get dataset IDs
        dset_ids = []
        for (field, ftype, fid) in self.fields:
            if fid is None:
                if dset_ids:
                    fid = np.max(dset_ids)+1
                else:
                    fid = 1
            dset_ids.append(fid)

        self.write_header(filename, dset_ids=dset_ids)
        self.write_fields(filename)

    def write_header(self, filename, dset_ids=None):
        """Write header of cube-file."""
        if self.nfields > 1 and dset_ids is None:
            dset_ids = range(1, self.nfields+1)
        with open(filename, 'w') as f:
            f.write('%s\n' % self.title)
            f.write('%s\n' % self.comment)
            if self.nfields > 1:
                f.write('%5d' % -self.cell.natm)
            else:
                f.write('%5d' % self.cell.natm)
            f.write('%12.6f%12.6f%12.6f' % tuple(self.origin))
            if self.nfields > 1:
                f.write('%5d' % self.nfields)
            f.write('\n')
            # Lattice vectors
            f.write('%5d%12.6f%12.6f%12.6f\n' % (self.nx, *(self.a[0]/(self.nx-1))))
            f.write('%5d%12.6f%12.6f%12.6f\n' % (self.ny, *(self.a[1]/(self.ny-1))))
            f.write('%5d%12.6f%12.6f%12.6f\n' % (self.nz, *(self.a[2]/(self.nz-1))))
            # Atoms
            for atm in range(self.cell.natm):
                sym = self.cell.atom_symbol(atm)
                f.write('%5d%12.6f' % (pyscf.gto.charge(sym), 0.0))
                f.write('%12.6f%12.6f%12.6f\n' % tuple(self.cell.atom_coords()[atm]))
            # Data set indices
            if self.nfields > 1:
                f.write('%5d' % self.nfields)
                for i in range(self.nfields):
                    f.write('%5d' % dset_ids[i])
                f.write('\n')

    def write_fields(self, filename):
        """Write voxel data of registered fields in `self.fields` to cube-file."""
        blksize = min(self.ncoords, 8000)
        with open(filename, 'a') as f:
            # Loop over x,y,z coordinates first, then fields!
            for blk0, blk1 in pyscf.lib.prange(0, self.ncoords, blksize):
                data = np.zeros((blk1-blk0, self.nfields))
                blk = np.s_[blk0:blk1]
                ao = self.cell.eval_gto('PBCGTOval', self.coords[blk])
                for i, (field, ftype, _) in enumerate(self.fields):
                    if ftype == 'orbital':
                        data[:,i] = np.dot(ao, field)
                    elif ftype == 'density':
                        data[:,i] = numint.eval_rho(self.cell, ao, field)
                    else:
                        raise ValueError('Unknown field type: %s' % ftype)
                data = data.flatten()
                for d0, d1 in pyscf.lib.prange(0, len(data), 6):
                    f.write(((d1-d0)*self.fmt + '\n') % tuple(data[d0:d1]))

if __name__ == '__main__':

    def make_graphene(a, c, atoms=["C", "C"], supercell=None):
        amat = np.asarray([
                [a, 0, 0],
                [a/2, a*np.sqrt(3.0)/2, 0],
                [0, 0, c]])
        coords_internal = np.asarray([
            [2.0, 2.0, 3.0],
            [4.0, 4.0, 3.0]])/6
        coords = np.dot(coords_internal, amat)

        if supercell is None:
            atom = [(atoms[0], coords[0]), (atoms[1], coords[1])]
        else:
            atom = []
            ncopy = supercell
            nr = 0
            for x in range(ncopy[0]):
                for y in range(ncopy[1]):
                    for z in range(ncopy[2]):
                        shift = x*amat[0] + y*amat[1] + z*amat[2]
                        atom.append((atoms[0]+str(nr), coords[0]+shift))
                        atom.append((atoms[1]+str(nr), coords[1]+shift))
                        nr += 1
            amat = np.einsum('i,ij->ij', ncopy, amat)
        return amat, atom


    from pyscf import pbc
    cell = pbc.gto.Cell(
        basis = 'gth-dzv',
        pseudo = 'gth-pade',
        dimension = 2,
        verbose=10)
    cell.a, cell.atom = make_graphene(2.46, 10.0, supercell=(2,2,1))
    hf = pbc.scf.HF(cell)
    hf = hf.density_fit()
    hf.kernel()

    with CubeFile(cell, "graphene.cube", crop={"c0" : 5.0, "c1" : 5.0}) as f:
        f.add_orbital(hf.mo_coeff[:,0])
        f.add_orbital(hf.mo_coeff[:,6:10])
        f.add_density([hf.make_rdm1(), np.linalg.inv(hf.get_ovlp())-hf.make_rdm1()])
