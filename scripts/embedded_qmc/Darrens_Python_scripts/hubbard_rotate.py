'''
robert.anderson@kcl.ac.uk

produces FCIDUMP files for 2D hubbard models, optionally with an arbitrary rotation of the basis
'''

import numpy
from itertools import product
from pyscf import tools, ao2mo, fci
from functools import reduce

def make_integrals(nx, ny, u, pbc=False):
    '''
    make the integral arrays for an arbitrary rectangular 2D hubbard model in the site basis

    Parameters:
    nx (int): number of sites in x direction
    ny (int): number of sites in y direction
    u (float): onsite repulsion as a multiple of the hopping strength
    pbc (bool): whether to build a hamiltonian with periodic boundary conditions

    Returns:
    h1e: 1-electron integral array
    eri: 2-electron integral array with 1-fold (no) symmetry
    '''
    
    '''
    need to map the 2d site space into a single integer index
    '''
    assert nx>0 and ny>0
    nsite = nx * ny

    def flat(ix, iy): return ix*ny+iy
    def left(ix): return None if (nx==1 or ix==0 and not pbc) else (ix-1)%nx
    def right(ix): return None if (nx==1 or ix==nx-1 and not pbc) else (ix+1)%nx
    def above(iy): return None if (ny==1 or iy==0 and not pbc) else (iy-1)%ny
    def below(iy): return None if (ny==1 or iy==ny-1 and not pbc) else (iy+1)%ny

    h1e = numpy.zeros((nsite, nsite))
    def set_h1e(ix, iy, jx, jy):
        i = flat(ix, iy)
        j = flat(jx, jy)
        assert i != j, 'there should be no on-site "hopping"!'
        h1e[i, j] = -1.0

    '''
    the interactions have the following stencil
     0  -t   0
    -t   0  -t
     0  -t   0
    '''
    for ix, iy in product(range(nx), range(ny)):
        if left(ix) is not None: set_h1e(ix, iy, left(ix), iy)
        if right(ix) is not None: set_h1e(ix, iy, right(ix), iy)
        if above(iy) is not None: set_h1e(ix, iy, ix, above(iy))
        if below(iy) is not None: set_h1e(ix, iy, ix, below(iy))

    eri = numpy.zeros((nsite,nsite,nsite,nsite))

    for i in range(nsite):
        eri[i,i,i,i] = u

    return h1e, eri

def transform(h1e, eri, mo_coeff):
    '''
    rotate the integrals to a different basis

    Parameters:
    h1e (2d array): 1-electron integral array
    eri (1, 2, or 4d array): 2-electron integral array
    mo_coeff (2d array): transformation coefficients

    Returns:
    h1e: 1-electron integral array
    eri: 2-electron integral array with 8-fold symmetry
    '''
    h1e = reduce(numpy.dot, (mo_coeff.T, h1e, mo_coeff))
    eri = ao2mo.kernel(ao2mo.restore(8, eri, mo_coeff.shape[0]),
            mo_coeff, verbose=0, compact=False)
    return h1e, eri


def write_fcidump(h1e, eri, nelec, fname='FCIDUMP'):
    '''
    writes the provided integrals to a file in a standard format for FCI programs
    '''
    nsite = h1e.shape[0]
    if len(eri.shape)!=1:
        # ERIs must have 8-fold symmetry restored
        eri = ao2mo.restore(8, eri, nsite)
    tools.fcidump.from_integrals(fname, h1e, eri, nsite, nelec, 0, 0, [1,]*nsite)

def get_fci(h1e, eri, nelec):
    '''
    compute the FCI energy and wavefunction exactly
    '''
    nsite = h1e.shape[0]
    return fci.direct_spin1.kernel(h1e, eri, nsite, nelec, verbose=6)

def get_fci_energy(h1e, eri, nelec):
    '''
    compute the FCI energy exactly
    '''
    return get_fci(h1e, eri, nelec)[0]
