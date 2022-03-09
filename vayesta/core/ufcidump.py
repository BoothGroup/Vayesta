#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
FCIDUMP functions (write, read) for real Hamiltonian
'''

import re
from functools import reduce
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import symm
from pyscf import __config__

DEFAULT_FLOAT_FORMAT = getattr(__config__, 'fcidump_float_format', ' %.16g')
TOL = getattr(__config__, 'fcidump_write_tol', 1e-15)

MOLPRO_ORBSYM = getattr(__config__, 'fcidump_molpro_orbsym', False)

# Mapping Pyscf symmetry numbering to Molpro symmetry numbering for each irrep.
# See also pyscf.symm.param.IRREP_ID_TABLE
# https://www.molpro.net/info/current/doc/manual/node36.html
ORBSYM_MAP = {
    'D2h': (1,         # Ag
            4,         # B1g
            6,         # B2g
            7,         # B3g
            8,         # Au
            5,         # B1u
            3,         # B2u
            2),        # B3u
    'C2v': (1,         # A1
            4,         # A2
            2,         # B1
            3),        # B2
    'C2h': (1,         # Ag
            4,         # Bg
            2,         # Au
            3),        # Bu
    'D2' : (1,         # A
            4,         # B1
            3,         # B2
            2),        # B3
    'Cs' : (1,         # A'
            2),        # A"
    'C2' : (1,         # A
            2),        # B
    'Ci' : (1,         # Ag
            2),        # Au
    'C1' : (1,)
}

def write_head(fout, nmo, nelec, ms=0, orbsym=None):
    if not isinstance(nelec, (int, numpy.number)):
        ms = abs(nelec[0] - nelec[1])
        nelec = nelec[0] + nelec[1]
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nmo, nelec, ms))
    if orbsym is not None and len(orbsym) > 0:
        fout.write('  ORBSYM=%s\n' % ','.join([str(x) for x in orbsym]))
    else:
        fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  UHF=.TRUE.,\n')
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')


def write_eri_spat(fout, eri, nmo, offset1, offset2, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    npair = nmo*(nmo+1)//2
    output_format = float_format + ' %4d %4d %4d %4d\n'
    if eri.size == nmo**4:
        eri = ao2mo.restore(8, eri, nmo)

    if eri.ndim == 2: # 4-fold symmetry
        assert(eri.size == npair**2)
        ij = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, nmo):
                    for l in range(0, k+1):
                        if abs(eri[ij,kl]) > tol:
                            fout.write(output_format % (eri[ij,kl], 2*i+1+offset1, 2*j+1+offset1, 2*k+1+offset2, 2*l+1+offset2))
                        kl += 1
                ij += 1
    else:  # 8-fold symmetry
        assert(eri.size == npair*(npair+1)//2)
        ij = 0
        ijkl = 0
        for i in range(nmo):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, i+1):
                    for l in range(0, k+1):
                        if ij >= kl:
                            if abs(eri[ijkl]) > tol:
                                fout.write(output_format % (eri[ijkl], 2*i+1+offset1, 2*j+1+offset1, 2*k+1+offset2, 2*l+1+offset2))
                            ijkl += 1
                        kl += 1
                ij += 1

def write_hcore_spat(fout, h, nmo, offset, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    h = h.reshape(nmo,nmo)
    output_format = float_format + ' %4d %4d  0  0\n'
    for i in range(nmo):
        for j in range(0, i+1):
            if abs(h[i,j]) > tol:
                fout.write(output_format % (h[i,j], 2*i+1+offset, 2*j+1+offset))

def from_integrals(filename, h1e, h2e, nmo, nelec, nuc=0, ms=0, orbsym=None,
                   tol=TOL, float_format=DEFAULT_FLOAT_FORMAT):
    '''Convert the given 1-electron and 2-electron integrals to FCIDUMP format'''
    if not isinstance(nmo, (int, numpy.number)):
        assert(nmo[0] == nmo[1])
        nmo = nmo[0]
    with open(filename, 'w') as fout:
        write_head(fout, nmo, nelec, ms, orbsym)
        write_eri_spat(fout, h2e[0], nmo, 0, 0, tol=tol, float_format=float_format)
        write_eri_spat(fout, h2e[1], nmo, 0, 1, tol=tol, float_format=float_format)
        write_eri_spat(fout, h2e[2], nmo, 1, 1, tol=tol, float_format=float_format)

        write_hcore_spat(fout, h1e[0], nmo, 0, tol=tol, float_format=float_format)
        write_hcore_spat(fout, h1e[1], nmo, 1, tol=tol, float_format=float_format)
        output_format = float_format + '  0  0  0  0\n'
        fout.write(output_format % nuc)

def from_scf(mf, filename, tol=TOL, float_format=DEFAULT_FLOAT_FORMAT,
             molpro_orbsym=MOLPRO_ORBSYM):
    '''Use the given SCF object to transfrom the 1-electron and 2-electron
    integrals then dump them to FCIDUMP.

    Kwargs:
        molpro_orbsym (bool): Whether to dump the orbsym in Molpro orbsym
            convention as documented in
            https://www.molpro.net/info/current/doc/manual/node36.html
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert mo_coeff.dtype == numpy.double

    h1e = tuple([reduce(numpy.dot, (mo_coeff[i].T, mf.get_hcore()[i], mo_coeff[i])) for i in [0,1]])
    if mf._eri is None:
        if getattr(mf, 'exxdiv', None):  # PBC system
            eri = mf.with_df.ao2mo(mo_coeff)
        else:
            eri = ao2mo.full(mf.mol, mo_coeff)
    else:  # Handle cached integrals or customized systems
        eri = ao2mo.full(mf._eri, mo_coeff)
    orbsym = getattr(mo_coeff, 'orbsym', None)
    if molpro_orbsym and orbsym is not None:
        orbsym = [ORBSYM_MAP[mol.groupname][i] for i in orbsym]
    nuc = mf.energy_nuc()
    from_integrals(filename, h1e, eri, h1e[0].shape[0], mf.mol.nelec, nuc, 0, orbsym,
                   tol, float_format)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert chkfile to FCIDUMP')
    parser.add_argument('chkfile', help='pyscf chkfile')
    parser.add_argument('fcidump', help='FCIDUMP file')
    args = parser.parse_args()

    # fcidump.py chkfile output
    #from_chkfile(args.fcidump, args.chkfile)
