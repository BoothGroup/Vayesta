# PySCF
from pyscf import lib
from pyscf.agf2 import mpi_helper
from pyscf.pbc import gto, scf, dft, gw
from pyscf.pbc.gw.krgw_ac import KRGWAC

# NumPy
import numpy as np

# Vayesta
from vayesta.misc.gdf import GDF
from vayesta import log, vlog

# Standard library
import os
import sys

# Test set
from gmtkn import sets
systems = sets['GAPS'].systems
keys = sorted(systems.keys())


nk = [int(sys.argv[1])] * 3
nao = (0, 32)
skip = ["LiH", "Kr", "Ne", "Ar", "Xe"]
skip_atoms = ["Pb", "Te"]
exp_to_discard = 0.0
precision = 1e-9
exxdiv = None
basis = 'gth-dzvp-molopt-sr'
pseudo = 'gth-pade'
xc = sys.argv[2]
method_name = 'krg0w0-%s' % xc
method = KRGWAC

log.handlers.clear()
fmt = vlog.VFormatter(indent=True)

for key in keys:
    if key in skip or any([a in key for a in skip_atoms]):
        continue
    mpi_helper.barrier()

    try:
        cell = gto.Cell()
        cell.atom = list(zip(systems[key]['atoms'], systems[key]['coords']))
        cell.a = systems[key]['a']
        cell.basis = basis
        cell.pseudo = pseudo
        cell.exp_to_discard = exp_to_discard
        cell.precision = precision
        cell.max_memory = 1e9
        cell.verbose = 0
        cell.build()
    except Exception as e:
        sys.stderr.write('Error in cell: %s\n' % e)
        sys.stderr.flush()
        continue

    if cell.nao < nao[0] or cell.nao >= nao[1] or cell.nelec[0] != cell.nelec[1]:
        continue

    log.handlers.clear()
    log.addHandler(vlog.VFileHandler('%s_%s_%s_%s%s%s.out' % (method_name, key, basis, *nk), formatter=fmt))
    if mpi_helper.rank != 0:
        log.setLevel(50)
    mpi_helper.barrier()

    mf = dft.KRKS(cell, xc=xc)
    mf.kpts = cell.make_kpts(nk)
    mf.with_df = GDF(cell, mf.kpts)
    mf.with_df.build()
    mf.exxdiv = exxdiv
    mf.chkfile = '%s_%s_%s_%s%s%s.chk' % (method_name, key, basis, *nk) if mpi_helper.rank == 0 else None
    mf.kernel()

    if mpi_helper.rank == 0:
        lib.chkfile.dump(mf.chkfile, 'scf/converged', mf.converged)
    mpi_helper.barrier()

    for k in range(len(mf.kpts)):
        mf.mo_energy[k] = mpi_helper.bcast(mf.mo_energy[k])
        mf.mo_coeff[k] = mpi_helper.bcast(mf.mo_coeff[k])
        mf.mo_occ[k] = mpi_helper.bcast(mf.mo_occ[k])
        mpi_helper.barrier()
    for k1 in range(len(mf.kpts)):
        for k2 in range(len(mf.kpts)):
            mf.with_df._cderi[k1, k2] = mpi_helper.bcast(mf.with_df._cderi[k1, k2])
            mpi_helper.barrier()
    mf.e_tot = mpi_helper.bcast(mf.e_tot)
    mpi_helper.barrier()

    try:
        kgw = KRGWAC(mf)
        kgw.ac = 'pade'
        kgw.linearized = False
        kgw.fc = exxdiv is not None
        start = min(kgw.nocc-3, 0)
        end = max(kgw.nocc+3, kgw.nmo)
        kgw.kernel(kptlist=[0], orbs=range(start, end))
        if mpi_helper.rank == 0:
            lib.chkfile.dump(mf.chkfile, 'kgw/mo_energy', kgw.mo_energy[0])
            lib.chkfile.dump(mf.chkfile, 'kgw/mo_coeff', kgw.mo_coeff[0])
            lib.chkfile.dump(mf.chkfile, 'kgw/mo_occ', kgw.mo_occ[0])
            lib.chkfile.dump(mf.chkfile, 'kgw/converged', kgw.converged)
    except Exception as e:
        sys.stderr.write('Error in GW: %s\n' % e)
        sys.stderr.flush()
        continue
