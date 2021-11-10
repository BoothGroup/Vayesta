# PySCF
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


nk = [3, 3, 3]
nao = (0, 32)
exp_to_discard = 0.0
precision = 1e-9
exxdiv = None
basis = 'gth-dzvp-molopt-sr'
pseudo = 'gth-pade'
xc = sys.argv[1]
method_name = 'krg0w0-%s' % xc
method = KRGWAC

log.handlers.clear()
fmt = vlog.VFormatter(indent=True)

for key in keys:
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
        print(key, e)
        continue

    if cell.nao < nao[0] or cell.nao >= nao[1] or cell.nelec[0] != cell.nelec[1]:
        continue

    log.handlers.clear()
    try:
        os.remove('%s_%s_%s_%s%s%s.out' % (method_name, key, basis, *nk))
    except:
        pass
    log.addHandler(vlog.VFileHandler('%s_%s_%s_%s%s%s.out' % (method_name, key, basis, *nk), formatter=fmt))

    mf = dft.KRKS(cell)
    mf.kpts = cell.make_kpts(nk)
    mf.with_df = GDF(cell, mf.kpts)
    mf.with_df.build()
    mf.exxdiv = exxdiv
    mf.chkfile = '%s_%s_%s_%s%s%s.chk' % (method_name, key, basis, *nk)

    if xc == 'hf':
        hf = scf.KRHF(cell)
        hf.kpts = cell.make_kpts(nk)
        hf.with_df = mf.with_df
        hf.exxdiv = exxdiv
        hf.kernel()
        mf.__dict__.update({key: getattr(mf, key) for key in ['e_tot', 'mo_energy', 'mo_coeff', 'mo_occ', 'converged']})
        mf.dump_chk(mf.__dict__)

    else:
        mf.kernel()

    try:
        kgw = KRGWAC(cell)
        kgw.ac = 'pade'
        kgw.linearized = False
        kgw.fc = exxdiv is not None
        start = min(gw.nocc-3, 0)
        end = max(gw.nocc+3, gw.nmo)
        gw.kernel(kptlist=[0], orbs=range(start, end))
        chkfile.dump(mf.chkfile, 'gw/mo_energy', gw.mo_energy[0])
        chkfile.dump(mf.chkfile, 'gw/mo_coeff', gw.mo_coeff[0])
        chkfile.dump(mf.chkfile, 'gw/mo_occ', gw.mo_occ[0])
    except Exception as e:
        print(key, e)
        continue
