from pyscf import gto, scf, solvent
from vayesta.eagf2 import UAGF2
from vayesta import log, vlog
import numpy as np
import sys
import os

#system = 'ttm_d0.xyz'
#basis = '6-31g**'
system = sys.argv[1]
basis = sys.argv[2]

# Init input and output files:
proj_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(proj_dir, 'data')
input_file = os.path.join(proj_dir, system)
ints_file = os.path.join(data_dir, system.replace('.xyz', '_%s.ints' % basis))
chk_file = os.path.join(data_dir, system.replace('.xyz', '_%s.chk' % basis))
os.system('mkdir -p %s' % data_dir)

# Define molecule:
mol = gto.Mole()
mol.atom = input_file
mol.basis = basis
mol.spin = 1
mol.max_memory = 1e9
mol.verbose = 0
#mol.output = "%s_%s.out" % (system.replace(".xyz", ""), basis)
mol.build()

#print('Size:')
#print('nao   = %d' % mol.nao)
#print('nalph = %d' % mol.nelec[0])
#print('nbeta = %d' % mol.nelec[1])

# Init ROHF:
mf = scf.ROHF(mol)
mf.conv_tol = 1e-11

# Add solvent model:
mf = solvent.DDCOSMO(mf)
mf.with_solvent.eps = 4.7113  # chloroform

# Add density fitting:
mf = mf.density_fit()
if os.path.isfile(ints_file):
    mf.with_df._cderi = ints_file
    mf.with_df.build()
else:
    mf.with_df._cderi_to_save = ints_file
    mf.with_df.build()
with_df = mf.with_df

# Add checkpoint file:
mf.chkfile = chk_file
if os.path.isfile(chk_file):
    stability_check = False
    mf.__dict__.update(scf.chkfile.load(chk_file, 'scf'))
else:
    mf.kernel()

    # Check MF stability:
    internal = mf.stability()[0]
    if not np.allclose(mf.mo_coeff, internal):
        dm = mf.make_rdm1(mo_coeff=internal)
        mf.kernel(dm0=dm)

# Convert to UHF for AGF2 solver:
mf = scf.addons.convert_to_uhf(mf)
mf.with_df = with_df

#ip_α = -mf.mo_energy[0][mf.mo_occ[0] > 0].max()
#ea_α = mf.mo_energy[0][mf.mo_occ[0] == 0].min()
#ip_β = -mf.mo_energy[1][mf.mo_occ[1] > 0].max()
#ea_β = mf.mo_energy[1][mf.mo_occ[1] == 0].min()
#print('UHF:')
#print('E(tot) = %20.12f' % mf.e_tot)
#print('α:')
#print('E(ip)  = %20.12f' % ip_α)
#print('E(ea)  = %20.12f' % ea_α)
#print('E(gap) = %20.12f' % (ip_α+ea_α))
#print('β:')
#print('E(ip)  = %20.12f' % ip_β)
#print('E(ea)  = %20.12f' % ea_β)
#print('E(gap) = %20.12f' % (ip_β+ea_β))
#print('<S^2>  = %20.12f' % mf.spin_square()[0])

# Run AGF2:
log.handlers.clear()
fmt = vlog.VFormatter(indent=True)
log.addHandler(vlog.VFileHandler("%s_%s.out" % (system.replace(".xyz", ""), basis), formatter=fmt))
gf2 = UAGF2(
        mf,
        conv_tol=1e-5,
        conv_tol_rdm1=1e-10,
        conv_tol_nelec=1e-10,
        conv_tol_nelec_factor=1e-3,
        max_cycle=30,
        max_cycle_outer=5,
        max_cycle_inner=100,
        weight_tol=1e-14,
        damping=0.25,
)
gf2.kernel()
