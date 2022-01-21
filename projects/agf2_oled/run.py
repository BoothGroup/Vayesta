from pyscf import gto, scf, solvent, agf2
import numpy as np
import os

system = 'ttm_d0.xyz'
basis = '6-31g**'

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
mol.verbose = 9
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
gf2 = agf2.dfuagf2.DFUAGF2(mf)
gf2.conv_tol = 1e-5
gf2.conv_tol_rdm1 = 1e-10
gf2.conv_tol_nelec = 1e-10
gf2.max_cycle = 30
gf2.max_cycle_outer = 5
gf2.max_cycle_inner = 100
gf2.weight_tol = 1e-14
gf2.damping = 0.25
gf2.kernel()

