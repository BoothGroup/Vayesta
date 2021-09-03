import numpy as np
from numpy import einsum
import numpy as np
from numpy import einsum
import vayesta
import vayesta.ewf
import vayesta.lattmod
import pyscf.tools
import pyscf.tools.ring
from functools import reduce
import numpy
from pyscf import gto, scf, ao2mo, fci

import vayesta
import vayesta.ewf
import vayesta.lattmod

from pyscf import lib

from scmf import SelfConsistentMF

# Hydrogen chain calculation

nsite = 12 # 10 atom H chain
nimp = 4# Size of atomic fragments
doping =0
boundary_cond = ('PBC')
basis='STO-6G'

R_min = 0.6 # Minimum symmetric stretching [A]
R_max = 2.2# Maximum symmetric stretching [A]
R_step = 0.05 # Resolution


R_range = np.arange(R_min, R_max+R_step, R_step)

Ridx = 0
for R in R_range:
    mol = pyscf.gto.Mole()
    atom = pyscf.tools.ring.make(nsite, R)
    atom = ['H %f %f %f' % xyz for xyz in atom]
    mol.atom = atom
    mol.basis = basis
    mol.output = 'pyscf_out.txt'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    stable = False
    while not stable:
        mo1 = mf.stability()[0]
        stable = (mo1 is mf.mo_coeff)
        if stable:
            print("HF stable!")
            break
        print("HF unstable...")
        dm1 = mf.make_rdm1(mo_coeff=mo1)
        mf.kernel(dm0=dm1)
    
    # Run Brueckner self-consistency
    print('Symmetric Hydrogen Ring stretching')
    print('----------------------------------')
    print('Radius [A]', R)
    print('HF energy [Ha]', mf.e_tot)
    print('HF 1-body DM main diag.')
    print(np.diag(mf.make_rdm1()))
    print('HF 1-body DM 1st diag.')
    print(np.diag(mf.make_rdm1(), k=1))
    
    pdmet_solver = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=np.inf, fragment_type='Lowdin-AO', make_rdm1=True, make_rdm2=True, nelectron_target=nimp)


    for s in range(0, nsite, nimp):
        f=pdmet_solver.make_atom_fragment(list(range(s, s+nimp)))
    pdmet_solver.kernel()
    assert len(pdmet_solver.fragments) == nsite//nimp
    
    
    pdmet_solver = pdmet_solver.pdmet_scmf()
    pdmet_solver.kernel()
    
    pdmet_converged = False
    pdmet_ewf = np.nan
    breuck_dmet = np.nan
    if (pdmet_solver.with_scmf.converged):
        pdmet_ewf = pdmet_solver.get_e_tot() / mol.nelectron
        pdmet_dmet = pdmet_solver.get_dmet_energy() / mol.nelectron
        pdmet_converged = True
    else:
        print('SCMF not converged')

    docc = np.zeros((nsite,))
    for f, s in enumerate(range(0, nsite, nimp)):
        imp = pdmet_solver.fragments[f].c_active[s:s+nimp]
        d = einsum("ijkl,xi,xj,xk,xl->x", pdmet_solver.fragments[f].results.dm2, *(4*[imp]))/2
        docc[s:s+nimp] = d

    with open('H-energies-amp-pdmet.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "pdmet"))
        f.write('%12.3f  %16.10f  \n' % (R, pdmet_ewf))
    with open('H-energies-dm-pdmet.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "pdmet"))
        f.write('%12.3f  %16.10f  \n' % (R, pdmet_dmet))

    s = nimp // 2
    with open('H-docc-pdmet.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "pdmet"))
        f.write('%12.3f  %16.10f \n' % (R, docc[s]))
    
    with open('H-convergence-pdmet.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "pdmet"))
        f.write('%12.3f %16.10s %d \n' % (R,  str(pdmet_converged), 1))

    Ridx += 1
