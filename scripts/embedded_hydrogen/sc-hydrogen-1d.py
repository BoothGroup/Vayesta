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

from scmf import SelfConsistentMF

import matplotlib.pyplot as plt
params = {
   'axes.labelsize': 40,
   'font.size': 40,
   'legend.fontsize': 40,
   'lines.linewidth' : 2,
   'lines.markersize' : 23,
   'xtick.labelsize': 40,
   'ytick.labelsize': 40,
   'figure.figsize': [40, 20]
   }
plt.rcParams.update(params)

def get_energy(mf, frags):
    # DMET energy
    e = np.asarray([get_e_dmet(mf, f) for f in frags])
    with open('Fragment_energies.txt', 'a') as f:
        f.write(str(mf.mol.atom)+'\n'+str(e)+'\n')
    print(e)
    assert abs(e.min()-e.max()) < 1e-6
    e = np.sum(e)
    return e

def get_e_dmet(mf,f):
    """DMET (projected DM) energy of fragment f."""
    c = f.c_active
    p = f.get_fragment_projector(c)
    #print(c.shape)
    #print(p.shape)
    #print(f.results.dm1.shape)
    #print(f.results.dm2.shape)
    #print(mf._eri.shape)
    
    n = mf.mol.nelectron
    eri = ao2mo.addons.restore('s1', mf._eri, n).reshape(n, n, n, n)
    print(eri.shape)
    pdm1 = np.einsum('ix,xj,ai,bj->ab', p, f.results.dm1, c, c, optimize=True)
    pdm2 = np.einsum('ix,xjkl,ai,bj,ck,dl->abcd', p, f.results.dm2, c, c, c, c, optimize=True)
    e1 = np.einsum('ij,ij->', mf.get_hcore(), pdm1, optimize=True)
    e2 = np.einsum('ijkl,ijkl->', eri, pdm2,optimize=True)/2
    return e1 + e2

# Hydrogen chain calculation

nsite = 10 # 10 atom H chain
nimp = 2# Size of atomic fragments

R_min = 0.6 # Minimum symmetric stretching [A]
R_max = 2.2 # Maximum symmetric stretching [A]
R_step = 0.05 # Resolution


R_range = np.arange(R_min, R_max+R_step, R_step)

mo_brueck = None
mo_pdmet = None

iteration = 0
for R in R_range:
    
    e_dmet = 0.0
    e_pdmet = 0.0
    e_oneshot = 0.0
    e_brueck = 0.0

    ring = pyscf.tools.ring.make(nsite, R)
    #print(ring)
    atom_config = [('H %f %f %f') % xyz for xyz in ring]

    # Create atomic chain configuration along x axis
    #for aidx, atom in enumerate(range(nsite)):
    #    atom_config.append(['H', (R*(aidx+1), 0, 0)])
    ##print(['H', (R*(aidx+1), 0, 0)])
    #print(atom_config)

    # Use minimal basis set (single 1s orbital per atom)
    mol = gto.Mole(atom=atom_config, basis='STO-6G', verbose=3, output='pyscf.out').build()
    # Run Hartree-Fock mean-field calculation

    mf = scf.RHF(mol)
    mf.conv_tol = 1.e-13

    mf.run(verbose=1)

    '''
    c = mf.mo_coeff
    stable = False
    print('Starting stability loop')
    it = 0
    while not stable and it <= 10:
        #print(mf.stability())
        c1 = mf.stability(verbose=1)[1]
        stable = (c1 is mf.mo_coeff)
        print('Loop completed')
        it += 1
        if stable:
            break
    dm = mf.make_rdm1(mo_coeff=c1)
    mf.kernel(dm=dm)
    '''
    c = mf.mo_coeff
    
    # Run PySCF FCI calculation
    '''
    h1e = reduce(numpy.dot, (c.T, mf.get_hcore(), c))
    eri = ao2mo.incore.full(mf._eri, c)
    e_fci, civec = fci.direct_spin0.kernel(h1e, eri, c.shape[1], mol.nelectron,tol=1e-14, lindep=1e-15, max_cycle=100)
    '''
    fci_solver = fci.FCI(mol, mo=c)
    e_fci, civec = fci_solver.kernel()
    
    '''
    stable = False
    while not stable:
        c1 = mf.stability()[0]
        stable = (c1 is mf.mo_coeff)
        if stable:
            break
    dm = mf.make_rdm1(mo_coeff=c1)
    mf.kernel(dm=dm)
    '''
    
    
    # Run embedded wavefunction calculations
    # bno_threshold MUST BE np.inf, otherwise additional bath orbitals are included (ie. we run an FCI calculation)
    oneshot_solver = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=np.inf, fragment_type='Lowdin-AO', make_rdm1=True, make_rdm2=True, sc_maxiter=1)
    #fragment = oneshot_solver.make_atom_fragment(list(range(nimp)))
    #oneshot_solver.kernel()
    #symfrags = fragment.make_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
    #assert len(symfrags) + 1 == ncells//nimp
    for s in range(0, nsite, nimp):
        f = oneshot_solver.make_atom_fragment(list(range(s, s+nimp)))
    oneshot_solver.kernel()
    assert len(oneshot_solver.fragments) == nsite//nimp
    for f in oneshot_solver.fragments:
        assert f.results.converged
    

    oneshot_dmet = get_energy(mf, oneshot_solver.fragments) + mol.energy_nuc()
        
    # Run v-DMET (correlation potential) calculation
        
    fci_dmet = vayesta.dmet.DMET(mf, fragment_type='Lowdin-AO', solver='FCI', make_rdm1=True, make_rdm2=True, charge_consistent=False)
    for s in range(0, nsite, nimp):
        f = fci_dmet.make_atom_fragment(list(range(s, s+nimp)))
    #f_dmet = fci_dmet.make_atom_fragment(list(range(nimp)))
    #frags_dmet = f_dmet.make_tsymmetric_fragments(tvecs=tvecs)
    '''
    if (uidx!=0):
        fci_dmet.vcorr = vcorr_prev
    '''
    
    fci_dmet.kernel()
    assert (len(fci_dmet.fragments) == nsite//nimp)
    for f in fci_dmet.fragments:
        assert f.results.converged
        
    dmet_e_tot = fci_dmet.e_tot
    dmet_docc = np.zeros((nsite,))
    for f, s in enumerate(range(0, nsite, nimp)):
        imp = fci_dmet.fragments[f].c_active[s:s+nimp]
        d = einsum("ijkl,xi,xj,xk,xl->x", fci_dmet.fragments[f].results.dm2, *(4*[imp]))/2
        dmet_docc[s:s+nimp] = d
    dmet_converged = fci_dmet.converged
    
        # Run Brueckner self-consistency
    
    sc_brueck = SelfConsistentMF(mf, sc_type='brueckner', tvecs=None, ab_initio=True)
    sc_brueck.kernel(nimp=nimp, mo_coeff=mo_brueck)
    mo_brueck = sc_brueck.mf.mo_coeff
    sc_brueck = SelfConsistentMF(mf, sc_type='pdmet', tvecs=None, ab_initio=True)
    sc_brueck.kernel(nimp=nimp, mo_coeff=mo_pdmet)
    mo_brueck = sc_pdmet.mf.mo_coeff
    
    print('Hydrogen ring bond stretching')
    print('No. electrons ', mol.nelectron)
    print('-----------------------------')
    print('H Ring stretching [A]', R)
    #print(mol.nelectron)
    print('HF energy per e- [Hartrees] ', mf.e_tot/mol.nelectron)
    #E_FCI.append(e_fci/mol.nelectron)
    print('FCI energy per e- [Hartrees]', e_fci/mol.nelectron)
    #e_oneshot = get_energy(mf, oneshot_solver.fragments)'
    


    print('Oneshot EWF AMP energy per e- [Hartrees]', oneshot_solver.e_tot/mol.nelectron)
    #print('Oneshot EWF RDM energy per e- [Hartrees]', oneshot_solver.e_tot/mol.nelectron)
    print('Oneshot EWF RDM energy per e- [Hartrees]', oneshot_dmet/mol.nelectron)
    #print('Oneshot EWF RDM energy per e- [Hartrees]', oneshot_solver.e_tot/mol.nelectron)
    print('Brueckner EWF AMP energy per e- [Hartrees]', sc_brueck.e_dmet)
    print('Brueckner EWF RDM energy per e- [Hartrees]', sc_brueck.e_ewf)
    
    print('DMET RDM energy per e- [Hartrees]', fci_dmet.e_tot/mol.nelectron)
    #print('Oneshot EWF RDM energy per e- [Hartrees]', oneshot_solver.e_tot/mol.nelectron)
    
    
    with open('H-energies-rdm.txt', 'a') as f:
        if (iteration == 0):
            f.write('# %s %s\n' % ("nsite", "nimp"))
            f.write('# %1d %1d \n' % (nsite, nimp))
            f.write('# %s %s %s %s %s %s %s\n' % ("Bond", "HF", "FCI", "Oneshot", "P-DMET", "Brueckner", "DMET-RDM proj."))
        
        f.write('%1.15f %1.15f %1.15f %1.15f %1.15f %1.15f %1.15f \n' % (R, mf.e_tot/mol.nelectron, e_fci/mol.nelectron, oneshot_dmet/mol.nelectron, -0.0001, sc_brueck.e_dmet, dmet_e_tot/mol.nelectron))
    with open('H-energies-ewf.txt', 'a') as f:
        if (iteration == 0):
            f.write('# %s %s\n' % ("nsite", "nimp"))
            f.write('# %1d %1d \n' % (nsite, nimp))
            f.write('# %s %s %s %s %s %s %s\n' % ("Bond", "HF", "FCI", "Oneshot", "P-DMET", "Brueckner", "DMET-RDM proj."))
        
        f.write('%1.15f %1.15f %1.15f %1.15f %1.15f %1.15f %1.15f \n' % (R, mf.e_tot/mol.nelectron, e_fci/mol.nelectron, oneshot_solver.e_tot/mol.nelectron, -0.0001, sc_brueck.e_ewf, fci_dmet.e_tot/mol.nelectron))
            
    iteration += 1
    
