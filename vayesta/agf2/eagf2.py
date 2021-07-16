import logging
import collections

import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.agf2
import pyscf.ao2mo

import vayesta
import vayesta.ewf
import vayesta.lattmod
from vayesta.agf2 import ragf2, ewdmet_bath

try:
    import dyson
except ImportError:
    pass

fakelog = logging.getLogger('fake')
fakelog.setLevel(logging.CRITICAL)

log = vayesta.log

# --- Settings
hubbard = 0             # if True, use hubbard model
ewdmet = 0              # if False, use DMET + MP2
bno_threshold = 1e-6    # for DMET + MP2
nmom_ewdmet = 8         # for EwDMET
fock_loop = 0           # do Fock loop on combined SE
nmom_projection = 0     # number of moments for projection, >0 requires dyson
democratic = 1          # democratic partitioning

# --- System
if hubbard:
    mol = vayesta.lattmod.Hubbard1D(
            nsite=20,
            nelectron=20,
            hubbard_u=2.0,
            verbose=0,
    )
else:
    mol = pyscf.gto.M(
            atom='C 0 0 0',
            basis='cc-pvdz',
            verbose=0,
    )

# --- Mean-field
if isinstance(mol, vayesta.lattmod.LatticeMole):
    mf = vayesta.lattmod.LatticeMF(mol)
else:
    mf = pyscf.scf.RHF(mol)
mf.max_memory = mol.max_memory = 1e9
mf.conv_tol = 1e-12
mf.kernel()
homo = np.max(mf.mo_energy[mf.mo_occ > 0])
lumo = np.min(mf.mo_energy[mf.mo_occ == 0])
log.info("Mean-field")
log.info("**********")
log.changeIndentLevel(1)
log.info("  > E(mf) = %14.8f", mf.e_tot)
log.info("  > IP    = %14.8f", -homo)
log.info("  > EA    = %14.8f", lumo)
log.info("  > Gap   = %14.8f", lumo - homo)
log.changeIndentLevel(-1)

# --- Fragmentation
if isinstance(mol, vayesta.lattmod.LatticeMole):
    ewf = vayesta.ewf.EWF(mf, bno_threshold=bno_threshold, fragment_type='site', log=fakelog)
    #for i in range(mol.natm//2):
    #    ewf.make_atom_fragment([i*2, i*2+1])
    for i in range(mol.natm):
        ewf.make_atom_fragment(i)
else:                                                 
    ewf = vayesta.ewf.EWF(mf, bno_threshold=bno_threshold, fragment_type='Lowdin-AO', log=fakelog)
    for i in range(mol.natm):
        ewf.make_atom_fragment(i)

# --- Loop over fragments
data = collections.defaultdict(list)
for x, frag in enumerate(ewf.fragments):
    log.info("Fragment %d", x)
    log.info("*********" + "*"*len(str(x)))
    log.changeIndentLevel(1)

    # --- Generate orbitals
    if ewdmet:
        c_ewdmet, c_froz_occ, c_froz_vir = ewdmet_bath.make_ewdmet_bath(frag, frag.c_env, nmom=nmom_ewdmet)
        c_act_occ, c_act_vir = frag.diagonalize_cluster_dm(frag.c_frag, c_ewdmet)
    else:
        frag.make_bath()
        c_nbo_occ, c_froz_occ = frag.truncate_bno(frag.c_no_occ, frag.n_no_occ, bno_threshold)
        c_nbo_vir, c_froz_vir = frag.truncate_bno(frag.c_no_vir, frag.n_no_vir, bno_threshold)
        c_act_occ = frag.canonicalize_mo(frag.c_cluster_occ, c_nbo_occ)[0]
        c_act_vir = frag.canonicalize_mo(frag.c_cluster_vir, c_nbo_vir)[0]
    c_act = np.hstack((c_act_occ, c_act_vir))
    c_froz = np.hstack((c_froz_occ, c_froz_vir))
    c_occ = np.hstack((c_froz_occ, c_act_occ))
    c_vir = np.hstack((c_act_vir, c_froz_vir))
    mo_coeff = np.hstack((c_occ, c_vir))
    rdm1_froz = np.dot(c_froz_occ, c_froz_occ.T) * 2.0
    rdm1_act = np.dot(c_act_occ, c_act_occ.T) * 2.0
    log.info("Orbital dimensions:")
    log.info("  > Active:  nocc = %-4d nvir = %-4d nelec = %-4.2f", c_act_occ.shape[1], c_act_vir.shape[1], np.trace(rdm1_act))
    log.info("  > Frozen:  nocc = %-4d nvir = %-4d nelec = %-4.2f", c_froz_occ.shape[1], c_froz_vir.shape[1], np.trace(rdm1_froz))
    log.info("  > Total:   nocc = %-4d nvir = %-4d nelec = %-4.2f", c_occ.shape[1], c_vir.shape[1], np.trace(mf.make_rdm1()))

    # --- Get the MOs
    fock = np.einsum('pq,pi,qj->ij', mf.get_fock(), mo_coeff, mo_coeff)
    mo_energy, r = np.linalg.eigh(fock)
    mo_occ = np.array(c_occ.shape[1]*[2] + c_vir.shape[1]*[0])

    # --- Get Veff due to the frozen density
    veff = mf.get_veff(dm=rdm1_froz)
    veff = np.einsum('pq,pi,qj->ij', veff, mo_coeff, mo_coeff)

    # -- Get the ERIs
    eri = pyscf.ao2mo.incore.full(mf._eri, c_act, compact=False)
    eri = eri.reshape((c_act.shape[1],) * 4)

    # --- Run the solver
    gf2 = ragf2.RAGF2(
            mf,
            mo_energy=mo_energy,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            frozen=(c_froz_occ.shape[1], c_froz_vir.shape[1]),
            eri=eri,
            veff=veff,
            conv_tol=1e-5,
            log=fakelog,
    )
    gf2.kernel()
    data['solvers'].append(gf2)
    fock = gf2.get_fock(with_frozen=False)
    se = gf2.se
    mom = np.array((se.get_occupied().moment(range(2*nmom_projection+2)), se.get_virtual().moment(range(2*nmom_projection+2))))
    log.info("AGF2 results:")
    log.info("  > Converged: %r", gf2.converged)
    log.info("  > E(1b)  = %14.8f   E(2b)   = %12.8f", gf2.e_1b, gf2.e_2b)
    log.info("  > E(tot) = %14.8f   E(corr) = %12.8f", gf2.e_tot, gf2.e_corr)
    log.info("  > IP     = %14.8f   EA      = %12.8f", gf2.e_ip, gf2.e_ea)
    log.info("  > Gap    = %14.8f", gf2.e_ip + gf2.e_ea)

    # --- Build projectors
    c = np.einsum('pa,pq,qi->ai', c_act, mf.get_ovlp(), frag.c_frag)
    p_frag = np.dot(c, c.T)
    if democratic:
        # mo_coeff here can be replace by np.hstack((frag.c_frag, frag.c_env))
        c = np.einsum('pa,pq,qi->ai', c_act, mf.get_ovlp(), mf.mo_coeff)
        p_full = np.dot(c, c.T)

    if democratic:
        # --- Project onto (fragment | full) in the basis of active cluster MOs
        fock = np.einsum('pq,pi,qj->ij', fock, p_frag, p_full)
        mom = np.einsum('...pq,pi,qj->...ij', mom, p_frag, p_full)

        # --- Partition democratically
        fock = 0.5 * (fock + fock.T)
        mom = 0.5 * (mom + mom.swapaxes(2, 3))

    else:
        # --- Project onto fragment orbitals in the basis of active cluster MOs
        fock = np.einsum('pq,pi,qj->ij', fock, p_frag, p_frag)
        se.coupling = np.dot(p_frag.T, se.coupling)
        mom = np.einsum('...pq,pi,qj->...ij', mom, p_frag, p_frag)

    # --- Transform into original MO basis
    c = np.einsum('pa,pq,qi->ai', c_act, mf.get_ovlp(), mf.mo_coeff)
    fock = np.einsum('pq,pi,qj->ij', fock, c, c)
    if not democratic:
        se.coupling = np.dot(c.T, se.coupling)
    mom = np.einsum('...pq,pi,qj->...ij', mom, c, c)

    # --- Store results
    data['fock'].append(fock)
    if not democratic:
        data['se'].append(se)
    data['mom'].append(mom)
    log.changeIndentLevel(-1)

# --- Combine fragment results
fock = sum(data['fock'])
mom = sum(data['mom'])
if not democratic:
    se = pyscf.agf2.SelfEnergy(
            np.concatenate([s.energy for s in data['se']]),
            np.concatenate([s.coupling for s in data['se']], axis=1),
    )
    #FIXME: If the fragments have very different chempots this may be tricky:
    #       If so, then track occupancy of poles instead of a global chempot.
    assert np.allclose(mom[0], se.get_occupied().moment(range(2*nmom_projection+2)))
    assert np.allclose(mom[1], se.get_virtual().moment(range(2*nmom_projection+2)))

# --- Construct the compressed SE
if nmom_projection == 0:
    se_occ = pyscf.agf2.SelfEnergy(*pyscf.agf2._agf2.cholesky_build(*mom[0]))
    se_vir = pyscf.agf2.SelfEnergy(*pyscf.agf2._agf2.cholesky_build(*mom[1]))
    se = pyscf.agf2.aux.combine(se_occ, se_vir)
else:
    e, v = dyson.kernel_se(mom[0], mom[1], nmom_lanczos=nmom_projection)
    se = pyscf.agf2.SelfEnergy(e, v)

# --- Get results
gf2 = ragf2.RAGF2(mf, log=fakelog)
gf2.se = se
gf2.se.remove_uncoupled(tol=1e-9)
gf2.gf = se.get_greens_function(fock)
gf2.gf.remove_uncoupled(tol=1e-9)
if fock_loop:
    gf2.gf, gf2.se = gf2.fock_loop()
else:
    gf2.se.chempot = gf2.gf.chempot = pyscf.agf2.chempot.binsearch_chempot(se.eig(fock), se.nphys, mol.nelectron)[0]
gf2.e_1b = gf2.energy_1body()
gf2.e_2b = gf2.energy_2body()
log.info("Output (Emebedded AGF2)")
log.info("***********************")
log.changeIndentLevel(1)
log.info("  > E(1b)  = %14.8f   E(2b)   = %12.8f", gf2.e_1b, gf2.e_2b)
log.info("  > E(tot) = %14.8f   E(corr) = %12.8f", gf2.e_tot, gf2.e_corr)
log.info("  > IP     = %14.8f   EA      = %12.8f", gf2.e_ip, gf2.e_ea)
log.info("  > Gap    = %14.8f", gf2.e_ip + gf2.e_ea)
log.changeIndentLevel(-1)

# --- Standard AGF2 output
gf2 = ragf2.RAGF2(mf, log=fakelog)
gf2.kernel()
log.info("Output (Standard AGF2)")
log.info("**********************")
log.changeIndentLevel(1)
log.info("  > E(1b)  = %14.8f   E(2b)   = %12.8f", gf2.e_1b, gf2.e_2b)
log.info("  > E(tot) = %14.8f   E(corr) = %12.8f", gf2.e_tot, gf2.e_corr)
log.info("  > IP     = %14.8f   EA      = %12.8f", gf2.e_ip, gf2.e_ea)
log.info("  > Gap    = %14.8f", gf2.e_ip + gf2.e_ea)
log.changeIndentLevel(-1)
