# PySCF
from pyscf import lib
from pyscf.pbc import gto, scf
from pyscf.agf2 import mpi_helper

# NumPy
import numpy as np

# Vayesta
from vayesta.eagf2 import KRAGF2
from vayesta.misc.gdf import GDF
from vayesta import log, vlog

# Standard library
import sys
import os

# Test set
from gmtkn import sets
systems = sets['GAPS'].systems
keys = sorted(systems.keys())


nk_sample = [5, 5, 5]
nk_interp = [int(sys.argv[1])] * 3
nao = (0, 32)
skip = ["LiH", "Kr", "Ne", "Ar", "Xe"]
skip_atoms = ["Pb", "Te"]
nao = (0, 32)
exp_to_discard = 0.0
precision = 1e-9
exxdiv = None
basis = 'gth-dzvp-molopt-sr'
pseudo = 'gth-pade'
method_name = 'interp-kragf2'
method = KRAGF2

options = dict(
        log=log,
        damping=0.66,
        diis_space=10,
        weight_tol=0.0,
        extra_cycle=True,
        fock_basis='ao',
        dump_chkfile=True,
        max_cycle=20,
        max_cycle_inner=100,
        max_cycle_outer=5,
        conv_tol=1e-5,
        conv_tol_rdm1=1e-10,
        conv_tol_nelec=1e-6,
        conv_tol_nelec_factor=1e-3,
)

def interpolate(arr_k, cell, kpts, kpts_fine, mo_coeff=None, mo_coeff_fine=None):
    # Interpolate some function evaluated on kpts onto kpts_fine
    # Assumed to be AO basis unless mo_coeff is provided
    # k = small mesh, l = large mesh, R = real space

    r_vec_abs = foldscf.translation_vectors_for_kmesh(cell, foldscf.kpts_to_kmesh(cell, kpts))
    nr = len(r_vec_abs)
    kR = np.exp(1.0j * np.dot(kpts, r_vec_abs.T)) / np.sqrt(nr)

    r_vec_abs = foldscf.translation_vectors_for_kmesh(cell, foldscf.kpts_to_kmesh(cell, kpts_fine))
    nr = len(r_vec_abs)
    kL = np.exp(1.0j * np.dot(kpts_fine, r_vec_abs.T)) / np.sqrt(nr)

    if mo_coeff is not None:
        arr_k_ao = np.einsum("...kij,kpi,kqj->...kpq", arr_k, mo_coeff, np.conj(mo_coeff))
    else:
        arr_k_ao = arr_k

    arr_sc_ao = foldscf.k2bvk_2d(arr_k_ao, kR)
    arr_sc_ao = scipy.linalg.block_diag(*[arr_sc_ao] * (kpts_fine.size // kpts.size))
    arr_l_ao = foldscf.bvk2k_2d(arr_sc_ao, kL)
    
    if mo_coeff_fine is not None:
        arr_l = np.einsum("...lpq,lpi,lqj->...lij", arr_l_ao, np.conj(mo_coeff_fine), mo_coeff_fine)
    else:
        arr_l = arr_l_ao

    return arr_l

log.handlers.clear()
fmt = vlog.VFormatter(indent=True)
mpi_helper.barrier()

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

    mf_sample = scf.KRHF(cell)
    mf_sample.kpts = cell.make_kpts(nk_sample)
    mf_sample.with_df = GDF(cell, mf_sample.kpts)
    mf_sample.with_df.build()
    mf_sample.exxdiv = exxdiv

    for i in range(len(mpi_helper.size)):
        if mpi_helper.rank == i:
            chkfile = "kragf2_%s_%s_%s%s%s.chk" % (key, basis, *nk_sample)
            mf_sample.mo_energy = lib.chkfile.load(chkfile, "scf/mo_energy")
            mf_sample.mo_coeff = lib.chkfile.load(chkfile, "scf/mo_coeff")
            mf_sample.mo_occ = lib.chkfile.load(chkfile, "scf/mo_occ")
            mf_sample.e_tot = lib.chkfile.load(chkfile, "scf/e_tot")
            mf_sample.e_tot = lib.chkfile.load(chkfile, "scf/converged")
        mpi_helper.barrier()

    mf_interp = scf.KRHF(cell)
    mf_interp.kpts = cell.make_kpts(nk_interp)
    mf_interp.with_df = GDF(cell, mf_interp.kpts)
    mf_interp.with_df.build()
    mf_interp.exxdiv = exxdiv
    mf_interp.chkfile = '%s_%s_%s_%s%s%s.chk' % (method_name, key, basis, *nk_interp) if mpi_helper.rank == 0 else None
    log.output("Doing MF")
    mf_interp.kernel()

    if mpi_helper.rank == 0:
        lib.chkfile.dump(mf_interp.chkfile, "scf/converged", mf_interp.converged)
    mpi_helper.barrier()

    for k in range(len(mf_interp.kpts)):
        mf_interp.mo_energy[k] = mpi_helper.bcast(mf_interp.mo_energy[k])
        mf_interp.mo_coeff[k] = mpi_helper.bcast(mf_interp.mo_coeff[k])
        mf_interp.mo_occ[k] = mpi_helper.bcast(mf_interp.mo_occ[k])
        mpi_helper.barrier()
    for k1 in range(len(mf_interp.kpts)):
        for k2 in range(len(mf_interp.kpts)):
            mf_interp.with_df._cderi[k1, k2] = mpi_helper.bcast(mf_interp.with_df._cderi[k1, k2])
            mpi_helper.barrier()
    mf_interp.e_tot = mpi_helper.bcast(mf_interp.e_tot)
    mpi_helper.barrier()

    gf2_sample = method(mf_sample, **options)
    for i in range(mpi_helper.size):
        if i == mpi_helper.rank:
            gf2_sample.update_from_chk(chkfile="kragf2_%s_%s_%s%s%s.chk" % (key, basis, *nk_sample))
        mpi_helper.barrier()

    t0_occ_sample = [se.get_occupied().moment(0) for se in gf2_sample.se]
    t1_occ_sample = [se.get_occupied().moment(1) for se in gf2_sample.se]
    t0_vir_sample = [se.get_virtual().moment(0) for se in gf2_sample.se]
    t1_vir_sample = [se.get_virtual().moment(1) for se in gf2_sample.se]

    t0_occ_interp = interpolate(t0_occ_sample, cell, kpts_sample, kpts_interp, mf_sample.mo_coeff, np.einsum("kpq,kqi->kpi", mf_interp.get_ovlp(), mf_interp.mo_coeff))
    t1_occ_interp = interpolate(t1_occ_sample, cell, kpts_sample, kpts_interp, mf_sample.mo_coeff, np.einsum("kpq,kqi->kpi", mf_interp.get_ovlp(), mf_interp.mo_coeff))
    t0_vir_interp = interpolate(t0_vir_sample, cell, kpts_sample, kpts_interp, mf_sample.mo_coeff, np.einsum("kpq,kqi->kpi", mf_interp.get_ovlp(), mf_interp.mo_coeff))
    t1_vir_interp = interpolate(t1_vir_sample, cell, kpts_sample, kpts_interp, mf_sample.mo_coeff, np.einsum("kpq,kqi->kpi", mf_interp.get_ovlp(), mf_interp.mo_coeff))

    se_occ_interp = [gf2._build_se_from_moments(t) for t in zip(t0_occ_interp, t1_occ_interp)]
    se_vir_interp = [gf2._build_se_from_moments(t) for t in zip(t0_vir_interp, t1_vir_interp)]
    se_interp = [agf2.aux.combine(o, v) for o, v in zip(se_occ_interp, se_vir_interp)]
    gf_interp = [s.get_greens_function(np.diag(e)) for s, e in zip(se_interp, mf_interp.mo_energy)]

    gf2_interp = method(mf_interp, eri=np.empty(()), veff=np.empty(()), **options)
    gf2_interp.gf_interp, gf2_interp.se_interp, converged = gf2_interp.fock_loop(gf=gf_interp, se=se_interp)
    gf2.converged = gf2_sample.converged and mf_interp.converged
    gf2.dump_chk()



