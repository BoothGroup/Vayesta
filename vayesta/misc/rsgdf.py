'''
Functionally equivalent to PySCF RSGDF, with all storage incore and
MPI parallelism.

Adapted from pyscf.pbc.df.rsdf

Ref:
'''

import logging
import collections
import numpy as np

from pyscf import lib
from pyscf.gto import moleintor
from pyscf.data.nist import BOHR
from pyscf.pbc.df import ft_ao, rsdf, rsdf_helper
from pyscf.pbc.lib.kpts_helper import unique, is_zero, gamma_point, KPT_DIFF_TOL, get_kconserv
from pyscf.ao2mo.outcore import balance_partition
from pyscf.agf2 import mpi_helper

import vayesta
from vayesta.misc.gdf import _kconserve_indices, _cholesky_decomposed_metric, _get_kpt_hash, make_chgcell, fuse_auxcell, GDF
from vayesta.core.util import time_string

try:
    from mpi4py import MPI
    timer = MPI.Wtime
except ImportError:
    from timeit import default_timer as timer

#TODO eqn ref, paper ref


def _get_2c2e(with_df, uniq_kpts, log=None):
    '''
    Get the bare two-center two-electron interaction.
    '''

    t0 = timer()
    log = log or logging.getLogger(__name__)

    int2c2e = rsdf_helper.intor_j2c(with_df.auxcell, with_df.omega_j2c, kpts=uniq_kpts)

    log.timing("Time for 2c2e:  %s", time_string(timer()-t0))

    return int2c2e


def _get_j2c(with_df, int2c2e, uniq_kpts, log=None):
    '''
    Build j2c using the 2c2e interaction, int2c2e
    '''

    t0 = timer()
    log = log or logging.getLogger(__name__)

    qaux = rsdf.get_aux_chg(with_df.auxcell)
    g0_j2c = np.pi / with_df.omega_j2c**2 / with_df.cell.vol
    mesh_j2c = with_df.mesh_j2c
    Gv, Gvbase, kws = with_df.cell.get_Gv_weights(mesh_j2c)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    b = with_df.cell.reciprocal_vectors()

    j2c = int2c2e

    for k, kpt in enumerate(uniq_kpts):
        t1 = timer()

        if is_zero(kpt):
            j2c[k] -= np.outer(qaux, qaux) * g0_j2c

        coulG_lr = rsdf.weighted_coulG(with_df.cell, with_df.omega_j2c, kpt, False, mesh_j2c)
        G_chg = ft_ao.ft_ao(with_df.auxcell, Gv, b=b, gxyz=gxyz, Gvbase=Gvbase, kpt=kpt).T
        G_aux = G_chg * coulG_lr

        if is_zero(kpt):
            j2c[k] += np.dot(G_aux.real, G_chg.T.real)
            j2c[k] += np.dot(G_aux.imag, G_chg.T.imag)
        else:
            j2c[k] += np.dot(G_aux.conj(), G_chg.T)

        del G_chg, G_aux

        log.timingv("Time for j2c [kpt %d] part:  %s", k, time_string(timer()-t0))

    log.timing("Time for j2c:  %s", time_string(timer()-t0))

    return j2c


def _aux_e2_nospltbas(
        cell,
        auxcell,
        omega,
        kptij_lst,
        shls_slice=None,
        bvk_kmesh=None,
        precision=None,
        fac_type='ME',
        eta_correct=True,
        R_correct=True,
        vol_correct_d=False,
        vol_correct_R=False,
        dstep=1,  # Angstroms
        log=None,
):
    '''
    Incore version of rsdf_helper._aux_e2_nospltbas specifically for RSGDF.
    '''

    log = log or logging.getLogger(__name__)

    if precision is None:
        precision = cell.precision

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    nkptij = len(kptij_lst)

    if gamma_point(kptij_lst):
        dtype = np.float64
    else:
        dtype = np.complex128

    # Get unique maps for the shells in the cell and auxiliary cell:
    refuniqshl_map, uniq_atms, uniq_bas, uniq_bas_loc = rsdf_helper.get_refuniq_map(cell)
    auxuniqshl_map, uniq_atms, uniq_basaux, uniq_basaux_loc = rsdf_helper.get_refuniq_map(auxcell)

    # Get the Schwarz inequalities for two-centre integrals:
    Qauxs = rsdf_helper.get_schwartz_data(uniq_basaux, omega, keep1ctr=True, safe=True)

    # Determine the cutoff radius for the Schwartz inequality between each unique shell:
    precision_schwarz = precision / np.max(Qauxs)
    dcuts = rsdf_helper.get_schwartz_dcut(
            uniq_bas,
            cell.vol,
            omega,
            precision_schwarz,
            r0=cell.rcut,  #TODO from RSGDF object?
            vol_correct=vol_correct_d,
    )

    dijs_lst = rsdf_helper.make_dijs_lst(dcuts, dstep/BOHR)
    dijs_loc = np.cumsum([0]+[len(dijs) for dijs in dijs_lst]).astype(np.int32)

    # Get the Schwarz inequalities for four-centre integrals:
    if fac_type.upper() in ["ISFQ0", "ISFQL"]:
        Qs_lst = get_schwartz_data(uniq_bas, omega, dijs_lst, keep1ctr=True, safe=True)
    else:
        Qs_lst = [np.zeros_like(dijs) for dijs in dijs_lst]

    # Determine the cutoff radius for shell pairs:
    bas_exps = np.array([np.asarray(b[1:])[:,0].min() for b in uniq_bas])
    Rcuts = rsdf_helper.get_3c2e_Rcuts(
            uniq_bas, uniq_basaux, dijs_lst, cell.vol,
            omega, precision, fac_type, Qs_lst,
            vol_correct=vol_correct_R,
            eta_correct=eta_correct,
            R_correct=R_correct,
    )

    # Determine the cutoff radius for each atom:
    atom_Rcuts = rsdf_helper.get_atom_Rcuts_3c(
            Rcuts, dijs_lst, bas_exps, uniq_bas_loc, uniq_basaux_loc,
    )

    # Determine the cutoff radius for the cell:
    cell_rcut = atom_Rcuts.max()

    # Collect prescreening data:
    Ls = rsdf_helper.get_Lsmin(cell, atom_Rcuts, uniq_atms)
    prescreen = (
            refuniqshl_map, auxuniqshl_map, len(uniq_basaux), bas_exps,
            dcuts**2, dstep/BOHR, Rcuts**2, dijs_loc, Ls,
    )

    log.debug("j3c prescreening:")
    log.debug("    cell rcut = %.2f Bohr", cell_rcut)
    log.debug("    keep %d imgs", Ls.shape[0])

    # Get the intor:
    intor, comp = moleintor._get_intor_and_comp(cell._add_suffix('int3c2e'), None)

    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas, 0, auxcell.nbas)
    shlpr_mask = np.ones((shls_slice[1]-shls_slice[0],
                          shls_slice[3]-shls_slice[2]),
                          dtype=np.int8, order="C")

    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)[:shls_slice[5]+1]
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    ngrids = aux_loc[shls_slice[5]] - aux_loc[shls_slice[4]]
    nij = ni * nj
    nii = (ao_loc[shls_slice[1]]*(ao_loc[shls_slice[1]]+1)//2 -
           ao_loc[shls_slice[0]]*(ao_loc[shls_slice[0]]+1)//2)

    if gamma_point(kptij_lst):
        nao_pair = nii
    else:
        nao_pair = nij

    int3c = rsdf_helper.wrap_int3c_nospltbas(
            cell, auxcell, omega, shlpr_mask, prescreen,
            intor=intor,
            aosym='s2ij',
            comp=comp,
            kptij_lst=kptij_lst,
            bvk_kmesh=bvk_kmesh,
    )

    int3c2e = np.zeros((nkptij, ngrids, nij), dtype=dtype)

    for p0, p1 in mpi_helper.prange(shls_slice[4], shls_slice[5], shls_slice[5]-shls_slice[4]):
        shls_slice_part = shls_slice[:4] + (shls_slice[4]+p0, shls_slice[4]+p1)
        q0, q1 = aux_loc[p0], aux_loc[p1]

        int3c2e_part = np.zeros((nkptij, nao_pair, q1-q0), dtype=dtype)
        int3c2e_part = int3c(shls_slice_part, int3c2e_part)
        int3c2e_part = lib.transpose(int3c2e_part, axes=(0, 2, 1))

        if int3c2e_part.shape[-1] != nij:
            int3c2e_part = int3c2e_part.reshape(-1, nii)
            int3c2e_part = lib.unpack_tril(int3c2e_part, lib.HERMITIAN, axis=-1)

        int3c2e_part = int3c2e_part.reshape(nkptij, q1-q0, nij)
        int3c2e[:, q0:q1] = int3c2e_part

    mpi_helper.allreduce_safe_inplace(int3c2e)
    mpi_helper.barrier()

    return int3c2e 


def _get_3c2e(with_df, kptij_list, log=None):
    '''
    Get the bare three-center two-electron interaction
    '''

    t0 = timer()
    log = log or logging.getLogger(__name__)

    if with_df.use_bvk:
        bvk_kmesh = rsdf.kpts_to_kmesh(with_df.cell, with_df.kpts)
        if bvk_kmesh is None:
            log.debug("Non-Gamma-inclusive k-mesh is found. bvk_kmesh is not used.")
        else:
            log.debug("Using bvk_kmesh = [%d %d %d]", *bvk_kmesh)
    else:
        bvk_kmesh = None

    int3c2e = _aux_e2_nospltbas(
            with_df.cell,
            with_df.auxcell,
            with_df.omega,
            kptij_list,
            #max_memory=with_df.max_memory,  #TODO
            bvk_kmesh=bvk_kmesh,
            precision=with_df.precision_R,
    )

    log.timing("Time for 3c2e:  %s", time_string(timer()-t0))

    return int3c2e


def _get_j3c(with_df, j2c, int3c2e, uniq_kpts, uniq_inverse_dict, kptij_lst, log=None, out=None):
    '''
    Use j2c and int3c2e to construct j3c.
    '''

    log = log or logging.getLogger(__name__)
    t0 = timer()

    cell = with_df.cell
    auxcell = with_df.auxcell

    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    Gv, Gvbase, kws = cell.get_Gv_weights(with_df.mesh_j2c)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]
    b = cell.reciprocal_vectors()
    j2c_chol = lambda v: v
    kpts = with_df.kpts
    nkpts = len(kpts)
    qaux = rsdf.get_aux_chg(with_df.auxcell)
    g0 = np.pi / with_df.omega**2 / cell.vol

    #FIXME for Γ
    dtype = np.complex128
    #if gamma_point(kptij_lst):
    #    dtype = np.double
    #else:
    #    dtype = np.complex128

    if with_df.use_bvk:
        bvk_kmesh = rsdf.kpts_to_kmesh(cell, with_df.kpts)
        if bvk_kmesh is None:
            log.debug("Non-Gamma-inclusive k-mesh is found. bvk_kmesh is not used.")
        else:
            log.debug("Using bvk_kmesh = [%d %d %d]", *bvk_kmesh)
    else:
        bvk_kmesh = None

    if out is None:
        out = np.zeros((nkpts, nkpts, naux, nao, nao), dtype=dtype)

    for uniq_kpt_ji_id in mpi_helper.nrange(len(uniq_kpts)):
        t1 = timer()
        log.debug("Constructing j3c [kpt %d].", uniq_kpt_ji_id)
        with log.withIndentLevel(1):

            kpt = uniq_kpts[uniq_kpt_ji_id]
            adapted_ji_idx = uniq_inverse_dict[uniq_kpt_ji_id]
            adapted_kptjs = kptij_lst[:, 1][adapted_ji_idx]

            log.debug("kpt = %s", kpt)
            log.debug("adapted_ji_idx = %s", adapted_ji_idx)

            # Prepare cholesky decomposition of j2c
            if j2c is not None:
                kptji_id = _kconserve_indices(cell, uniq_kpts, -kpt)[0]
                j2c_chol = _cholesky_decomposed_metric(with_df, j2c, kptji_id, log)
                log.debug("kptji_id = %s", kptji_id)

            shls_slice = (0, auxcell.nbas)
            G_chg = ft_ao.ft_ao(auxcell, Gv, shls_slice=shls_slice, 
                                b=b, gxyz=gxyz, Gvbase=Gvbase, kpt=kpt)
            G_chg *= rsdf.weighted_coulG(cell, with_df.omega, kpt, False, with_df.mesh_j2c).reshape(-1, 1)
            log.debugv("Norm of FT for RS cell:  %.12g", np.linalg.norm(G_chg))

            if is_zero(kpt):
                log.debug("Including net charge of AO products")
                vbar = qaux * g0
                ovlp = cell.pbc_intor('int1e_ovlp', hermi=0, kpts=adapted_kptjs)
                ovlp = [np.ravel(s) for s in ovlp]

            bstart, bend, ncol = balance_partition(cell.ao_loc_nr()*nao, nao*nao)[0]
            shls_slice = (bstart, bend, 0, cell.nbas)
            G_ao = ft_ao.ft_aopair_kpts(cell, Gv, shls_slice=shls_slice, aosym='s1', 
                                        b=b, gxyz=gxyz, Gvbase=Gvbase, q=kpt, 
                                        kptjs=adapted_kptjs, bvk_kmesh=bvk_kmesh)
            G_ao = G_ao.reshape(-1, ngrids, ncol)
            log.debugv("Norm of FT for AO cell:  %.12g", np.linalg.norm(G_ao))

            for kji, ji in enumerate(adapted_ji_idx):
                v = int3c2e[ji].astype(out.dtype)

                if is_zero(kpt):
                    for i in np.where(vbar != 0)[0]:
                        v[i] -= vbar[i] * ovlp[kji]

                v += np.dot(G_chg.T.conj(), G_ao[kji])
                v = j2c_chol(v)
                v = v.reshape(-1, nao, nao)

                for ki in with_df.kpt_hash[_get_kpt_hash(kptij_lst[ji][0])]:
                    for kj in with_df.kpt_hash[_get_kpt_hash(kptij_lst[ji][1])]:
                        out[ki, kj, :v.shape[0]] += v
                        log.debug("Filled j3c for kpt [%d, %d]", ki, kj)
                        if ki != kj:
                            out[kj, ki, :v.shape[0]] += lib.transpose(v, axes=(0, 2, 1)).conj()
                            log.debug("Filled j3c for kpt [%d, %d]", kj, ki)

            log.timingv("Time for j3c [kpt %d]:  %s", uniq_kpt_ji_id, time_string(timer()-t1))

    mpi_helper.allreduce_safe_inplace(out)
    mpi_helper.barrier()

    log.timing("Time for j3c:  %s", time_string(timer()-t0))

    return out

def _make_j3c(with_df, kptij_lst):
    '''
    Build the j3c array.
    '''

    t0 = timer()
    log = with_df.log

    if with_df.cell.dimension < 3:
        raise ValueError('GDF does not support low-dimension cells')

    kptis = kptij_lst[:, 0]
    kptjs = kptij_lst[:, 1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    uniq_inverse_dict = {k: np.where(uniq_inverse == k)[0] for k in range(len(uniq_kpts))}

    log.info("Number of unique kpts (kj - ki): %d", len(uniq_kpts))
    log.debugv("uniq_kpts:\n", uniq_kpts)

    # Get the 2c2e interaction:
    int2c2e = _get_2c2e(with_df, uniq_kpts, log=log)

    # Get j2c:
    j2c = _get_j2c(with_df, int2c2e, uniq_kpts, log=log)

    # Get the 3c2e interaction:
    int3c2e = _get_3c2e(with_df, kptij_lst, log=log)

    # Get j3c:
    j3c = _get_j3c(with_df, j2c, int3c2e, uniq_kpts, uniq_inverse_dict, kptij_lst, log=log)

    log.timing("Total time for j3c construction:  %s", time_string(timer()-t0))

    return j3c


class RSGDF(GDF):
    def __init__(self, cell, kpts=np.zeros((1, 3)), log=None):
        super().__init__(cell, kpts=kpts, log=log)

        # If True and kpts are Γ-inclusive, use BVK cell trick for j3c:
        self.use_bvk = True

        # Precision for real-space lattice sum (R) and reciprocal-space Fourier transform (G):
        self.precision_R = cell.precision * 1e-2
        self.precision_G = cell.precision

        # One of {omega, npw_max} must be provided and the other will be
        # determined. Priority when both are given is omega > npw_max.
        # If omega is determined, omega = max(omega, self._omega_min).
        # Default npw_max = 350 ≡ 7x7x7 PWs for 3D isotropic systems.
        # mesh_compact is determined from omega for given accuracy.
        self.npw_max = 350
        self._omega_min = 0.3
        self.omega = None
        self.ke_cutoff = None
        self.mesh_compact = None

        # Use a fixed higher omega for j2c since it is inverted.
        self.omega_j2c = 0.4
        self.mesh_j2c = None
        self.precision_j2c = 1e-4 * self.precision_G

        self._keys.update(self.__dict__.keys())

    def dump_flags(self):
        super().dump_flags()
        for key in [
                'precision_R', 'precision_G', 'npw_max', '_omega_min', 'omega',
                'mesh_compact', 'omega_j2c', 'mesh_j2c', 'precision_j2c',
        ]:
            self.log.info("  > %-24s %r", key + ":", getattr(self, key))
        return self

    _make_j3c = _make_j3c

    def build(self, j_only=None, with_j3c=True):
        j_only = j_only or self._j_only
        if j_only:
            self.log.warn('j_only=True has not effect on overhead in %s', self.__class__)
        if self.kpts_band is not None:
            raise ValueError('%s does not support kwarg kpts_band' % self.__class__)

        self.check_sanity()
        self.dump_flags()

        rsdf.RSDF._rsh_build(self)

        self.chgcell = make_chgcell(self.auxcell, self.eta, rcut=self.rcut_smooth)
        self.kconserv = get_kconserv(self.cell, self.kpts)

        kpts = np.asarray(self.kpts)[unique(self.kpts)[1]]
        kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
        kptij_lst = np.asarray(kptij_lst)

        self.kpt_hash = collections.defaultdict(list)
        for k, kpt in enumerate(self.kpts):
            val = _get_kpt_hash(kpt)
            self.kpt_hash[val].append(k)

        self.kptij_hash = collections.defaultdict(list)
        for k, kpt in enumerate(kptij_lst):
            val = _get_kpt_hash(kpt)
            self.kptij_hash[val].append(k)

        if with_j3c:
            self._cderi = self._make_j3c(kptij_lst)

        return self


if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from vayesta.misc.gdf import GDF

    cell = gto.Cell()
    cell.atom = 'He 0 0 0; He 1 1 1'
    cell.basis = 'cc-pvdz'
    cell.a = np.eye(3) * 3
    cell.verbose = 0
    cell.max_memory = 1e9
    cell.build()

    kpts = cell.make_kpts([3, 1, 1])
    kpts += 1e-4

    with_df_0 = rsdf.RSDF(cell, kpts)
    with_df_0.build()

    with_df_1 = RSGDF(cell, kpts)
    with_df_1.build()

    dm = np.random.random((len(kpts), cell.nao, cell.nao))
    dm += dm.swapaxes(1,2).conj()

    for ki, kpti in enumerate(kpts):
        for kj, kptj in enumerate(kpts):
            for (r1, i1, _), (r2, i2, _) in zip(
                    list(with_df_0.sr_loop((kpti, kptj))),
                    list(with_df_1.sr_loop((kpti, kptj))),
            ):
                assert np.allclose(r1, r2), (lib.fp(r1), lib.fp(r2))
                assert np.allclose(i1, i2), (lib.fp(i1), lib.fp(i2))
