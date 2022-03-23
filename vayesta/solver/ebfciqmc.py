import dataclasses

from .fciqmc import FCIQMCSolver, UFCIQMCSolver, header
from .rdm_utils import load_spinfree_ladder_rdm_from_m7

import numpy as np
from vayesta.core.util import *


class EBFCIQMCSolver(FCIQMCSolver):
    @dataclasses.dataclass
    class Options(FCIQMCSolver.Options):
        threads: int = 1
        lindep: float = None
        conv_tol: float = None
        max_boson_occ: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_M7(self, *args, **kwargs):

        self.EBDUMP_name = 'EBDUMP_cluster' + str(int(self.fragment.id))
        self.BOSDUMP_name = 'BOSDUMP_cluster' + str(int(self.fragment.id))


        write_bosdump(self.fragment.bos_freqs, fname=self.BOSDUMP_name)
        write_ebdump(np.array(self.fragment.couplings), fname=self.EBDUMP_name)
        M7_config_obj = super().setup_M7(*args, **kwargs)

        # adding the pure boson number-conserving 1RDM (0011) since it is a prerequisite of variational energy estimation
        if self.opts.make_rdm_ladder:
            M7_config_obj.M7_config_dict['av_ests']['rdm']['ranks'] = ['1', '2', '1110', '1101', '0011']
            M7_config_obj.M7_config_dict['av_ests']['rdm']['archive']['save'] = 'yes'
        M7_config_obj.M7_config_dict['hamiltonian']['nboson_max'] = self.opts.max_boson_occ
        return M7_config_obj

    def make_rdm_eb(self):
        rdm_1110 = load_spinfree_ladder_rdm_from_m7(self.h5_name, True)
        rdm_1101 = load_spinfree_ladder_rdm_from_m7(self.h5_name, False)
        # only holds for stochastic: no
        # assert np.allclose(rdm_1110, rdm_1101.transpose(1,0,2))
        # average over arrays that are equivalent due to hermiticity symmetry
        return (rdm_1110 + rdm_1101.transpose(1, 0, 2)) / 2.0


class UEBFCIQMCSolver(EBFCIQMCSolver, UFCIQMCSolver):
    pass


def write_ebdump(v, v_unc=None, fname='EBDUMP', thresh=1e-12):
    '''
    write the coeffients of boson "excitations" and "de-excitations" which correspond to
    single fermion number-conserving excitations (ranksigs 1110, and 1101 in M7 nomenclature)
    and those which do not couple to fermion degrees of freedom at all (0010, and 0001),

    these coefficients are given here by the v, and v_unc (uncoupled) args respectively
    '''
    if (len(v.shape) == 4):
        assert v.shape[0]==2
        spin_resolved = True
        nmode = v.shape[1]
        norb = v.shape[2]
    else:
        assert len(v.shape) == 3
        spin_resolved = False
        nmode = v.shape[0]
        norb = v.shape[1]

    if v_unc is None:
        v_unc = np.zeros(nmode)
    elif len(np.shape(v_unc)) == 0:
        v_unc = np.ones(nmode) * v_unc
    else:
        assert len(np.shape(v_unc)) == 1

    with open(fname, 'w') as f:
        f.write(header(norb, nmode, spin_resolved))
        if spin_resolved:
            for i in range(2):
                for n in range(nmode):
                    for p in range(norb):
                        for q in range(norb):
                            if (abs(v[i, n, p, q]) < thresh): continue
                            f.write('{}    {}    {}    {}\n'.format(v[i, n, p, q], n + 1, 2 * p + 1 + i, 2 * q + 1 + i))
        else:
            for n in range(nmode):
                for p in range(norb):
                    for q in range(norb):
                        if (abs(v[n, p, q]) < thresh): continue
                        f.write('{}    {}    {}    {}\n'.format(v[n, p, q], n + 1, p + 1, q + 1))

        for n in range(nmode):
            if (v_unc[n] == 0.0): continue
            f.write('{}    {}    {}    {}\n'.format(v_unc[n], n + 1, 0, 0))


def write_bosdump(w, fname='BOSDUMP', thresh=1e-12):
    '''
    write the coeffients of boson number-conserving operators (ranksig 0011 in M7 nomenclature)
    '''
    nmode = w.shape[0]
    ndim = len(w.shape)
    if ndim == 1: w = np.diag(w)
    ndim = len(w.shape)
    assert ndim == 2
    with open(fname, 'w') as f:
        f.write(header(None, nmode))
        for n in range(nmode):
            for m in range(nmode):
                if (abs(w[n, m]) < thresh): continue
                # w is a coeff array for boson number-conserving singles, 
                # but the BOSDUMP format supports upto doubles
                f.write('{}    {}    {}    0     0\n'.format(w[n, m], n + 1, m + 1))


def write_all(tmat, Umat, Vmat, Vmat_unc, Omat, ecore, nelec, fname_fcidump='FCIDUMP', fname_ebdump='EBDUMP',
              fname_bosdump='BOSDUMP'):
    '''
    include the delta_zpm value in the core energy of the FCIDUMP
    '''
    from pyscf import tools, ao2mo
    nsite = tmat.shape[0]
    tools.fcidump.from_integrals(fname_fcidump, tmat,
                                 ao2mo.restore(8, Umat, nsite), nsite, nelec, ecore, 0, [1, ] * nsite)

    write_ebdump(Vmat, Vmat_unc, fname_ebdump)
    write_bosdump(Omat, fname_bosdump)
