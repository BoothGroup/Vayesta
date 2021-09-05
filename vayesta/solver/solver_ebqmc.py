
import dataclasses

from .solver_qmc import FCIQMCSolver
from .rdm_utils import load_spinfree_ladder_rdm_from_m7

import numpy as np
from vayesta.core.util import *



header = ''' &FCI NORB= {}
 &END
'''


class EBFCIQMCSolver(FCIQMCSolver):
    @dataclasses.dataclass
    class Options(FCIQMCSolver.Options):
        threads: int = 1
        lindep: float = None
        conv_tol: float = None
        bos_occ_cutoff: int = NotSet
        make_rdm_ladder: bool = True

    @dataclasses.dataclass
    class Results(FCIQMCSolver.Results):
        # CI coefficients
        c0: float = None
        c1: np.array = None
        c2: np.array = None
        rdm_eb: np.array = None


    def __init__(self, freqs, couplings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bos_freqs = freqs
        self.eb_couplings = couplings

    def setup_M7(self, *args):
        M7_config_obj = super().setup_M7(*args)

        write_bosdump(self.bos_freqs)
        write_ebdump(self.eb_couplings)


        # adding the pure boson number-conserving 1RDM (0011) since it is a prerequisite of variational energy estimation
        if self.opts.make_rdm_ladder:
            M7_config_obj.M7_config_dict['av_ests']['rdm']['ranks'] .extend(['1110', '1101', '0011'])
            M7_config_obj.M7_config_dict['av_ests']['rdm']['archive']['save'] = 'yes'
        M7_config_obj.M7_config_dict['hamiltonian']['nboson_max'] = self.opts.bos_occ_cutoff

    def gen_M7_results(self, h5_name, *args):# ham_pkl_name, M7_config_obj, coeff_pkl_name, eris):
        """Generate M7 results object."""
        results = super().gen_M7_results(h5_name, *args)
        if self.opts.make_rdm_ladder:
            rdm_1110 = load_spinfree_ladder_rdm_from_m7(h5_name, True)
            rdm_1101 = load_spinfree_ladder_rdm_from_m7(h5_name, False)
            # average over arrays that are equivalent due to hermiticity symmetry
            results.dm_ladder = (rdm_1110 + rdm_1101.transpose(0, 2, 1)) / 2.0
        return results

def write_ebdump(v, v_unc=None, fname='EBDUMP'):
    '''
    write the coeffients of boson "excitations" and "de-excitations" which correspond to
    single fermion number-conserving excitations (ranksigs 1110, and 1101 in M7 nomenclature)
    and those which do not couple to fermion degrees of freedom at all (0010, and 0001),

    these coefficients are given here by the v, and v_unc (uncoupled) args respectively
    '''
    assert len(v.shape )==3
    nsite = v.shape[0]
    if v_unc is None:
        v_unc = np.zeros(nsite)
    elif len(np.shape(v_unc) )==0:
        v_unc = np.ones(nsite ) *v_unc
    else: assert len(np.shape(v_unc) )==1

    with open(fname, 'w') as f:
        f.write(header.format(nsite))
        for n in range(nsite):
            for p in range(nsite):
                for q in range(nsite):
                    if (v[n, p, q ]==0.0): continue
                    f.write('{}    {}    {}    {}\n'.format(v[n, p, q], n+ 1, p + 1, q + 1))
        for n in range(nsite):
            if (v_unc[n] == 0.0): continue
            f.write('{}    {}    {}    {}\n'.format(v_unc[n], n + 1, 0, 0))

def write_bosdump(w, fname='BOSDUMP'):
    '''
    write the coeffients of boson number-conserving operators (ranksig 0011 in M7 nomenclature)
    '''
    nsite = w.shape[0]
    ndim = len(w.shape)
    if ndim == 1: w = np.diag(w)
    ndim = len(w.shape)
    assert ndim == 2
    with open(fname, 'w') as f:
        f.write(header)
        for n in range(nsite):
            for m in range(nsite):
                if (w[n, m] == 0.0): continue
                f.write('{}    {}    {}\n'.format(w[n, m], n + 1, m + 1))


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