#import copy
import numpy as np

import pyscf
import pyscf.mp
import pyscf.cc

#from pyscf.cc.ccsd import _ChemistERIs as CCSD_ERIs
#from pyscf.cc.rccsd import _ChemistERIs as RCCSD_ERIs
#from pyscf.cc.dfccsd import _ChemistERIs as DFCCSD_ERIs
#from pyscf.mp.mp2 import _ChemistERIs as MP2_ERIs

from vayesta.core.util import *


def postscf_ao2mo(postscf, mo_coeff=None, fock=None, mo_energy=None, e_hf=None):
    """Use as ao2mo(cc, mo_coeff) (instead of cc.ao2mo(mo_coeff))

    Supports:

    cc.ccsd.CCSD
    cc.ccsd.RCCSD
    cc.ccsd.DFCCSD
    mp.mp2.MP2
    mp.mp2.DFMP2
    """
    replace = {}
    if fock is not None:
        replace['get_fock'] = (fock if callable(fock) else lambda *args, **kwargs: fock)
    if e_hf is not None:
        replace['energy_tot'] = (e_hf if callable(e_hf) else lambda *args, **kwargs: e_hf)
    if (fock is not None and e_hf is not None):
        # make_rdm1 and get_veff are called within postscf.ao2mo, to generate
        # the Fock matrix and SCF energy - since we set these directly,
        # we can avoid performing any computation in these functions:
        do_nothing = lambda *args, **kwargs: None
        replace['make_rdm1'] = do_nothing
        replace['get_veff'] = do_nothing

    # Replace attributes in `replace` temporarily for the potscf.ao2mo call;
    # after the with statement, the attributes are reset to their intial values.
    with replace_attr(postscf._scf, **replace):
        eris = postscf.ao2mo(mo_coeff)

    if mo_energy is not None:
        eris.mo_energy = mo_energy

    return eris

    ## Important: Check specialised classes before parent classes (since DFCCSD is also CCSD!)
    #if isinstance(postscf, cc.dfccsd.DFCCSD):
    #    eris = DFCCSD_ERIs()
    #if isinstance(postscf, cc.rccsd.RCCSD):
    #if isinstance(postscf, cc.ccsd.CCSD):
    #if isinstance(postscf, mp.mp2.DFMP2):
    #if isinstance(postscf, mp.mp2.MP2):
