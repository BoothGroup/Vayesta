"""Package for wave function objects.

TODO: spin-off projections (delegation, inheritance?)
"""

from vayesta.core.types.wf.wf import WaveFunction
from vayesta.core.types.wf.hf import HF_WaveFunction, RHF_WaveFunction, UHF_WaveFunction
from vayesta.core.types.wf.mp2 import MP2_WaveFunction, RMP2_WaveFunction, UMP2_WaveFunction
from vayesta.core.types.wf.cisd import CISD_WaveFunction, RCISD_WaveFunction, UCISD_WaveFunction
from vayesta.core.types.wf.ccsd import CCSD_WaveFunction, RCCSD_WaveFunction, UCCSD_WaveFunction
from vayesta.core.types.wf.fci import FCI_WaveFunction, RFCI_WaveFunction, UFCI_WaveFunction

# WIP:
from vayesta.core.types.wf.cisdtq import CISDTQ_WaveFunction, RCISDTQ_WaveFunction, UCISDTQ_WaveFunction
from vayesta.core.types.wf.ccsdtq import CCSDTQ_WaveFunction, RCCSDTQ_WaveFunction, UCCSDTQ_WaveFunction


__all__ = [
    "WaveFunction",
    "HF_WaveFunction",
    "RHF_WaveFunction",
    "UHF_WaveFunction",
    "MP2_WaveFunction",
    "RMP2_WaveFunction",
    "UMP2_WaveFunction",
    "CISD_WaveFunction",
    "RCISD_WaveFunction",
    "UCISD_WaveFunction",
    "CCSD_WaveFunction",
    "RCCSD_WaveFunction",
    "UCCSD_WaveFunction",
    "FCI_WaveFunction",
    "RFCI_WaveFunction",
    "UFCI_WaveFunction",
    "CISDTQ_WaveFunction",
    "RCISDTQ_WaveFunction",
    "UCISDTQ_WaveFunction",
    "CCSDTQ_WaveFunction",
    "RCCSDTQ_WaveFunction",
    "UCCSDTQ_WaveFunction",
]
