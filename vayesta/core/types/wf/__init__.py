"""Package for wave function objects.

TODO: spin-off projections (delegation, inheritance?)
"""

from .wf import WaveFunction
from .hf import HF_WaveFunction, RHF_WaveFunction, UHF_WaveFunction
from .mp2 import MP2_WaveFunction, RMP2_WaveFunction, UMP2_WaveFunction
from .cisd import CISD_WaveFunction, RCISD_WaveFunction, UCISD_WaveFunction
from .ccsd import CCSD_WaveFunction, RCCSD_WaveFunction, UCCSD_WaveFunction
from .fci import FCI_WaveFunction, RFCI_WaveFunction, UFCI_WaveFunction


__all__ = [
        'WaveFunction',
        'HF_WaveFunction', 'RHF_WaveFunction', 'UHF_WaveFunction',
        'MP2_WaveFunction', 'RMP2_WaveFunction', 'UMP2_WaveFunction',
        'CCSD_WaveFunction', 'RCCSD_WaveFunction', 'UCCSD_WaveFunction',
        'CISD_WaveFunction', 'RCISD_WaveFunction', 'UCISD_WaveFunction',
        'FCI_WaveFunction', 'RFCI_WaveFunction', 'UFCI_WaveFunction',
        ]
