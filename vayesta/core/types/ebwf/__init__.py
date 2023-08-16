from vayesta.core.types.ebwf.ebwf import EBWavefunction
try:
        from vayesta.core.types.ebwf.ebcc import EBCC_WaveFunction, REBCC_WaveFunction, UEBCC_WaveFunction
except ImportError:
        _has_ebcc = False
else:
        _has_ebcc = True
