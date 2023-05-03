from .ccsd import RCCSD_Solver, UCCSD_Solver
import numpy as np
import dataclasses

from typing import Optional, List, Any

from . import coupling


class coupledRCCSD_Solver(RCCSD_Solver):
    @dataclasses.dataclass
    class Options(RCCSD_Solver.Options):
        # Couple CCSD in other fragments
        fragments: Optional[List] = None

    def set_coupled_fragments(self, fragments):
        self.opts.fragments = fragments

    def get_callback(self):
        if self.opts.fragments is None:
            raise ValueError("Please specify fragments to couple CCSD calculation with.")
        return coupling.couple_ccsd_iterations(self, self.opts.fragments)
