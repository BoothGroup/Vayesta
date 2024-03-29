import dataclasses
from typing import Optional, List

from vayesta.solver.coupling import couple_ccsd_iterations
from vayesta.solver.ccsd import RCCSD_Solver


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
        return couple_ccsd_iterations(self, self.opts.fragments)
