import dataclasses
from typing import Optional, List, Any

import numpy as np

from vayesta.solver.coupling import tailor_with_fragments, externally_correct
from .ccsd import RCCSD_Solver, UCCSD_Solver


class extRCCSD_Solver(RCCSD_Solver):
    @dataclasses.dataclass
    class Options(RCCSD_Solver.Options):
        # Tailor/externally correct CCSD with other fragments
        external_corrections: Optional[List[Any]] = dataclasses.field(default_factory=list)

    def get_callback(self):
        # Tailoring of T1 and T2
        tailors = [ec for ec in self.opts.external_corrections if (ec[1] == 'tailor')]
        externals = [ec for ec in self.opts.external_corrections if (ec[1] in ('external', 'delta-tailor'))]
        if tailors and externals:
            raise NotImplementedError
        if tailors:
            tailor_frags = self.hamil._fragment.base.get_fragments(id=[t[0] for t in tailors])
            proj = tailors[0][2]
            if np.any([(t[2] != proj) for t in tailors]):
                raise NotImplementedError
            self.log.info("Tailoring CCSD from %d fragments (projectors= %d)", len(tailor_frags), proj)
            return tailor_with_fragments(self, tailor_frags, project=proj)
        # External correction of T1 and T2
        if externals:
            self.log.info("Externally correct CCSD from %d fragments", len(externals))
            return externally_correct(self, externals, hamil=self.hamil)
        # No correction applied; no callback function to apply.
        return None


class extUCCSD_Solver(UCCSD_Solver, extRCCSD_Solver):
    @dataclasses.dataclass
    class Options(UCCSD_Solver.Options, extRCCSD_Solver.Options):
        pass
