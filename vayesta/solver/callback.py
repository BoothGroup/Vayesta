import dataclasses
from typing import Callable
import numpy as np

from vayesta.solver.solver import ClusterSolver

class CallbackSolver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        # Need to specify a type for this to work
        callback: int = None

    # def __init__(self, hamil, log=None, **kwargs):
    #     #super().__init__(hamil, log, **kwargs)
    #     self.opts = self.Options()
    #     self.callback = kwargs['callback']
    #     self.hamil = hamil
    #     print("AAAAAAA %s"%kwargs['callback'])
        # self.results = None

    def kernel(self, *args, **kwargs):
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=True)
        #print(self.opts.callback)
        self.results = self.opts.callback(mf_clus)