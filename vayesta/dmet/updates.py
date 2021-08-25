import numpy as np
from pyscf.lib import diis

class Update:
    def __init__(self, param_shape):
        self.param_shape = param_shape
        self.prev_params = np.zeros((sum([np.product(x) for x in param_shape]),))

    def _flatten_params(self, params):
        for (p,s) in zip(params, self.param_shape):
            assert(p.shape == s)
        return np.concatenate([x.ravel() for x in params])

    def _unflatten_params(self, params):
        res = []
        x = 0
        for s in self.param_shape:
            nval = np.product(s)
            res += [params[x:x+nval].reshape(s)]
            x += nval
        return res

class DIISUpdate(Update):
    def __init__(self, param_shape, space_size = 6, min_space_size = 1):
        super().__init__(param_shape)
        # Force incore on DIIS, otherwise we'll demolish our storage for large enough systems.
        self.adiis = diis.DIIS(incore=True)
        self.adiis.space = space_size
        self.adiis.min_space = min_space_size

    def update(self, params):
        flat_params = self._flatten_params(params)
        diff = sum((flat_params - self.prev_params)**2)**(0.5)
        update = self.adiis.update(flat_params)
        self.prev_params = flat_params
        return self._unflatten_params(update), diff

class MixUpdate(Update):
    def __init__(self, param_shape, alpha = 1.0):
        super().__init__(param_shape)
        self.alpha = alpha

    def update(self, params):
        flat_params = self._flatten_params(params)
        diff = sum((flat_params - self.prev_params) ** 2) ** (0.5)
        update = (1.0 - self.alpha) * self.prev_params + self.alpha * flat_params
        self.prev_params = flat_params
        return self._unflatten_params(update), diff
