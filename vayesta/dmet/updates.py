import numpy as np

from pyscf.lib import diis


class Update:
    def __init__(self):
        self.param_shape = None
        self.prev_params = None

    def _flatten_params(self, params):

        def get_shape(x):
            return [get_shape(y) if type(y) != np.ndarray else y.shape for y in x]

        def get_flat(x):
            return np.concatenate([get_flat(y) if type(y) != np.ndarray else y.ravel() for y in x])

        flat_params = get_flat(params)

        if self.param_shape is None:
            self.param_shape = get_shape(params)
            self.prev_params = np.zeros_like(flat_params)

        return flat_params

    def _unflatten_params(self, params):
        x = 0

        def get_nonflat(flat_params, shapes, x):
            if type(shapes[0]) == int:
                return flat_params[x:x + np.product(shapes)].reshape(shapes), x + np.product(shapes)
            else:
                finres = []
                for shape in shapes:
                    res, x = get_nonflat(flat_params, shape, x)
                    finres += [res]
                return finres, x

        nonflat_params, x = get_nonflat(params, self.param_shape, x)
        assert (x == len(params))
        return nonflat_params


class DIISUpdate(Update):
    def __init__(self, space_size=6, min_space_size=1):
        super().__init__()
        # Force incore on DIIS, otherwise we'll demolish our storage for large enough systems.
        self.adiis = diis.DIIS(incore=True)
        self.adiis.space = space_size
        self.adiis.min_space = min_space_size

    def update(self, params):
        flat_params = self._flatten_params(params)
        diff = sum((flat_params - self.prev_params) ** 2) ** (0.5)
        update = self.adiis.update(flat_params)
        self.prev_params = flat_params
        return self._unflatten_params(update), diff


class MixUpdate(Update):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def update(self, params):
        flat_params = self._flatten_params(params)
        diff = sum((flat_params - self.prev_params) ** 2) ** (0.5)
        update = (1.0 - self.alpha) * self.prev_params + self.alpha * flat_params
        self.prev_params = flat_params
        return self._unflatten_params(update), diff
