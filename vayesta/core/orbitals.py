import numpy as np

from vayesta.core.util import *

def is_rhf(coeff):
    return (coeff[0].ndim == 1)

def stack_coeffs(*coeffs):
    # RHF
    if (coeffs[0][0].ndim == 1):
        assert np.all([c.ndim == 2 for c in coeffs])
        return hstack(*coeffs)
    # UHF
    assert np.all([len(c) == 2 for c in coeffs])
    assert np.all([(c[0].ndim == c[1].ndim == 2) for c in coeffs])
    return (hstack(*[c[0] for c in coeffs]),
            hstack(*[c[1] for c in coeffs]))

class BaseOrbitalSpace:
    pass

def OrbitalSpace(coeff, name):
    if is_rhf(coeff):
        return SpatialOrbitalSpace(coeff, name=name)
    return SpinOrbitalSpace(coeff, name=name)

class SpatialOrbitalSpace(BaseOrbitalSpace):

    def __init__(self, coeff, name=''):
        self.coeff = coeff
        self.name = name

    def __repr__(self):
        return "SpatialOrbitals(%s)" % self.name

    @property
    def size(self):
        return self.coeff.shape[-1]

    def __add__(self, other):
        if other is None:
            # Return copy to avoid unintended side-effects
            return self.copy()
        coeff = hstack(self.coeff, other.coeff)
        name = '+'.join(filter(None, [self.name, other.name]))
        return type(self)(coeff, name=name)

    def copy(self, name=''):
        return type(self)(self.coeff.copy(), name=(name or self.name))

    def transform(self, trafo, inplace=False):
        copy = self.copy() if not inplace else self
        copy.coeff = trafo(copy.coeff)
        return copy

class SpinOrbitalSpace(BaseOrbitalSpace):

    def __init__(self, coeff, name=''):
        self.alpha = SpatialOrbitalSpace(coeff[0], 'alpha')
        self.beta = SpatialOrbitalSpace(coeff[1], 'beta')
        self.name = name

    def __repr__(self):
        return "SpinOrbitals(%s)" % self.name

    @property
    def size(self):
        return np.asarray([self.alpha.size, self.beta.size])

    @property
    def coeff(self):
        return (self.alpha.coeff, self.beta.coeff)

    def __add__(self, other):
        if other is None:
            # Return copy to avoid unintended side-effects
            return self.copy()
        coeff = ((self.alpha + other.alpha).coeff,
                 (self.beta + other.beta).coeff)
        name = ','.join(filter(None, [self.name, other.name]))
        return type(self)(coeff, name=name)

    def copy(self, name=''):
        return type(self)((self.alpha.coeff.copy(), self.beta.coeff.copy()), name=(name or self.name))

    def transform(self, trafo, inplace=False):
        if not hasattr(trafo, '__len__'):
            trafo = (trafo, trafo)
        copy = self.copy() if not inplace else self
        copy.alpha.transform(trafo[0], inplace=True)
        copy.beta.transform(trafo[1], inplace=True)
        return copy

#class OrbitalCollection(BaseOrbitals):
#
#    def __init__(self, **orbitals):
#        #self._names = []
#        self._orbitals = {}
#        for name, coeff in orbitals.items():
#            orb = Orbitals(coeff, name=name)
#            #self._names.append(name)
#            #self._orbitals.append(orb)
#            self._orbitals[name] = orb
#            setattr(self, name, orb)
#
#    def __repr__(self):
#        return "OrbitalCollection%s" % self._orbitals
#
#    def __iter__(self):
#        yield from self._orbitals.values()
#
#    def __len__(self):
#        return sum([len(x) for x in self])
#
#    def __add__(self, other):
#     cluster   < dmet
#
#
#    #    if other is None:
#    #        # Return copy to avoid unintended side-effects
#    #        return SpinnedOrbitalSpace((self.alpha.coeff, self.beta.coeff))
#    #    if isinstance(other, SpinnedOrbitalSpace):
#    #        other = other.alpha, other.beta
#
#    def copy(self):
#        return OrbitalCollection(**{k : v.copy() for k, v in self._orbitals.items()})


if __name__ == '__main__':

    nao = 10
    nfrag = 2
    nbath = 3
    e, v = np.linalg.eigh(np.random.rand(nao, nao))

    def test(c_frag, c_bath):
        frag = Orbitals(c_frag, "Fragment")
        bath = Orbitals(c_bath, "DMET-bath")
        cluster = frag + bath

        print("str(frag)=       %r" % frag)
        print("str(bath)=       %r" % cluster)
        print("str(cluster)=    %r" % cluster)
        try:
            print("cluster.size=    %r" % cluster.size)
            print("cluster.coeff=   %r" % cluster.coeff)
        except TypeError:
            print("cluster.size=    %r %r" % cluster.size)
            print("cluster.coeff=   %r %r" % cluster.coeff)

    # RHF
    c_frag = v[:,:nfrag]
    c_bath = v[:,nfrag:nfrag+nbath]
    test(c_frag, c_bath)

    # UHF
    c_frag = (c_frag, c_frag)
    c_bath = (c_bath, v[:,nfrag:nfrag+nbath+2])
    test(c_frag, c_bath)


