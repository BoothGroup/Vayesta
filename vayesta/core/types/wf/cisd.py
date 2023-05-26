import numpy as np
import pyscf
import pyscf.ci
from vayesta.core import spinalg
from vayesta.core.util import callif, einsum
from vayesta.core.types import wf as wf_types
from vayesta.core.types.wf.project import (project_c1, project_c2, project_uc1, project_uc2, symmetrize_c2,
                                           symmetrize_uc2)


def CISD_WaveFunction(mo, c0, c1, c2, **kwargs):
    if mo.nspin == 1:
        cls = RCISD_WaveFunction
    elif mo.nspin == 2:
        cls = UCISD_WaveFunction
    return cls(mo, c0, c1, c2, **kwargs)


class RCISD_WaveFunction(wf_types.WaveFunction):

    def __init__(self, mo, c0, c1, c2, projector=None):
        super().__init__(mo, projector=projector)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.c1 = project_c1(wf.c1, projector)
        wf.c2 = project_c2(wf.c2, projector)
        wf.projector = projector
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None: projector = self.projector
        wf = self.project(projector.T, inplace=inplace)
        wf.projector = None
        if not sym:
            return wf
        wf.c2 = symmetrize_c2(wf.c2)
        return wf

    def copy(self):
        c0 = self.c0
        c1 = spinalg.copy(self.c1)
        c2 = spinalg.copy(self.c2)
        proj = callif(spinalg.copy, self.projector)
        return type(self)(self.mo.copy(), c0, c1, c2, projector=proj)

    def as_mp2(self):
        raise NotImplementedError

    def as_cisd(self, c0=None):
        if c0 is None:
            return self
        c1 = self.c1 * c0/self.c0
        c2 = self.c2 * c0/self.c0
        return RCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_ccsd(self):
        proj = self.projector
        if proj is not None:
            self = self.restore()
        t1 = self.c1/self.c0
        t2 = self.c2/self.c0 - einsum('ia,jb->ijab', t1, t1)
        l1, l2 = t1, t2
        wf = wf_types.RCCSD_WaveFunction(self.mo, t1, t2, l1=l1, l2=l2, projector=self.projector)
        if proj is not None:
            wf = wf.project(proj)
        return wf

    def get_cisdvec(self):
        if self.projector is not None:
            raise NotImplementedError
        return np.hstack((self.c0, self.c1.ravel(), self.c2.ravel()))

    def as_fci(self):
        ci = pyscf.ci.cisd.to_fcivec(self.get_cisdvec(), self.mo.norb, self.mo.nelec)
        return wf_types.RFCI_WaveFunction(self.mo, ci, projector=self.projector)


class UCISD_WaveFunction(RCISD_WaveFunction):

    @property
    def c1a(self):
        return self.c1[0]

    @property
    def c1b(self):
        return self.c1[1]

    @property
    def c2aa(self):
        return self.c2[0]

    @property
    def c2ab(self):
        return self.c2[1]

    @property
    def c2ba(self):
        if len(self.c2) == 4:
            return self.c2[2]
        return self.c2ab.transpose(1,0,3,2)

    @property
    def c2bb(self):
        return self.c2[-1]

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.c1 = project_uc1(wf.c1, projector)
        wf.c2 = project_uc2(wf.c2, projector)
        wf.projector = projector
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None: projector = self.projector
        wf = self.project((projector[0].T, projector[1].T), inplace=inplace)
        wf.projector = None
        if not sym:
            return wf
        wf.c2 = symmetrize_uc2(wf.c2)
        return wf

    def as_mp2(self):
        raise NotImplementedError

    def as_cisd(self, c0=None):
        if c0 is None:
            return self
        c1a = self.c1a * c0/self.c0
        c1b = self.c1b * c0/self.c0
        c2aa = self.c2aa * c0/self.c0
        c2ab = self.c2ab * c0/self.c0
        c2bb = self.c2bb * c0/self.c0
        c1 = (c1a, c1b)
        if len(self.c2) == 3:
            c2 = (c2aa, c2ab, c2bb)
        elif len(self.c2) == 4:
            c2ba = self.c2ba * c0/self.c0
            c2 = (c2aa, c2ab, c2ba, c2bb)
        return wf_types.UCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_ccsd(self):
        proj = self.projector
        if proj is not None:
            self = self.restore()
        t1a = self.c1a/self.c0
        t1b = self.c1b/self.c0
        t1 = (t1a, t1b)
        t2aa = self.c2aa/self.c0 - einsum('ia,jb->ijab', t1a, t1a) + einsum('ib,ja->ijab', t1a, t1a)
        t2bb = self.c2bb/self.c0 - einsum('ia,jb->ijab', t1b, t1b) + einsum('ib,ja->ijab', t1b, t1b)
        t2ab = self.c2ab/self.c0 - einsum('ia,jb->ijab', t1a, t1b)
        if len(self.c2) == 3:
            t2 = (t2aa, t2ab, t2bb)
        elif len(self.c2) == 4:
            t2ba = self.c2ab/self.c0 - einsum('ia,jb->ijab', t1b, t1a)
            t2 = (t2aa, t2ab, t2ba, t2bb)
        l1, l2 = t1, t2
        wf = wf_types.UCCSD_WaveFunction(self.mo, t1, t2, l1=l1, l2=l2, projector=self.projector)
        if proj is not None:
            wf = wf.project(proj)
        return wf

    def get_cisdvec(self):
        if self.projector is not None:
            raise NotImplementedError
        return pyscf.ci.ucisd.amplitudes_to_cisdvec(self.c0, self.c1, self.c2)

    def as_fci(self):
        norb = self.mo.norb
        if norb[0] != norb[1]:
            # TODO: Allow padding via frozen argument?
            raise NotImplementedError
        ci = pyscf.ci.ucisd.to_fcivec(self.get_cisdvec(), norb[0], self.mo.nelec)
        return wf_types.RFCI_WaveFunction(self.mo, ci, projector=self.projector)
