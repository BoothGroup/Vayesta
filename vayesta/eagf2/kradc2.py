# Standard library
import dataclasses

# NumPy
import numpy as np

# PySCF
from pyscf import lib, agf2
from pyscf.agf2 import mpi_helper
from pyscf.pbc.lib import kpts_helper

# Vayesta
import vayesta
from vayesta.core.util import time_string, OptionsBase
from vayesta.eagf2.kragf2 import KRAGF2

# Timings
if mpi_helper.mpi == None:
    from timeit import default_timer as timer
else:
    timer = mpi_helper.mpi.Wtime


@dataclasses.dataclass
class KRADC2Options(OptionsBase):
    """Options for KRADC2 calculations.
    """

    # --- Theory
    non_dyson: bool = True

    # --- Convergence
    nroots: int = 5
    which: str = "ip"
    conv_tol: float = 1e-9
    max_cycle: int = 100
    max_space: int = 12

    # --- Analysis
    excitation_tol: float = 0.1
    excitation_number: int = 5

    # --- Output
    dump_chkfile: bool = True
    chkfile: str = None


class KRADC2:

    Options = KRADC2Options

    def __init__(
            self,
            mf,
            log=None,
            options=None,
            mo_energy=None,
            mo_coeff=None,
            mo_occ=None,
            eri=None,
            **kwargs,
    ):
        """Restricted algebraic diagrammatic construction theory of
        the second-order single particle Green's function.
        """

        # Logging:
        self.log = log or vayesta.log
        self.log.info("Initializing " + self.__class__.__name__)
        self.log.info("*************" + "*" * len(str(self.__class__.__name__)))

        # Options:
        if options is None:
            self.opts = self.Options(**kwargs)
        else:
            self.opts = options.replace(kwargs)
        self.log.info(self.__class__.__name__ + " parameters:")
        for key, val in self.opts.items():
            self.log.info("  > %-28s %r", key + ":", val)

        # k-space objects
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)

        # Set attributes:
        self.mol = mf.mol
        self.mf = mf
        self.mo_energy = mo_energy if mo_energy is not None else mf.mo_energy
        self.mo_coeff = mo_coeff if mo_coeff is not None else mf.mo_coeff
        self.mo_occ = mo_occ if mo_occ is not None else mf.get_occ(self.mo_energy, self.mo_coeff)
        self.eri = eri
        self.e = None
        self.v = None
        self._nmo = None
        self._nocc = None
        self.chkfile = self.opts.chkfile or self.mf.chkfile

        # Print system information:
        self._print_sizes()

        # ERIs:
        if self.eri is not None:
            if self.eri.ndim == 3:
                self.log.info("ERIs passed by kwarg and will be density fitted")
            else:
                self.log.info("ERIs passed by kwarg and will be four centered")
        else:
            self.ao2mo()

    _print_sizes = KRAGF2._print_sizes
    ao2mo = KRAGF2.ao2mo

    def build(self, eri=None):
        """Build the matrix-vector product function and the diagonal
        of the matrix.
        """

        if eri is None:
            eri = self.eri

        kconserv = self.khelper.kconserv
        kp = 0

        nocc = [np.sum(occ > 0) for occ in self.mo_occ]
        nvir = [np.sum(occ == 0) for occ in self.mo_occ]
        ei = [mo[occ > 0] for mo, occ in zip(self.mo_energy, self.mo_occ)]
        ea = [mo[occ == 0] for mo, occ in zip(self.mo_energy, self.mo_occ)]

        if self.opts.which != "ip":
            ei, ea = ea, ei
            nocc, nvir = nvir, nocc

        if self.opts.non_dyson:
            phys = slice(None, nocc[kp])
            aux = slice(nocc[kp], None)
        else:
            phys = slice(None, nocc[kp]+nvir[kp])
            aux = slice(nocc[kp]+nvir[kp], None)

        def kpt_iter(mpi=True):
            lst = mpi_helper.nrange(self.nkpts**2) if mpi else range(self.nkpts**2)
            for kqr in lst:
                kq, kr = divmod(kqr, self.nkpts)
                ks = kconserv[kp, kq, kr]
                yield kq, kr, ks

        naux = np.zeros((self.nkpts, self.nkpts), dtype=int)
        for kq, kr, ks in kpt_iter():
            naux[kq, kr] = nvir[kq] * nocc[kr] * nvir[ks]

        h1 = 0.0
        for kq, kr, ks in kpt_iter():
            Δ = 1.0 / lib.direct_sum("x-a+i-b->xaib", ei[kp], ea[kq], ei[kr], ea[ks])

            Lia = eri[kp, kq, :, phys, nocc[kq]:]
            Ljb = eri[kr, ks, :, :nocc[kr], nocc[ks]:]
            iajb = lib.einsum("Lia,Ljb->iajb", Lia, Ljb)

            Lib = eri[kp, ks, :, phys, nocc[ks]:]
            Lja = eri[kr, kq, :, :nocc[kr], nocc[kq]:]
            ibja = lib.einsum("Lib,Lja->iajb", Lib, Lja)

            h1 += lib.einsum("iakb,jakb,iakb->ij", iajb, iajb, Δ)
            h1 -= lib.einsum("iakb,jakb,iakb->ij", iajb, ibja, Δ) * 0.5

        mpi_helper.barrier()
        h1 = mpi_helper.allreduce(h1)
        h1 = lib.hermi_sum(h1)

        if self.opts.non_dyson:
            h1 += np.diag(ei[kp])
        else:
            h1 += np.diag(self.mo_energy[kp])

        def matvec(v):
            # +---+-------+   +---+       +---+
            # | h |   v   |   | x |       | α |
            # +---+-------+   +---+       +---+
            # |   |       |   |   |  -->  |   |
            # | u |   e   |   | y |       | β |
            # |   |       |   |   |       |   |
            # +---+-------+   +---+       +---+

            x = v[phys]
            y = v[aux]

            hx = np.dot(h1, x)
            vy = np.zeros(x.shape, dtype=np.complex128)
            ux = np.zeros(y.shape, dtype=np.complex128)
            ey = np.zeros(y.shape, dtype=np.complex128)

            for kq, kr, ks in kpt_iter():
                p0 = sum(naux.ravel()[:kq*self.nkpts+kr])
                p1 = sum(naux.ravel()[:kq*self.nkpts+kr+1])

                Lij = eri[kp, kq, :, :nocc[kp], :nocc[kq]]
                Lak = eri[kr, ks, :, nocc[kr]:, :nocc[ks]]
                ijak = lib.einsum("Lij,Lak->ijak", Lij, Lak)

                Lik = eri[kp, ks, :, :nocc[kp], :nocc[ks]]
                Laj = eri[kr, kq, :, nocc[kr]:, :nocc[kq]]
                ikaj = lib.einsum("Lik,Laj->ijak", Lik, Laj)

                e_iaj = lib.direct_sum("i-a+j->iaj", ei[kq], ea[kr], ei[ks])

                vy += lib.einsum("kiaj,iaj->k", ijak, y[p0:p1].reshape(ijak.shape[1:])) * 2.0
                vy -= lib.einsum("kiaj,iaj->k", ikaj, y[p0:p1].reshape(ikaj.shape[1:]))

                ux.ravel()[p0:p1] += lib.einsum("kiaj,k->iaj", ijak.conj(), x).ravel()

                ey.ravel()[p0:p1] += e_iaj.ravel() * y.ravel()[p0:p1]

            mpi_helper.barrier()
            vy = mpi_helper.allreduce(vy)
            ux = mpi_helper.allreduce(ux)
            ey = mpi_helper.allreduce(ey)

            return np.concatenate([hx + vy, ux + ey], axis=0)

        d = [np.diag(h1)]
        for kq, kr, ks in kpt_iter(mpi=False):
            d.append(lib.direct_sum("i-a+j->iaj", ei[kq], ea[kr], ei[ks]).ravel())
        diag = np.concatenate(d)

        return matvec, diag

    def kernel(self):
        """Solve ADC.
        """

        t0 = timer()

        matvec, diag = self.build()
        matvecs = lambda xs: [matvec(x) for x in xs]

        guesses = np.zeros((self.opts.nroots, diag.size), dtype=diag.dtype)
        arg = np.argsort(np.absolute(diag))

        for root, guess in enumerate(arg[:self.opts.nroots]):
            guesses[root, guess] = 1.0
        guesses = list(guesses)

        def pick(w, v, nroots, envs):
            w, v, idx = lib.linalg_helper.pick_real_eigs(w, v, nroots, envs)
            mask = np.argsort(np.abs(w))
            return w[mask], v[:, mask], idx

        self.conv, self.e, self.v = lib.davidson_nosym1(
                matvecs,
                guesses,
                diag,
                pick=pick,
                nroots=self.opts.nroots,
                tol=self.opts.conv_tol,
                max_cycle=self.opts.max_cycle,
                max_space=self.opts.max_space,
                verbose=0,
        )

        if self.opts.which == "ip":
            self.e *= -1

        self.print_excitations(e=self.e, v=self.v, output=True)

        if self.opts.dump_chkfile and self.chkfile is not None:
            self.log.debug("Dumping output to chkfile")
            self.dump_chk()

        self.log.info("Time elapsed:  %s", time_string(timer() - t0))

        return self.conv, self.e, self.v

    def print_excitations(self, e=None, v=None, title="Excitations", output=False):
        """Print the excitations and some information on their character.
        """

        if e is None:
            e = self.e
        if v is None:
            v = self.v

        log = self.log.info if not output else self.log.output

        log(title)
        log("*" * len(title))
        self.log.changeIndentLevel(1)

        if self.opts.non_dyson and self.opts.which == "ip":
            phys = slice(None, np.sum(self.mo_occ[0] > 0))
        elif self.opts.non_dyson and self.opts.which == "ea":
            phys = slice(None, np.sum(self.mo_occ[0] == 0))
        elif not self.opts.non_dyson:
            phys = slice(None, self.mo_occ[0].size)

        self.log.info("%2s %12s %12s %s", "", "Energy", "QP weight", " Dominant MOs")
        for n in range(min(len(e), self.opts.excitation_number)):
            en = e[n]
            vn = v[n][phys]
            qpwt = np.linalg.norm(vn)**2
            char_string = ""
            num = 0
            for i in np.argsort(vn**2)[::-1]:
                if vn[i]**2 > self.opts.excitation_tol:
                    if num == 3:
                        char_string += " ..."
                        break
                    char_string += "%3d (%7.3f %%) " % (i, np.abs(vn[i]**2)*100)
                    num += 1
            log("%2d %12.6f %12.6f  %s", n, en, qpwt, char_string)

        self.log.changeIndentLevel(-1)

    def dump_chk(self, chkfile=None, key="kadc2"):
        """Save the calculation state.
        """

        chkfile = chkfile or self.chkfile

        if chkfile is None:
            return self

        lib.chkfile.dump(chkfile, "%s/conv" % key, self.conv)
        lib.chkfile.dump(chkfile, "%s/e" % key, self.e)
        lib.chkfile.dump(chkfile, "%s/v" % key, self.v)

        return self

    @property
    def frozen(self):
        return getattr(self, '_frozen', [(0, 0),]*self.nkpts)
    @frozen.setter
    def frozen(self, frozen):
        if frozen == None or frozen == 0 or frozen == (0, 0):
            self._frozen = (0, 0)
        elif isinstance(frozen, int):
            self._frozen = (frozen,  0)
        else:
            self._frozen = frozen
        if isinstance(self._frozen, tuple):
            self._frozen = [self._frozen,] * self.nkpts

    @property
    def cell(self): return self.mol

    @property
    def act(self): return [slice(None)] * self.nkpts
    @property
    def froz(self): return [slice(0, 0)] * self.nkpts

    @property
    def nmo(self): return self._nmo or self.mo_occ[0].size
    @property
    def nocc(self): return self._nocc or [np.sum(x > 0) for x in self.mo_occ]
    @property
    def nvir(self): return [self.nmo-x for x in self.nocc]
    @property
    def nfroz(self): return [f[0]+f[1] for f in self.frozen]
    @property
    def nact(self): return [self.nmo-x for x in self.nfroz]
    @property
    def nkpts(self): return len(self.kpts)
