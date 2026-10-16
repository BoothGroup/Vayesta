# Vayesta code review

Scope: the whole `vayesta/` package (~37k lines) plus tests, CI and packaging, at commit `f64be6c`.
Method: a manual read of the core (`core/qemb`, `ewf`, `core/util`, `core/types/cluster.py`, `solver/solver.py`, bath base classes), parallel reviews of the other subpackages, and static/runtime verification of every claim listed here. "Run" means the behaviour was reproduced with a script; everything else was traced in the source. Items are ranked within each section.

## 1. Executive summary

The physics code is largely sound; the problems are structural. Four themes account for most findings:

1. **Everything holds a reference to everything.** `Embedding` ↔ `Fragment` ↔ `Bath` ↔ `ClusterHamiltonian` ↔ `ClusterSolver` are mutually linked and reach through each other (`hamil._fragment.base.get_fock()`, `bath.fragment.base.opts.bath_options[...]`). No layer below `Embedding` can be constructed or tested from plain integrals.
2. **Restricted and unrestricted are parallel class hierarchies joined by diamond inheritance** (`UEWF(REWF, UEmbedding)`, `UDMET(UEmbedding, RDMET)`, …). This has already produced silent method shadowing, R/U behavioural drift and copy-paste bugs in the U variants (see §3).
3. **Control flow by monkey-patching.** `optimize_chempot`, `pdmet_scmf`, `brueckner_scmf`, `qpewdmet_scmf`, `ClusterSolver.optimize_cpt` and `Embedding(overwrite=...)` all replace `self.kernel` on the instance; DMET/EDMET instead re-implement the whole kernel loop. Three self-consistency mechanisms coexist and are not composable.
4. **Shared mutable state in the helpers.** Option-dict defaults are shared between all instances, method caches are class-level, `log_method` freezes the first caller's logger, and `import vayesta` rewrites `sys.argv`, mutates the stdlib `logging` module and writes log files into the working directory.

About 60 concrete bugs were confirmed; the most consequential are the ones that silently change numbers (§3.1) rather than the ones that crash.

## 2. Design and dependency problems

### 2.1 Coupling between layers

| Layer | Depends on | Evidence |
|---|---|---|
| `ClusterHamiltonian` | `Fragment` and, through it, `Embedding` | `solver/hamiltonian.py:132,185` (`self._fragment.base.get_fock/get_eris_array`), `:118` (`fragment.get_fragment_projector`) |
| `ClusterSolver` | `ClusterHamiltonian` → `Fragment` → `Embedding`; results are side effects (`wf`, `converged`, `civec`, `callback_results`) read back by name-matching the solver string in `ewf/fragment.py:211-300` | `solver/solver.py:17-37`, `coupling.py:103-110` even uses `solver.base/cluster/fragment`, which no solver has |
| `Bath` classes | `fragment.base.*` including private `_check_orthonormal`, `opts.bath_options`, `fragment._dmet_bath` | `core/bath/dmet.py:49-50`, `bosonic_bath/qba_rpa.py:127,172` |
| `Fragmentation` | the full `Embedding` (`register`, `Fragment`, `symmetry`, `fragments`, `create_*sym_fragments`) | `core/fragmentation/fragmentation.py:36-96` |
| `ewf.fragment` | `ewf.ewf` (circular import, `ewf/fragment.py:29`, only used by dead counterpoise code) | |

The real inputs of a bath are `S`, `F`, `dm1`, `c_frag`, `c_env`, an integral getter and two options; of a solver, `(h_eff, eris/cderi, nelec, projector)`. Passing these explicitly (with `Fragment.make_bath`/`get_solver` as the only adapters) would remove most of the back-references and make the pieces unit-testable.

### 2.2 Restricted/unrestricted split

* `UDMET`'s MRO is `[UDMET, UEmbedding, DMET, Embedding]`, so `UEmbedding.get_corrfunc` shadows `DMET.get_corrfunc` (run). The diamonds only work because `UEmbedding`/`UFragment` happen to define no `Options`/`Results`; adding one would silently shadow DMET/EWF options.
* Drift: `UEmbedding.check_fragment_symmetry` raises `RuntimeError` while the base raises `SymmetryError`, and `core/scmf/scmf.py:112` catches only the latter, so UHF self-consistency crashes where RHF recovers. `UEmbedding._check_fragment_nelectron` has a different signature and semantics from the base. `DMETFragment.get_dmet_energy_contrib` (R) uses an uncached `self.mf.get_fock()` while the U version uses the cached `base.get_fock()`.
* 26 ad-hoc spin checks of the form `np.ndim(mo_coeff[0]) == 1` are scattered over 12 modules although `spinsym` exists.
* Near-copies: `ssrpa.py`/`ssurpa.py`, `RIRPA.py`/`RIURPA.py`, `ebfci_slow.py`/`uebfci_slow.py` (~80% identical), `ewf/rdm.py`/`urdm.py`, `UMP2_BNO_Bath._make_t2`, the `compress_cderi` closure (copied with a bug, §3.1).

`core/spinalg.py` already provides the tuple-aware primitives; most R/U pairs could collapse to one spin-generic implementation.

### 2.3 Options plumbing

* Every `Fragment` deep-copies the complete `Embedding.Options` at construction (`core/qemb/fragment.py:200`), and `Fragment.Options` mirrors `Embedding.Options` with `None` meaning "inherit". `Embedding.change_options` has to propagate by hand.
* `OptionsBase.dict_with_defaults` returns the *same* dict object for every instance (run): `emb.opts.bath_options["threshold"] = x` changes the defaults of every later `Embedding` in the process. `asdict()` returns `self.__dict__` itself.
* `OptionsBase.update` silently drops unknown keys (`core/util.py:704`). Consequences found: `tcc_fci_opts` vs `fci_opts` (TCCSD FCI options never applied), `filename` vs `dumpfile`, `sc_mode`, `coupled_fragments`, EBFCI `max_cycle`, EDMET `Options.solver` are all dead. `get_solver_options` is a string-matching translation layer ("TODO: fix this mess").
* Two unrelated symmetry weights, `opts.sym_factor` (user) and `symmetry_factor` (children count), are multiplied in some energy paths (`ewf.py:422,465`) and not in others (`get_wf_corr_energy`, `get_fbc_energy`).

### 2.4 Kernel replacement and self-consistency

`qemb.py:1680-1752` and `solver/solver.py:193` rebind `self.kernel`; `SCMF.__init__` captures `emb.kernel` at construction, so the outcome depends on the order in which `optimize_chempot()` and `*_scmf()` are called. `DMET.kernel` and `EDMET.kernel` are ~100 copy-pasted lines of chemical-potential bracketing that have already diverged (§3), while `Embedding.optimize_chempot` + `misc/cptbisect.py` implement the same search a third time. A hook-based `Embedding.kernel` (prepare → run fragments → energy) with DMET/EDMET overriding hooks, one `ChempotOptimizer`, and SCMF as a plain outer loop would replace all of this.

### 2.5 Library hygiene

* `import vayesta` parses and **rewrites `sys.argv`** (`core/cmdargs.py:50-53`: `-v`, `-q`, `-o`, `--log`, `--mpi` are stolen from the host program), initialises MPI, attaches `vlog.txt`/`verr.txt` file handlers in the CWD (run), renames stdlib level 30 to `DEPRECATED` process-wide and sets `logging.FATAL = 100` (run), calls `numpy.set_printoptions`, and dies with `FileNotFoundError` if `git` is not on `PATH` (`__init__.py:108-115` catches only `CalledProcessError`).
* `util.cache` wraps methods in a class-level `functools.lru_cache(16)` keyed on `self` (run): the 16-entry cache is shared by all fragments (useless beyond a handful), `fx.get_overlap.cache_clear()` in `Fragment.reset` clears every fragment, and `EWF._make_rdm1_ccsd_global_wf_cached` keeps up to 16 whole embeddings alive.
* `log_method` uses `nonlocal message, log` and so bakes in the first caller's class name and logger (run: a second embedding with its own `log=` logs into the first one's).
* `VFormatter.format` does `record.msg % record.args` itself (`vlog.py:156`): `log.info("50% done")` raises, and `exc_info`/`log.exception` tracebacks are dropped. `log.deprecated` is `lru_cache`-wrapped, so array arguments raise `TypeError`.
* The MPI abstraction leaks through `if mpi:` truthiness (56 call sites) and import-time decorators.

## 3. Bugs

### 3.1 Silently wrong numbers (highest priority)

| Where | Problem |
|---|---|
| `solver/hamiltonian.py:737` | UHF CD-ERI compression: `(compress_cderi(cderi[0]), compress_cderi(cderi[0]))` replaces the beta block with alpha. Trigger: `UMP2` with `compress_cderi=True`. |
| `rpa/rirpa/__init__.py:28` + `edmet/edmet.py:232,389` | The factory returns the dRPA-only class unless `rixc` is a *keyword*; EDMET passes the xc kernel positionally, so DF-EDMET self-consistency runs plain dRPA every iteration. |
| `rpa/rirpa/RIRPA.py:208`, `RIdRRPA.py:26,104` | `kernel_moms` ignores `alpha`; adiabatic-connection integration is evaluated at α=1 at every point. |
| `rpa/rirpa/RIRPA.py:780-789` | `compress_low_rank` uses `eig` + `rot.T` on non-symmetric factors (every `rixc` case): complex/negative modes dropped, `rot.T` is not the inverse (run: rank-5 test loses 0.4 in max error). |
| `rpa/ssrpa.py:196-202` | `_gen_eps` mutates `self.ov_rot` on every call; the second `eigh` flips signs so `ApB` and `full_mom0` are in different bases when `ov_rot` and `xc_kernel` are both used. `screening_crpa.py:211` relies on this mutation. |
| `ewf/ewf.py:360` | `get_dm_corr_energy` passes `t_as_lambda=None, with_exxdiv=None` to the 1-DM term instead of its arguments; `energy_functional="dm-t2only"` therefore mixes Λ for the 1-DM part with T for the 2-DM part. |
| `core/qemb/qemb.py:1120-1135` | `get_fragments(options=..., flags=...)`: the `skip` flag from the options loop is overwritten by the flags loop, so fragments failing the options filter are returned (run). |
| `core/bath/bno.py:41-43` | `truncation="electron-percent"`/`"truncation"` compare per-spin BNO occupations with the spin-summed electron count: the allowed loss is twice the requested fraction. |
| `core/bosonic_bath/projected_interactions.py:224-244` | PBC DF branch has no `return`; execution falls into the non-PBC loop and exchange couplings are accumulated twice (compare `core/eris.py:56`). |
| `solver/coupling.py:655-684` | External-correction amplitudes are divided by energy denominators iff the *last* correction in the list is `"external"`; mixing `delta-tailor` and `external` gives wrong corrections. |
| `solver/hamiltonian.py:612,650` | `UClusterHamiltonian` ignores `match_fock` (uses screened/bare ERIs the other way round from the R class), so R and U build different cluster Hamiltonians with screening. |
| `dmet/dmet.py:161,173`, `edmet/edmet.py:148,160` | After a CCSD `LinAlgError` the bracket step `cpt - (new_cpt - cpt)*3/8` has the wrong sign and walks away from the root. |
| `solver/cisd.py:34`, `solver/ebfci.py`, `edmet/fragment.py:729` | `converged = True` is hard-coded (CISD), never set (EBFCI) or forced (EDMET); unconverged clusters seed CCSD/FCI silently. |
| `lattmod/latt.py` (`get_index` vs `atom_coords`/`get_tiles_order`) | `Hubbard2D` builds `h1e` with the first dimension slow but coordinates/tiles with it fast; on rectangular lattices tiles and translation symmetry describe a different lattice than the Hamiltonian (run: 24 of 56 neighbour pairs disagree for `(2,4)`). Only square lattices are tested. |
| `solver/hamiltonian.py:874-879` | Polaritonic spin-symmetry breaking is logged at `critical` and execution continues with the alpha shift; the sibling check raises. |
| `rpa/rirpa/RIdRRPA.py:124,177` | Numerical-integration error estimate discarded (`(None, None)` returned); `check_errors` never fires on the default RHF route. `NI_eval.py:290-297` "nested" error estimate uses non-nested grids. |
| `tests/common.py:18-21` | `assertAllclose(rtol=allclose_atol, atol=allclose_rtol)` is cross-wired and bound at definition time, so subclass tolerance overrides are ignored; 278 of 358 calls rely on these defaults. |

### 3.2 Crashes and dead paths

| Where | Problem |
|---|---|
| `core/types/wf/ccsd.py:390` | `[lambda c: dot(c, x) for x in (...)]` captures `x` late: both spins use the beta rotation (run: `ValueError` when n_α ≠ n_β). Reached via `UCCSD_WaveFunction.rotate` from `solver/ebcc.py:289,367`. |
| `core/scmf/scmf.py:106` | Calls `emb.get_e_corr()`, which only EWF defines; `pdmet_scmf`/`brueckner_scmf` crash with DMET/EDMET. |
| `edmet/edmet.py:40` + `edmet/fragment.py:741` | Default `oneshot=False` forces `make_dd_moments`, but `wf.make_dd_moms` exists only for EBFCI; EDMET with any EBCC solver and default options raises. Tests pass `oneshot=True`. |
| `edmet/ufragment.py:169-171` | Both spins reshape to `(nbos, nocc[0], nvir[1])`; any open-shell UHF EDMET raises. The UHF test is closed-shell. |
| `edmet/edmet.py:241,253` | Zero-boson fallback arrays are 2-D but indexed `[0, sl, :]`. |
| `dmet/dmet.py:178-180`, `edmet/edmet.py:165` | "Could not find bracket" `break`s the outer loop with `e_corr=None` (`TypeError` later) / doesn't break and feeds `brentq` a non-bracket. |
| `core/bath/rpa.py:16` | `RPA_BNO_Bath` inherits `get_finite_bath_correction`, which needs `_make_t2` (MP2-only): `bathtype="rpa"` + `get_fbc_energy()` raises. |
| `core/bath/bno.py:214-219` | `addbuffer=True` is RHF-only but reachable from UHF (`c_env_occ` is a tuple). Also makes the virtual-BNO MP2 allocate a full-system `t2`. |
| `core/bath/dmet.py:253-270` | Calls `self.project_ref_orbitals` on the bath; the method lives on `Fragment`. The `c_ref`, `nbath`, `make_dmet_bath_fast` and `transform_mp2_eris` paths are dead. |
| `core/screening/screening_crpa.py:79-80` | cRPA fallback returns a 4-tuple of full-system `nmo⁴` zeros vs the 3-tuple of cluster arrays on the normal path → `MemoryError`/broadcast error in `hamiltonian.py:501`. |
| `core/bosonic_bath/qba_rpa.py:188-195` | Bosonic SVD via Gram matrix + `occup**-0.5` → `inf`/`nan` for zero eigenvalues. |
| `core/types/cluster.py:15` | `Cluster.__init__` ignores its `bosons` argument; `copy()`/`basis_transform(inplace=False)` on a bosonic cluster raises. |
| `ewf/ewf.py:379` | `_make_rdm1_ccsd_1p1l` does not exist (`dm1="1p1l"`). |
| `ewf/fragment.py:555-605` | Counterpoise/BSSE uses `self.bno_threshold` and `ecc.make_atom_cluster`, neither of which exists. |
| `ewf/fragment.py:112` | `Fragment._reset` is never called (base method is `reset`), so CAS orbitals are never cleared between iterations. |
| `core/qemb/qemb.py:1236` | `_absorb_fragments` resets `_occ_bath_factory/_vir_bath_factory`; the real attributes are `_bath_factory_occ/_vir`. |
| `ewf/icmp2.py:125,136` | Fallback uses `x.bath`, which no fragment has. |
| `solver/coupling.py:103-110`, `coupled_ccsd.py` | `coupledCCSD` uses `solver.base/cluster/fragment` and `cc.conv_flag/force_iter` which do not exist; dead but still registered in `solver/__init__.py`. |
| `solver/callback.py:38-48` | Unrecognised callback results → `UnboundLocalError` on `wf`. |
| `solver/tccsd.py:25` + `hamiltonian.py:552` | `with_new_cluster` copies `screening` but not `_seris`; `TCCSD` + screening raises. |
| `solver/solver.py:111,146` | `optimize_cpt` assigns `results` but returns `result` (always `None`) and never warm-starts (`get_init_guess` returns `{}`). |
| `solver/_uccsd_eris.py:34-47` | Memory guard against pyscf's default `max_memory` refuses UCCSD clusters above ~145 orbitals although the dense ERIs already exist. |
| `rpa/rirpa/RIRPA.py:427,307` | `get_apb_eri_ri() / np.sqrt(2)` divides an `(array, None)` tuple; `2 * niworker.kernel_adaptive()` doubles a tuple. `direct_AC_integration`/`adaptive_quad` are dead. |
| `core/util.py:350` | `einsum(..., driver=...)` never pops `driver`, so it is forwarded to `np.einsum` and rejected. |
| `core/ao2mo/postscf_ao2mo.py:24` | Element-wise version check: pyscf 3.0.0 compares as "older than 2.1"; dev suffixes raise at import. |
| `core/ao2mo/kao2gmo.py:162` | `tensordot(phase, cderi_mo, axes=1)` contracts the wrong axis; correct only because `phase` is symmetric for unshifted meshes. |
| `mpi/scf.py:531` | `mpi.gdf()` raises for pyscf's default `_cderi_to_save` (a `NamedTemporaryFile`). |
| `core/symmetry/*` | `Symmetry` class fails for molecules and is unused; `group.py:83` clears the wrong attribute; `SymmetryTranslation.inverse` passes `mol` as `group`; `tsymmetry.py` is dead and rounds before dividing; `operation.py:303` treats `"open"` boundaries as anti-periodic. |
| Format strings with missing arguments | `qemb.py:771`, `core/qemb/fragment.py:346,1179`, `rpa/ssrpa.py:181`. |
| Misc | `qemb.py:120` default key `addbuffer_dmet_vir` (should be `addbuffer_vir`); `fragment.py:938` accepts `"rpacorr"` that `make_bath` never creates; duplicate `elif symtype == "inversion"` at `qemb.py:882`; `make_cluster` occupation checks are `assert`s; `EWF.kernel` swallows `AttributeError` from the energy evaluation (`ewf.py:185`) and reports `nan`; `Fragment.kernel` returns `None` for solver `"HF"` without setting results, so `_all_converged` fails; `dmet.py:283` `print_clusters` uses nonexistent `frag.size`; `examples/edmet/01-1d-hubbard-comparison.py:36` calls nonexistent `make_tsymmetric_fragments`; `vlog.py` and `latt.py` small issues listed by the infrastructure review (`Hubbard1D.get_index(int)`, invalid boundary → `UnboundLocalError`, `gto_helper.py:325` missing `continue`, `tools/eos.py:171` `NameError`). |

### 3.3 Wavefunction / RDM layer

_(pending: findings from the review of `core/types`, `core/qemb/rdm.py`, `ewf/rdm.py`, `ewf/urdm.py`, `ewf/amplitudes.py`, `ewf/icmp2.py`, `ewf/ccsd_t.py`, `core/vpyscf` will be added here)_

## 4. Algorithmic and performance issues

1. **Density-fitting blocking is non-functional**: `blksize = int(1e9 / naux * nao * nao * 8)` (`core/eris.py:41,92`, `bosonic_bath/projected_interactions.py:222`) is `(1e9/naux)·nao²·8` instead of `1e9/(nao²·8)`, so the whole `(naux, nao, nao)` tensor is unpacked at once (≈3 GB at nao=500, naux=1500).
2. **Dense 4-index cluster ERIs everywhere**: `get_fragment_energy`, `get_fragment_dmet_energy`, the Hamiltonian's `get_fock` screening correction and `get_heff` all go through `get_eris_bare()` (`N_cl⁴` memory, `N_cl⁴·N_aux` time) even when `cderi` is available. With `store_eris=False` the AO→MO transform is repeated per block and per pyscf `get_fock` call (`hamiltonian.py:142,189,366`).
3. **MP2 bath stores the full `t2`** (`bno.py:326`, `nocc_cl²·N_vir²`) although it is built block-wise; the natural-orbital density could be accumulated per block. Two `N⁵` einsums where one (`2t − tᵀ`) suffices (`bno.py:451`).
4. **RPA bath/EDMET recompute full-system `(L|ov)`** per fragment and per occ/vir (`bath/rpa.py:210`), and `RIRPA.kernel_energy` with `lov=None` transforms three times (five per EDMET iteration); `target_rot = einsum("ij,ab->iajb", eye, eye)` materialises an `O(N³)`-element identity.
5. **Chemical-potential loops**: `optimize_cpt` re-solves cold at every `brentq` step (10-30× solver cost); DMET's `DMETFragment.kernel` rebuilds the cluster per evaluation and calls the uncached `self.mf.get_fock()` (a full J/K build) per fragment per evaluation (`dmet/fragment.py:159`); one extra full solver pass after convergence (`dmet.py:189`).
6. **EDMET non-DF** `get_eri_couplings` does a full-system `N⁵` transform per fragment per call at three call sites per iteration, keeping only `(ov|ov)`; the xc kernel is three dense `nao⁴` arrays with no size guard.
7. **Repeated solves**: `delta-tailor` re-solves fragment *y*'s CCSD inside every consumer *x* (`coupling.py:506`).
8. Smaller: `Fragmentation.__exit__` runs each `O(N³)` completeness check twice (`fragmentation.py:88-96`); `e_mf` recomputes `mf.energy_tot` on every access and `e_tot` re-evaluates it; `Fragment.nelectron` rebuilds `mf.make_rdm1()` per call inside loops; `ssrpa.py:238` forms full `nao⁴` MO ERIs to slice `(ov|ov)`; `projected_interactions.py:169` builds an `nao²·ncl²` intermediate; finite-bath correction re-transforms integrals that could be rotated.

## 5. Tests, CI and packaging

* Fast suite passes here without the C extension (373 passed, 56 skipped); the three `kao2gmo driver="c"` tests fail with `AttributeError` on `None` instead of a clear "libcore not built" message.
* `veryslow` tests never run anywhere (`addopts -m 'not veryslow'`, CI never passes `--with-veryslow`); `test_self_energy.py` (3) and `test_tailoring.TestExternalCorrection` (6, "TODO: replace test values") are permanently skipped; 45 tests are gated on `ebcc`/`dyson`; there is no MPI test and the `mpi` extra is never installed in CI.
* Nothing exercises: open-shell UHF EDMET, `maxiter>1` EDMET, rectangular Hubbard2D, `bathtype="rpa"` with FBC, `electron-percent` truncation, `get_fragments(options=, flags=)`, SCMF with DMET, the symmetry-fragment creation API, `optimize_chempot` outside SCMF, or most of the public energy/DM methods of `EWF` listed by the test review.
* Lint and type gates cannot catch real errors: ruff ignores `F401, F811, F821, F841` (undefined names currently pass, but the gate is off) and pyright downgrades `reportUndefinedVariable`, `reportAttributeAccessIssue`, `reportCallIssue` to warnings and excludes tests.
* Packaging: `cvxpy` is both a hard dependency and the `dmet` extra (the extra only adds an upper bound; `pytest.skip("Requires cvxpy")` guards are dead); `setup.py` imports the removed-in-future `setuptools.command.test`; `CMAKE_BUILD_ARGS` is appended to the configure args after configure ran and never reaches `--build`; `libs/df` and `libs/ewf` C sources are shipped but never built and have no callers; `j3c_jk.c` is compiled and uncalled; the sdist ships 900+ test files and GMTKN55 data but not `examples/`; three different version strings (`1.0.1`, docs `1.0.0`, CHANGELOG `1.0.2` dated 2023); no macOS CI job despite darwin-specific code paths.
* `testsystems.py` builds ~60 molecules at import and memoises mean-fields with `functools.cache` on methods (one global cache per method, no disk persistence, so every xdist worker recomputes); every `import vayesta` writes `vlog.txt` into the CWD.

## 6. Recommendations (in order of leverage)

1. **Fix the silent-number bugs in §3.1 first** and add a regression test for each (UMP2 `compress_cderi`, DF-EDMET with xc kernel, `get_fragments` filters, `electron-percent`, rectangular Hubbard2D, UHF EBCC rotation, `dm-t2only`).
2. **Make `OptionsBase` safe**: `default_factory=lambda: dict(kwargs)`, copying `asdict`, and `update` that raises (or validates once) on unknown keys. This alone exposes the dead options.
3. **Replace `util.cache`/`log_method`** with per-instance caches (`cached_property` or an explicit dict with invalidation) and a per-call logger lookup.
4. **Move import-time behaviour into an explicit `vayesta.configure(...)`**: no argv parsing, no file handlers, no stdlib `logging` mutation (use `setLoggerClass`, keep level 30 as WARNING), no `git` subprocess at import.
5. **Decouple the lower layers**: give `Bath`, `ClusterHamiltonian` and `ClusterSolver` explicit inputs and typed results (`SolverResult(wf, converged, moments, extras)`), and make the Fragment the only adapter. Unify `get_bath` signatures behind one `Truncation` object so `make_cluster` stops branching on `bathtype`.
6. **One self-consistency mechanism**: a hook-based `Embedding.kernel`, a single `ChempotOptimizer` shared by `optimize_chempot`, DMET and EDMET, and SCMF as an outer loop that does not rebind methods.
7. **Collapse R/U pairs** onto spin-generic code using `spinalg`, starting with `rdm.py`/`urdm.py`, `ssrpa`/`ssurpa`, `RIRPA`/`RIURPA` and the EB-FCI kernels; replace the 26 `np.ndim(...)` spin checks with `spinsym`.
8. **Delete dead code**: counterpoise/BSSE, `coupledCCSD`, `direct_AC_integration` and the unused `calc_energy_correction` versions, `make_dmet_bath_fast`/`c_ref`/`nbath` bath paths, `tsymmetry.py` and the `Symmetry` class, `core/helper.py`, the unbuilt `libs/df`, `libs/ewf` sources, `DMETResults`/`EDMETResults`, and the commented-out blocks.
9. **Tighten the gates**: enable `F821`/`F841` in ruff, promote the pyright rules to errors module by module, fix `assertAllclose`, run `veryslow` nightly, add an MPI job and a macOS job, and make the `dmet` extra either real or removed.
