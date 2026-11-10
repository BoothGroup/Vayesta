# RIRPA in Vayesta: current state, cost, and a plan for HHT/CDERI improvements

Notation used throughout:

| symbol | meaning |
| --- | --- |
| `o`, `v` | occupied / virtual orbitals of the full system |
| `ov` | number of spatial particle-hole pairs, `o*v` (spin-orbital codes use `2ov`) |
| `N_aux` | number of auxiliary (density-fitting) functions; `O(N)` |
| `L` | the occupied-virtual CDERIs, shape `(N_aux, ov)` |
| `D` | diagonal of orbital-energy differences `e_a - e_i`, length `ov` |
| `n_t` | number of rows in `target_rot`, the space the moment is projected onto |
| `N_p` | number of quadrature points (Clenshaw-Curtis) or poles (HHT) |
| `N_frag` | number of fragments in an embedding calculation |

All scaling statements count floating-point work only. Nothing here was run for this note.
The measured numbers quoted in section 3 come from the `monomial_GW` and `Davidson-gw`
records, and the note names the source each time.

---

## 1. Vayesta as a whole

Vayesta is a Python quantum-embedding package built on PySCF. It partitions a molecule or
solid into fragments, builds a cluster for each fragment (fragment orbitals plus a bath),
solves each cluster with a correlated solver (CCSD, FCI, MP2, EBFCI, ...), and assembles
global properties from the fragment results. The main subpackages are:

- `core/`: fragmentation, bath construction (`core/bath/`: DMET, MP2 and RPA bath natural
  orbitals, bosonic baths), integrals (`core/eris.py`, where `get_cderi` lives), screening
  (`core/screening/`), and the generic embedding driver `core/qemb/`.
- `ewf/`, `dmet/`, `edmet/`: the embedding methods themselves (wave-function embedding,
  DMET, and extended/bosonic DMET).
- `solver/`: cluster solvers, including the electron-boson ones that EDMET needs.
- `rpa/`: full-system RPA. `rpa.py`, `ssrpa.py` and `ssurpa.py` are dense `O(N^6)`
  reference implementations. `rirpa/` is the density-fitted, moment-based code that the
  embedding layers use in production.

RIRPA matters because it supplies **global long-range information to local clusters**. It is
called from four places:

| caller | what it asks RIRPA for | `target_rot` |
| --- | --- | --- |
| `core/screening/screening_moment.py` (`mrpa` screened interactions, cRPA) | zeroth dd moment, then only `T_f eta0 T_f^T` per fragment | stacked cluster ph spaces, spin-orbital, `n_t = sum_f ov_f^clus` |
| `core/bath/rpa.py` (RPA bath natural orbitals) | zeroth moment, then `m0 T^T` | `rot_occ (x) rot_vir`: `o * v_clus` or `o_clus * v` rows, **one RIRPA object per fragment** |
| `core/qemb/qemb.py` (bosonic bath; external dRPA correction) | zeroth moment / total energy | bosonic targets; or `L` itself (the cumulant energy) |
| `edmet/edmet.py` | zeroth moment and energy **with an xc kernel** (`rixc`) | cluster ph and bosonic spaces |

---

## 2. How `vayesta/rpa/rirpa` is implemented

### 2.1 Files and classes

| file | contents |
| --- | --- |
| `__init__.py` | `ssRIRPA(mf, ...)` factory: UHF goes to `ssRIURPA`, a non-`None` `rixc` goes to `ssRIRRPA`, and everything else goes to `ssRIdRRPA` |
| `RIRPA.py` | `ssRIRRPA`: the general (xc-kernel-capable) restricted class, which works in the **spin-orbital** `2ov` space; also `construct_product_RI`, `construct_inverse_RI` and `compress_low_rank` |
| `RIdRRPA.py` | `ssRIdRRPA`: specialised direct-RPA RHF path in **spatial** `ov` with explicit spin factors (the path most embedding calls take) |
| `RIURPA.py` | `ssRIURPA`: unrestricted variant; overrides only the integral/xc construction |
| `NI_eval.py` | numerical-integration framework: Clenshaw-Curtis on infinite and semi-infinite ranges with a nested error estimate, Gauss-Laguerre, and the optimisation of the grid scale `a` |
| `momzero_NI.py` | integrands for the zeroth moment (`MomzeroDeductNone/D/HigherOrder`, plus the `_dRHF` variant) and the analytic-offset integrand (`MomzeroOffsetCalcGaussLag`) |
| `energy_NI.py` | integrand for `Tr (MP)^{1/2}` (`NITrRootMP`, plus `_dRHF`) |

### 2.2 The mathematics as coded

This follows Furche (PRB 64, 195120) and arXiv:2301.09107. For restricted dRPA in spatial
orbitals,

```text
A - B = D                              (diagonal)
A + B = D + 4 L^T L                    (vayesta stores this as (sqrt2 L)^T (sqrt2 L) with a further 2 for spin)
M P   = (A - B)(A + B) = D^2 + S_L^T S_R,   S_L = sqrt2 L D-scaled, S_R = sqrt2 L   (rank N_aux)
eta0  = (M P)^{1/2} (A + B)^{-1}
```

`(MP)^{1/2}` is evaluated as a frequency integral, using

```text
(MP)^{1/2} = D + (1/pi) Int_{-inf}^{inf} [ I - w^2 (MP + w^2)^{-1} ] - [ diagonal ] dw
```

and the resolvent is reduced to auxiliary space with Woodbury. At each frequency,

```text
F(w)   = (D^2 + w^2)^{-1}                      (diagonal)
Q(w)   = S_R F(w) S_L^T                        (N_aux x N_aux)     <- the N_aux^2 ov step
term   = (T F) S_L^T [ (I + Q)^{-1} - I ] S_R F   (n_t x ov)
```

The `"HO"` (default) scheme subtracts the first-order term `-I` inside the integrand and adds
it back analytically. That analytic term,

```text
offset = T * Int_0^inf e^{-wD} S_L^T S_R e^{-wD} dw      (exact value: (D_p + D_q)^{-1} (S_L^T S_R)_pq)
```

would cost `O(n_t ov^2)` if done exactly. Vayesta therefore evaluates it with a **second,
separate 48-point Gauss-Laguerre integral**, whose scale is also optimised.

The remaining steps of the zeroth moment are as follows.

1. Fit the grid scale `a` for each integral (`opt_quadrature_diag`). The fit uses an
   exponential search over 14 decades, then Newton or bounded minimisation, so that the rule
   integrates a diagonal approximation exactly. Each probe costs `O(N_p ov)`.
2. Integrate with Clenshaw-Curtis (`npoints=48`, rounded up to a multiple of 4). The
   1/2- and 1/4-grids are nested inside it, and they give the error estimate.
3. Form the explicit inverse `np.linalg.inv(I + Q)` at every point.
4. Multiply by `(A+B)^{-1}`, itself built as an RI correction
   (`construct_inverse_RI`: an `N_aux x N_aux` inverse plus an SVD; or, in the dRPA path, an
   explicit `inv(I + 2 L D^-1 L^T)`).
5. Optionally compress (`compress > 0`) by a non-symmetric `eig` of `S_L S_R^T` with an
   absolute `1e-12` cut. This is off by default.

Higher moments use the exact recurrence `eta_{n+2} = eta_n D^2 + (eta_n S_R^T) S_L`. Each
order is two GEMMs of `O(n_t N_aux ov)`, which is already optimal.

The energy (`kernel_energy`) is `E = 1/2 [Tr (MP)^{1/2} - Tr A]`. The trace is another
optimised Clenshaw-Curtis integral (`NITrRootMP`), and its diagonal and first-order parts are
added analytically.

The xc-kernel path (`ssRIRRPA` with `rixc`, used by EDMET) differs in two ways. First,
`A - B` is no longer diagonal, so `construct_product_RI` builds the non-symmetric product
with rank `N_aux(A+B) + N_aux(A-B)`, which is up to `2-3x` the CDERI rank. Second, the whole
calculation runs in the `2ov` spin-orbital space.

`get_gap` runs a **non-symmetric** Davidson (`pyscf.lib.eig`, left and right vectors) on
`MP`.

### 2.3 Current cost

Per call of `kernel_moms(0, target_rot)` on the dRPA RHF path (`ssRIdRRPA`), with `N_p = 48`:

| step | cost | notes |
| --- | --- | --- |
| CDERI transform `get_cderi` (if `lov` not passed) | `O(N_aux nao^2 o)` | `N^4`; repeated by every caller that does not pass `lov` |
| build `S_L, S_R` | `O(N_aux ov)` | two to three `N_aux x ov` arrays held at once |
| grid fit, both integrals | `O(N_it N_p ov)` | cheap, but runs every call |
| main integral: `Q(w)` | `N_p * N_aux^2 ov` | **dominant for small `n_t`** |
| main integral: `inv(I+Q)` | `N_p * N_aux^3` | explicit inverse |
| main integral: target contractions | `N_p * (2 n_t N_aux ov + n_t N_aux^2)` | **dominant for large `n_t`** |
| offset integral (Gauss-Laguerre) | `N_p * 2 n_t N_aux ov` | a second pass, as expensive as the target part of the main integral |
| `(A+B)^{-1}` | `N_aux^2 ov + N_aux^3 + 2 n_t N_aux ov` | needed only because of the `(MP)^{1/2}(A+B)^{-1}` form |
| spin stacking (spin-orbital targets) | `O(n_t^2 ov + n_t^3)` | `compress_target_rot` forms an `n_t x n_t` Gram |
| higher moments | `2 n_t N_aux ov` per order | fine |

The leading terms are therefore

```text
cost(eta0) ~ 48 N_aux^2 ov  +  ~190 n_t N_aux ov  +  48 N_aux^3
memory     ~ (2-3) N_aux ov + (2 + nmom) n_t ov
```

This is `O(N^4)` when `n_t = O(N)`, and `O(N^5)` for the full moment (`n_t = ov`, which the
code warns about). The energy costs `~96 N_aux^2 ov` (`Q` plus the back contraction at every
point). For the xc-kernel path, replace `N_aux` by the 2-3x larger product rank and `ov` by
`2ov`.

How the call pattern turns this into total cost:

- **RPA bath**: a new `ssRIdRRPA` is built for every fragment. So the fragment-independent
  `48 N_aux^2 ov` work (the `Q(w)` matrices and inverses) and the grid fit are repeated
  `N_frag` times. With a dense Kronecker target of `n_t ~ o * v_clus` rows, each fragment is
  `O(N^4)`, and the calculation is `O(N^5)` overall.
- **Screening**: one call with all fragments stacked. It forms the full `n_t x 2ov` result
  and then keeps only the diagonal fragment blocks `T_f eta0 T_f^T`, so most of the
  `n_t x ov` back-contraction is thrown away.
- **Accuracy**: the error is controlled only by the nested-Clenshaw-Curtis estimate and a
  warning above `err_tol=1e-6` per element. It is not certified. Section 3.1 gives
  `momentGW`'s measurement of this same route at about `1e-9` relative error on a small-gap
  system, and an unfitted 48-point Clenshaw-Curtis rule measured `5e-7` on a larger one.

---

## 3. Bringing in the monomial / Davidson improvements

### 3.1 What those projects established

Both `monomial_GW` (a momentGW fork whose legacy Clenshaw-Curtis eta0 route *is* this
Vayesta route, Gauss-Laguerre offset included) and `Davidson-gw` (`gwfast`) replaced the
frequency integral with the **Hale-Higham-Trefethen (HHT) / Zolotarev rational approximation
of the inverse square root**. They write it in the symmetric, `(A+B)^{-1}`-free form

```text
eta0  = D^{1/2} Mt^{-1/2} D^{1/2},    Mt = D^2 + 4 W W^T,   W = D^{1/2} L^T      (symmetric PD)
x^{-1/2} ~= sum_j w_j / (x + s_j)      on a certified [lmin, lmax] enclosing spec(Mt)
```

Each pole is one resolvent, reduced exactly to auxiliary space by Woodbury:

```text
F_j = D / (D^2 + s_j)                        (diagonal)
C_j = I + 4 L F_j L^T                        (N_aux x N_aux, SPD)  <- one Gram, N_aux^2 ov
eta0 ~= sum_j w_j [ F_j  -  4 F_j L^T C_j^{-1} L F_j ]
```

The components, all implemented and tested in at least one of the two repositories:

1. **Certified spectral interval.** `lmin = min(D)^2` and
   `lmax = max(D^2) + 4 min(||W||_F^2, ||W||_1 ||W||_inf)`. `gwfast` also has a tighter
   `"spectral"` option, which uses the top eigenvalue of one `N_aux x N_aux` Gram. Both ends
   are padded outwards.
2. **Pole count from a measured scalar error.** The rule is the smallest `N_p` for which
   `max |1 - sqrt(x) r(x)| <= tol` over the interval. `N_p` grows only like
   `log(kappa) log(1/tol)`. The coefficients come from AGM and descending-Landen elliptic
   functions in extended precision, so `1 - k^2` is never formed (`momentGW/eta0.py`,
   `gwfast/quadrature.py`).
3. **No grid fit and no offset integral.** The rational sum absorbs the bare and first-order
   terms: at zero coupling each solve is the identity. This removes Vayesta's
   `"HO"`/Gauss-Laguerre second pass and its `a`-optimisation.
4. **Cholesky solves with residual checks instead of `inv`.** `momentGW` measured that this
   is an accuracy rule rather than a speed rule, because both are dominated by the
   `N_aux^2 ov` apply.
5. **No `ov x ov` intermediate.** The code records intermediate shapes and a test asserts
   that none is formed.
6. **Natural auxiliary functions** (`gwfast.integral.natural_auxiliary_functions`). These
   rotate the CDERIs to the eigenvectors of `L L^T` above a threshold. The rotation is
   symmetric (`eigh`) and does not change the result above that threshold. Compare Vayesta's
   off-by-default, non-symmetric `eig` compression.
7. **Square root and inverse square root together** (`gwfast.rpa.get_root_operators`).
   `Mt^{1/2} ~= sum_j w_j [I - s_j (Mt + s_j)^{-1}]` reuses the same `C_j`, so one pass over
   the poles gives both.

Measured results, quoted from those projects:

| source | comparison | result |
| --- | --- | --- |
| `monomial_GW/ROADMAP.md`, Milestone 2 | ozone/PBE (condition `6.7e4`) eta0 vs a dense oracle | Clenshaw-Curtis-48: `1.15e-9`; HHT-26: `2.1e-13` |
| same | benzene/cc-pVDZ eta0 stage | 20 poles vs 48 points, **1.82x** faster |
| `Davidson-gw/docs/hht-quadrature-benchmark-2026-09-22.md` | C8H18/def2-TZVPP (`ov` 15411, `N_aux` 1148) | unfitted Clenshaw-Curtis-48: `5e-7`; HHT-20: `3e-14`; **1.95x** faster at 20 poles, 2.72x at 16; peak memory roughly halved |

### 3.2 Applying it to Vayesta's projected moments

Vayesta needs `T eta0` for a general target `T` (`n_t x ov`), not just `eta0 L^T` as in GW.
The Woodbury form still applies:

```text
T eta0 ~= T * diag(sum_j w_j F_j)  -  4 sum_j w_j (T F_j L^T) C_j^{-1} (L F_j)
           \_____ ~= T, O(n_t ov) _/        \ n_t N_aux ov /  \solve/  \ n_t N_aux ov /
```

Per pole this costs `N_aux^2 ov` (Gram) `+ N_aux^3/3` (Cholesky) `+ 2 n_t N_aux ov + n_t N_aux^2`.
The scaling class is the same as today. The gain comes from having fewer passes and dropping
the redundant ones:

| | current (CC-48, `"HO"`) | HHT, `N_p ~ 20` | ratio |
| --- | --- | --- | --- |
| `N_aux^2 ov` work | `48` | `20` | 2.4x |
| `n_t N_aux ov` work | `~96 (main) + ~96 (offset) + 2` | `~40` | ~4.8x |
| `N_aux^3` | 48 explicit inverses + SVD | 20 Cholesky factorisations | ~7x |
| grid fitting, `(A+B)^{-1}` | yes | not needed | - |
| error control | heuristic, about `1e-7` to `1e-9` | certified to `tol` (default `1e-14` in `momentGW`) | - |

The expected eta0 speedup is therefore about **2.4x when `n_t << N_aux`, about 4x when
`n_t ~ N_aux`, and about 4.8x when `n_t >> N_aux`**, with accuracy improving by 5 to 7
orders of magnitude. These are operation-count estimates and must be measured. The one
directly comparable measurement is 1.82x on the GW eta0 stage, where `n_t = N_aux`.

The same pole factors give further savings that depend on what each caller actually needs:

1. **Cache the per-pole factors across targets and fragments.** `F_j`, `C_j` and its
   Cholesky factor depend only on the mean field, not on `T`. Build them once per
   calculation, at `N_p N_aux^2 ov` and `N_p N_aux^2` memory, and pass them to every caller.
   For the **RPA bath** this changes the fragment-independent part from
   `N_frag * 48 N_aux^2 ov` to `20 N_aux^2 ov` once. That is a reduction of
   `~2.4 N_frag`x on that term, and it changes the overall scaling from `O(N^5)` to
   `O(N^4)` plus a per-fragment target term.
2. **Two-sided projections.** Screening keeps only `T_f eta0 T_f^T`, and the RPA bath uses
   `m0 T^T`. Both are given by

   ```text
   T eta0 T^T ~= T T^T  -  4 sum_j w_j  B_j C_j^{-1} B_j^T,     B_j = T F_j L^T   (n_t x N_aux)
   ```

   This never forms the `n_t x ov` result. Per pole and fragment the cost is
   `n_f N_aux ov + n_f N_aux^2 + n_f^2 N_aux`, instead of `2 n_t N_aux ov` for the stacked
   target followed by the projection. For screening it also stops computing the
   off-diagonal fragment blocks, which are discarded today.
3. **Kronecker-structured targets.** The bath target is `R_o (x) R_v`, but it is stored
   densely, and `screening_moment` does the same per spin. Applying it in factored form,
   `B_j[I,A,Q] = sum_{ia} R_o[I,i] R_v[A,a] (F_j L^T)[ia,Q]` done as two small GEMMs, saves
   a factor of about `o` (or `v`) per contraction. With item 1, the bath then costs about
   `O(o v v_clus N_aux)` per fragment and pole, which is `O(N^3)` per fragment and
   `O(N^4)` overall.
4. **Targets inside the RI span.** For the cumulant energy in `qemb` the target is `L`
   itself, and `T F_j L^T` *is* the Gram (`(C_j - I)/4`). This is exactly the `momentGW`
   kernel, at `~2 N_aux^2 ov` per pole and no `n_t` term.
5. **Energy from the same poles.** Applying the rule to `sqrt(x) = x * x^{-1/2}` for both
   `Mt` and `D^2`, and differencing so that the leading terms cancel identically, gives

   ```text
   Tr(Mt^{1/2}) - Tr(D) ~= 4 sum_j w_j s_j Tr[ C_j^{-1} (L D (D^2+s_j)^{-2} L^T) ]
   ```

   This costs `~2 N_aux^2 ov` per pole, against `~96 N_aux^2 ov` for `NITrRootMP` today:
   about 2.4x faster and certified. One caveat: the bound is relative per eigenvalue, so
   the absolute energy error is at most about `2 tol Tr(D)`. Choose `tol` against the
   required `E_corr` accuracy. At `1e-14` this is well under a microhartree for any system
   Vayesta handles.
6. **Natural auxiliary functions.** Rotating `L` once to `N_aux' <= N_aux` NAFs saves
   `(N_aux/N_aux')^2` on every Gram and `N_aux/N_aux'` on every target contraction. The
   saving depends on the system and threshold and has not been measured here. The
   rotation must use a scale-aware PSD criterion and report the rank and discarded norm,
   which is `monomial_GW` Milestone 4's open item. The current absolute `1e-12` cut does
   neither.
7. **Symmetric gap and eigenvectors.** `Mt` is symmetric with eigenvalues `Omega^2`. `get_gap`
   can therefore use the symmetric `pyscf.lib.davidson` on
   `x -> D^2 x + 4 W (W^T x)` (`O(N_aux ov)` per matvec) instead of the left/right
   non-symmetric `eig` on `MP` followed by re-biorthogonalisation.

Rough projected totals for the two main embedding uses (operation counts, before NAF):

| use | today | with the changes | gain |
| --- | --- | --- | --- |
| RPA bath, `N_frag` fragments | `N_frag (48 N_aux^2 ov + ~190 o v_c N_aux ov)` ~ `O(N^5)` | `20 N_aux^2 ov + N_frag * 20 (o v v_c N_aux + ...)` ~ `O(N^4)` | grows with system size; already about `N_frag x o` on the target term |
| `mrpa` screening | `48 N_aux^2 ov + ~190 n_t N_aux ov` (`n_t = sum_f 2 ov_f`) | `20 N_aux^2 ov + 20 sum_f (2 n_f N_aux ov / k + n_f N_aux^2)`, where `k` is the Kronecker saving | about 5x, and more with Kronecker targets |
| total dRPA energy | `~96 N_aux^2 ov` | `~40 N_aux^2 ov` | about 2.4x, certified |

### 3.3 Proposed sequence

This mirrors `monomial_GW`'s ordering: restricted dRPA first, measured before it becomes the
default, and the old route kept as an independent reference.

1. **Baseline.** Record eta0, the projected moments, energies, screened interactions and
   bath occupations for a small set (H2O, a small-gap molecule, one Hubbard case, one
   solid), using the existing `vayesta/tests/rpa/test_rirpa_*` systems plus a dense
   `ssRPA` oracle.
2. **HHT kernel for `ssRIdRRPA`.** Port `momentGW/eta0.py` (interval, poles, scalar
   certificate) and a projected apply that accepts `T` in one-sided, two-sided and
   in-RI-span forms. Add an `eta0_method={"clencur","hht"}` option. Keep Clenshaw-Curtis
   as the default until the baseline agrees within the certified error, then switch.
3. **Pole-factor cache and caller changes.** Build the factors once per embedding object.
   Move `bath/rpa.py` and `screening_moment.py` to the two-sided, Kronecker-factored apply,
   and stop recomputing CDERIs per fragment by passing `lov` everywhere.
4. **Energy and gap** from the same poles (items 5 and 7 above).
5. **NAF/scale-aware compression** behind an option, reporting rank, discarded norm and
   the observed effect on eta0 and energy.
6. **Deferred: the UHF and xc-kernel paths.** `ssRIURPA` should follow with the
   stacked-spin convention that `gwfast` already validated (spin factor 1 over `2ov`).
   EDMET's `rixc` path cannot use the symmetric `D^{1/2} Mt^{-1/2} D^{1/2}` form, because
   `A - B` is not diagonal. It needs either a non-symmetric Woodbury rational apply (LU
   solves) on `MP`, whose interval is not theorem-backed, or `(A-B)^{1/2}` as a
   diagonal-plus-low-rank function. Keep it on Clenshaw-Curtis until that is designed.
   Periodic `kdf` cases also wait, for the reasons `monomial_GW` Milestone 6 records.

### 3.4 Points to settle before implementation

- **Spin conventions.** `ssRIdRRPA` mixes spatial integrals with spin-orbital outputs
  (`stack_spin`, `return_spatial`). The factor `4` in `Mt` and the split of the
  spin-diagonal identity term must be checked against the dense `ssRPA` before anything else.
- **Negative CDERIs** (`lov_neg`, from non-PSD metrics in some periodic setups). `A+B` then
  takes an indefinite low-rank part, `C_j` is no longer SPD, and the certified interval
  argument fails. The HHT path should refuse them at first.
- **Memory for the cache.** Storing the factors costs `N_p N_aux^2` (about 0.2 GB at
  `N_aux = 1000`, `N_p = 20`). For very large `N_aux`, rebuild per pole; `gwfast`'s
  `PRECOMPUTE_S` switch is exactly this trade.
- **Higher moments** are already exact recurrences at `O(n_t N_aux ov)` per order.
  `monomial_GW` Milestone 3 found that its high-order instabilities were in the downstream
  realisation, not in this recurrence, so no change is proposed there.
