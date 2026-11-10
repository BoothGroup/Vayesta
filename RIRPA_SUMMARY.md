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
2. **Two-sided projections.** Several callers contract the `ov` index of `eta0` on both
   sides, and the left and right targets need not be the same, or even have the same
   number of rows. With a left target `X` (`n_X x ov`) and a right target `Y` (`n_Y x ov`),
   `eta0` is symmetric, so

   ```text
   X eta0 Y^T ~= X diag(sum_j w_j F_j) Y^T  -  4 sum_j w_j  B_j^X C_j^{-1} (B_j^Y)^T,
   B_j^X = X F_j L^T   (n_X x N_aux),     B_j^Y = Y F_j L^T   (n_Y x N_aux)
   ```

   The result carries only target labels (`n_X x n_Y`), with no particle-hole index left.
   When the two target labels are contracted against each other it is a single number,
   labelled only by the pair `(X, Y)`. That is a trace in auxiliary space:

   ```text
   Tr[X eta0 Y^T] ~= Tr[X diag(sum_j w_j F_j) Y^T]  -  4 sum_j w_j Tr[ C_j^{-1} (B_j^Y)^T B_j^X ]
   ```

   Neither form ever builds the `n_X x ov` intermediate `X eta0`. Per pole the cost is
   `(n_X + n_Y) N_aux ov` for the two `B` matrices, which drops to `n_X N_aux ov` when
   `Y = X`. Then the open form costs `n_X N_aux^2 + n_X n_Y N_aux`, and the trace costs
   `min(n_X, n_Y) N_aux^2 + N_aux^3` instead. Today the cost is `2 n_X N_aux ov` (main
   integral) plus `2 n_X N_aux ov` (offset) per point, and then a further `n_X n_Y ov` for
   the projection.

   Where this applies in Vayesta:

   | caller | `X` | `Y` | form |
   | --- | --- | --- | --- |
   | `screening_moment.py` (`dot(momzero[n:n+nov], rot.T)`) | `T_f` | `T_f` | open, diagonal fragment blocks only; the off-diagonal blocks computed today are discarded |
   | `bath/rpa.py` (`dot(m0, target_rot.T)`) | `R_o (x) R_v` | the same | open |
   | `qemb.py` cumulant energy (`einsum("pq,pq->", m0 - l_, l_2)`) | `l_` | `l_2`, which differs from `l_` by the sign of the negative-CDERI rows | trace |
   | `RIRPA.kernel_energy`, `"linear"` xc correction | `ri_b_xc[0]` | `ri_b_xc[1]` | trace |
   | `RIRPA.direct_AC_integration` | `lrot` | `rrot` | trace |

   When either side lies in the RI span the matching `B` is already known. For `Y = L`,
   `B_j^Y = L F_j L^T = (C_j - I)/4`. The cumulant energy without negative CDERIs has
   `X = Y = L`, and its trace reduces to `-(1/4) sum_j w_j Tr[C_j - 2I + C_j^{-1}]` plus the
   diagonal term. That is one Gram and one Cholesky factorisation per pole, with no target
   contraction at all.

   Keep the rational diagonal `sum_j w_j F_j` rather than replacing it by `I`. The callers
   subtract the bare target (`m0 - l_`, `eta0 - lrot`), and the consistent diagonal makes
   that difference come out of the low-rank term alone instead of from a cancellation of
   two large numbers.
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

---

## 4. Local and complementary projections (draft)

**Status: first draft for discussion.** All costs are operation counts. Nothing here has been
implemented or measured. It builds on the pole form of section 3 and on the discussion of
local spaces labelled by `x`.

### 4.1 Setup

Each label `x` defines a local occupied and a local virtual space through orthonormal rows of
`C^x`:

```text
c_o  (o_x x o),   c_v  (v_x x v),        c_o c_o^T = I,  c_v c_v^T = I
P_o = c_o^T c_o,  P_v = c_v^T c_v,        Q_o = I - P_o,  Q_v = I - P_v
e_o, e_v          complement rows,       e_o^T e_o = Q_o,  e_v^T e_v = Q_v
X   = c_o (x) c_v                         local ph target, n_x = o_x v_x rows
```

`n_x` is `O(1)` in system size. There are `O(N)` labels for molecules and `O(1)` for solids.

The right-hand side of the quantities discussed is a **complement** of the local space. Two
complements have come up, and they are different quantities, not two representations of one:

| name | projector on the right | blocks kept | example |
| --- | --- | --- | --- |
| ph complement | `Pi_bar = I - P_o (x) P_v` | everything except local-local, including the mixed local-occ/env-vir and env-occ/local-vir blocks | `eta~^x_{i_bar a_bar, kc}` (full MO indices on the right) |
| strict environment | `Q_o (x) Q_v` | env-env only | `eta^x_{i_bar a_bar, k_bar c_bar}` with complement barred indices |

Either can be **represented** in the full MO basis (apply the projector) or in the complement
basis (rotate with `e_o (x) e_v`). As George notes, the basis is a choice. Section 4.3 shows
that the MO-basis representation is the cheaper one to compute.

**To decide:** which complement each use needs.

### 4.2 Shared and per-label factors

Shared factors, built once per mean field, as in section 3:

```text
F_j = D (D^2 + s_j)^{-1}                  (diagonal, N_p ov)
C_j = I + 4 L F_j L^T  ->  chol(C_j)      (N_p N_aux^2 memory)
cost: N_p (N_aux^2 ov + N_aux^3/3)
eta0 ~= sum_j w_j [ F_j - 4 F_j L^T C_j^{-1} L F_j ]
```

Per-label left factors, built for each `x` and each pole:

```text
B_j^x = X F_j L^T             (n_x x N_aux)   Kronecker: (o_x o v + o_x v_x v) N_aux
G_j^x = w_j B_j^x C_j^{-1}    (n_x x N_aux)   one Cholesky solve: n_x N_aux^2
```

`B_j^x` is applied in factored form: reshape `F_j L^T` to `(o, v, N_aux)` and contract `c_o`,
then `c_v`. It is never formed as a dense `n_x x ov` target.

Every quantity below is built from these factors, through one of three kernels.

### 4.3 Kernel A: one-sided, with a projector on the right

```text
X eta0 R,        R in { I,  Pi_bar,  Q_o (x) Q_v }
```

**Bare term.** It is `X D_r R` with `D_r = diag(sum_j w_j F_j) ~= I`. For `R = I` it is about
`X`. For either complement `X R = 0` exactly, so the bare term is `X (D_r - I) R`, which is
`O(tol)`. It can be dropped, leaving only the correlated part, with no large cancellation.

**Low-rank term.** Sum over the poles *before* projecting:

```text
S^x = -4 sum_j G_j^x (L F_j)               (n_x x ov)      N_p n_x N_aux ov
X eta0 R ~= S^x R                          (R applied once per label, not per pole)
```

The projector has to come after the pole sum. `F_j` is diagonal in the MO basis and does not
commute with `R`, so applying `R` inside the sum would cost one application per pole.

How `R` is applied once:

| `R` | how | cost per label |
| --- | --- | --- |
| `Pi_bar` | `S - (S X^T) X`, where `S X^T = -4 sum_j G_j^x (B_j^x)^T` is the two-sided block (kernel B, case a), so it is already available | `N_p n_x^2 N_aux` + `~n_x ov o_x` to expand |
| `Q_o (x) Q_v`, MO basis | `S - S(P_o (x) I) - S(I (x) P_v) + S(P_o (x) P_v)`, each a low-rank subtraction | `~2 n_x ov (o_x + v_x)` |
| `Q_o (x) Q_v`, complement basis | rotate `S` by `e_o^T`, `e_v^T` | `n_x (o_env o v + o_env v_env v)`; larger, so use only if that basis is needed downstream |

**Cost per label:**

```text
N_p [ (o_x o v + o_x v_x v) N_aux  +  n_x N_aux^2  +  n_x N_aux ov ]  +  O(n_x ov (o_x + v_x))
                                                      ^^^^^^^^^^^^^ dominant, output-bound
```

The output has `n_x ov` entries, so the `n_x N_aux ov` back-multiply per pole cannot be
avoided while the result is kept open on the right. The only levers on it are `N_p` and
`N_aux` (natural auxiliary functions).

**Batching across labels.** Stack `G_j^x` over a batch of labels, so that each pole is one GEMM
`(sum_x n_x x N_aux) . (N_aux x ov)`. The flops are the same but the efficiency is better.
Consume or store each `S^x R` before the next batch, because holding all labels is `O(N^3)`
memory for molecules.

### 4.4 Kernel B: two-sided, `X eta0 Y^T` with a small `Y`

```text
X eta0 Y^T ~= X D_r Y^T  -  4 sum_j G_j^x (B_j^y)^T,     B_j^y = Y F_j L^T   (n_Y x N_aux)
```

Per pole this costs forming `B_j^y` plus `n_x N_aux n_Y`. The open `n_x x ov` block is never
built. Kernel B pays off only when `n_Y << ov`, or when `B_j^y` is known without a sweep over
`ov`. That is the case when the right-hand index is contracted with something small *before*
anything is stored.

| case | `Y` | `B_j^y` | extra cost per pole | uses |
| --- | --- | --- | --- | --- |
| (a) same space | `X` | `B_j^x` (already built) | `n_x^2 N_aux` | screening `T_f eta0 T_f^T`, RPA bath `m0 T^T`, the `Pi_bar` correction in kernel A |
| (b) RI span | `L` | `L F_j L^T = (C_j - I)/4`, free | none: `X eta0 L^T ~= X D_r L^T - sum_j w_j B_j^x (I - C_j^{-1})` | GW-type `eta0 V`, interactions built from RI vectors |
| (c) RI span, projected | `L Pi_bar` | `(C_j - I)/4 - (L X^T) B_j^x` | `N_aux^2 n_x`; `L X^T` once per label | the RHS of `eta~^x` contracted directly with `(kc\|Q)` |
| (c') RI span, env-env | `L (Q_o (x) Q_v)` | `(C_j - I)/4` minus the `P_o`, `P_v` and `P_o (x) P_v` corrections, each a factored low-rank product | `~(o_x + v_x) ov N_aux + (o_x v + o v_x) N_aux^2` | the RHS of `eta^x_{i_bar a_bar, k_bar c_bar}` contracted with integrals |
| (d) another label | `X' = c_o' (x) c_v'` | `B_j^{x'}`, already built for `x'` | `n_x N_aux n_x'` | cross-label couplings, if needed |

For (c) and (c') the bare term `X D_r R L^T` is again `O(tol)` and can be dropped.

Whether kernel B beats kernel A for (c') depends on the sizes:
`N_aux (o_x v + o v_x)` against `o_x v_x o v` per pole. For small local spaces in a large
system, kernel B is expected to win, but this has to be measured.

### 4.5 Kernel C: fully contracted, `Tr[X eta0 Y^T]`

```text
Tr[X eta0 Y^T] ~= Tr[X D_r Y^T]  -  4 sum_j w_j Tr[ C_j^{-1} (B_j^y)^T B_j^x ]
```

This costs `min(n_X, n_Y) N_aux^2 + N_aux^3` per pole once the `B` matrices are built. When
`X = Y = L`, it is `-(1/4) sum_j w_j Tr[C_j - 2I + C_j^{-1}]` plus the diagonal term: one Gram
and one Cholesky factorisation per pole, with no target contraction. The uses are the
cumulant energy, the `"linear"` xc energy correction and `direct_AC_integration` (section 3.2,
item 2).

### 4.6 Summary by use

| use | `X` | right side | output | kernel | dominant cost per label | molecules / solids |
| --- | --- | --- | --- | --- | --- | --- |
| `eta~^x_{i_bar a_bar, kc}` | local | `Pi_bar`, MO basis | `n_x x ov` | A | `N_p n_x N_aux ov` | `O(N^4)` / shared `O(N^4)` + `O(N^3)` per label |
| `eta^x_{i_bar a_bar, k_bar c_bar}` stored | local | `Q_o (x) Q_v` | `n_x x n_env` | A, then rotate if the env basis is needed | `N_p n_x N_aux ov` | as above |
| either, RHS contracted with `L` next | local | `L Pi_bar` or `L (Q_o (x) Q_v)` | `n_x x N_aux` | B (c)/(c') | `N_aux^2 (o_x v + o v_x)` | no `n_x ov` term |
| screening, RPA bath | `T_f` | `T_f` | `n_x x n_x` | B (a) | `n_x N_aux^2` + factored forward | |
| energies | various | various | scalar | C | `N_aux^2` per pole | |

Today's route costs about `48 N_aux^2 ov + 192 sum_x n_x N_aux ov` for any of these, and
always builds the `n_x x ov` block. With the shared factors, kernel A costs
`20 N_aux^2 ov + 20 sum_x n_x N_aux ov`, about 2.4x on the shared term and about 10x on the
per-label term. Kernels B and C remove the per-label `ov` sweep where the use allows it.

### 4.7 Proposed kernel interface

```python
poles = build_pole_factors(D, L, tol)                   # s_j, w_j, F_j, chol(C_j); certified interval
left  = local_left_factors(poles, c_o, c_v)             # B_j^x, G_j^x (Kronecker-factored)
one_sided(poles, left, right=None | "ph_complement" | "env", basis="mo" | "env")   # kernel A
two_sided(poles, left, right)                           # kernel B; right = left' | "ri" | ("ri", projector)
trace(poles, left, right)                               # kernel C
```

Every function also takes a batch of labels, and each records the per-pole Cholesky residuals
and the intermediate shapes, following `monomial_GW` Milestone 2.

### 4.8 Open questions

1. **Which complement** (`Pi_bar` or `Q_o (x) Q_v`) each use needs.
2. **What the right-hand index is contracted with next** in each use. This decides between
   kernel A and kernel B, and it is the main source of savings beyond the constant factors.
3. **Whether the outputs are stored or consumed** per label. This sets the memory plan for
   kernel A.
4. **Orthonormality of `C^x` rows.** The `X R = 0` cancellation and the cheap projector
   applications rely on it.
5. **Spin.** Restricted spatial forms are given here, with the factor 4 still to be checked
   against the dense `ssRPA`. The spin-orbital stacking used by the existing callers has to be
   mapped onto them.
6. **Dropping the `O(tol)` bare term.** It is consistent with the certificate on `eta0`, but
   the error bound for the projected quantity should be stated explicitly, for example by
   Cauchy-Schwarz with `||X|| = 1`.
