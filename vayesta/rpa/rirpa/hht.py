"""Zeroth dRPA moment by the Hale-Higham-Trefethen (HHT) rational approximation.

For a restricted closed-shell reference the singlet zeroth density-density moment is

    eta = D^1/2 M^-1/2 D^1/2,    M = D^2 + 4 D^1/2 V^T V D^1/2,

with D the particle-hole energy differences and V the occupied-virtual CDERIs. The inverse
square root is approximated on a guaranteed enclosure [lmin, lmax] of the spectrum of M by

    x^-1/2 ~= sum_k w_k / (x + s_k)        (Hale, Higham and Trefethen, SINUM 46, 2505 (2008)),

and every resolvent is reduced to an N_aux x N_aux Cholesky solve by the Woodbury identity.
The scalar relative error of the rule over the enclosure bounds the relative error of the
operator approximation in every eigendirection.
"""

import numpy as np
import scipy.linalg
import scipy.special


def hht_poles(
    lmin: float, lmax: float, tol: float = 1e-12, n_poles: int | None = None, n_max: int = 64, n_grid: int = 4096
) -> tuple[np.ndarray, np.ndarray, float]:
    """Shifts and weights of the HHT rule for x^{-1/2} on [lmin, lmax].

    Parameters
    ----------
    lmin, lmax : float
        Enclosure of the spectrum, 0 < lmin < lmax.
    tol : float, optional
        Requested scalar relative error. The smallest pole count meeting it is used.
    n_poles : int, optional
        Fixed pole count; overrides `tol`.
    n_max : int, optional
        Largest pole count tried before giving up.
    n_grid : int, optional
        Points of the logarithmic grid on which the scalar error is measured.

    Returns
    -------
    shifts, weights : np.ndarray
        Positive shifts s_k and weights w_k.
    error : float
        max |1 - sqrt(x) sum_k w_k / (x + s_k)| over the grid.

    Raises
    ------
    ValueError
        If the interval is not 0 < lmin < lmax.
    RuntimeError
        If `tol` cannot be met with at most `n_max` poles.
    """
    if not 0.0 < lmin < lmax:
        raise ValueError("Require 0 < lmin < lmax, got lmin=%r, lmax=%r." % (lmin, lmax))
    m = 1.0 - lmin / lmax
    kp = scipy.special.ellipkm1(lmin / lmax)  # complete elliptic integral K(m)
    x = np.geomspace(lmin, lmax, n_grid)

    def rule(n: int) -> tuple[np.ndarray, np.ndarray, float]:
        t = (np.arange(1, n + 1) - 0.5) * kp / n
        sn, cn, dn, _ = scipy.special.ellipj(t, m)
        shifts = lmin * (sn / cn) ** 2
        weights = 2.0 * kp * np.sqrt(lmin) / (np.pi * n) * dn / cn**2
        r = (weights[None] / (x[:, None] + shifts[None])).sum(1)
        return shifts, weights, float(np.max(np.abs(1.0 - np.sqrt(x) * r)))

    if n_poles is not None:
        return rule(n_poles)
    # Start just below the pole count predicted by the convergence rate exp(-2 pi^2 n / (ln kappa + c)).
    rate = 2.0 * np.pi**2 / (np.log(lmax / lmin) + 6.0)
    n = max(1, int(np.ceil(np.log(10.0 / tol) / rate)) - 4)
    while n <= n_max:
        shifts, weights, err = rule(n)
        if err <= tol:
            return shifts, weights, err
        n += 1
    raise RuntimeError("HHT rule cannot reach tol=%.1e with %d poles (lmax/lmin=%.1e)." % (tol, n_max, lmax / lmin))


def _inverse_from_cholesky(cho: tuple[np.ndarray, bool]) -> np.ndarray:
    """Full symmetric inverse from a `scipy.linalg.cho_factor` result, via LAPACK potri.

    Parameters
    ----------
    cho : tuple
        Factor and `lower` flag, as returned by `scipy.linalg.cho_factor`.

    Returns
    -------
    inv : np.ndarray
        The inverse of the factorised matrix.
    """
    c, lower = cho
    inv, info = scipy.linalg.lapack.dpotri(c, lower=lower)
    if info != 0:
        raise np.linalg.LinAlgError("potri failed with info=%d" % info)
    tri = np.tril(inv) if lower else np.triu(inv)
    return tri + tri.T - np.diag(np.diag(tri))


def hht_mom0_spinindependent(
    eps: np.ndarray, V: np.ndarray, target_rot: np.ndarray, tol: float = 1e-12, n_poles: int | None = None
) -> tuple[np.ndarray, dict]:
    """Return 0.5 * (T eta - T) for spatial target rows T (singlet dRPA, restricted reference).

    This is the array `mom0_spinindependent` of `ssRIdRRPA._kernel_mom0`. The diagonal part of the rational
    approximation, sum_k w_k F_k, equals the identity to within `tol` and is not formed.

    Parameters
    ----------
    eps : np.ndarray, shape (ov,)
        Particle-hole energy differences D.
    V : np.ndarray, shape (naux, ov)
        Occupied-virtual CDERIs, already multiplied by sqrt(alpha).
    target_rot : np.ndarray, shape (ntarget, ov)
        Spatial target rows T.
    tol : float, optional
        Scalar relative error of the rational approximation.
    n_poles : int, optional
        Fixed pole count; overrides `tol`.

    Returns
    -------
    mom0_spinindependent : np.ndarray, shape (ntarget, ov)
    info : dict
        Pole count, scalar error, enclosure, its ratio and the largest relative residual of the Cholesky solves.

    Raises
    ------
    ValueError
        If any particle-hole energy difference is not strictly positive.
    """
    if np.any(eps <= 0.0):
        raise ValueError("HHT eta0 requires strictly positive particle-hole energy differences.")
    d2 = eps**2
    V_sqD = V * np.sqrt(eps)[None]
    # Guaranteed enclosure of the spectrum of M (the coupling term is positive semidefinite), padded outwards.
    coupling = max(float(np.linalg.eigvalsh(np.dot(V_sqD, V_sqD.T))[-1]), 0.0)
    lmin = float(d2.min()) * (1.0 - 1e-8)
    lmax = (float(d2.max()) + 4.0 * coupling) * (1.0 + 1e-8)
    shifts, weights, err = hht_poles(lmin, lmax, tol=tol, n_poles=n_poles)
    eye = np.eye(V.shape[0])
    use_inverse = target_rot.shape[0] > V.shape[0]
    out = np.zeros(target_rot.shape)
    residual = 0.0
    for s, w in zip(shifts, weights):
        f = eps / (d2 + s)  # F_k (positive)
        vf = V * f[None]  # V F_k
        y = V * np.sqrt(f)[None]
        g = eye + 4.0 * np.dot(y, y.T)  # G_k as a symmetric product (BLAS syrk, half the work of a GEMM)
        cho = scipy.linalg.cho_factor(g, check_finite=False)
        y_k = np.dot(target_rot, vf.T)  # T F_k V^T
        if use_inverse:
            # Many right-hand sides: one N_aux^3 inverse and a GEMM is faster than two triangular solves.
            # G_k has all eigenvalues >= 1, so the explicit inverse is well conditioned.
            z = np.dot(_inverse_from_cholesky(cho), y_k.T)
        else:
            z = scipy.linalg.cho_solve(cho, y_k.T, check_finite=False)  # G_k^-1 (T F_k V^T)^T
        if y_k.size:
            # Residual of the solve on at most 32 columns: G_k is well conditioned, so a sample suffices as a check.
            ref = y_k.T[:, :32]
            res = np.linalg.norm(np.dot(g, z[:, :32]) - ref) / max(np.linalg.norm(ref), 1e-300)
            residual = max(residual, float(res))
        out -= 2.0 * w * np.dot(z.T, vf)
    info = dict(
        n_poles=len(shifts),
        scalar_error=err,
        interval=(lmin, lmax),
        kappa=lmax / lmin,
        max_cholesky_residual=residual,
    )
    return out, info
