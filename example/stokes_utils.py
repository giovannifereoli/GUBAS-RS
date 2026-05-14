"""
stokes_utils.py
===============
Python-side Stokes coefficient utilities for gravity-field OD post-processing.

Implements the Tricarico 2008 linear map between symmetric inertia integrals
T_{ijk} and spherical harmonic Stokes coefficients C_{lm}, S_{lm}.

Key functions
-------------
stokes_matrix(theta_indices, min_degree, max_degree, normalized=False)
    Build M  (N_cs × N_theta) such that  C/S_flat = M · N_theta_flat,
    where N_{ijk} = T_{ijk} / (M · r₀ˡ).

convert_phi_xt_to_cs(phi_xt, theta_indices, min_degree, max_degree, normalized=False)
    phi_xt : (nsteps, 30, N_theta) or (30, N_theta)
    Returns phi_xC, phi_xS each (nsteps, 30, N_cs) via  Φ_xCS = Φ_xθ · M⁺

cs_labels(min_degree, max_degree)
    Returns list of strings: ["C20", "C21", "S21", "C22", "S22", ...]
"""

from math import factorial, comb
import numpy as np


# ── Tricarico 2008 combinatorics ──────────────────────────────────────────────

def _pochhammer(a, m):
    """Rising factorial (a)_m = a(a+1)…(a+m−1).  Returns 1 for m=0."""
    result = 1.0
    for k in range(m):
        result *= (a + k)
    return result


def _norm_factor(l, m):
    """Geodesy fully-normalized factor √((2l+1)(2−δ₀ₘ)(l−m)!/(l+m)!)."""
    delta = 1.0 if m == 0 else 0.0
    return np.sqrt((2*l + 1) * (2 - delta) * factorial(l - m) / factorial(l + m))


def _clm_contrib(l, m, ix, iy, iz):
    """Sum of Tricarico Eq. 16 coefficients for C_{lm} at index (ix, iy, iz).
    Returns α such that C_{lm} += 2^{−l} · α · N_{ix,iy,iz}."""
    if ix + iy + iz != l:
        return 0.0
    coeff = 0.0
    for p in range(l // 2 + 1):
        poch = _pochhammer(l - m - 2*p + 1, m)
        if poch == 0.0:
            continue
        clp = comb(l, p) * comb(2*l - 2*p, l) * poch
        for q in range(m // 2 + 1):
            c_m_2q = comb(m, 2*q)
            if c_m_2q == 0:
                continue
            sign = (-1)**(p + q)
            for nux in range(p + 1):
                for nuy in range(p - nux + 1):
                    nuz = p - nux - nuy
                    multi = factorial(p) / (factorial(nux) * factorial(nuy) * factorial(nuz))
                    ex = m - 2*q + 2*nux
                    ey =     2*q + 2*nuy
                    ez = l - m  - 2*nux - 2*nuy
                    if ex == ix and ey == iy and ez == iz:
                        coeff += sign * clp * c_m_2q * multi
    return coeff


def _slm_contrib(l, m, ix, iy, iz):
    """Sum of Tricarico Eq. 17 coefficients for S_{lm} at index (ix, iy, iz)."""
    if m == 0 or ix + iy + iz != l:
        return 0.0
    coeff = 0.0
    for p in range(l // 2 + 1):
        poch = _pochhammer(l - m - 2*p + 1, m)
        if poch == 0.0:
            continue
        clp = comb(l, p) * comb(2*l - 2*p, l) * poch
        for q in range((m - 1) // 2 + 1):
            c_m_2q1 = comb(m, 2*q + 1)
            if c_m_2q1 == 0:
                continue
            sign = (-1)**(p + q)
            for nux in range(p + 1):
                for nuy in range(p - nux + 1):
                    nuz = p - nux - nuy
                    multi = factorial(p) / (factorial(nux) * factorial(nuy) * factorial(nuz))
                    ex = m - 2*q - 1 + 2*nux
                    ey =     2*q + 1 + 2*nuy
                    ez = l - m      - 2*nux - 2*nuy
                    if ex == ix and ey == iy and ez == iz:
                        coeff += sign * clp * c_m_2q1 * multi
    return coeff


# ── Public API ────────────────────────────────────────────────────────────────

def cs_labels(min_degree, max_degree):
    """Ordered list of C/S label strings matching rows of stokes_matrix."""
    labels = []
    for l in range(min_degree, max_degree + 1):
        labels.append(f"C{l}0")
        for m in range(1, l + 1):
            labels.append(f"C{l}{m}")
            labels.append(f"S{l}{m}")
    return labels


def inertia_labels(theta_indices):
    """Label strings for θ columns: 'T_{ijk}'."""
    return [f"T{i}{j}{k}" for (i, j, k) in theta_indices]


def stokes_matrix(theta_indices, min_degree, max_degree, normalized=False):
    """
    Build the linear Stokes sensitivity matrix M of shape (N_cs, N_theta).

    M[row, k] = ∂C_row / ∂N_{ijk_k}   (or ∂S / ∂N)

    where N_{ijk} = T_{ijk} / (M_body · r₀ˡ) are the dimensionless moments.
    The C/S returned by the Rust run_stm_augmented use r₀=1 km, so to get
    physical Stokes coefficients multiply N by 1/(mass * r0^l).

    Row ordering (interleaved per degree):
        l=min_degree: C_{l,0}, C_{l,1}, S_{l,1}, …, C_{l,l}, S_{l,l}
        l=min_degree+1: …

    Parameters
    ----------
    theta_indices : list of (i, j, k) tuples   (N_theta entries)
    min_degree, max_degree : int
    normalized : bool  — if True, apply geodesy fully-normalized factor

    Returns
    -------
    M : ndarray (N_cs, N_theta)
    """
    N_theta = len(theta_indices)
    # Row ordering
    rows = []
    for l in range(min_degree, max_degree + 1):
        rows.append((l, 0, False))
        for m in range(1, l + 1):
            rows.append((l, m, False))
            rows.append((l, m, True))
    N_cs = len(rows)

    M = np.zeros((N_cs, N_theta))
    for r, (l, m, is_sin) in enumerate(rows):
        pre = 0.5**l * (_norm_factor(l, m) if normalized else 1.0)
        for k, (ii, jj, kk) in enumerate(theta_indices):
            if is_sin:
                M[r, k] = pre * _slm_contrib(l, m, ii, jj, kk)
            else:
                M[r, k] = pre * _clm_contrib(l, m, ii, jj, kk)
    return M


def stokes_pseudoinverse(M):
    """
    Compute the right pseudoinverse M⁺ = Mᵀ (M Mᵀ)⁻¹ of the Stokes matrix.

    Since M is (N_cs × N_theta) with full row rank (N_cs ≤ N_theta),
    M⁺ is (N_theta × N_cs) and satisfies M M⁺ = I_{N_cs}.

    This lets us compute  Φ_xCS = Φ_xθ · M⁺  (shape 30 × N_cs).
    """
    try:
        return M.T @ np.linalg.inv(M @ M.T)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def convert_phi_xt_to_cs(phi_xt, theta_indices, min_degree, max_degree,
                          normalized=False):
    """
    Convert Φ_xθ (sensitivity to inertia integrals) to Φ_xCS
    (sensitivity to Stokes coefficients C_{lm}, S_{lm}).

    Φ_xCS = Φ_xθ · M⁺   where M⁺ = right pseudoinverse of Stokes matrix.

    Parameters
    ----------
    phi_xt       : ndarray (nsteps, 30, N_theta)  or  (30, N_theta)
    theta_indices: list of (i,j,k) — must match last dim of phi_xt
    min_degree, max_degree : int
    normalized   : bool

    Returns
    -------
    phi_xcs : ndarray same leading dims as phi_xt but last dim = N_cs
    M        : ndarray (N_cs, N_theta) — the Stokes matrix used
    Mplus    : ndarray (N_theta, N_cs) — the pseudoinverse used
    """
    M     = stokes_matrix(theta_indices, min_degree, max_degree, normalized)
    Mplus = stokes_pseudoinverse(M)
    phi_xcs = phi_xt @ Mplus  # (..., 30, N_theta) @ (N_theta, N_cs) → (..., 30, N_cs)
    return phi_xcs, M, Mplus