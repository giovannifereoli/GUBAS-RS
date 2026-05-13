import math
import numpy as np
from collections import defaultdict


def pochhammer(a, m):
    out = 1.0
    for k in range(m):
        out *= a + k
    return out


def binom(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def multinomial3(p, nx, ny):
    nz = p - nx - ny
    if nx < 0 or ny < 0 or nz < 0:
        return 0.0
    return math.factorial(p) / (
        math.factorial(nx) * math.factorial(ny) * math.factorial(nz)
    )


def all_N_indices(order):
    """All (i,j,k) such that i+j+k = order."""
    out = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            k = order - i - j
            out.append((i, j, k))
    return out


def gamma_coeff(l, m):
    """
    Coefficients for:
        C_lm = sum_{ijk} gamma[(i,j,k)] N_ijk
    Eq. (16)-style formula from Tricarico appendix.
    """
    coeff = defaultdict(float)

    for p in range(l // 2 + 1):
        for q in range(m // 2 + 1):
            base = (
                2.0 ** (-l)
                * (-1.0) ** (p + q)
                * binom(l, p)
                * binom(2 * l - 2 * p, l)
                * pochhammer(l - m - 2 * p + 1, m)
                * binom(m, 2 * q)
            )

            for nx in range(p + 1):
                for ny in range(p - nx + 1):
                    i = m - 2 * q + 2 * nx
                    j = 2 * q + 2 * ny
                    k = l - m - 2 * nx - 2 * ny

                    if i >= 0 and j >= 0 and k >= 0 and i + j + k == l:
                        coeff[(i, j, k)] += base * multinomial3(p, nx, ny)

    return dict(coeff)


def sigma_coeff(l, m):
    """
    Coefficients for:
        S_lm = sum_{ijk} sigma[(i,j,k)] N_ijk
    Eq. (17)-style formula from Tricarico appendix.
    """
    coeff = defaultdict(float)

    if m == 0:
        return {}

    for p in range(l // 2 + 1):
        for q in range((m - 1) // 2 + 1):
            base = (
                2.0 ** (-l)
                * (-1.0) ** (p + q)
                * binom(l, p)
                * binom(2 * l - 2 * p, l)
                * pochhammer(l - m - 2 * p + 1, m)
                * binom(m, 2 * q + 1)
            )

            for nx in range(p + 1):
                for ny in range(p - nx + 1):
                    i = m - 2 * q - 1 + 2 * nx
                    j = 2 * q + 1 + 2 * ny
                    k = l - m - 2 * nx - 2 * ny

                    if i >= 0 and j >= 0 and k >= 0 and i + j + k == l:
                        coeff[(i, j, k)] += base * multinomial3(p, nx, ny)

    return dict(coeff)


def N_to_SH(N, lmax):
    """
    Convert normalized inertia products N_ijk to unnormalized spherical harmonics.

    Parameters
    ----------
    N : dict
        Dictionary keyed by (i,j,k).
    lmax : int
        Maximum degree.

    Returns
    -------
    C, S : dict
        Dictionaries keyed by (l,m).
    """
    C = {}
    S = {}

    for l in range(lmax + 1):
        for m in range(l + 1):
            gc = gamma_coeff(l, m)
            sc = sigma_coeff(l, m)

            C[(l, m)] = sum(a * N.get(idx, 0.0) for idx, a in gc.items())
            S[(l, m)] = sum(a * N.get(idx, 0.0) for idx, a in sc.items())

    return C, S


def build_order_matrix(l, parity):
    """
    Build linear map for one degree l.

    parity='C': rows are C_lm, m=0..l, columns have even j.
    parity='S': rows are S_lm, m=1..l, columns have odd j.
    """
    if parity == "C":
        rows = [(l, m) for m in range(l + 1)]
        cols = [idx for idx in all_N_indices(l) if idx[1] % 2 == 0]
        coeff_fun = gamma_coeff
    elif parity == "S":
        rows = [(l, m) for m in range(1, l + 1)]
        cols = [idx for idx in all_N_indices(l) if idx[1] % 2 == 1]
        coeff_fun = sigma_coeff
    else:
        raise ValueError("parity must be 'C' or 'S'")

    A = np.zeros((len(rows), len(cols)))

    for r, (_, m) in enumerate(rows):
        coeffs = coeff_fun(l, m)
        for c, idx in enumerate(cols):
            A[r, c] = coeffs.get(idx, 0.0)

    return A, rows, cols


def SH_to_N_min_norm(C, S, lmax):
    """
    Minimum-norm inverse: spherical harmonics -> one possible N_ijk set.

    Important: the inverse is generally underdetermined.
    """
    N_est = {}

    for l in range(lmax + 1):
        # C block: even-j N_ijk
        A, rows, cols = build_order_matrix(l, "C")
        b = np.array([C.get(row, 0.0) for row in rows])
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        for idx, val in zip(cols, x):
            N_est[idx] = val

        # S block: odd-j N_ijk
        if l >= 1:
            A, rows, cols = build_order_matrix(l, "S")
            b = np.array([S.get(row, 0.0) for row in rows])
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            for idx, val in zip(cols, x):
                N_est[idx] = val

    return N_est


def ellipsoid_N(a, b, c, r0, lmax):
    """
    N_ijk for a homogeneous triaxial ellipsoid aligned with principal axes.

    Only even i,j,k survive by symmetry.
    """
    N = {}

    for n in range(lmax + 1):
        for i, j, k in all_N_indices(n):
            if i % 2 or j % 2 or k % 2:
                N[(i, j, k)] = 0.0
                continue

            # Formula:
            # N_ijk = 3 a^i b^j c^k / r0^(i+j+k)
            #         * prod odd(i) prod odd(j) prod odd(k)
            #         / prod_{u=1}^{(i+j+k)/2+2} (2u-1)
            num = 3.0 * (a**i) * (b**j) * (c**k) / (r0**n)

            for p in range(1, i // 2 + 1):
                num *= 2 * p - 1
            for q in range(1, j // 2 + 1):
                num *= 2 * q - 1
            for s in range(1, k // 2 + 1):
                num *= 2 * s - 1

            den = 1.0
            for u in range(1, n // 2 + 3):
                den *= 2 * u - 1

            N[(i, j, k)] = num / den

    return N


def verify_low_order_identities(N, C, S):
    """
    Check explicit formulas through degree 4 from the appendix.
    """
    checks = {
        "C20": C[(2, 0)] - (N[(0, 0, 2)] - 0.5 * (N[(0, 2, 0)] + N[(2, 0, 0)])),
        "C21": C[(2, 1)] - 3 * N[(1, 0, 1)],
        "C22": C[(2, 2)] - 3 * (N[(2, 0, 0)] - N[(0, 2, 0)]),
        "S21": S[(2, 1)] - 3 * N[(0, 1, 1)],
        "S22": S[(2, 2)] - 6 * N[(1, 1, 0)],
        "C40": C[(4, 0)]
        - (
            N[(0, 0, 4)]
            - 3 * (N[(2, 0, 2)] + N[(0, 2, 2)])
            + 0.75 * N[(2, 2, 0)]
            + 0.375 * (N[(0, 4, 0)] + N[(4, 0, 0)])
        ),
        "C44": C[(4, 4)] - (105 * (N[(4, 0, 0)] + N[(0, 4, 0)]) - 630 * N[(2, 2, 0)]),
        "S44": S[(4, 4)] - 420 * (N[(3, 1, 0)] - N[(1, 3, 0)]),
    }

    max_err = max(abs(v) for v in checks.values())
    return checks, max_err


def build_conversion_jacobian(lmax):
    """
    Build dense Jacobians:

        dC/dN
        dS/dN

    where:

        C_vec = Jc @ N_vec
        S_vec = Js @ N_vec

    Returns
    -------
    N_keys : list[(i,j,k)]
    C_keys : list[(l,m)]
    S_keys : list[(l,m)]
    Jc : ndarray
    Js : ndarray
    """

    # -------------------------
    # All Nijk indices
    # -------------------------
    N_keys = []

    for l in range(lmax + 1):
        N_keys.extend(all_N_indices(l))

    # remove duplicates preserving order
    seen = set()
    N_keys = [x for x in N_keys if not (x in seen or seen.add(x))]

    N_map = {k: i for i, k in enumerate(N_keys)}

    # -------------------------
    # SH keys
    # -------------------------
    C_keys = []
    S_keys = []

    for l in range(lmax + 1):
        for m in range(l + 1):
            C_keys.append((l, m))

            if m > 0:
                S_keys.append((l, m))

    # -------------------------
    # Allocate Jacobians
    # -------------------------
    Jc = np.zeros((len(C_keys), len(N_keys)))
    Js = np.zeros((len(S_keys), len(N_keys)))

    # -------------------------
    # Fill dC/dN
    # -------------------------
    for r, (l, m) in enumerate(C_keys):

        coeffs = gamma_coeff(l, m)

        for idx, val in coeffs.items():
            c = N_map[idx]
            Jc[r, c] = val

    # -------------------------
    # Fill dS/dN
    # -------------------------
    for r, (l, m) in enumerate(S_keys):

        coeffs = sigma_coeff(l, m)

        for idx, val in coeffs.items():
            c = N_map[idx]
            Js[r, c] = val

    return N_keys, C_keys, S_keys, Jc, Js


if __name__ == "__main__":
    lmax = 4

    # Example: homogeneous triaxial ellipsoid
    a = 3.0
    b = 2.0
    c = 1.0
    r0 = 3.0

    N_true = ellipsoid_N(a, b, c, r0, lmax)
    C, S = N_to_SH(N_true, lmax)

    print("\n=== Forward conversion: N_ijk -> C_lm, S_lm ===")
    for l in range(lmax + 1):
        for m in range(l + 1):
            print(f"C[{l},{m}] = {C[(l,m)]: .12e}    S[{l},{m}] = {S[(l,m)]: .12e}")

    print("\n=== Verification against explicit Appendix formulas ===")
    checks, max_err = verify_low_order_identities(N_true, C, S)
    for name, err in checks.items():
        print(f"{name:4s} residual = {err: .3e}")
    print(f"max residual = {max_err:.3e}")

    assert max_err < 1e-12

    print("\n=== Inverse check: SH -> one minimum-norm N_ijk solution ===")
    N_inv = SH_to_N_min_norm(C, S, lmax)
    C2, S2 = N_to_SH(N_inv, lmax)

    max_C_err = max(
        abs(C[(l, m)] - C2[(l, m)]) for l in range(lmax + 1) for m in range(l + 1)
    )
    max_S_err = max(
        abs(S[(l, m)] - S2[(l, m)]) for l in range(lmax + 1) for m in range(l + 1)
    )

    print(f"max |C - C_reconstructed| = {max_C_err:.3e}")
    print(f"max |S - S_reconstructed| = {max_S_err:.3e}")

    assert max_C_err < 1e-12
    assert max_S_err < 1e-12

    print("\nNOTE:")
    print("The recovered N_ijk are not generally equal to the original N_ijk.")
    print(
        "The inverse problem is underdetermined; this returns one minimum-norm solution."
    )

    print("\n=== Jacobian example ===")

    N_keys, C_keys, S_keys, Jc, Js = build_conversion_jacobian(lmax=4)

    # Example:
    # dC_40 / dN_220

    row = C_keys.index((4, 0))
    col = N_keys.index((2, 2, 0))

    print("dC40/dN220 =", Jc[row, col])

    # should be:
    # +3/4 = 0.75

    # Example:
    # dC44 / dN220

    row = C_keys.index((4, 4))
    col = N_keys.index((2, 2, 0))

    print("dC44/dN220 =", Jc[row, col])

    # should be:
    # -630

    # Example:
    # dS44 / dN310

    row = S_keys.index((4, 4))
    col = N_keys.index((3, 1, 0))

    print("dS44/dN310 =", Js[row, col])

    # should be:
    # +420
