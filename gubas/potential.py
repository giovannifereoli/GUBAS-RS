"""Mutual potential and partial derivatives (Python reference implementation).

All functions expect km-kg-s units.
See Hou 2016 for equation definitions.  ``e`` must always be a row vector of
shape ``(1, 3)``.
"""

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _u_tilde(n, t, a, b, e, ta, tbp):
    n = int(n)
    u = np.zeros((t.size, 1))
    for k in range(n, -1, -2):
        for i1 in range(k + 1):
            for i2 in range(k - i1 + 1):
                for i3 in range(k - i1 - i2 + 1):
                    for i4 in range(k - i1 - i2 - i3 + 1):
                        for i5 in range(k - i1 - i2 - i3 - i4 + 1):
                            i6 = k - i1 - i2 - i3 - i4 - i5
                            for j1 in range(n - k + 1):
                                for j2 in range(n - k + 1 - j1):
                                    for j3 in range(n - k + 1 - j1 - j2):
                                        for j4 in range(n - k + 1 - j1 - j2 - j3):
                                            for j5 in range(n - k + 1 - j1 - j2 - j3 - j4):
                                                j6 = n - k - j1 - j2 - j3 - j4 - j5
                                                u[k // 2] += (
                                                    a[k, i1, i2, i3, i4, i5, i6]
                                                    * b[n-k, j1, j2, j3, j4, j5, j6]
                                                    * e[0,0]**(i1+i4) * e[0,1]**(i2+i5) * e[0,2]**(i3+i6)
                                                    * ta[i1+j1, i2+j2, i3+j3]
                                                    * tbp[i4+j4, i5+j5, i6+j6]
                                                )
        u[k // 2] *= t[k // 2]
    return float(u.sum())


def _de_dx(e, r, de, dx):
    x = r * e
    if de == dx:
        others = [v for v in [0, 1, 2] if v != dx]
        return (x[0, others[0]]**2 + x[0, others[1]]**2) / r**3
    return -x[0, de] * x[0, dx] / r**3


def _du_dx_tilde(n, t, a, b, e, r, dx, ta, tbp):
    n = int(n)
    de0 = _de_dx(e, r, 0, dx)
    de1 = _de_dx(e, r, 1, dx)
    de2 = _de_dx(e, r, 2, dx)
    du = np.zeros((t.size, 1))
    for k in range(n, -1, -2):
        for i1 in range(k + 1):
            for i2 in range(k - i1 + 1):
                for i3 in range(k - i1 - i2 + 1):
                    for i4 in range(k - i1 - i2 - i3 + 1):
                        for i5 in range(k - i1 - i2 - i3 - i4 + 1):
                            i6 = k - i1 - i2 - i3 - i4 - i5
                            for j1 in range(n - k + 1):
                                for j2 in range(n - k + 1 - j1):
                                    for j3 in range(n - k + 1 - j1 - j2):
                                        for j4 in range(n - k + 1 - j1 - j2 - j3):
                                            for j5 in range(n - k + 1 - j1 - j2 - j3 - j4):
                                                j6 = n - k - j1 - j2 - j3 - j4 - j5
                                                s = i1+i4; p = i2+i5; q = i3+i6
                                                ce = 0.0
                                                if s > 0:
                                                    ce += s * e[0,0]**(s-1) * e[0,1]**p * e[0,2]**q * de0
                                                if p > 0:
                                                    ce += p * e[0,0]**s * e[0,1]**(p-1) * e[0,2]**q * de1
                                                if q > 0:
                                                    ce += q * e[0,0]**s * e[0,1]**p * e[0,2]**(q-1) * de2
                                                du[k // 2] += (
                                                    a[k, i1, i2, i3, i4, i5, i6]
                                                    * b[n-k, j1, j2, j3, j4, j5, j6]
                                                    * ta[i1+j1, i2+j2, i3+j3]
                                                    * tbp[i4+j4, i5+j5, i6+j6]
                                                    * ce
                                                )
        du[k // 2] *= t[k // 2]
    return float(du.sum())


# ── public API ────────────────────────────────────────────────────────────────

def du_x(g, m, tk, a, b, e, r, dx, ta, tbp):
    """Partial of the mutual potential with respect to position element *dx*.

    Args:
        g   : gravitational constant (km³/(kg·s²))
        m   : truncation order
        tk  : tk coefficient array from ``tk_calc``
        a,b : coefficient arrays from ``a_calc`` / ``b_calc``
        e   : unit relative-position vector, shape (1,3)
        r   : relative-position magnitude (km)
        dx  : index (0, 1, or 2) of position component
        ta  : primary inertia integrals
        tbp : secondary inertia integrals rotated into A frame

    Returns:
        scalar partial dU/dx_dx (km-kg-s)
    """
    m = int(m)
    du = 0.0
    for n in range(m + 1):
        x_dx = r * e[0, dx]
        du += (-(n + 1.0) * x_dx / r**(n + 3.0)) * _u_tilde(n, tk[n], a, b, e, ta, tbp) \
            + (1.0 / r**(n + 1.0)) * _du_dx_tilde(n, tk[n], a, b, e, r, dx, ta, tbp)
    return -g * du


def du_c(g, m, tk, a, b, e, r, ta, tbp, dt):
    """Partial of the mutual potential with respect to a rotation matrix element.

    Args:
        dt  : partial of the rotated secondary inertia integrals w.r.t. C(i,j)
              (compute with ``dT_dc`` from ``potential.py``)

    Returns:
        scalar partial
    """
    m = int(m)
    du = 0.0
    for n in range(m + 1):
        t = tk[n]
        u_val = np.zeros((t.size, 1))
        for k in range(n, -1, -2):
            for i1 in range(k + 1):
                for i2 in range(k - i1 + 1):
                    for i3 in range(k - i1 - i2 + 1):
                        for i4 in range(k - i1 - i2 - i3 + 1):
                            for i5 in range(k - i1 - i2 - i3 - i4 + 1):
                                i6 = k - i1 - i2 - i3 - i4 - i5
                                for j1 in range(n - k + 1):
                                    for j2 in range(n - k + 1 - j1):
                                        for j3 in range(n - k + 1 - j1 - j2):
                                            for j4 in range(n - k + 1 - j1 - j2 - j3):
                                                for j5 in range(n - k + 1 - j1 - j2 - j3 - j4):
                                                    j6 = n - k - j1 - j2 - j3 - j4 - j5
                                                    u_val[k // 2] += (
                                                        a[k, i1, i2, i3, i4, i5, i6]
                                                        * b[n-k, j1, j2, j3, j4, j5, j6]
                                                        * e[0,0]**(i1+i4) * e[0,1]**(i2+i5) * e[0,2]**(i3+i6)
                                                        * ta[i1+j1, i2+j2, i3+j3]
                                                        * dt[i4+j4, i5+j5, i6+j6]
                                                    )
            u_val[k // 2] *= t[k // 2]
        du += float(u_val.sum()) / r**(n + 1.0)
    return -g * du


def potential(g, m, tk, a, b, e, r, ta, tbp):
    """Compute the mutual gravitational potential.

    Returns:
        U (scalar, km-kg-s units — negative means bound)
    """
    m = int(m)
    u = 0.0
    for n in range(m + 1):
        u += _u_tilde(n, tk[n], a, b, e, ta, tbp) / r**(n + 1.0)
    return -g * u
