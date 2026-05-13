"""Hou 2016 expansion coefficients (tk, a, b) — Python reference implementation.

See Hou 2016 for equation definitions and derivations.
"""

import math
import numpy as np


def tk_calc(m):
    """Generate tk expansion coefficients up to truncation order *m*.

    Returns a 2-D array of shape ``(m+1, m//2 + 2)``.
    """
    t = np.zeros((m + 1, m // 2 + 2))
    for n in range(m + 1):
        if n % 2:  # odd
            t[n, 0] = ((-1.0) ** ((n - 1) // 2) * math.factorial(n)
                       / (2.0 ** (n - 1) * math.factorial((n - 1) // 2) ** 2))
        else:      # even
            t[n, 0] = ((-1.0) ** (n // 2) * math.factorial(n)
                       / (2.0 ** n * math.factorial(n // 2) ** 2))
        k = float(n % 2)
        i = 1
        while k <= n:
            t[n, i] = -(n - k) * (n + k + 1.0) * t[n, i - 1] / ((k + 2.0) * (k + 1.0))
            k += 2.0
            i += 1
    return t


def a_calc(n):
    """Generate *a* expansion coefficients up to truncation order *n*.

    Returns a 7-D array of shape ``(n+1,) * 7``.
    """
    n = int(n)
    a = np.zeros([n + 1] * 7)
    a[0, 0, 0, 0, 0, 0, 0] = 1.0
    if n > 0:
        a[1, 1, 0, 0, 0, 0, 0] =  1.0
        a[1, 0, 1, 0, 0, 0, 0] =  1.0
        a[1, 0, 0, 1, 0, 0, 0] =  1.0
        a[1, 0, 0, 0, 1, 0, 0] = -1.0
        a[1, 0, 0, 0, 0, 1, 0] = -1.0
        a[1, 0, 0, 0, 0, 0, 1] = -1.0
        for k in range(2, n + 1):
            for i1 in range(k + 1):
                for i2 in range(k - i1 + 1):
                    for i3 in range(k - i1 - i2 + 1):
                        for i4 in range(k - i1 - i2 - i3 + 1):
                            for i5 in range(k - i1 - i2 - i3 - i4 + 1):
                                for i6 in range(k - i1 - i2 - i3 - i4 - i5 + 1):
                                    if i1 > 0: a[k,i1,i2,i3,i4,i5,i6] += a[k-1,i1-1,i2,i3,i4,i5,i6]
                                    if i2 > 0: a[k,i1,i2,i3,i4,i5,i6] += a[k-1,i1,i2-1,i3,i4,i5,i6]
                                    if i3 > 0: a[k,i1,i2,i3,i4,i5,i6] += a[k-1,i1,i2,i3-1,i4,i5,i6]
                                    if i4 > 0: a[k,i1,i2,i3,i4,i5,i6] -= a[k-1,i1,i2,i3,i4-1,i5,i6]
                                    if i5 > 0: a[k,i1,i2,i3,i4,i5,i6] -= a[k-1,i1,i2,i3,i4,i5-1,i6]
                                    if i6 > 0: a[k,i1,i2,i3,i4,i5,i6] -= a[k-1,i1,i2,i3,i4,i5,i6-1]
    return a


def b_calc(n):
    """Generate *b* expansion coefficients up to truncation order *n*.

    Returns a 7-D array of shape ``(n+1,) * 7``.
    """
    n = int(n)
    b = np.zeros([n + 1] * 7)
    b[0, 0, 0, 0, 0, 0, 0] = 1.0
    if n > 1:
        b[2, 2, 0, 0, 0, 0, 0] =  1.0
        b[2, 0, 2, 0, 0, 0, 0] =  1.0
        b[2, 0, 0, 2, 0, 0, 0] =  1.0
        b[2, 0, 0, 0, 2, 0, 0] =  1.0
        b[2, 0, 0, 0, 0, 2, 0] =  1.0
        b[2, 0, 0, 0, 0, 0, 2] =  1.0
        b[2, 1, 0, 0, 1, 0, 0] = -2.0
        b[2, 0, 1, 0, 0, 1, 0] = -2.0
        b[2, 0, 0, 1, 0, 0, 1] = -2.0
        for k in range(n, -1, -1):
            for j1 in range(n - k + 1):
                for j2 in range(n - k + 1 - j1):
                    for j3 in range(n - k + 1 - j1 - j2):
                        for j4 in range(n - k + 1 - j1 - j2 - j3):
                            for j5 in range(n - k + 1 - j1 - j2 - j3 - j4):
                                for j6 in range(n - k + 1 - j1 - j2 - j3 - j4 - j5):
                                    if n - k > 2:
                                        if j1 > 0 and j4 > 0:
                                            b[n-k,j1,j2,j3,j4,j5,j6] += -2.0 * b[n-k-2,j1-1,j2,j3,j4-1,j5,j6]
                                        if j2 > 0 and j5 > 0:
                                            b[n-k,j1,j2,j3,j4,j5,j6] += -2.0 * b[n-k-2,j1,j2-1,j3,j4,j5-1,j6]
                                        if j3 > 0 and j6 > 0:
                                            b[n-k,j1,j2,j3,j4,j5,j6] += -2.0 * b[n-k-2,j1,j2,j3-1,j4,j5,j6-1]
                                        if j1 > 1: b[n-k,j1,j2,j3,j4,j5,j6] += b[n-k-2,j1-2,j2,j3,j4,j5,j6]
                                        if j2 > 1: b[n-k,j1,j2,j3,j4,j5,j6] += b[n-k-2,j1,j2-2,j3,j4,j5,j6]
                                        if j3 > 1: b[n-k,j1,j2,j3,j4,j5,j6] += b[n-k-2,j1,j2,j3-2,j4,j5,j6]
                                        if j4 > 1: b[n-k,j1,j2,j3,j4,j5,j6] += b[n-k-2,j1,j2,j3,j4-2,j5,j6]
                                        if j5 > 1: b[n-k,j1,j2,j3,j4,j5,j6] += b[n-k-2,j1,j2,j3,j4,j5-2,j6]
                                        if j6 > 1: b[n-k,j1,j2,j3,j4,j5,j6] += b[n-k-2,j1,j2,j3,j4,j5,j6-2]
    return b
