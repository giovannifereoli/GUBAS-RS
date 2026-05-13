"""Python inertia integral and moment-of-inertia functions.

These are used for post-processing; the fast equivalents live in the Rust core.
See Hou 2016 for equation derivations.
"""

import math
import numpy as np
from numpy import linalg as la


def _q_ijk(i, j, k):
    return math.factorial(i) * math.factorial(j) * math.factorial(k) / math.factorial(i + j + k + 3)


def _tet_sums(l, m, n, x1, x2, x3, y1, y2, y3, z1, z2, z3):
    total = 0.0
    for i1 in range(l + 1):
        for j1 in range(l - i1 + 1):
            for i2 in range(m + 1):
                for j2 in range(m - i2 + 1):
                    for i3 in range(n + 1):
                        for j3 in range(n - i3 + 1):
                            total += (
                                math.factorial(l) / (math.factorial(i1) * math.factorial(j1) * math.factorial(l - i1 - j1))
                                * math.factorial(m) / (math.factorial(i2) * math.factorial(j2) * math.factorial(m - i2 - j2))
                                * math.factorial(n) / (math.factorial(i3) * math.factorial(j3) * math.factorial(n - i3 - j3))
                                * x1**i1 * x2**j1 * x3**(l - i1 - j1)
                                * y1**i2 * y2**j2 * y3**(m - i2 - j2)
                                * z1**i3 * z2**j3 * z3**(n - i3 - j3)
                                * _q_ijk(i1 + i2 + i3, j1 + j2 + j3, l + m + n - i1 - i2 - i3 - j1 - j2 - j3)
                            )
    return total


def poly_inertia(q, rho, tet_file, vert_file):
    """Compute inertia integrals up to order *q* from polyhedron mesh files.

    Args:
        q        : truncation order
        rho      : density in kg/km³
        tet_file : CSV file listing tetrahedron face indices (3 columns)
        vert_file: CSV file with vertex coordinates in metres (4 columns, first is index)

    Returns:
        T : ndarray of shape (q+1, q+1, q+1)
    """
    tet  = np.genfromtxt(tet_file,  delimiter=",") - 1
    vert = np.genfromtxt(vert_file, delimiter=",") / 1000.0  # m → km
    tet  = tet[:, ~np.all(np.isnan(tet), axis=0)]

    t = np.zeros((q + 1, q + 1, q + 1))
    for l in range(q + 1):
        for m in range(q + 1 - l):
            for n in range(q + 1 - m - l):
                for a in range(tet.shape[0]):
                    v1 = vert[int(tet[a, 0]), 1:4]
                    v2 = vert[int(tet[a, 1]), 1:4]
                    v3 = vert[int(tet[a, 2]), 1:4]
                    ta = abs(la.det(np.column_stack([v1, v2, v3])))
                    t[l, m, n] += rho * ta * _tet_sums(
                        l, m, n,
                        v1[0], v2[0], v3[0],
                        v1[1], v2[1], v3[1],
                        v1[2], v2[2], v3[2],
                    )
    return t


def inertia_rot(c, q, t):
    """Rotate inertia integral set *t* by rotation matrix *c*.

    Args:
        c : (3,3) rotation matrix
        q : truncation order
        t : ndarray of shape (q+1, q+1, q+1)

    Returns:
        tp : rotated inertia integral set, same shape as *t*
    """
    tp = np.zeros((q + 1, q + 1, q + 1))
    for l in range(q + 1):
        for m in range(q + 1 - l):
            for n in range(q + 1 - l - m):
                for i1 in range(l + 1):
                    for j1 in range(l - i1 + 1):
                        for i2 in range(m + 1):
                            for j2 in range(m - i2 + 1):
                                for i3 in range(n + 1):
                                    for j3 in range(n - i3 + 1):
                                        if (i1+i2+i3 <= q) and (j1+j2+j3 <= q) and \
                                                (l+m+n - i1-i2-i3 - j1-j2-j3 <= q):
                                            coef = (
                                                math.factorial(l) / (math.factorial(i1) * math.factorial(j1) * math.factorial(l - i1 - j1))
                                                * math.factorial(m) / (math.factorial(i2) * math.factorial(j2) * math.factorial(m - i2 - j2))
                                                * math.factorial(n) / (math.factorial(i3) * math.factorial(j3) * math.factorial(n - i3 - j3))
                                                * c[0,0]**i1 * c[0,1]**j1 * c[0,2]**(l-i1-j1)
                                                * c[1,0]**i2 * c[1,1]**j2 * c[1,2]**(m-i2-j2)
                                                * c[2,0]**i3 * c[2,1]**j3 * c[2,2]**(n-i3-j3)
                                            )
                                            tp[l, m, n] += coef * t[i1+i2+i3, j1+j2+j3, l+m+n-i1-i2-i3-j1-j2-j3]
    return tp


def poly_moi(rho, tet_file, vert_file):
    """Compute principal moments of inertia from a polyhedron mesh.

    Args:
        rho      : density in kg/km³
        tet_file : CSV face index file
        vert_file: CSV vertex file in metres

    Returns:
        i_diag : ndarray [Ixx, Iyy, Izz] in kg·km²
        mass   : total mass in kg
    """
    tet  = np.genfromtxt(tet_file,  delimiter=",") - 1
    vert = np.genfromtxt(vert_file, delimiter=",") / 1000.0
    tet  = tet[:, ~np.all(np.isnan(tet), axis=0)]

    i_diag = np.zeros(3)
    mass   = 0.0
    for a in range(tet.shape[0]):
        p2 = vert[int(tet[a, 0]), 1:4]
        p3 = vert[int(tet[a, 1]), 1:4]
        p4 = vert[int(tet[a, 2]), 1:4]
        vol = rho * abs(la.det(np.column_stack([p2, p3, p4]))) / 6.0
        i_diag[0] += vol * (
            p2[1]**2 + p2[1]*p3[1] + p3[1]**2 + p2[1]*p4[1] + p3[1]*p4[1] + p4[1]**2 +
            p2[2]**2 + p2[2]*p3[2] + p3[2]**2 + p2[2]*p4[2] + p3[2]*p4[2] + p4[2]**2
        ) / 10.0
        i_diag[1] += vol * (
            p2[0]**2 + p2[0]*p3[0] + p3[0]**2 + p2[0]*p4[0] + p3[0]*p4[0] + p4[0]**2 +
            p2[2]**2 + p2[2]*p3[2] + p3[2]**2 + p2[2]*p4[2] + p3[2]*p4[2] + p4[2]**2
        ) / 10.0
        i_diag[2] += vol * (
            p2[0]**2 + p2[0]*p3[0] + p3[0]**2 + p2[0]*p4[0] + p3[0]*p4[0] + p4[0]**2 +
            p2[1]**2 + p2[1]*p3[1] + p3[1]**2 + p2[1]*p4[1] + p3[1]*p4[1] + p4[1]**2
        ) / 10.0
        mass += vol
    return i_diag, mass
