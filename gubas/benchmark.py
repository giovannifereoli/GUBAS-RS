"""Reader for Fahnestock-format system parameter and initial-state files."""

import numpy as np
from numpy import linalg as la


def read_bench(param_file, state_file):
    """Read a Fahnestock-format binary asteroid parameter/state file pair.

    Args:
        param_file : path to the system-data file (e.g. ``systemdata_standard_MKS_units``)
        state_file : path to the initial-state file (e.g. ``initstate_standard_MKS_units``)

    Returns:
        (g, rho_a, rho_b, x0) all in km-kg-s units.
    """
    params = np.genfromtxt(param_file)
    states = np.genfromtxt(state_file)

    rho_a = params[0] * 1000.0**3   # kg/m³ → kg/km³
    rho_b = params[1] * 1000.0**3
    ia    = np.reshape(params[4:13],  (3, 3))
    ib    = np.reshape(params[13:22], (3, 3))
    mc    = params[22]
    ms    = params[23]
    m     = params[24]
    g     = params[31] / 1000.0**3   # m³/(kg·s²) → km³/(kg·s²)

    r0  = states[0:3] / 1000.0              # m → km
    v0  = states[3:6] / m / 1000.0          # m → km, per-reduced-mass
    wc0 = la.solve(ia, states[6:9])         # primary angular velocity in A
    c0  = states[12:21]                     # B→A rotation, row-wrapped
    cc0 = states[21:30]                     # A→N rotation, column-wrapped
    ws0 = c0.reshape(3, 3) @ la.solve(ib, c0.reshape(3, 3).T @ states[9:12])

    # A→N is stored column-wrapped → reorder to row-wrapped
    x0 = [
        r0[0],  r0[1],  r0[2],
        v0[0],  v0[1],  v0[2],
        wc0[0], wc0[1], wc0[2],
        ws0[0], ws0[1], ws0[2],
        cc0[0], cc0[3], cc0[6],
        cc0[1], cc0[4], cc0[7],
        cc0[2], cc0[5], cc0[8],
        c0[0],  c0[1],  c0[2],
        c0[3],  c0[4],  c0[5],
        c0[6],  c0[7],  c0[8],
    ]
    return g, rho_a, rho_b, x0
