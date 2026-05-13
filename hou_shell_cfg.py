"""GUBAS main driver script.

Reads ``hou_config.cfg``, writes ``ic_input.txt``, calls the Rust integrator,
and optionally runs post-processing to produce CSV output files.

Usage:
    python hou_shell_cfg.py                  # uses ./hou_config.cfg
    python hou_shell_cfg.py my_config.cfg    # uses a custom config file

The Rust integrator is invoked via the compiled ``hou_cpp_final`` binary
(subprocess fallback) or the ``gubas_rs`` Python extension if available.
"""

import csv
import datetime
import os
import struct
import subprocess
import sys
from decimal import Decimal

import numpy as np
from numpy import linalg as la

from gubas.config import hou_config_read
from gubas.icfile import write_icfile
from gubas.coefficients import a_calc, b_calc, tk_calc
from gubas.inertia import inertia_rot
from gubas.potential import potential
from gubas.postprocess import postprocess

# ── read config ───────────────────────────────────────────────────────────────

cfg_file = sys.argv[1] if len(sys.argv) > 1 else "hou_config.cfg"
print(datetime.datetime.now())

(
    g, n, na, nb, aa, ba, ca, ab, bb, cb,
    a_shape, b_shape, rho_a, rho_b, t0, tf,
    tet_file_a, vert_file_a, tet_file_b, vert_file_b,
    x0, tgen, integ, h, tol,
    out_freq, out_time_name, case,
    flyby_toggle, helio_toggle, sg_toggle, tt_toggle,
    mplanet, a_hyp, e_hyp, i_hyp, raan_hyp, om_hyp, tau_hyp,
    msolar, a_helio, e_helio, i_helio, raan_helio, om_helio, tau_helio,
    sol_rad, au_def,
    love1, love2, refrad1, refrad2, eps1, eps2,
    msun, post_processing,
) = hou_config_read(cfg_file)

# mutual-potential order must be at least as large as each body's inertia order
n = max(n, na, nb)
if a_shape == 2 and na < n:
    na = n
    print("### Expansion Order of Primary Shape Increased to Match Integrated Order ###")
if b_shape == 2 and nb < n:
    nb = n
    print("### Expansion Order of Secondary Shape Increased to Match Integrated Order ###")

print(f"\n### Expansion Order Set To: {n} ###\n")

# ── validate output frequency ─────────────────────────────────────────────────

if out_freq > 0 and integ != 3:
    mod_check = Decimal(str(out_freq)) % Decimal(str(h))
    if out_freq < h or mod_check != 0:
        raise ValueError("Output frequency is not commensurate with the integration step size")

# ── announce selections ───────────────────────────────────────────────────────

_integ_names = {1: "RK4", 2: "LGVI", 3: "RK 7(8)", 4: "A-B-M"}
if integ not in _integ_names:
    raise ValueError(f"Invalid Integrator Selection: {integ}")
if integ == 3:
    print(f"### Integrator Set to {_integ_names[integ]} from {t0}s to {tf}s with tolerance {tol} ###")
else:
    print(f"### Integrator Set to {_integ_names[integ]} from {t0}s to {tf}s with step {h}s ###")

_shape_names = {0: "Sphere", 1: "Ellipsoid", 2: "Full Shape Model"}
if a_shape not in _shape_names:
    raise ValueError(f"Bad Shape Selection for Primary: {a_shape}")
if b_shape not in _shape_names:
    raise ValueError(f"Bad Shape Selection for Secondary: {b_shape}")

print(f"\n### Primary: {_shape_names[a_shape]} (order {na}) ###")
print(f"### Secondary: {_shape_names[b_shape]} (order {nb}) ###\n")

# ── write ic_input.txt ────────────────────────────────────────────────────────

write_icfile(
    g, n, na, nb, aa, ba, ca, ab, bb, cb,
    a_shape, b_shape, rho_a, rho_b, t0, tf,
    f"TDP_{n}.csv", f"TDS_{n}.csv", "IDP.csv", "IDS.csv",
    tet_file_a, vert_file_a, tet_file_b, vert_file_b,
    x0, tgen, integ, h, tol,
    flyby_toggle, helio_toggle, sg_toggle, tt_toggle,
    mplanet, a_hyp, e_hyp, i_hyp, raan_hyp, om_hyp, tau_hyp,
    msolar, a_helio, e_helio, i_helio, raan_helio, om_helio, tau_helio,
    sol_rad, au_def,
    love1, love2, refrad1, refrad2, eps1, eps2,
    msun,
)

# ── run Rust integrator ───────────────────────────────────────────────────────

try:
    import gubas_rs
    gubas_rs.run()
except ImportError:
    # Fall back to the compiled binary (drop-in replacement)
    subprocess.run(["./hou_cpp_final"], check=True)

print(datetime.datetime.now())

# ── load inertia integral output ──────────────────────────────────────────────

ta_raw = np.genfromtxt(f"TDP_{n}.csv", delimiter=",")
tb_raw = np.genfromtxt(f"TDS_{n}.csv", delimiter=",")

# Handle scalar case (order-0 sphere)
if ta_raw.ndim == 0: ta_raw = np.array([[[float(ta_raw)]]])
if tb_raw.ndim == 0: tb_raw = np.array([[[float(tb_raw)]]])

# Rust saves as (n+1)² rows × (n+1) cols; reshape then transpose axes
side = len(ta_raw.T)
ta_raw = ta_raw.reshape(side, side, side)
tb_raw = tb_raw.reshape(side, side, side)

ta = np.zeros_like(ta_raw)
tb = np.zeros_like(tb_raw)
for f1 in range(n + 1):
    for f2 in range(n + 1):
        for f3 in range(n + 1):
            ta[f1, f2, f3] = ta_raw[f3, f1, f2]
            tb[f1, f2, f3] = tb_raw[f3, f1, f2]

ia_diag = np.genfromtxt("IDP.csv", delimiter=",")
ib_diag = np.genfromtxt("IDS.csv", delimiter=",")

mc = ta[0, 0, 0]
ms = tb[0, 0, 0]
m  = mc * ms / (mc + ms)

# ── post-processing ───────────────────────────────────────────────────────────

if not post_processing:
    print("Skipping post-processing; binary output files written.")
    print(datetime.datetime.now())
    sys.exit(0)

print("Post Processing...")
tk = tk_calc(n)
a  = a_calc(n)
b  = b_calc(n)

postprocess(
    g=g, n=n, tk=tk, a=a, b=b,
    ta=ta, tb=tb,
    ia_diag=ia_diag, ib_diag=ib_diag,
    mc=mc, ms=ms, m=m,
    tf=tf, out_freq=out_freq, h=h, integ=integ, case=case,
    out_time_name=out_time_name,
    flyby_toggle=flyby_toggle, helio_toggle=helio_toggle,
)

print(datetime.datetime.now())
