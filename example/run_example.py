"""GUBAS example: Didymos binary system (2-hour LGVI integration).

This script runs a short simulation of the Didymos binary asteroid system
using the Rust integrator and plots the relative-position trajectory and
energy/angular-momentum conservation.

The example uses:
  - Primary  : Didymos-A polyhedron shape (order-4 inertia integrals)
  - Secondary: Dimorphos ellipsoid (order-4)
  - Integrator: LGVI (symplectic, fixed 30-second step)
  - Duration : 2 hours (7200 seconds)

Run from the example/ directory:
    cd example
    python run_example.py

Or from the repo root:
    python example/run_example.py
"""

import os
import struct
import subprocess
import sys

import numpy as np

# ── locate the example directory and switch into it ───────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Make sure the repo root's gubas package is importable
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)  # append so venv site-packages take priority

from gubas.config import hou_config_read
from gubas.icfile import write_icfile
from gubas.coefficients import a_calc, b_calc, tk_calc
from gubas.inertia import inertia_rot
from gubas.potential import potential

# ── read the example config ───────────────────────────────────────────────────

print("Reading hou_config.cfg ...")
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
) = hou_config_read("hou_config.cfg")

n = max(n, na, nb)
if a_shape == 2: na = max(na, n)
if b_shape == 2: nb = max(nb, n)

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

# ── run the integrator ────────────────────────────────────────────────────────

print(f"Integrating {tf:.0f} s with LGVI (step = {h:.0f} s) ...")
os.makedirs("output_t", exist_ok=True)
os.makedirs("output_x", exist_ok=True)

try:
    import gubas_rs          # maturin-built Python extension
    gubas_rs.run()
except (ImportError, AttributeError):
    # Fall back to the standalone binary
    binary = os.path.join(REPO_ROOT, "gubas_rs", "target", "release", "hou_cpp_final")
    if not os.path.isfile(binary):
        binary = "./hou_cpp_final"
    subprocess.run([binary], check=True)

print("Integration done.")

# ── read binary output ────────────────────────────────────────────────────────

STATE_LEN = 30

def read_bin_t(path):
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.float64)

def read_bin_x(path, nsteps):
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.float64).reshape(nsteps, STATE_LEN)

times = read_bin_t("output_t/t_out.bin")
nsteps = len(times)
states = read_bin_x("output_x/x_out.bin", nsteps)

print(f"Read {nsteps} steps.")

# ── load inertia integrals ────────────────────────────────────────────────────

import numpy as np
ta_raw = np.genfromtxt(f"TDP_{n}.csv", delimiter=",")
tb_raw = np.genfromtxt(f"TDS_{n}.csv", delimiter=",")
if ta_raw.ndim == 0: ta_raw = np.array([[[float(ta_raw)]]])
if tb_raw.ndim == 0: tb_raw = np.array([[[float(tb_raw)]]])

side = len(ta_raw.T)
ta_raw = ta_raw.reshape(side, side, side)
tb_raw = tb_raw.reshape(side, side, side)
ta = np.zeros_like(ta_raw); tb = np.zeros_like(tb_raw)
for f1 in range(n + 1):
    for f2 in range(n + 1):
        for f3 in range(n + 1):
            ta[f1, f2, f3] = ta_raw[f3, f1, f2]
            tb[f1, f2, f3] = tb_raw[f3, f1, f2]

ia_diag = np.genfromtxt("IDP.csv", delimiter=",")
ib_diag = np.genfromtxt("IDS.csv", delimiter=",")
mc = ta[0, 0, 0]; ms = tb[0, 0, 0]; m = mc * ms / (mc + ms)
ia = np.diag(ia_diag); ib = np.diag(ib_diag)

# ── compute energy and angular momentum ───────────────────────────────────────

tk = tk_calc(n)
a_c  = a_calc(n)
b_c  = b_calc(n)

rel_pos_km = np.zeros((nsteps, 3))
energies   = np.zeros(nsteps)
h_norms    = np.zeros(nsteps)

print("Computing energy/angular momentum (this may take a moment) ...")
for i, u in enumerate(states):
    cc = u[12:21].reshape(3, 3)
    c  = u[21:30].reshape(3, 3)
    cs = cc @ c
    r_a = u[0:3]
    r_n = cc @ r_a
    v_n = cc @ u[3:6]
    wc  = u[6:9]
    ws  = c.T @ u[9:12]

    r_mag = np.linalg.norm(r_n)
    e_vec = (cc.T @ (r_n / r_mag)).reshape(1, 3)
    tbp   = inertia_rot(c, n, tb)
    u_pot = potential(g, n, tk, a_c, b_c, e_vec, r_mag, ta, tbp)

    kt  = 0.5 * m * v_n @ v_n
    kr1 = 0.5 * wc @ ia @ wc
    kr2 = 0.5 * ws @ ib @ ws
    h_vec = m * np.cross(r_n, v_n) + cc @ (ia @ wc) + cs @ (ib @ ws)

    rel_pos_km[i] = r_a * 1000.0   # km → m for display
    energies[i]   = u_pot + kt + kr1 + kr2
    h_norms[i]    = np.linalg.norm(h_vec)

de = (energies[0] - energies) / abs(energies[0])
dh = (h_norms[0]  - h_norms)  / h_norms[0]

print(f"  Max |ΔE/E0| = {np.max(np.abs(de)):.2e}")
print(f"  Max |ΔH/H0| = {np.max(np.abs(dh)):.2e}")

# ── plot (optional, requires matplotlib) ─────────────────────────────────────

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # relative position trajectory (X-Y plane, metres)
    axes[0].plot(rel_pos_km[:, 0], rel_pos_km[:, 1], lw=0.8)
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)")
    axes[0].set_title("Relative Position (A frame)")
    axes[0].set_aspect("equal")

    # energy conservation
    axes[1].plot(times / 3600, de)
    axes[1].set_xlabel("Time (hr)"); axes[1].set_ylabel("ΔE / E₀")
    axes[1].set_title("Energy Conservation")

    # angular momentum conservation
    axes[2].plot(times / 3600, dh)
    axes[2].set_xlabel("Time (hr)"); axes[2].set_ylabel("ΔH / H₀")
    axes[2].set_title("Angular Momentum Conservation")

    plt.tight_layout()
    plt.savefig("example_results.png", dpi=150)
    print("Plot saved to example_results.png")
    plt.show()

except ImportError:
    print("matplotlib not found — skipping plots.")
    print("Install with: pip install matplotlib")
