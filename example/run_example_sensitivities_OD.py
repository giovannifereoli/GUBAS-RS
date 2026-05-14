"""Didymos-Dimorphos: STM and OD Sensitivities.

Demonstrates orbit-determination (OD) partials for the Didymos binary system:

  1. Computes the 30×30 State Transition Matrix Φ(t, t₀) using the Rust
     propagator with an exact forward-mode AD Jacobian.
  2. Compares AD vs FD Jacobian at t₀ — AD gives machine-precision derivatives,
     FD has O(h) truncation error (~1e-7).
  3. Plots the evolution of key OD partial blocks and the full STM structure.

State vector layout (30 elements):
  [0:3]   r     relative position (km, A-frame)
  [3:6]   v     relative velocity (km/s, A-frame, = Cc^T ṙ_I)
  [6:9]   ωc    primary spin (rad/s, principal frame)
  [9:12]  ωs    secondary spin (rad/s, principal frame)
  [12:21] Cc    primary attitude DCM (row-major, inertial→body)
  [21:30] C     secondary-relative attitude DCM

Run from the example/ directory:
    python run_example_sensitivities_OD.py
"""

import os, struct, subprocess, sys, time
import numpy as np

# ── setup paths ───────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)  # append so venv site-packages take priority

BINARY = os.path.join(REPO_ROOT, "gubas_rs", "target", "release", "hou_cpp_final")

def run_rust(flag=None):
    """Try the Python extension first, fall back to the release binary."""
    try:
        import gubas_rs
        if flag == "--stm":
            gubas_rs.run_stm_py()
        else:
            gubas_rs.run()
        return
    except (ImportError, AttributeError):
        pass
    cmd = [BINARY] + ([flag] if flag else [])
    subprocess.run(cmd, check=True)

# ── step 1: write ic_input.txt from hou_config.cfg ────────────────────────────

from gubas.config import hou_config_read
from gubas.icfile import write_icfile

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

# ── step 2: run STM propagation ───────────────────────────────────────────────

os.makedirs("output_phi", exist_ok=True)
print(f"\nRunning STM propagation ({tf:.0f} s, h={h:.0f} s, RK4 + AD) ...")
t_start = time.time()
run_rust("--stm")
print(f"  done in {time.time() - t_start:.1f} s")

# ── step 3: load results ───────────────────────────────────────────────────────

def load_f64(path):
    return np.frombuffer(open(path, "rb").read(), dtype="<f8")

ts     = load_f64("output_phi/phi_t_out.bin")                    # (nsteps,)
phis   = load_f64("output_phi/phi_out.bin").reshape(-1, 30, 30)  # (nsteps, 30, 30)
jac_ad = load_f64("output_phi/jac_ad.bin").reshape(30, 30)       # (30, 30)
jac_fd = load_f64("output_phi/jac_fd.bin").reshape(30, 30)       # (30, 30)
xdots  = load_f64("output_phi/xdot_out.bin").reshape(-1, 30)     # (nsteps, 30)
As     = load_f64("output_phi/A_out.bin").reshape(-1, 30, 30)    # (nsteps, 30, 30)

print(f"\nLoaded {len(ts)} STM snapshots  (t = {ts[0]:.0f} s → {ts[-1]:.0f} s)")

# ── step 4: compute diagnostics ───────────────────────────────────────────────

# Frobenius norms of key OD partial blocks over time
# ∂r(t)/∂r₀  : Φ[0:3, 0:3]
# ∂r(t)/∂v₀  : Φ[0:3, 3:6]
# ∂r(t)/∂ωc₀ : Φ[0:3, 6:9]
# ∂r(t)/∂ωs₀ : Φ[0:3, 9:12]
# ∂r(t)/∂Cc₀ : Φ[0:3, 12:21]
# ∂ωc(t)/∂ωc₀: Φ[6:9, 6:9]

block_norms = {
    r"$\partial r / \partial r_0$":   np.linalg.norm(phis[:, 0:3, 0:3],  ord="fro", axis=(1,2)),
    r"$\partial r / \partial v_0$":   np.linalg.norm(phis[:, 0:3, 3:6],  ord="fro", axis=(1,2)),
    r"$\partial r / \partial \omega_{c0}$": np.linalg.norm(phis[:, 0:3, 6:9],  ord="fro", axis=(1,2)),
    r"$\partial r / \partial \omega_{s0}$": np.linalg.norm(phis[:, 0:3, 9:12], ord="fro", axis=(1,2)),
    r"$\partial \omega_c / \partial \omega_{c0}$": np.linalg.norm(phis[:, 6:9, 6:9], ord="fro", axis=(1,2)),
}

phi_frob = np.linalg.norm(phis, ord="fro", axis=(1,2))

# AD vs FD comparison
jac_diff = np.abs(jac_ad - jac_fd)
max_err_col = jac_diff.max(axis=0)          # per state-variable column
max_err_total = jac_diff.max()
print(f"\n  Jacobian A(x₀, t₀):")
print(f"    AD vs FD  max |Δ|        = {max_err_total:.3e}")
print(f"    AD vs FD  mean |Δ| (nonzero cols) = "
      f"{max_err_col[max_err_col > 0].mean():.3e}")

# STM condition number at final time
cond = np.linalg.cond(phis[-1])
print(f"    Φ(tf) condition number   = {cond:.2e}")
print(f"    Φ(t0) should = I — max |Φ₀ - I| = {np.abs(phis[0] - np.eye(30)).max():.2e}")

# Verify Φ̇ ≈ A·Φ at each epoch via finite difference
#   Φ̇_fd[k] = (Φ[k+1] - Φ[k]) / (t[k+1] - t[k])
#   Φ̇_theory[k] = A[k] · Φ[k]
phidot_fd  = np.diff(phis, axis=0) / np.diff(ts)[:, None, None]   # (nsteps-1, 30, 30)
phidot_th  = np.einsum("nij,njk->nik", As[:-1], phis[:-1])         # (nsteps-1, 30, 30)
phidot_err = np.linalg.norm(phidot_fd - phidot_th, ord="fro", axis=(1, 2))
A_frob     = np.linalg.norm(As, ord="fro", axis=(1, 2))            # (nsteps,)
print(f"\n  Mode 2 — dynamics + Jacobian at epochs:")
print(f"    max translational accel |ẋ[3:6]| = {np.linalg.norm(xdots[:, 3:6], axis=1).max():.4e} km/s²")
print(f"    mean ||A(t)||_F                  = {A_frob.mean():.4e}")
print(f"    max ||Φ̇_fd - A·Φ||_F            = {phidot_err.max():.4e}  (FD noise expected)")

# ── step 5: plot ──────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.rcParams.update({"font.size": 9})
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm, SymLogNorm

    hours = ts / 3600.0

    fig = plt.figure(figsize=(16, 15))
    fig.suptitle("Didymos–Dimorphos  |  STM & OD Sensitivities  (RK7(8) + AD)",
                 fontsize=12, fontweight="bold")
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.38)

    # ── row 0: OD partial block norms ─────────────────────────────────────────

    ax0 = fig.add_subplot(gs[0, :2])
    for label, vals in block_norms.items():
        ax0.semilogy(hours, vals, label=label, lw=1.2)
    ax0.set_xlabel("Time (hr)")
    ax0.set_ylabel(r"$\|$block$\|_F$")
    ax0.set_title("OD partial block norms (Frobenius)")
    ax0.legend(fontsize=7, ncol=2)
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 2:])
    ax1.semilogy(hours, phi_frob, color="k", lw=1.2)
    ax1.set_xlabel("Time (hr)")
    ax1.set_ylabel(r"$\|\Phi\|_F$")
    ax1.set_title("Full STM Frobenius norm")
    ax1.grid(True, alpha=0.3)

    # ── row 1: Jacobian heatmaps ───────────────────────────────────────────────

    # state labels for axis ticks
    labels = (["r"] * 3 + ["v"] * 3 + [r"$\omega_c$"] * 3 + [r"$\omega_s$"] * 3
              + ["Cc"] * 9 + ["C"] * 9)
    tick_pos  = [1, 4, 7, 10, 16, 25]
    tick_labs = [r"$r$", r"$v$", r"$\omega_c$", r"$\omega_s$", r"$C_c$", r"$C$"]

    def jac_heatmap(ax, mat, title, vmin=None, vmax=None, cmap="RdBu_r", symmetric=True):
        if symmetric:
            vlim = np.abs(mat).max() or 1e-30
            im = ax.imshow(mat, cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
        else:
            mat_safe = np.where(mat > 0, mat, np.nan)
            im = ax.imshow(mat_safe, cmap="viridis", norm=LogNorm(
                vmin=mat_safe[np.isfinite(mat_safe)].min() if np.any(np.isfinite(mat_safe)) else 1e-20,
                vmax=mat_safe[np.isfinite(mat_safe)].max() if np.any(np.isfinite(mat_safe)) else 1),
                aspect="auto")
        plt.colorbar(im, ax=ax, pad=0.02)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_labs, fontsize=6)
        ax.set_yticks(tick_pos); ax.set_yticklabels(tick_labs, fontsize=6)
        ax.set_xlabel("state j  (∂/∂x_j)", fontsize=7)
        ax.set_ylabel("ODE component i  (f_i)", fontsize=7)

    ax2 = fig.add_subplot(gs[1, 0])
    jac_heatmap(ax2, jac_ad, "Jacobian A  (AD, exact)", symmetric=True)

    ax3 = fig.add_subplot(gs[1, 1])
    jac_heatmap(ax3, jac_fd, "Jacobian A  (FD, h≈1.5e-8)", symmetric=True)

    ax4 = fig.add_subplot(gs[1, 2])
    diff_sym = jac_ad - jac_fd
    vlim = np.abs(diff_sym).max() or 1.0
    im4 = ax4.imshow(diff_sym, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    plt.colorbar(im4, ax=ax4, pad=0.02)
    ax4.set_title(f"AD − FD  (max |Δ|={max_err_total:.1e})", fontsize=8)
    ax4.set_xticks(tick_pos); ax4.set_xticklabels(tick_labs, fontsize=6)
    ax4.set_yticks(tick_pos); ax4.set_yticklabels(tick_labs, fontsize=6)

    ax5 = fig.add_subplot(gs[1, 3])
    ax5.bar(range(30), max_err_col, color="steelblue", width=0.8)
    ax5.axhline(1e-7, ls="--", color="r", lw=0.8, label="1e-7 FD floor")
    ax5.set_yscale("log")
    ax5.set_xlabel("State index j"); ax5.set_ylabel("max |AD−FD| over rows i")
    ax5.set_title("AD vs FD error per column", fontsize=8)
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3, axis="y")

    # ── row 2: STM snapshot at final time ─────────────────────────────────────

    ax6 = fig.add_subplot(gs[2, :2])
    phi_final = phis[-1]
    vlim = np.abs(phi_final).max() or 1.0
    im6 = ax6.imshow(phi_final, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    plt.colorbar(im6, ax=ax6, pad=0.02)
    ax6.set_title(f"Φ(t₀→{hours[-1]:.1f} hr)  —  full 30×30 STM  (cond = {cond:.1e})", fontsize=8)
    ax6.set_xticks(tick_pos); ax6.set_xticklabels(tick_labs, fontsize=7)
    ax6.set_yticks(tick_pos); ax6.set_yticklabels(tick_labs, fontsize=7)
    ax6.set_xlabel("initial state x₀  (∂x₀)", fontsize=7)
    ax6.set_ylabel("final state x(t)  (∂x)", fontsize=7)

    ax7 = fig.add_subplot(gs[2, 2:])
    # Show singular values of final STM (OD observability)
    sv = np.linalg.svd(phi_final, compute_uv=False)
    ax7.semilogy(sv, "o-", ms=4, color="darkgreen")
    ax7.set_xlabel("Singular value index")
    ax7.set_ylabel("Singular value")
    ax7.set_title("Singular values of Φ(tf)  — OD observability", fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.axhline(1.0, ls="--", color="gray", lw=0.8)

    # ── row 3: Mode 2 — dynamics ẋ and Jacobian A ────────────────────────────

    hours_mid = (ts[:-1] + ts[1:]) / 2.0 / 3600.0   # midpoints for FD Φ̇

    ax8 = fig.add_subplot(gs[3, :2])
    labels_xdot = [r"$\dot{r}_x$", r"$\dot{r}_y$", r"$\dot{r}_z$",
                   r"$\ddot{r}_x$", r"$\ddot{r}_y$", r"$\ddot{r}_z$"]
    for k, lbl in zip([0, 1, 2, 3, 4, 5], labels_xdot):
        ax8.plot(hours, xdots[:, k], lw=1.0, label=lbl)
    ax8.set_xlabel("Time (hr)")
    ax8.set_ylabel("ẋ  (km or km/s per s)")
    ax8.set_title("ODE RHS  ẋ(t) — translational & velocity components")
    ax8.legend(fontsize=7, ncol=3)
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[3, 2])
    ax9.semilogy(hours, A_frob, color="darkorange", lw=1.2)
    ax9.set_xlabel("Time (hr)")
    ax9.set_ylabel(r"$\|A(t)\|_F$")
    ax9.set_title(r"Jacobian $A=\partial f/\partial x$ Frobenius norm")
    ax9.grid(True, alpha=0.3)

    ax10 = fig.add_subplot(gs[3, 3])
    ax10.semilogy(hours_mid, phidot_err, color="purple", lw=1.0)
    ax10.set_xlabel("Time (hr)")
    ax10.set_ylabel(r"$\|\dot\Phi_{FD} - A\Phi\|_F$")
    ax10.set_title(r"Verify $\dot\Phi = A\Phi$  (FD noise)")
    ax10.grid(True, alpha=0.3)

    plt.savefig("sensitivities_OD.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to sensitivities_OD.png")
    plt.show()

except ImportError:
    print("matplotlib not found — skipping plots.  pip install matplotlib")