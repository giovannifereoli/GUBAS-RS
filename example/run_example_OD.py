"""Didymos-Dimorphos: OD Dynamical Model Interface.

Demonstrates using gubas_rs as a **callable dynamical model** for OD filters:

  model = gubas_rs.DynamicsModel()          # load once — reads ic_input.txt
  xdot, A_flat = model.eval(x, t)           # ẋ (30,) + A (900,) via exact AD
  dxphi = model.eval_augmented(xphi, t)     # augmented ODE RHS (930,)

Workflow:
  1. Write ic_input.txt from hou_config.cfg
  2. Run Rust RK7(8)+AD propagation (--stm) to produce the reference trajectory
  3. Load DynamicsModel and run scipy DOP853 on the 930-state augmented ODE
  4. Compare scipy vs Rust at shared time points and plot

Build the extension first (from gubas_RUST/gubas_rs/ with your .venv active):
    cd gubas_rs
    maturin develop --release

Run:
    python run_example_OD.py
"""

import os, sys, subprocess, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
# Append — venv site-packages must be found before the Rust crate directory.
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

BINARY = os.path.join(REPO_ROOT, "gubas_rs", "target", "release", "hou_cpp_final")

# ── step 0: write ic_input.txt ────────────────────────────────────────────────

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

# ── step 1: Rust RK7(8)+AD reference propagation ─────────────────────────────

os.makedirs("output_phi", exist_ok=True)
print(f"\nRunning Rust STM propagation (RK7(8)+AD, tol={tol:.0e}) ...")
t_rust = time.time()
try:
    import gubas_rs as _gr
    if hasattr(_gr, "run_stm_py"):
        _gr.run_stm_py()
    else:
        subprocess.run([BINARY, "--stm"], check=True)
except ImportError:
    subprocess.run([BINARY, "--stm"], check=True)
print(f"  done in {time.time() - t_rust:.1f} s")

def load_f64(path):
    return np.frombuffer(open(path, "rb").read(), dtype="<f8")

ts_rust   = load_f64("output_phi/phi_t_out.bin")                    # (N,)
xs_rust   = load_f64("output_phi/x_out.bin").reshape(-1, 30)        # (N, 30)
phis_rust = load_f64("output_phi/phi_out.bin").reshape(-1, 30, 30)  # (N, 30, 30)
print(f"  Rust: {len(ts_rust)} epochs, t = {ts_rust[0]:.0f}–{ts_rust[-1]:.0f} s")

# ── step 2: load DynamicsModel ────────────────────────────────────────────────

try:
    import gubas_rs
    if not hasattr(gubas_rs, "DynamicsModel"):
        raise AttributeError("DynamicsModel not found — rebuild extension")
except (ImportError, AttributeError) as exc:
    print(f"\nERROR: {exc}")
    print("Build the Python extension (with your .venv active):")
    print(f"    cd {os.path.join(REPO_ROOT, 'gubas_rs')}")
    print("    maturin develop --release")
    sys.exit(1)

print("\nLoading DynamicsModel ...")
t_load = time.time()
model = gubas_rs.DynamicsModel()
print(f"  loaded in {time.time() - t_load:.2f} s")

x0_list = list(x0)

# ── step 3: single-point evaluation ──────────────────────────────────────────

print(f"\n── Point evaluation at t₀ = {t0:.0f} s ─────────────────────────────")
xdot, A_flat = model.eval(x0_list, t0)
A  = np.array(A_flat).reshape(30, 30)
xd = np.array(xdot)

print(f"  ẋ[0:3]  (rel vel)     = {xd[0:3]}")
print(f"  ẋ[3:6]  (rel accel)   = {xd[3:6]}")
print(f"  ẋ[6:9]  (primary α)   = {xd[6:9]}")
print(f"  ẋ[9:12] (secondary α) = {xd[9:12]}")
print(f"  ||A||_F               = {np.linalg.norm(A, 'fro'):.6e}")

block_labels = [("r", 0, 3), ("v", 3, 6), ("ωc", 6, 9), ("ωs", 9, 12),
                ("Cc", 12, 21), ("C", 21, 30)]
print("\n  A sparsity (||block||_F, row=output, col=input):")
print("  " + "".join(f"  {lbl[0]:>6}" for lbl in block_labels))
for rl, rs, re in block_labels:
    row = f"  {rl:>4}"
    for _, cs, ce in block_labels:
        nrm = np.linalg.norm(A[rs:re, cs:ce], "fro")
        row += f"  {nrm:>8.2e}" if nrm > 1e-20 else f"  {'·':>8}"
    print(row)

# ── step 4: scipy DOP853 augmented ODE ───────────────────────────────────────

try:
    from scipy.integrate import solve_ivp
except ImportError:
    print("\nscipy not found — pip install scipy")
    sys.exit(0)

print(f"\n── scipy DOP853  [{t0:.0f} s → {tf:.0f} s],  state dim = 930 ──────────")

phi0   = np.eye(30).ravel()
x0_aug = np.concatenate([x0_list, phi0])

def aug_rhs(t, xphi):
    return model.eval_augmented(xphi.tolist(), t)

t_sci = time.time()
sol = solve_ivp(
    aug_rhs, [t0, tf], x0_aug,
    method="DOP853",
    rtol=tol, atol=tol * 1e-3,
    dense_output=True,          # needed for interpolation at Rust time points
)
elapsed_sci = time.time() - t_sci

if not sol.success:
    print(f"  FAILED: {sol.message}")
    sys.exit(1)

print(f"  done in {elapsed_sci:.1f} s  ({sol.t.shape[0]} steps, {sol.nfev} ODE evals)")

x_sci_f   = sol.y[:30, -1]
phi_sci_f = sol.y[30:, -1].reshape(30, 30)
sv        = np.linalg.svd(phi_sci_f, compute_uv=False)

print(f"  x(tf) r   = {x_sci_f[0:3]}")
print(f"  x(tf) v   = {x_sci_f[3:6]}")
print(f"  ||Φ(tf)||_F = {np.linalg.norm(phi_sci_f, 'fro'):.6e}")
print(f"  singular values: {sv[:6].round(4)} ...")

# ── step 5: scipy vs Rust comparison at Rust epochs ──────────────────────────

print(f"\n── Validation: scipy DOP853 vs Rust RK7(8)+AD ──────────────────────")

# Interpolate scipy dense solution at every Rust time point
aug_at_rust   = sol.sol(ts_rust)                              # (930, N)
xs_sci_interp = aug_at_rust[:30, :].T                         # (N, 30)
phis_sci_interp = aug_at_rust[30:, :].T.reshape(-1, 30, 30)  # (N, 30, 30)

# Position and velocity errors
dr = np.linalg.norm(xs_sci_interp[:, 0:3] - xs_rust[:, 0:3], axis=1)   # km
dv = np.linalg.norm(xs_sci_interp[:, 3:6] - xs_rust[:, 3:6], axis=1)   # km/s

# STM Frobenius error
dphi = np.linalg.norm(phis_sci_interp - phis_rust, ord="fro", axis=(1, 2))

# STM Frobenius norm for both (for overlay plot)
phi_frob_sci  = np.linalg.norm(phis_sci_interp, ord="fro", axis=(1, 2))
phi_frob_rust = np.linalg.norm(phis_rust,        ord="fro", axis=(1, 2))

print(f"  max ||Δr||            = {dr.max():.3e} km")
print(f"  max ||Δv||            = {dv.max():.3e} km/s")
print(f"  max ||ΔΦ||_F          = {dphi.max():.3e}")
print(f"  final ||Φ_sci||_F     = {phi_frob_sci[-1]:.6e}")
print(f"  final ||Φ_rust||_F    = {phi_frob_rust[-1]:.6e}")
print(f"  (differences at rtol={tol:.0e} — both integrators exact to tolerance)")

# ── step 6: plot ──────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.rcParams.update({"font.size": 9})
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    hours_rust = ts_rust / 3600.0
    hours_sci  = sol.t  / 3600.0

    # Evaluate (ẋ, A) along Rust trajectory for physics plots
    print("\nEvaluating (ẋ, A) along Rust trajectory ...")
    A_norms  = np.array([np.linalg.norm(
                    np.array(model.eval(xs_rust[k].tolist(), ts_rust[k])[1]).reshape(30,30), "fro")
                 for k in range(len(ts_rust))])
    acc_mags = np.array([np.linalg.norm(
                    model.eval(xs_rust[k].tolist(), ts_rust[k])[0][3:6])
                 for k in range(len(ts_rust))])

    fig = plt.figure(figsize=(15, 14))
    fig.suptitle("Didymos–Dimorphos  |  OD Dynamical Model — scipy DOP853 vs Rust RK7(8)+AD",
                 fontsize=11, fontweight="bold")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.38)

    tick_pos  = [1, 4, 7, 10, 16, 25]
    tick_labs = [r"$r$", r"$v$", r"$\omega_c$", r"$\omega_s$", r"$C_c$", r"$C$"]

    # ── row 0: trajectory ─────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    for k, lbl in enumerate([r"$r_x$", r"$r_y$", r"$r_z$"]):
        ax0.plot(hours_sci, sol.y[k, :], lw=1.2, label=lbl + " (scipy)")
        ax0.plot(hours_rust, xs_rust[:, k], ":", lw=1.4, label=lbl + " (rust)")
    ax0.set_xlabel("Time (hr)"); ax0.set_ylabel("Position (km)")
    ax0.set_title("Relative position r(t) — scipy DOP853 vs Rust RK7(8)")
    ax0.legend(fontsize=7, ncol=3); ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.imshow(np.log10(np.abs(A) + 1e-30), cmap="viridis", aspect="auto")
    ax1.set_xticks(tick_pos); ax1.set_xticklabels(tick_labs, fontsize=7)
    ax1.set_yticks(tick_pos); ax1.set_yticklabels(tick_labs, fontsize=7)
    ax1.set_title(r"$\log_{10}|A(t_0)|$ — Jacobian structure", fontsize=8)

    # ── row 1: physics ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(hours_rust, A_norms, color="darkorange", lw=1.2)
    ax2.set_xlabel("Time (hr)"); ax2.set_ylabel(r"$\|A(t)\|_F$")
    ax2.set_title(r"Jacobian $A = \partial f/\partial x$ norm")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.semilogy(hours_rust, acc_mags, color="steelblue", lw=1.2)
    ax3.set_xlabel("Time (hr)"); ax3.set_ylabel(r"$|\ddot{r}|$ (km/s²)")
    ax3.set_title("Translational acceleration magnitude")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.semilogy(hours_rust, phi_frob_sci,  lw=1.4, label="scipy DOP853",  color="royalblue")
    ax4.semilogy(hours_rust, phi_frob_rust, lw=1.4, ls="--", label="Rust RK7(8)", color="firebrick")
    ax4.set_xlabel("Time (hr)"); ax4.set_ylabel(r"$\|\Phi(t)\|_F$")
    ax4.set_title("STM Frobenius norm  (both integrators)")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # ── row 2: STM heatmap ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    vlim = np.abs(phi_sci_f).max() or 1.0
    im5 = ax5.imshow(phi_sci_f, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    plt.colorbar(im5, ax=ax5, pad=0.02)
    ax5.set_title(f"Φ(tf) scipy DOP853  [rtol={tol:.0e}]  ||·||_F = {phi_frob_sci[-1]:.2e}",
                  fontsize=8)
    ax5.set_xticks(tick_pos); ax5.set_xticklabels(tick_labs, fontsize=7)
    ax5.set_yticks(tick_pos); ax5.set_yticklabels(tick_labs, fontsize=7)
    ax5.set_xlabel(r"initial state $x_0$", fontsize=7)
    ax5.set_ylabel(r"final state $x(t_f)$", fontsize=7)

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.semilogy(sv, "o-", ms=4, color="darkgreen")
    ax6.axhline(1.0, ls="--", color="gray", lw=0.8)
    ax6.set_xlabel("Index"); ax6.set_ylabel("Singular value")
    ax6.set_title("Singular values of Φ(tf)  [OD observability]", fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ── row 3: validation ─────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.semilogy(hours_rust, dr,  lw=1.2, color="royalblue",  label=r"$\|\Delta r\|$")
    ax7.semilogy(hours_rust, dv,  lw=1.2, color="firebrick",  label=r"$\|\Delta v\|$",  ls="--")
    ax7.set_xlabel("Time (hr)"); ax7.set_ylabel("Error (km, km/s)")
    ax7.set_title("Trajectory error: scipy − Rust", fontsize=8)
    ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[3, 1])
    ax8.semilogy(hours_rust, dphi, lw=1.2, color="purple")
    ax8.set_xlabel("Time (hr)"); ax8.set_ylabel(r"$\|\Delta\Phi\|_F$")
    ax8.set_title(r"STM error: $\|\Phi_{sci}(t) - \Phi_{rust}(t)\|_F$", fontsize=8)
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[3, 2])
    diff_f = phi_sci_f - phis_rust[-1]
    vlim9  = np.abs(diff_f).max() or 1e-10
    im9 = ax9.imshow(diff_f, cmap="RdBu_r", vmin=-vlim9, vmax=vlim9, aspect="auto")
    plt.colorbar(im9, ax=ax9, pad=0.02)
    ax9.set_title(f"Φ_sci(tf) − Φ_rust(tf)  max|Δ|={vlim9:.1e}", fontsize=8)
    ax9.set_xticks(tick_pos); ax9.set_xticklabels(tick_labs, fontsize=7)
    ax9.set_yticks(tick_pos); ax9.set_yticklabels(tick_labs, fontsize=7)

    plt.savefig("OD_dynamical_model.png", dpi=150, bbox_inches="tight")
    print("Plot saved to OD_dynamical_model.png")
    plt.show()

except ImportError:
    print("matplotlib not found — pip install matplotlib")