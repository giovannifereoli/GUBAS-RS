"""
run_example_OD_stokes.py
========================
Two OD interfaces exposed by AugmentedDynamicsModel, validated against Rust.

Interface A — model.eval(z, t) → (ż, A_aug)
    Single-point evaluation: ODE RHS and full Jacobian at (z, t).
    For an EKF/UKF: you call this at each step to get A, then propagate
    your own covariance  Ṗ = A·P + P·Aᵀ + Q  with your own integrator.

Interface B — model.eval_stm(z_phi_flat, t) → rhs_flat
    Bundles [ż; A_aug·Φ_aug] into a single vector for solve_ivp.
    For a batch propagator: scipy integrates z and Φ_aug together in one call.

After propagation, Φ_xθ (from the top-right block of Φ_aug) is converted
to ∂x/∂C_{lm} and ∂x/∂S_{lm} via the Stokes linear map.

Validation: scipy DOP853 vs Rust RK7(8) — trajectory and STM errors.

Usage
-----
  cd <repo_root>/example
  python run_example_OD_stokes.py
  python run_example_OD_stokes.py --min 2 --max 3
"""

import sys, os, argparse, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_ROOT)
sys.path.append(EXAMPLE_DIR)

from stokes_utils import convert_phi_xt_to_cs, cs_labels, inertia_labels

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--min",  type=int, default=2, help="min harmonic degree (default 2)")
p.add_argument("--max",  type=int, default=2, help="max harmonic degree (default 2)")
p.add_argument("--body", type=int, default=0, help="0=primary, 1=secondary")
p.add_argument("--rtol", type=float, default=1e-10, help="scipy relative tolerance")
p.add_argument("--skip-rust", action="store_true",
               help="skip Rust run, use existing output_phi_aug/")
args = p.parse_args()
MIN_DEG, MAX_DEG, WHICH_BODY = args.min, args.max, args.body

os.chdir(EXAMPLE_DIR)

try:
    import gubas_rs
except ImportError:
    print("ERROR: gubas_rs not installed.")
    print(f"  cd {REPO_ROOT}/gubas_rs")
    print("  VIRTUAL_ENV=../.venv PATH=../.venv/bin:$PATH maturin develop --release")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Build AugmentedDynamicsModel
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: Build AugmentedDynamicsModel")
print("=" * 60)

model = gubas_rs.AugmentedDynamicsModel(
    min_degree=MIN_DEG, max_degree=MAX_DEG, which_body=WHICH_BODY)

N_AUG     = model.n_aug          # 30 + N_theta
N_THETA   = N_AUG - 30
theta_idx = model.theta_indices   # list of (i,j,k)
theta0    = np.array(model.theta_nominal)  # actual T_{ijk} values

print(f"  n_aug   = {N_AUG}  (30 state + {N_THETA} θ parameters)")
print(f"  theta   = {inertia_labels(theta_idx)}")
print(f"  |θ|_inf = {np.abs(theta0).max():.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Run Rust augmented propagation (reference)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2: Rust built-in augmented propagation (reference)")
print("=" * 60)

OUT = "output_phi_aug"

if not args.skip_rust:
    t0_rust = time.perf_counter()
    gubas_rs.run_stm_augmented_py(
        min_degree=MIN_DEG, max_degree=MAX_DEG, which_body=WHICH_BODY)
    print(f"  Rust wall time: {time.perf_counter() - t0_rust:.2f} s")

ts_rust   = np.fromfile(f"{OUT}/phi_t_out.bin", dtype="<f8")
xs_rust   = np.fromfile(f"{OUT}/x_out.bin",     dtype="<f8").reshape(-1, 30)
phi_aug_r = np.fromfile(f"{OUT}/phi_aug_out.bin",dtype="<f8").reshape(-1, N_AUG, N_AUG)
nsteps    = len(ts_rust)
t0, tf    = ts_rust[0], ts_rust[-1]
x0        = xs_rust[0]

print(f"  {nsteps} snapshots, t ∈ [{t0:.0f}, {tf:.0f}] s")


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — Point evaluation demo
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: Point evaluation  model.eval(z, t) → (ż, A_aug)")
print("=" * 60)

z0    = np.concatenate([x0, theta0])
zdot, A_flat = model.eval(z0.tolist(), t0)
A_aug = np.array(A_flat).reshape(N_AUG, N_AUG)

print(f"  z0[:3]  (pos, km) = {z0[:3]}")
print(f"  ż0[:3]  (vel, km/s) = {zdot[:3]}")
print(f"  A_aug shape = {A_aug.shape}")
print(f"  A_aug[:3,:3] (∂ṙ/∂r):\n{A_aug[:3,:3]}")
print(f"  ||B (∂ẋ/∂θ)|| = {np.linalg.norm(A_aug[:30, 30:]):.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — scipy DOP853 propagation via model.eval_stm
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4: scipy DOP853  model.eval_stm(z_phi_flat, t)")
print("=" * 60)

# Initial condition: z0 = [x0; θ0],  Φ_aug(t0) = I
phi0_flat = np.eye(N_AUG).ravel()
y0        = np.concatenate([z0, phi0_flat])  # length N_AUG + N_AUG²

def aug_ode(t, y):
    return np.array(model.eval_stm(y.tolist(), t))

print(f"  y0 length = {len(y0)} = {N_AUG} + {N_AUG}²")
print(f"  Propagating t=[{t0:.0f}, {tf:.0f}] s with rtol={args.rtol:.0e} …")

t_sci = time.perf_counter()
sol = solve_ivp(
    aug_ode, [t0, tf], y0,
    method="DOP853",
    rtol=args.rtol, atol=args.rtol * 1e-3,
    t_eval=ts_rust,           # evaluate at same times as Rust
    dense_output=False,
)
dt_sci = time.perf_counter() - t_sci
print(f"  scipy: success={sol.success},  {sol.t.shape[0]} points,  "
      f"wall time={dt_sci:.2f} s,  {sol.nfev} func evals")

if not sol.success:
    print(f"  WARNING: {sol.message}")

# Unpack scipy solution at Rust time points
y_sci   = sol.y.T                              # (nsteps, N_AUG + N_AUG²)
xs_sci  = y_sci[:, :30]                        # (nsteps, 30) — trajectory
phi_sci = y_sci[:, N_AUG:].reshape(-1, N_AUG, N_AUG)  # (nsteps, N_AUG, N_AUG)

# Actual number of matching points (scipy may have stopped early)
n_match = min(len(ts_rust), len(sol.t))
ts_cmp  = ts_rust[:n_match]
t_day   = ts_cmp / 86400.0


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 — Validation: Rust vs scipy
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: Validation — Rust vs scipy")
print("=" * 60)

dr   = np.linalg.norm(xs_sci[:n_match, :3]  - xs_rust[:n_match, :3],  axis=1)
dv   = np.linalg.norm(xs_sci[:n_match, 3:6] - xs_rust[:n_match, 3:6], axis=1)
dphi = np.array([np.linalg.norm(phi_sci[s] - phi_aug_r[s], "fro")
                 for s in range(n_match)])

print(f"  max Δr      = {dr.max():.3e} km")
print(f"  max Δv      = {dv.max():.3e} km/s")
print(f"  max ΔΦ_aug  = {dphi.max():.3e}  (Frobenius)")


# ═════════════════════════════════════════════════════════════════════════════
# Step 6 — Stokes conversion on scipy result
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 6: Stokes conversion  Φ_xθ → Φ_xCS")
print("=" * 60)

# Extract Φ_xθ from the full Φ_aug (top-30 rows, right N_theta columns)
phi_xt_sci  = phi_sci[:n_match, :30, 30:]           # (n_match, 30, N_theta)
phi_xt_rust = phi_aug_r[:n_match, :30, 30:]         # (n_match, 30, N_theta)

phi_xcs_sci,  M, Mplus = convert_phi_xt_to_cs(phi_xt_sci,  theta_idx, MIN_DEG, MAX_DEG)
phi_xcs_rust, _, _     = convert_phi_xt_to_cs(phi_xt_rust, theta_idx, MIN_DEG, MAX_DEG)

col_cs    = cs_labels(MIN_DEG, MAX_DEG)
col_theta = inertia_labels(theta_idx)
N_CS      = M.shape[0]

print(f"  M (Stokes matrix): shape {M.shape},  rank {np.linalg.matrix_rank(M)}")
print(f"  Final ‖Φ_xCS‖_F (scipy) = {np.linalg.norm(phi_xcs_sci[-1]):.4e}")
print(f"  Final ‖Φ_xCS‖_F (Rust)  = {np.linalg.norm(phi_xcs_rust[-1]):.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 7 — Plot
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 7: Plotting")
print("=" * 60)

fig = plt.figure(figsize=(18, 16))
fig.suptitle(
    f"AugmentedDynamicsModel OD demo — degrees [{MIN_DEG},{MAX_DEG}], "
    f"N_θ={N_THETA}, N_aug={N_AUG}\n"
    f"scipy DOP853 (rtol={args.rtol:.0e}) vs Rust RK7(8) reference",
    fontsize=12)
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

# ── Row 0: trajectory overlay ────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0:2])
ax.plot(t_day, np.linalg.norm(xs_rust[:n_match, :3],  axis=1), "C0",    lw=2,   label="Rust")
ax.plot(t_day, np.linalg.norm(xs_sci[:n_match, :3],   axis=1), "C1--",  lw=1.2, label="scipy")
ax.set_xlabel("time (days)"); ax.set_ylabel("‖r‖ (km)")
ax.set_title("Separation"); ax.legend()

ax = fig.add_subplot(gs[0, 2:4])
ax.plot(t_day, np.linalg.norm(xs_rust[:n_match, 3:6], axis=1), "C0",    lw=2,   label="Rust")
ax.plot(t_day, np.linalg.norm(xs_sci[:n_match, 3:6],  axis=1), "C1--",  lw=1.2, label="scipy")
ax.set_xlabel("time (days)"); ax.set_ylabel("‖v‖ (km/s)")
ax.set_title("Velocity magnitude"); ax.legend()

# ── Row 1: trajectory and STM errors ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0:2])
ax.semilogy(t_day, dr + 1e-30, "C0", label="Δr (km)")
ax.semilogy(t_day, dv + 1e-30, "C1", label="Δv (km/s)")
ax.set_xlabel("time (days)"); ax.set_ylabel("error")
ax.set_title("Trajectory error  Rust vs scipy"); ax.legend()

ax = fig.add_subplot(gs[1, 2:4])
ax.semilogy(t_day, dphi + 1e-30, "k", lw=1.5, label="‖ΔΦ_aug‖_F")
# Also show per-block errors
dphi_xx = np.array([np.linalg.norm(phi_sci[s,:30,:30] - phi_aug_r[s,:30,:30], "fro")
                    for s in range(n_match)])
dphi_xt = np.array([np.linalg.norm(phi_sci[s,:30,30:] - phi_aug_r[s,:30,30:], "fro")
                    for s in range(n_match)])
ax.semilogy(t_day, dphi_xx + 1e-30, "C0--", lw=1, label="‖ΔΦ_xx‖_F")
ax.semilogy(t_day, dphi_xt + 1e-30, "C1--", lw=1, label="‖ΔΦ_xθ‖_F")
ax.set_xlabel("time (days)"); ax.set_ylabel("Frobenius error")
ax.set_title("STM error  Rust vs scipy"); ax.legend(fontsize=8)

# ── Row 2: Stokes sensitivities over time ─────────────────────────────────────
ax = fig.add_subplot(gs[2, 0:2])
xcs_r_norm = np.linalg.norm(phi_xcs_rust[:, :3, :], axis=1)  # pos block (n, N_cs)
xcs_s_norm = np.linalg.norm(phi_xcs_sci[:, :3, :],  axis=1)
for j, lbl in enumerate(col_cs):
    ax.semilogy(t_day, xcs_r_norm[:, j] + 1e-30, lw=1.5, label=f"Rust {lbl}")
    ax.semilogy(t_day, xcs_s_norm[:, j] + 1e-30, "--", lw=1,  label=f"sci  {lbl}")
ax.set_xlabel("time (days)"); ax.set_ylabel("‖∂r/∂(C or S)‖")
ax.set_title("Position sensitivity to Stokes C/S")
ax.legend(fontsize=6, ncol=2)

ax = fig.add_subplot(gs[2, 2:4])
dcs = np.array([np.linalg.norm(phi_xcs_sci[s] - phi_xcs_rust[s], "fro")
                for s in range(n_match)])
ax.semilogy(t_day, dcs + 1e-30, "k")
ax.set_xlabel("time (days)"); ax.set_ylabel("‖ΔΦ_xCS‖_F")
ax.set_title("Stokes sensitivity error  Rust vs scipy")

# ── Row 3: final-time heatmaps ────────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 0:2])
pxt_final = phi_xcs_rust[-1]   # (30, N_cs)
vmax = np.abs(pxt_final).max() or 1.0
im = ax.imshow(pxt_final, aspect="auto", cmap="RdBu_r",
               vmin=-vmax, vmax=vmax, extent=[0, N_CS, 30, 0])
ax.set_xticks(np.arange(N_CS) + 0.5)
ax.set_xticklabels(col_cs, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("state index")
ax.set_title(f"Φ_xCS (Rust) at t={t_day[-1]:.2f} d")
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[3, 2:4])
# Full Φ_aug (Rust) at final time
im2 = ax.imshow(np.log10(np.abs(phi_aug_r[-1]) + 1e-30),
                aspect="auto", cmap="viridis",
                extent=[0, N_AUG, N_AUG, 0])
ax.axhline(30, color="w", lw=0.8, ls="--")
ax.axvline(30, color="w", lw=0.8, ls="--")
ax.set_xlabel(f"column (0-29: state,  30+: θ)")
ax.set_ylabel(f"row    (0-29: state,  30+: θ)")
ax.set_title(f"log₁₀|Φ_aug| (Rust, {N_AUG}×{N_AUG}) at final time")
plt.colorbar(im2, ax=ax)

out_png = os.path.join(EXAMPLE_DIR, "example_results_stokes.png")
fig.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"  Saved → {out_png}")
plt.show()