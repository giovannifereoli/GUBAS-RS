"""
run_example_OD_stokes_2.py
==========================
AugmentedDynamicsModel OD interface — both bodies, separate degree/order.

Augmented state:  z = [x(30); θ_a(N_a); θ_b(N_b)]
  θ_a = primary   T_a inertia integrals at degrees [min_degree_a, max_degree_a]
  θ_b = secondary T_b inertia integrals at degrees [min_degree_b, max_degree_b]

Interface A — model.eval(z, t) → (ż, A_aug)
    Single-point evaluation: ODE RHS and full (30+N_a+N_b)² Jacobian.
    For an EKF/UKF: call this at each step, propagate Ṗ = A·P + P·Aᵀ + Q yourself.

Interface B — model.eval_stm(z_phi_flat, t) → rhs_flat
    Bundles [ż; A_aug·Φ_aug] into one vector for solve_ivp.
    For a batch propagator: scipy integrates z and Φ_aug together in one call.

Φ_aug block layout  (n_aug = 30 + N_a + N_b):
  ┌──────────┬────────────┬────────────┐
  │ Φ_xx     │ Φ_xθ_a    │ Φ_xθ_b    │  ← rows 0..30
  ├──────────┼────────────┼────────────┤
  │ 0        │ I_Na       │ 0          │  ← rows 30..30+N_a
  ├──────────┼────────────┼────────────┤
  │ 0        │ 0          │ I_Nb       │  ← rows 30+N_a..n_aug
  └──────────┴────────────┴────────────┘

After propagation Φ_xθ_a and Φ_xθ_b are converted independently to
∂x/∂C_{lm} and ∂x/∂S_{lm} via the Stokes linear map (stokes_utils.py).

Usage
-----
  cd <repo_root>/example
  python run_example_OD_stokes_2.py
  python run_example_OD_stokes_2.py --min-a 2 --max-a 3 --min-b 2 --max-b 2
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
p.add_argument("--min-a",  type=int, default=2, help="primary min degree (default 2)")
p.add_argument("--max-a",  type=int, default=2, help="primary max degree (default 2)")
p.add_argument("--min-b",  type=int, default=2, help="secondary min degree (default 2)")
p.add_argument("--max-b",  type=int, default=2, help="secondary max degree (default 2)")
p.add_argument("--rtol",   type=float, default=1e-10, help="scipy relative tolerance")
p.add_argument("--skip-rust", action="store_true",
               help="skip Rust run, use existing output_phi_aug/")
args = p.parse_args()
MIN_A, MAX_A = args.min_a, args.max_a
MIN_B, MAX_B = args.min_b, args.max_b

os.chdir(EXAMPLE_DIR)

try:
    import gubas_rs
except ImportError:
    print("ERROR: gubas_rs not installed.")
    print(f"  cd {REPO_ROOT}/gubas_rs")
    print("  VIRTUAL_ENV=../.venv PATH=../.venv/bin:$PATH maturin develop --release")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Build AugmentedDynamicsModel (both bodies)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: Build AugmentedDynamicsModel (both bodies)")
print("=" * 60)

model = gubas_rs.AugmentedDynamicsModel(
    min_degree_a=MIN_A, max_degree_a=MAX_A,
    min_degree_b=MIN_B, max_degree_b=MAX_B,
)

N_AUG   = model.n_aug
N_A     = model.n_theta_a
N_B     = model.n_theta_b
idx_a   = model.theta_indices_a   # list of (i,j,k) for T_a
idx_b   = model.theta_indices_b   # list of (i,j,k) for T_b
theta0  = np.array(model.theta_nominal)      # [θ_a; θ_b]
theta0a = np.array(model.theta_nominal_a)
theta0b = np.array(model.theta_nominal_b)

print(f"  n_aug = {N_AUG}  (30 state + {N_A} θ_a + {N_B} θ_b)")
print(f"  θ_a ({N_A}): {inertia_labels(idx_a)}")
print(f"  θ_b ({N_B}): {inertia_labels(idx_b)}")
print(f"  |θ_a|_inf = {np.abs(theta0a).max():.4e}  |θ_b|_inf = {np.abs(theta0b).max():.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Run Rust augmented propagation (both bodies, reference)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2: Rust built-in augmented propagation (reference)")
print("=" * 60)

OUT = "output_phi_aug"

if not args.skip_rust:
    t0_rust = time.perf_counter()
    gubas_rs.run_stm_augmented_both_py(
        min_degree_a=MIN_A, max_degree_a=MAX_A,
        min_degree_b=MIN_B, max_degree_b=MAX_B,
    )
    print(f"  Rust wall time: {time.perf_counter() - t0_rust:.2f} s")

ts_rust   = np.fromfile(f"{OUT}/phi_t_out.bin",   dtype="<f8")
xs_rust   = np.fromfile(f"{OUT}/x_out.bin",       dtype="<f8").reshape(-1, 30)
phi_aug_r = np.fromfile(f"{OUT}/phi_aug_out.bin", dtype="<f8").reshape(-1, N_AUG, N_AUG)
nsteps    = len(ts_rust)
t0, tf    = ts_rust[0], ts_rust[-1]
x0        = xs_rust[0]

print(f"  {nsteps} snapshots, t ∈ [{t0:.0f}, {tf:.0f}] s,  Φ_aug {N_AUG}×{N_AUG}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — scipy DOP853 via Interface B  model.eval_stm(z_phi_flat, t)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: scipy DOP853  model.eval_stm(z_phi_flat, t)")
print("=" * 60)

# z0 = [x0; θ_a0; θ_b0],  Φ_aug(t0) = I
z0        = np.concatenate([x0, theta0])
phi0_flat = np.eye(N_AUG).ravel()
y0        = np.concatenate([z0, phi0_flat])   # length N_AUG + N_AUG²

def aug_ode(t, y):
    return np.array(model.eval_stm(y.tolist(), t))

print(f"  y0 length = {len(y0)} = {N_AUG} + {N_AUG}²")
print(f"  Propagating t=[{t0:.0f}, {tf:.0f}] s,  rtol={args.rtol:.0e} …")

t_sci = time.perf_counter()
sol = solve_ivp(
    aug_ode, [t0, tf], y0,
    method="DOP853",
    rtol=args.rtol, atol=args.rtol * 1e-3,
    t_eval=ts_rust,
    dense_output=False,
)
dt_sci = time.perf_counter() - t_sci
print(f"  scipy: success={sol.success},  {sol.t.shape[0]} pts,  "
      f"{dt_sci:.2f} s,  {sol.nfev} func evals")

if not sol.success:
    print(f"  WARNING: {sol.message}")

y_sci   = sol.y.T                                          # (nsteps, N_AUG + N_AUG²)
xs_sci  = y_sci[:, :30]                                    # (nsteps, 30)
phi_sci = y_sci[:, N_AUG:].reshape(-1, N_AUG, N_AUG)      # (nsteps, N_AUG, N_AUG)

n_match = min(len(ts_rust), len(sol.t))
ts_cmp  = ts_rust[:n_match]
t_day   = ts_cmp / 86400.0


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — Validation: Rust vs scipy
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4: Validation — Rust vs scipy")
print("=" * 60)

dr      = np.linalg.norm(xs_sci[:n_match, :3]  - xs_rust[:n_match, :3],  axis=1)
dv      = np.linalg.norm(xs_sci[:n_match, 3:6] - xs_rust[:n_match, 3:6], axis=1)
dphi    = np.array([np.linalg.norm(phi_sci[s] - phi_aug_r[s], "fro")
                    for s in range(n_match)])
dphi_xx  = np.array([np.linalg.norm(
                phi_sci[s, :30, :30] - phi_aug_r[s, :30, :30], "fro")
                      for s in range(n_match)])
dphi_xta = np.array([np.linalg.norm(
                phi_sci[s, :30, 30:30+N_A] - phi_aug_r[s, :30, 30:30+N_A], "fro")
                      for s in range(n_match)])
dphi_xtb = np.array([np.linalg.norm(
                phi_sci[s, :30, 30+N_A:]   - phi_aug_r[s, :30, 30+N_A:],   "fro")
                      for s in range(n_match)])

print(f"  max Δr       = {dr.max():.3e} km")
print(f"  max Δv       = {dv.max():.3e} km/s")
print(f"  max ΔΦ_aug   = {dphi.max():.3e}  (Frobenius)")
print(f"  max ΔΦ_xx    = {dphi_xx.max():.3e}")
print(f"  max ΔΦ_xθ_a  = {dphi_xta.max():.3e}")
print(f"  max ΔΦ_xθ_b  = {dphi_xtb.max():.3e}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 — Stokes conversion: Φ_xθ_a → Φ_xCS_a,  Φ_xθ_b → Φ_xCS_b
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: Stokes conversion  Φ_xθ_a → Φ_xCS_a,  Φ_xθ_b → Φ_xCS_b")
print("=" * 60)

phi_xta_sci  = phi_sci[:n_match,  :30, 30:30+N_A]
phi_xtb_sci  = phi_sci[:n_match,  :30, 30+N_A:]
phi_xta_rust = phi_aug_r[:n_match, :30, 30:30+N_A]
phi_xtb_rust = phi_aug_r[:n_match, :30, 30+N_A:]

phi_xcs_a_sci,  M_a, _ = convert_phi_xt_to_cs(phi_xta_sci,  idx_a, MIN_A, MAX_A)
phi_xcs_a_rust, _,   _ = convert_phi_xt_to_cs(phi_xta_rust, idx_a, MIN_A, MAX_A)
phi_xcs_b_sci,  M_b, _ = convert_phi_xt_to_cs(phi_xtb_sci,  idx_b, MIN_B, MAX_B)
phi_xcs_b_rust, _,   _ = convert_phi_xt_to_cs(phi_xtb_rust, idx_b, MIN_B, MAX_B)

col_cs_a = cs_labels(MIN_A, MAX_A)
col_cs_b = cs_labels(MIN_B, MAX_B)

print(f"  M_a {M_a.shape}, rank {np.linalg.matrix_rank(M_a)}")
print(f"  M_b {M_b.shape}, rank {np.linalg.matrix_rank(M_b)}")
print(f"  Final ‖Φ_xCS_a‖_F  sci={np.linalg.norm(phi_xcs_a_sci[-1]):.4e}  "
      f"rust={np.linalg.norm(phi_xcs_a_rust[-1]):.4e}")
print(f"  Final ‖Φ_xCS_b‖_F  sci={np.linalg.norm(phi_xcs_b_sci[-1]):.4e}  "
      f"rust={np.linalg.norm(phi_xcs_b_rust[-1]):.4e}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 6 — Plot
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 6: Plotting")
print("=" * 60)

fig = plt.figure(figsize=(18, 18))
fig.suptitle(
    f"AugmentedDynamicsModel (both bodies) — primary [{MIN_A},{MAX_A}] N_a={N_A}, "
    f"secondary [{MIN_B},{MAX_B}] N_b={N_B},  N_aug={N_AUG}\n"
    f"scipy DOP853 (rtol={args.rtol:.0e}) vs Rust RK7(8)",
    fontsize=11)
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.35)

# ── Row 0: trajectory overlay ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0:2])
ax.plot(t_day, np.linalg.norm(xs_rust[:n_match, :3],  axis=1), "C0",   lw=2,   label="Rust")
ax.plot(t_day, np.linalg.norm(xs_sci[:n_match,  :3],  axis=1), "C1--", lw=1.2, label="scipy")
ax.set_xlabel("time (days)"); ax.set_ylabel("‖r‖ (km)")
ax.set_title("Separation"); ax.legend()

ax = fig.add_subplot(gs[0, 2:4])
ax.plot(t_day, np.linalg.norm(xs_rust[:n_match, 3:6], axis=1), "C0",   lw=2,   label="Rust")
ax.plot(t_day, np.linalg.norm(xs_sci[:n_match,  3:6], axis=1), "C1--", lw=1.2, label="scipy")
ax.set_xlabel("time (days)"); ax.set_ylabel("‖v‖ (km/s)")
ax.set_title("Velocity magnitude"); ax.legend()

# ── Row 1: integration errors ──────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0:2])
ax.semilogy(t_day, dr + 1e-30,  "C0", label="Δr (km)")
ax.semilogy(t_day, dv + 1e-30,  "C1", label="Δv (km/s)")
ax.set_xlabel("time (days)"); ax.set_ylabel("error")
ax.set_title("Trajectory error  Rust vs scipy"); ax.legend()

ax = fig.add_subplot(gs[1, 2:4])
ax.semilogy(t_day, dphi     + 1e-30, "k",    lw=1.5, label="‖ΔΦ_aug‖_F")
ax.semilogy(t_day, dphi_xx  + 1e-30, "C0--", lw=1,   label="‖ΔΦ_xx‖_F")
ax.semilogy(t_day, dphi_xta + 1e-30, "C1--", lw=1,   label="‖ΔΦ_xθ_a‖_F")
ax.semilogy(t_day, dphi_xtb + 1e-30, "C2--", lw=1,   label="‖ΔΦ_xθ_b‖_F")
ax.set_xlabel("time (days)"); ax.set_ylabel("Frobenius error")
ax.set_title("STM block errors  Rust vs scipy"); ax.legend(fontsize=8)

# ── Row 2: Stokes sensitivity norms ───────────────────────────────────────
ax = fig.add_subplot(gs[2, 0:2])
norms_a_r = np.linalg.norm(phi_xcs_a_rust[:, :3, :], axis=1)  # (n, N_cs_a)
norms_a_s = np.linalg.norm(phi_xcs_a_sci[:,  :3, :], axis=1)
for j, lbl in enumerate(col_cs_a):
    ax.semilogy(t_day, norms_a_r[:, j] + 1e-30, lw=1.5, label=f"A·{lbl}")
    ax.semilogy(t_day, norms_a_s[:, j] + 1e-30, "--", lw=0.8)
ax.set_xlabel("time (days)"); ax.set_ylabel("‖∂r/∂(C or S)_a‖")
ax.set_title("Position sensitivity — primary C/S")
ax.legend(fontsize=6, ncol=2)

ax = fig.add_subplot(gs[2, 2:4])
norms_b_r = np.linalg.norm(phi_xcs_b_rust[:, :3, :], axis=1)
norms_b_s = np.linalg.norm(phi_xcs_b_sci[:,  :3, :], axis=1)
for j, lbl in enumerate(col_cs_b):
    ax.semilogy(t_day, norms_b_r[:, j] + 1e-30, lw=1.5, label=f"B·{lbl}")
    ax.semilogy(t_day, norms_b_s[:, j] + 1e-30, "--", lw=0.8)
ax.set_xlabel("time (days)"); ax.set_ylabel("‖∂r/∂(C or S)_b‖")
ax.set_title("Position sensitivity — secondary C/S")
ax.legend(fontsize=6, ncol=2)

# ── Row 3: final-time heatmaps ─────────────────────────────────────────────
ax = fig.add_subplot(gs[3, 0:2])
pxcs_both = np.hstack([phi_xcs_a_rust[-1], phi_xcs_b_rust[-1]])  # (30, N_cs_a+N_cs_b)
vmax = np.abs(pxcs_both).max() or 1.0
im = ax.imshow(pxcs_both, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
n_cs_a = len(col_cs_a)
tick_pos = list(range(n_cs_a)) + [n_cs_a + j for j in range(len(col_cs_b))]
tick_lbl = [f"A:{l}" for l in col_cs_a] + [f"B:{l}" for l in col_cs_b]
ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=7)
ax.axvline(n_cs_a - 0.5, color="k", lw=1.5, ls="--")
ax.set_ylabel("state index")
ax.set_title(f"Φ_xCS (Rust) at t={t_day[-1]:.2f} d  [A | B]")
plt.colorbar(im, ax=ax)

ax = fig.add_subplot(gs[3, 2:4])
im2 = ax.imshow(np.log10(np.abs(phi_aug_r[-1]) + 1e-30),
                aspect="auto", cmap="viridis",
                extent=[0, N_AUG, N_AUG, 0])
ax.axhline(30,     color="w", lw=0.8, ls="--")
ax.axhline(30+N_A, color="w", lw=0.8, ls=":")
ax.axvline(30,     color="w", lw=0.8, ls="--")
ax.axvline(30+N_A, color="w", lw=0.8, ls=":")
ax.set_xlabel(f"col  (0-29: x,  30-{29+N_A}: θ_a,  {30+N_A}+: θ_b)")
ax.set_ylabel(f"row  (0-29: x,  30-{29+N_A}: θ_a,  {30+N_A}+: θ_b)")
ax.set_title(f"log₁₀|Φ_aug| (Rust, {N_AUG}×{N_AUG}) at final time")
plt.colorbar(im2, ax=ax)

out_png = os.path.join(EXAMPLE_DIR, "example_results_stokes_2.png")
fig.savefig(out_png, dpi=130, bbox_inches="tight")
print(f"  Saved → {out_png}")
plt.show()