# GUBAS-RS — General Use Binary Asteroid Simulator (Rust)

GUBAS-RS models the coupled translational and rotational dynamics of binary asteroid systems using the Hou 2016 Full Two-Body Problem (F2BP) formulation. Each body is represented by its inertia integrals T_{ijk} up to a user-defined expansion order; the mutual gravitational potential, forces, and torques are evaluated as a truncated series expansion.

This repository is a **Rust port** of the original C++ GUBAS integrator, extended with:

- **Exact Jacobians** via forward-mode automatic differentiation (dual numbers) — no finite differences
- **State Transition Matrix (STM)** propagation alongside the trajectory
- **Augmented parameter sensitivity** — simultaneous estimation sensitivity with respect to the inertia integrals T_{ijk} of **both bodies**, with independent harmonic degree/order selection per body
- **Stokes coefficient conversion** — linear map from ∂x/∂T_{ijk} to ∂x/∂C_{lm}, ∂x/∂S_{lm} (Tricarico 2008)
- **Python interface** via PyO3/maturin — callable from any OD filter without subprocess overhead

---

## Quick start

```bash
git clone https://github.com/<org>/gubas_RUST.git
cd gubas_RUST

# Sets up .venv, installs Python deps, builds Rust binary, and (optionally) the
# Python extension — all in one step.
python initialize.py --maturin

source .venv/bin/activate
cd example && python run_example.py
```

See [Installation](#installation) below for manual steps or Windows instructions.

---

## Repository layout

```
gubas_RUST/
├── gubas_rs/                   Rust crate (core library + Python extension)
│   ├── Cargo.toml
│   ├── pyproject.toml          maturin build config
│   └── src/
│       ├── lib.rs              library root, PyO3 module registration
│       ├── main.rs             standalone binary entry point
│       ├── dynamics.rs         F2BP equations of motion (generic over scalar type)
│       ├── stm.rs              STM and augmented STM propagators
│       ├── stokes.rs           Stokes coefficient computation (Tricarico 2008)
│       ├── dual.rs             forward-mode dual number type
│       ├── types.rs            Params<T>, Cube<T> — generic simulation structures
│       ├── integrators.rs      RK4, RK7(8), LGVI, ABM
│       ├── inertia.rs          inertia integral generation (ellipsoid + polyhedron)
│       ├── potential.rs        mutual gravitational potential
│       ├── coefficients.rs     Hou expansion coefficients (tk, a, b)
│       ├── orbit.rs            Keplerian orbit utilities
│       └── math3.rs            3-vector / 3×3-matrix primitives
├── tests/                      Python pytest suite (no Rust extension required)
│   ├── conftest.py             adds example/ to sys.path automatically
│   └── test_stokes_utils.py    24 tests for stokes_utils.py
├── example/
│   ├── ic_input.txt            Didymos–Dimorphos initial conditions
│   ├── Didymos_A_facet.csv     Primary shape model — tetrahedra
│   ├── Didymos_A_vert_met.csv  Primary shape model — vertices (metres)
│   ├── TDP_4.csv / TDS_4.csv   Precomputed inertia integrals (order 4)
│   ├── IDP.csv / IDS.csv       Principal moments of inertia
│   ├── stokes_utils.py         Stokes matrix + pseudoinverse + conversion utilities
│   ├── run_example.py          Basic trajectory example
│   ├── run_example_OD.py       STM propagation + OD-style interface
│   ├── run_example_OD_stokes.py   Single-body C/S sensitivity demo
│   └── run_example_OD_stokes_2.py Both-body C/S sensitivity demo (OD-ready)
├── pytest.ini                  Pytest configuration (testpaths = tests)
├── requirements.txt            Python dependencies
├── gubas/                      Legacy Python package (config, post-processing)
└── General_Use_Binary_Asteroid_Simulator_Tool_User_Guide.pdf
```

---

## Installation

### Prerequisites

| Tool | Minimum version | Where to get it |
|------|----------------|-----------------|
| Rust toolchain | ≥ 1.75 | [rustup.rs](https://rustup.rs) |
| Python | ≥ 3.8 | [python.org](https://python.org) |

### Automated setup (recommended)

`initialize.py` handles everything in one command:

```bash
# Binary + Python extension (recommended for OD / Python use):
python initialize.py --maturin

# Binary only (no Python extension):
python initialize.py
```

What it does:
1. Checks that `cargo` is available.
2. Creates `.venv` and installs `numpy scipy matplotlib maturin pytest`.
3. Builds `gubas_rs/target/release/hou_cpp_final` (`cargo build --release`).
4. `[--maturin]` Runs `maturin develop --release` to install `import gubas_rs`.

### Manual setup

```bash
# Python dependencies
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib maturin pytest

# Rust binary
cd gubas_rs && cargo build --release

# Python extension (for OD use)
maturin develop --release
python -c "import gubas_rs; print('ok')"
```

---

## Testing

### Rust unit tests — 108 tests across all 11 modules

```bash
cd gubas_rs
cargo test
```

Expected output:

```
test result: ok. 108 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Every module is covered with analytic reference checks:

| Module | Tests | What is verified |
|--------|------:|-----------------|
| `coefficients.rs` | 20 | `factorial`, `ifact`, `t_ind` indexing; `coeff_vec_len`; `tk_calc` analytic values; `a_calc`/`b_calc` seeds |
| `dual.rs` | 16 | Product/quotient/chain rules; sin, cos, exp, ln, sqrt, powi, tan, tanh, sinh derivatives |
| `math3.rs` | 17 | i×j=k, anticommutativity, tilde↔cross, mat_mul, transpose, det, inv3, norm, trace |
| `stokes.rs` | 12 | `inertia_indices` count; label ordering; C₀₀=1; sphere C₂ₘ=0; oblate C₂₀=−3/5; triaxial C₂₂ analytic; M·N = direct `nijk_to_clm_slm` |
| `inertia.rs` | 8 | Ellipsoid mass, T₂₀₀/T₀₂₀/T₀₀₂ analytic, MOI formulas; sphere moments equal; q_ijk symmetry |
| `potential.rs` | 7 | `de_dx` diagonal/off-diagonal; monopole U=−GM₁M₂/r; monopole force = GM₁M₂/r²; transverse force = 0 |
| `orbit.rs` | 8 | Kepler's equation at periapsis/quarter/half period; `kepler2cart` circular |r|=a, r·v=0, vis-viva; periapsis radius |
| `types.rs` | 9 | `Cube<T>` new/set/get/add/zeros/extremes/order; `compute_lgvi_inertia` diagonal formula, equal moments |
| `dynamics.rs` | 2 | Point-mass centripetal acceleration = −G(M₁+M₂)/r²; zero torques for n=0 |
| `lgvi.rs` | 3 | Outer product values and antisymmetry; monopole radial force via `map_partials` |
| `integrators.rs` | 4 | RK4, ABM, RK7(8), LGVI each produce non-NaN output of correct size |
| `stm.rs` | 2 | AD Jacobian exact to machine eps; max\|AD−FD\| < 1e-5 |

Filter by module or test name:

```bash
cargo test coefficients::   # only coefficients module
cargo test monopole         # any test whose name contains "monopole"
```

### Python pytest suite — 24 tests

No compiled Rust extension required. Run from the **repo root**:

```bash
pytest          # pytest.ini sets testpaths = tests automatically
pytest -v       # verbose — shows each test name
```

Expected output:

```
24 passed in 0.1s
```

| Class | Tests | What is verified |
|-------|------:|-----------------|
| `TestCsLabels` | 5 | Count at degree 2 and 4, exact label ordering, C/S interleaving |
| `TestInertiaLabels` | 1 | Label format `Tijk` |
| `TestStokesMatrix` | 7 | Shape (degree 2 and 2–4); sphere null space; oblate C₂₀=−3/5; triaxial C₂₂=3; S=0 for aligned body; normalized/unnormalized factor √5 |
| `TestStokesPseudoinverse` | 5 | M·M⁺=I for degrees 2 and 3; shape; full row rank; null space dim |
| `TestConvertPhiXtToCs` | 4 | Output shapes (3-D / 2-D input, multi-degree); chain-rule projection identity |
| `TestContribInternals` | 2 | S_{l,0}=0; wrong-degree contributions = 0 |

`tests/conftest.py` adds `example/` to `sys.path` automatically — no manual path setup needed.

---

## Input files

All runs read `ic_input.txt` from the **current working directory**. The file is read line-by-line; here is a commented overview of every field:

```
6.67e-20          # G (km³ kg⁻¹ s⁻²)
4                 # expansion order N (max degree i+j+k)
4                 # order_a (primary)
4                 # order_b (secondary)
576.7195          # primary semi-axis a_a (m) — used if a_shape=1
576.7195          # primary semi-axis b_a (m)
576.7195          # primary semi-axis c_a (m)
103.16            # secondary semi-axis a_b (m)
79.35             # secondary semi-axis b_b (m)
66.13             # secondary semi-axis c_b (m)
2                 # a_shape: 0=sphere, 1=ellipsoid, 2=polyhedron
1                 # b_shape: 0=sphere, 1=ellipsoid, 2=polyhedron
2203111036687.207 # primary density ρ_a (kg/km³)
2203111036687.207 # secondary density ρ_b (kg/km³)
0.0               # t0 (s)
7200.0            # tf (s)
TDP_4.csv         # primary inertia integrals CSV (read or written)
TDS_4.csv         # secondary inertia integrals CSV
IDP.csv           # primary principal moments of inertia CSV
IDS.csv           # secondary principal moments of inertia CSV
Didymos_A_facet.csv    # primary polyhedron facets (only if a_shape=2)
Didymos_A_vert_met.csv # primary polyhedron vertices in metres
Didymos_B_facet.csv    # secondary polyhedron facets (only if b_shape=2)
Didymos_B_vert_met.csv # secondary polyhedron vertices
# --- 30 initial state values (one per line) ---
# r (km): x, y, z
# v (km/s): vx, vy, vz
# ωc (rad/s): primary angular velocity
# ωs (rad/s): secondary angular velocity
# Cc (9 values, row-major): inertial-to-primary rotation matrix
# C  (9 values, row-major): secondary-to-primary rotation matrix
1                 # tgen: 1=generate inertia integrals from scratch, 0=read CSV
2                 # integ: 1=RK4, 2=LGVI, 3=RK7(8), 4=ABM
30.0              # step size h (s) — fixed-step integrators only
1e-15             # tolerance — adaptive integrators only
0                 # flyby_toggle (0=off, 1=on)
0                 # helio_toggle (0=off, 1=on)
0                 # sg_toggle — legacy Hill solar gravity (0=off, 1=on)
0                 # tt_toggle — tidal torques (0=off, 1=on)
# --- flyby (hyperbolic 3rd body) parameters ---
5.9722e+24        # mplanet (kg)
-11072.0          # a_hyp (km, negative for hyperbola)
9.1284            # e_hyp (>1)
0.1745            # i_hyp (rad)
0.1745            # raan_hyp (rad)
0.1745            # om_hyp (rad)
345600.0          # tau_hyp — time of periapsis (s)
# --- heliocentric orbit parameters ---
1.988e+30         # msolar (kg)
246013901.32      # a_helio (km)
0.38362664        # e_helio
...               # i, raan, om, tau (helio)
1.0               # sol_rad (AU)
149597870.7       # au_def (km/AU)
# --- tidal torque parameters ---
1e-05             # love1 — primary Love number
1e-05             # love2 — secondary Love number
660.0             # refrad1 — primary reference radius (m)
225.0             # refrad2 — secondary reference radius (m)
0.005             # eps1 — primary tidal lag
0.005             # eps2 — secondary tidal lag
1.989e+30         # msun — solar mass for tidal dissipation (kg)
```

### Inertia integrals: generate vs load

Set `tgen = 1` on the first run to compute T_{ijk} from the body model (ellipsoid or polyhedron) and save them to `TDP_{N}.csv` / `TDS_{N}.csv`. On subsequent runs set `tgen = 0` to load the cached CSV directly — this is much faster.

For the **polyhedron** case (`a_shape = 2` or `b_shape = 2`), provide the facet and vertex CSV files. The code tetrahedralizes the volume and numerically integrates ∫ x^i y^j z^k dm. For **ellipsoid** (`a_shape = 1`), the analytic closed form is used.

---

## State vector layout

The 30-element state vector in **A-frame** (primary body frame):

| Indices | Quantity | Units |
|---------|----------|-------|
| 0–2 | r — relative position (secondary w.r.t. primary CoM) | km |
| 3–5 | v — relative velocity | km/s |
| 6–8 | ωc — primary angular velocity | rad/s |
| 9–11 | ωs — secondary angular velocity (A-frame) | rad/s |
| 12–20 | Cc — inertial-to-A rotation matrix, row-major | — |
| 21–29 | C — B-to-A rotation matrix, row-major | — |

---

## Integrators

| Flag | Name | Type | Notes |
|------|------|------|-------|
| 1 | RK4 | Fixed step | General purpose |
| 2 | LGVI | Fixed step | Symplectic, conserves SO(3) |
| 3 | RK 7(8) | Adaptive | Dormand-Prince; use for accuracy-critical runs |
| 4 | ABM | Fixed step | Adams–Bashforth–Moulton 4th order |

---

## Perturbations

| Toggle | Effect |
|--------|--------|
| `flyby_toggle = 1` | 3rd body on a hyperbolic trajectory (e.g. Earth flyby). Adds tidal orbital acceleration and gravitational torques on both bodies. Parameters: `mplanet`, `a_hyp` (< 0), `e_hyp`, `i_hyp`, `raan_hyp`, `om_hyp`, `tau_hyp`. |
| `helio_toggle = 1` | Sun on a heliocentric elliptic orbit. Same tidal structure as flyby. For long timescale runs. |
| `sg_toggle = 1` | Legacy Hill-equation solar gravity (circular orbit approximation). |
| `tt_toggle = 1` | Maxwell viscoelastic tidal torques (Mignard-style). Parameters: Love numbers, reference radii, lag angles. |

---

## Python API — OD interface

After `maturin develop --release`, the following are available in Python:

### Basic trajectory

```python
import gubas_rs
gubas_rs.run()              # reads ic_input.txt in cwd
gubas_rs.run("/path/to/")   # or specify directory
```

### STM (∂x/∂x₀) — state-only sensitivity

```python
gubas_rs.run_stm_py()
# writes output_phi/phi_out.bin   (nsteps, 30, 30)
#         output_phi/phi_t_out.bin  (nsteps,)
#         output_phi/x_out.bin      (nsteps, 30)
#         output_phi/A_out.bin      (nsteps, 30, 30) — Jacobian at each epoch
```

Or propagate from Python with exact Jacobians:

```python
model = gubas_rs.DynamicsModel()

xdot, A_flat = model.eval(x.tolist(), t)       # ẋ and ∂f/∂x at one point
A = np.array(A_flat).reshape(30, 30)

rhs = model.eval_augmented(xphi.tolist(), t)   # [ẋ; Φ̇] for solve_ivp
```

### Augmented STM — parameter sensitivity for both bodies

The augmented state is **z = [x(30); θ_a(N_a); θ_b(N_b)]** where θ_a and θ_b are the inertia integral vectors T_{ijk} of the primary and secondary at the requested harmonic degrees.  θ̇ = 0 (parameters are constant); the augmented STM Φ_aug propagates ∂x/∂θ_a and ∂x/∂θ_b alongside the trajectory.

```python
import gubas_rs, numpy as np

# Build model — select harmonic degrees independently per body
model = gubas_rs.AugmentedDynamicsModel(
    min_degree_a=2, max_degree_a=2,   # primary:   degree-2 T_{ijk} (6 params)
    min_degree_b=2, max_degree_b=2,   # secondary: degree-2 T_{ijk} (6 params)
)

N_AUG = model.n_aug        # 30 + N_a + N_b  (e.g. 42 for deg-2 both bodies)
N_A   = model.n_theta_a    # number of primary parameters
N_B   = model.n_theta_b    # number of secondary parameters

# Nominal parameter values (use as θ₀)
theta0  = np.array(model.theta_nominal)   # [θ_a; θ_b], length N_a+N_b
theta0a = np.array(model.theta_nominal_a) # primary only
theta0b = np.array(model.theta_nominal_b) # secondary only

# Which T_{ijk} entries are included (list of (i,j,k) tuples)
idx_a = model.theta_indices_a
idx_b = model.theta_indices_b
```

#### Interface A — point evaluation for OD

Returns ż and A_aug at a single (z, t).  Use this to drive your own covariance propagation Ṗ = A·P + P·Aᵀ + Q.

```python
z0 = np.concatenate([x0, theta0])

zdot, A_aug_flat = model.eval(z0.tolist(), t)

zdot    = np.array(zdot)                    # length N_AUG
A_aug   = np.array(A_aug_flat).reshape(N_AUG, N_AUG)
# A_aug block structure:
#   A_aug[:30, :30]      = ∂f/∂x  (dynamics Jacobian)
#   A_aug[:30, 30:30+N_a] = ∂f/∂θ_a (sensitivity to primary integrals)
#   A_aug[:30, 30+N_a:]  = ∂f/∂θ_b (sensitivity to secondary integrals)
```

#### Interface B — full ODE RHS for scipy

Bundles [ż; A_aug·Φ_aug] into one vector so scipy integrates z and Φ_aug together.

```python
from scipy.integrate import solve_ivp

phi0_flat = np.eye(N_AUG).ravel()
y0        = np.concatenate([z0, phi0_flat])  # length N_AUG + N_AUG²

def aug_ode(t, y):
    return np.array(model.eval_stm(y.tolist(), t))

sol = solve_ivp(aug_ode, [t0, tf], y0, method="DOP853",
                rtol=1e-10, atol=1e-13, t_eval=t_eval)

y        = sol.y.T                              # (nsteps, N_AUG + N_AUG²)
xs       = y[:, :30]                            # trajectory
phi_aug  = y[:, N_AUG:].reshape(-1, N_AUG, N_AUG)  # Φ_aug history

# Extract sensitivity blocks
phi_xta  = phi_aug[:, :30, 30:30+N_A]          # (nsteps, 30, N_a) ∂x/∂θ_a
phi_xtb  = phi_aug[:, :30, 30+N_A:]            # (nsteps, 30, N_b) ∂x/∂θ_b
phi_xx   = phi_aug[:, :30, :30]                 # (nsteps, 30, 30)  ∂x/∂x₀
```

#### Interface C — Rust reference propagation (validation)

Runs the full augmented propagation inside Rust (fastest, used for validation):

```python
gubas_rs.run_stm_augmented_both_py(
    min_degree_a=2, max_degree_a=2,
    min_degree_b=2, max_degree_b=2,
)
# writes output_phi_aug/phi_aug_out.bin  (nsteps, N_AUG, N_AUG)
#         output_phi_aug/phi_t_out.bin   (nsteps,)
#         output_phi_aug/x_out.bin       (nsteps, 30)
#         output_phi_aug/theta_indices.txt  column key for θ_a / θ_b
#         output_phi_aug/stokes_out.txt     C/S at t₀ for both bodies

import numpy as np
phi_aug_r = np.fromfile("output_phi_aug/phi_aug_out.bin", dtype="<f8").reshape(-1, N_AUG, N_AUG)
```

### Stokes coefficient conversion

After propagation, convert the inertia-integral sensitivity blocks to spherical harmonic C/S sensitivity using `stokes_utils.py`:

```python
from stokes_utils import convert_phi_xt_to_cs, cs_labels

# Primary body
phi_xcs_a, M_a, Mplus_a = convert_phi_xt_to_cs(
    phi_xta, idx_a, min_degree=2, max_degree=2)
# phi_xcs_a: (nsteps, 30, N_cs_a)  — ∂x/∂(C/S) for primary

# Secondary body
phi_xcs_b, M_b, Mplus_b = convert_phi_xt_to_cs(
    phi_xtb, idx_b, min_degree=2, max_degree=2)
# phi_xcs_b: (nsteps, 30, N_cs_b)  — ∂x/∂(C/S) for secondary

col_a = cs_labels(2, 2)  # ["C20","C21","S21","C22","S22"]
col_b = cs_labels(2, 2)
```

The conversion uses the right pseudoinverse M⁺ = Mᵀ(MMᵀ)⁻¹ of the Stokes matrix M (N_cs × N_theta).  M is a fat matrix because T_{ijk} has more independent entries than C/S at the same degree — the null space corresponds to trace-like combinations that are gravitationally invisible.

#### Why M is fat

| degree l | N_theta (T_{ijk} entries) | N_cs (C/S coefficients) | null space |
|----------|--------------------------|-------------------------|------------|
| 2 | 6 | 5 | 1 |
| 3 | 10 | 7 | 3 |
| 4 | 15 | 9 | 6 |

---

## Complete OD pipeline example

This is the minimal script to get trajectory + full C/S partials for both bodies, ready to plug into an OD filter:

```python
import gubas_rs, numpy as np, os
from stokes_utils import convert_phi_xt_to_cs, cs_labels

os.chdir("example/")   # ic_input.txt must be in cwd

# 1. Build model
model = gubas_rs.AugmentedDynamicsModel(
    min_degree_a=2, max_degree_a=2,
    min_degree_b=2, max_degree_b=2,
)
N_AUG = model.n_aug       # 42
N_A   = model.n_theta_a   # 6
N_B   = model.n_theta_b   # 6
theta0 = np.array(model.theta_nominal)
idx_a  = model.theta_indices_a
idx_b  = model.theta_indices_b

# 2. Load initial conditions (from whatever source)
x0 = ...   # length-30 array (km, km/s, rad/s, rotation matrices)
t0, tf = 0.0, 7200.0
t_eval = np.linspace(t0, tf, 500)

# 3. Propagate z and Φ_aug with scipy
from scipy.integrate import solve_ivp

z0  = np.concatenate([x0, theta0])
y0  = np.concatenate([z0, np.eye(N_AUG).ravel()])

sol = solve_ivp(
    lambda t, y: np.array(model.eval_stm(y.tolist(), t)),
    [t0, tf], y0, method="DOP853",
    rtol=1e-10, atol=1e-13, t_eval=t_eval,
)

y       = sol.y.T
xs      = y[:, :30]
phi_aug = y[:, N_AUG:].reshape(-1, N_AUG, N_AUG)

# 4. Extract blocks
phi_xx   = phi_aug[:, :30, :30]                # ∂x/∂x₀  (30×30)
phi_xta  = phi_aug[:, :30, 30:30+N_A]          # ∂x/∂θ_a (30×6)
phi_xtb  = phi_aug[:, :30, 30+N_A:]            # ∂x/∂θ_b (30×6)

# 5. Convert to C/S partials
phi_xcs_a, _, _ = convert_phi_xt_to_cs(phi_xta, idx_a, 2, 2)
phi_xcs_b, _, _ = convert_phi_xt_to_cs(phi_xtb, idx_b, 2, 2)
# phi_xcs_a[k]: (30, 5) — partials ∂x(t_k)/∂[C20,C21,S21,C22,S22] primary
# phi_xcs_b[k]: (30, 5) — same for secondary

# For OD at step k — get A_aug for covariance propagation
z_k = np.concatenate([xs[k], theta0])
zdot, A_flat = model.eval(z_k.tolist(), t_eval[k])
A_aug = np.array(A_flat).reshape(N_AUG, N_AUG)
# Ṗ = A_aug @ P + P @ A_aug.T + Q
```

---

## Output files

### Trajectory only (`run()`)

| File | Shape | Contents |
|------|-------|----------|
| `output_t/t_out.bin` | (nsteps,) | Time vector (s) |
| `output_x/x_out.bin` | (nsteps, 30) | State history |

### STM (`run_stm_py()`)

| File | Shape | Contents |
|------|-------|----------|
| `output_phi/phi_out.bin` | (nsteps, 30, 30) | Φ_xx — state STM |
| `output_phi/x_out.bin` | (nsteps, 30) | State history |
| `output_phi/phi_t_out.bin` | (nsteps,) | Times |
| `output_phi/A_out.bin` | (nsteps, 30, 30) | Jacobian at each epoch |
| `output_phi/xdot_out.bin` | (nsteps, 30) | ẋ at each epoch |

### Augmented STM (`run_stm_augmented_both_py()`)

| File | Shape | Contents |
|------|-------|----------|
| `output_phi_aug/phi_aug_out.bin` | (nsteps, N_AUG, N_AUG) | Full Φ_aug |
| `output_phi_aug/phi_out.bin` | (nsteps, 30, 30) | Φ_xx block only |
| `output_phi_aug/x_out.bin` | (nsteps, 30) | State history |
| `output_phi_aug/phi_t_out.bin` | (nsteps,) | Times |
| `output_phi_aug/theta_indices.txt` | text | Column key: body, i, j, k per θ entry |
| `output_phi_aug/stokes_out.txt` | text | C/S at t₀ for both bodies |

All binary files are little-endian `float64`. Read in Python:

```python
phi_aug = np.fromfile("output_phi_aug/phi_aug_out.bin", dtype="<f8").reshape(-1, N_AUG, N_AUG)
```

---

## Example scripts

| Script | What it shows |
|--------|---------------|
| `run_example.py` | Basic trajectory, energy/angular momentum conservation |
| `run_example_OD.py` | STM propagation, `DynamicsModel.eval` vs Rust reference |
| `run_example_OD_stokes.py` | Single-body C/S partials demo |
| `run_example_OD_stokes_2.py` | **Both-body C/S partials** — Interface A/B demo, validation, plots |

Run from the `example/` directory:

```bash
cd example
python run_example_OD_stokes_2.py
python run_example_OD_stokes_2.py --min-a 2 --max-a 3 --min-b 2 --max-b 2
python run_example_OD_stokes_2.py --skip-rust   # reuse existing Rust output
```

---

## Dependencies

**Rust** (no external crates except for the Python extension):

| Crate | Purpose |
|-------|---------|
| `pyo3` | Python bindings (optional, enabled by maturin) |
| `num-traits` | Generic numeric trait for dual-number propagation |

**Python**:

```bash
pip install numpy scipy matplotlib maturin
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, request features, and
submit pull requests.  In brief: open a GitHub Issue with a minimal reproducible example
for bugs; run `cargo test` and `pytest` before opening a PR.

---

## Citation

If you use GUBAS-RS in a publication, please cite the original GUBAS paper:

> Alex B. Davis and Daniel J. Scheeres,
> *Doubly Synchronous Binary Asteroid Mass Parameter Observability*,
> Icarus, Vol. 341 (2020). https://doi.org/10.1016/j.icarus.2019.113439

If using the Stokes coefficient conversion:

> P. Tricarico,
> *Figure–figure interaction between bodies having arbitrary shapes and mass distributions: a power series expansion approach*,
> Celestial Mechanics and Dynamical Astronomy, Vol. 100 (2008). https://doi.org/10.1007/s10569-007-9106-5

If using the flyby or heliocentric perturbation:

> Alex J. Meyer and Daniel J. Scheeres,
> *The effect of planetary flybys on singly synchronous binary asteroids*,
> Icarus, Vol. 367 (2021). https://doi.org/10.1016/j.icarus.2021.114554

---

## License

MIT License

Copyright (c) 2024 Giovanni Fereoli

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Errata

User Guide Eq. 9: the provided definition of the inertia integrals is mass-normalised; the code and subsequent equations use the non-normalised form.

---

A JOSS paper (`paper.md` / `paper.bib`) is included in the repository root for formal
software citation.

---

*Original C++ code by Alex B. Davis & Alex J. Meyer. Rust port and OD extensions by Giovanni Fereoli.*