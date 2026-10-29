//! # GUBAS-RS — General Use Binary Asteroid Simulator (Rust)
//!
//! ---
//!
//! ## Statement of Need
//!
//! Binary asteroid systems — two gravitationally bound rocky bodies — are among
//! the most informative objects for planetary science: their mutual orbit encodes
//! both masses, while the coupled spin–orbit evolution constrains the internal
//! mass distribution of each body.  Modelling this system accurately requires the
//! **Full Two-Body Problem (F2BP)**, in which both bodies are treated as extended,
//! non-spherical masses with arbitrary gravity fields.
//!
//! Existing open tools either approximate one or both bodies as point masses, or
//! lack the sensitivity analysis machinery needed for **orbit determination (OD)**
//! from spacecraft tracking data.  GUBAS-RS fills this gap by providing:
//!
//! 1. **High-fidelity F2BP dynamics** — mutual gravitational potential evaluated
//!    as a Hou 2016 series in inertia integrals T\_{ijk}, valid for any body shape.
//! 2. **Exact State Transition Matrix** — forward-mode automatic differentiation
//!    ([`dual`]) replaces finite differences; no step-size tuning, no truncation error.
//! 3. **Parameter sensitivity for gravity field estimation** — the augmented STM
//!    propagates ∂x/∂T\_{ijk} simultaneously for *both* bodies ([`stm`]), enabling
//!    direct use in batch least-squares or Kalman filters.
//! 4. **Stokes coefficient conversion** — the linear map
//!    ∂x/∂T\_{ijk} → ∂x/∂C\_{lm}/S\_{lm} ([`stokes`]) delivers partials in the
//!    spherical harmonic basis used by standard geodesy tools.
//! 5. **Python interface** — compiled via PyO3/maturin so that any Python OD
//!    framework can call the Rust core with zero subprocess overhead.
//!
//! ### Target audience
//!
//! - **Planetary scientists** modelling binary asteroid dynamics (e.g. Didymos–
//!   Dimorphos post-DART, binary near-Earth asteroids).
//! - **Astrodynamicists** building OD pipelines that need trajectory + STM from
//!   a single consistent F2BP model.
//! - **Researchers** studying DART/Hera mission science: gravity field recovery
//!   from radiometric tracking.
//!
//! ---
//!
//! ## Installation
//!
//! ### Prerequisites
//!
//! | Tool | Minimum version |
//! |---|---|
//! | Rust toolchain | ≥ 1.75 — install via [rustup.rs](https://rustup.rs) |
//! | Python | ≥ 3.8 |
//!
//! Python packages (`numpy`, `scipy`, `matplotlib`, `maturin`, `pytest`) are
//! listed in `requirements.txt` and installed automatically by `initialize.py`.
//!
//! ### Recommended: one-command setup
//!
//! Run `initialize.py` from the **repository root** once after cloning:
//!
//! ```bash
//! # Binary + Python extension (needed for OD / Python use):
//! python initialize.py --maturin
//!
//! # Binary only (no Python extension):
//! python initialize.py
//! ```
//!
//! This creates `.venv`, installs all Python dependencies from `requirements.txt`,
//! builds `gubas_rs/target/release/hou_cpp_final`, and (with `--maturin`) runs
//! `maturin develop --release` so that `import gubas_rs` works inside `.venv`.
//!
//! ### Manual setup
//!
//! ```bash
//! # 1. Python environment
//! python -m venv .venv && source .venv/bin/activate
//! pip install -r requirements.txt          # numpy scipy matplotlib maturin pytest
//!
//! # 2. Rust binary
//! cd gubas_rs && cargo build --release
//! # → gubas_rs/target/release/hou_cpp_final
//!
//! # 3. Python extension (for OD use)
//! maturin develop --release
//! python -c "import gubas_rs; print('ok')"
//! ```
//!
//! ---
//!
//! ## Module Overview
//!
//! | Module | Role |
//! |---|---|
//! | [`dynamics`] | F2BP ODE right-hand side (`hou_ode`) — 30-element state |
//! | [`integrators`] | RK4, Adams-Bashforth-Moulton, adaptive RK7(8), LGVI |
//! | [`stm`] | STM and augmented-STM propagators (∂x/∂x₀ and ∂x/∂θ) |
//! | [`stokes`] | N\_{ijk} → C\_{lm}/S\_{lm} Stokes matrix (Tricarico 2008) |
//! | [`potential`] | Mutual gravitational potential and all partial derivatives |
//! | [`coefficients`] | Hou 2016 expansion coefficients (t\_k, a\_k, b\_k) |
//! | [`inertia`] | Inertia integrals T\_{ijk} from ellipsoid or polyhedron |
//! | [`orbit`] | Kepler's equation solver (elliptic + hyperbolic), elements → Cartesian |
//! | [`dual`] | Forward-mode dual number: `Dual = a + bε`, ε² = 0 |
//! | [`math3`] | 3-vector / 3×3-matrix primitives (`cross`, `inv`, `tilde`, …) |
//! | [`types`] | `Cube<T>`, `Params<T>` — generic simulation data structures |
//!
//! Every public function is documented with its arguments, return value, units,
//! and mathematical background.  See the sidebar for the full API reference.
//!
//! ---
//!
//! ## Example Usage
//!
//! ### Minimal Rust example — one RK4 step with point masses
//!
//! All physical units are **km / kg / s**.
//!
//! ```rust,ignore
//! use gubas_rs::{
//!     coefficients::{a_calc, b_calc, tk_calc},
//!     integrators::rk4_stack,
//!     math3::diag,
//!     types::{Cube, Params},
//! };
//!
//! // Body masses (Didymos system, kg)
//! let ma: f64 = 5.32e11;
//! let mb: f64 = 4.94e9;
//! let g:  f64 = 6.674e-20; // km³/(kg·s²)
//!
//! // Zeroth-order expansion: point masses
//! let n = 0usize;
//! let mut ta = Cube::new(n);  ta.set(0, 0, 0, ma);
//! let mut tb = Cube::new(n);  tb.set(0, 0, 0, mb);
//!
//! let mut params = Params {
//!     g, m: ma * mb / (ma + mb), nu: mb / (ma + mb),
//!     ta, tb,
//!     ia: diag([3.01e19, 3.01e19, 3.01e19]), // primary inertia tensor (kg·km²)
//!     ib: diag([5.0e16,  5.0e16,  5.0e16]),  // secondary inertia tensor
//!     n, tk: tk_calc(n), a: a_calc(n), b: b_calc(n),
//!     flyby_toggle: 0, helio_toggle: 0, sg_toggle: 0, tt_toggle: 0,
//! #   mplanet: 0.0, a_hyp: -1.0, e_hyp: 1.5,
//! #   i_hyp: 0.0, raan_hyp: 0.0, om_hyp: 0.0, tau_hyp: 0.0, n_hyp: 0.0,
//! #   msolar: 0.0, a_helio: 1.0, e_helio: 0.0,
//! #   i_helio: 0.0, raan_helio: 0.0, om_helio: 0.0, tau_helio: 0.0, n_helio: 0.0,
//! #   sol_rad: 0.0, au_def: 1.496e8, mean_motion: 0.0,
//! #   love1: 0.0, love2: 0.0, refrad1: 1.0, refrad2: 1.0,
//! #   rho_a: 1e12, rho_b: 1e12, eps1: 0.0, eps2: 0.0,
//! #   ida: [[0.0; 3]; 3], idb: [[0.0; 3]; 3], msun: 2e30,
//! };
//! params.compute_lgvi_inertia();
//!
//! // Circular orbit initial state: r = [1.19 km, 0, 0], Cc = C = I₃
//! let a_orb = 1.19_f64;
//! let v_c   = (g * (ma + mb) / a_orb).sqrt();
//! let eye   = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
//! let x0: [f64; 30] = [
//!     a_orb, 0.0, 0.0,  0.0, v_c, 0.0,       // r (km), v (km/s)
//!     0.0, 0.0, 0.0,    0.0, 0.0, 0.0,        // ωc, ωs (rad/s)
//!     eye[0], eye[1], eye[2], eye[3], eye[4],
//!     eye[5], eye[6], eye[7], eye[8],          // Cc (row-major)
//!     eye[0], eye[1], eye[2], eye[3], eye[4],
//!     eye[5], eye[6], eye[7], eye[8],          // C  (row-major)
//! ];
//!
//! // Integrate 100 s at h = 1 s — writes output_t/ and output_x/
//! rk4_stack(0.0, 100.0, x0, 1.0, &params);
//! ```
//!
//! ### Full OD pipeline in Python
//!
//! After building the extension with `maturin develop --release`:
//!
//! ```python
//! import gubas_rs, numpy as np
//! from stokes_utils import convert_phi_xt_to_cs, cs_labels
//!
//! # 1. Build the augmented model (degree-2 gravity for both bodies) from
//! #    example/ic_input.txt and the inertia files it names
//! model = gubas_rs.AugmentedDynamicsModel(
//!     min_degree_a=2, max_degree_a=2,
//!     min_degree_b=2, max_degree_b=2,
//!     work_dir="example/",
//! )
//! N_AUG = model.n_aug      # 30 + N_a + N_b  (e.g. 42 for deg-2 both)
//! theta0 = np.array(model.theta_nominal)
//! idx_a  = model.theta_indices_a  # list of (i,j,k) for primary T_{ijk}
//! idx_b  = model.theta_indices_b
//!
//! # 2. Build augmented initial conditions
//! x0  = ...  # 30-element state (km, km/s, rad/s, rotation matrices)
//! z0  = np.concatenate([x0, theta0])
//! y0  = np.concatenate([z0, np.eye(N_AUG).ravel()])   # state + Φ_aug
//!
//! # 3. Propagate with scipy (uses exact AD Jacobian internally)
//! from scipy.integrate import solve_ivp
//! sol = solve_ivp(
//!     lambda t, y: np.array(model.eval_stm(y.tolist(), t)),
//!     [0.0, 7200.0], y0, method="DOP853",
//!     rtol=1e-10, atol=1e-13,
//! )
//! phi_aug = sol.y.T[:, N_AUG:].reshape(-1, N_AUG, N_AUG)
//!
//! # 4. Convert inertia-integral partials to spherical harmonic C/S partials
//! phi_xta = phi_aug[:, :30, 30:30+model.n_theta_a]
//! phi_xcs_a, _, _ = convert_phi_xt_to_cs(phi_xta, idx_a, 2, 2)
//! # phi_xcs_a[k]: (30, 5) — ∂x(tₖ)/∂[C20,C21,S21,C22,S22] primary
//! ```
//!
//! More complete examples are in `example/`:
//!
//! | Script | Demonstrates |
//! |---|---|
//! | `run_example.py` | Trajectory, energy and angular momentum conservation |
//! | `run_example_OD.py` | STM propagation, `DynamicsModel.eval` |
//! | `run_example_OD_stokes.py` | Single-body C/S partials |
//! | `run_example_OD_stokes_2.py` | **Both-body C/S partials**, validation vs Rust reference |
//!
//! ---
//!
//! ## Automated Tests
//!
//! The test suite verifies every module against analytic reference values —
//! no floating-point tolerances are loosened beyond what the physics demands.
//!
//! ### Rust tests — 121 tests across all 11 modules
//!
//! ```bash
//! cd gubas_rs
//! cargo test
//! ```
//!
//! Expected:
//! ```text
//! running 121 tests
//! test coefficients::tests::a_calc_n0_monopole ... ok
//! ...
//! test result: ok. 121 passed; 0 failed; 0 ignored
//! ```
//!
//! Filter by module or test name:
//! ```bash
//! cargo test potential::     # only potential module
//! cargo test oblate_c20      # any test whose name contains "oblate_c20"
//! ```
//!
//! Key analytic checks performed (121 tests across all 11 modules):
//!
//! | Module | Tests | Representative checks |
//! |---|---:|---|
//! | [`coefficients`] | 20 | `factorial`, `t_ind` indexing, `tk_calc` analytic values, `a_calc`/`b_calc` seeds |
//! | [`dual`] | 16 | sin′ = cos, product/chain/quotient rules, powi at zero |
//! | [`math3`] | 17 | i×j = k, tilde↔cross, det(I) = 1, inv · self = I, norm, trace |
//! | [`stokes`] | 12 | sphere C₂ₘ = 0; oblate C₂₀ = −3/5; triaxial C₂₂ = 3; M·N = direct |
//! | [`inertia`] | 9 | Ellipsoid mass, T₂₀₀/T₀₂₀/T₀₀₂ analytic, sphere MOI equal, q\_ijk symmetry, inertia CSV formats |
//! | [`potential`] | 7 | order-0 recovers −G·M₁M₂/r and G·M₁M₂/r² force; transverse force = 0 |
//! | [`orbit`] | 8 | vis-viva; r·v = 0 at periapsis; circular |r| = a |
//! | [`types`] | 19 | `Cube<T>` set/get/add/zeros; `compute_lgvi_inertia`; `ic_input.txt` and CSV parsing errors; ellipsoid integrals imply its inertia; `with_theta` updates inertia and masses |
//! | [`dynamics`] | 4 | centripetal acceleration = −G(M₁+M₂)/r²; zero torques for n=0; free rotation with products of inertia conserves energy and \|L\|; AD ∂ẋ/∂T₂₀₀ matches FD through potential and inertia |
//! | [`lgvi`] | 3 | outer product values and antisymmetry; monopole radial force |
//! | [`integrators`] | 4 | all four integrators produce non-NaN output of correct size |
//! | [`stm`] | 2 | AD Jacobian exact to machine eps; max\|AD−FD\| < 1e-5 |
//!
//! ### Python tests — 24 tests (no Rust extension needed)
//!
//! ```bash
//! # from the repo root
//! python3 -m pytest
//! ```
//!
//! Expected:
//! ```text
//! collected 24 items
//! tests/test_stokes_utils.py::TestCsLabels::test_count_degree2 PASSED
//! ...
//! 24 passed in 1.3s
//! ```
//!
//! Filter by class or keyword:
//! ```bash
//! python3 -m pytest -k "TestStokesMatrix"
//! python3 -m pytest -k "oblate_c20"
//! ```
//!
//! `tests/conftest.py` adds `example/` to `sys.path` automatically — no
//! manual path setup is required.
//!
//! ### Rebuild and view the HTML documentation
//!
//! ```bash
//! cd gubas_rs
//! cargo doc --no-deps --open
//! # opens: gubas_rs/target/doc/gubas_rs/index.html
//! ```
//!
//! ---
//!
//! ## Contributing and Support
//!
//! See [`CONTRIBUTING.md`](https://github.com/giovannifereoli/GUBAS-RS/blob/master/CONTRIBUTING.md)
//! in the repository root for full guidelines.  In brief:
//!
//! - **Bug reports** — open a GitHub Issue with a minimal reproducible example.
//! - **Feature requests** — open an Issue describing the use-case.
//! - **Pull requests** — fork the repo, create a branch, add tests for any new
//!   code, run `cargo test` and `python3 -m pytest`, then open a PR against
//!   `master`.
//! - **Questions / support** — open a GitHub Discussion or contact the
//!   maintainer at `giovafere@gmail.com`.

#![allow(dead_code)]

pub mod coefficients;
pub mod dual;
pub mod dynamics;
pub mod inertia;
pub mod integrators;
pub mod lgvi;
pub mod math3;
pub mod orbit;
pub mod potential;
pub mod stm;
pub mod stokes;
pub mod types;

use std::path::Path;

use dynamics::augmented_ode;
use integrators::{abm, lgvi_integ, rk4_stack, rk87};
use types::{check_theta_indices, Initialization, Params};

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

// ── ic_read ───────────────────────────────────────────────────────────────────

/// Read `ic_input.txt` from the current working directory.
///
/// # Panics
/// If the file is missing or malformed.  [`Initialization::read`] returns the
/// error instead.
pub fn ic_read() -> Initialization {
    Initialization::read(Path::new(".")).unwrap_or_else(|e| panic!("{e}"))
}

/// Load the parameters from the inertia files named in `ics`, in the current
/// working directory, or panic with the reason.
fn load_params(ics: &Initialization) -> Params<f64> {
    Params::load(Path::new("."), ics).unwrap_or_else(|e| match e.kind() {
        std::io::ErrorKind::NotFound => panic!("{e}; run with Tgen = 1 to generate it"),
        _ => panic!("{e}"),
    })
}

// ── run_simulation ────────────────────────────────────────────────────────────

/// Run a full GUBAS simulation. Reads `ic_input.txt` from the current working
/// directory and writes binary output to `output_t/t_out.bin` and
/// `output_x/x_out.bin` in the same directory.
///
/// With `Tgen = 1` the inertia integrals and moments are computed from the
/// body shapes and saved to the files named in `ic_input.txt`; otherwise they
/// are loaded from those files.
pub fn run_simulation() {
    let ics = ic_read();
    let params = if ics.tgen == 1 {
        println!("Generating Inertia Integrals");
        Params::generate(Path::new("."), &ics).unwrap_or_else(|e| panic!("{e}"))
    } else {
        load_params(&ics)
    };

    println!("Integrating");
    match ics.integ {
        1 => { println!("RK4");     rk4_stack(ics.t0, ics.tf, ics.x0, ics.h, &params); }
        2 => { println!("LGVI");    lgvi_integ(ics.t0, ics.tf, ics.x0, ics.h, &params); }
        3 => { println!("RK 7(8)"); rk87(ics.t0, ics.tf, ics.x0, ics.tol, &params); }
        4 => { println!("ABM");     abm(ics.t0, ics.tf, ics.x0, ics.h, &params); }
        other => eprintln!("Unknown integrator flag: {}", other),
    }
}

// ── run_stm ───────────────────────────────────────────────────────────────────

/// Propagate the STM alongside the trajectory using forward-mode AD Jacobians.
///
/// Reads `ic_input.txt`, builds `Params<f64>` identically to `run_simulation`,
/// then:
///   1. Prints max |AD − FD| for the Jacobian at t₀ (sanity check).
///   2. Runs `propagate_stm_ad` (RK4 + exact dual-number Jacobian).
///   3. Writes binary output to `output_phi/`:
///      - `phi_out.bin`   — STM history Φ(t),  shape (nsteps, 30, 30) f64 LE
///      - `phi_t_out.bin` — times,              shape (nsteps,)        f64 LE
///      - `jac_ad.bin`    — Jacobian A at t₀ from AD, shape (30, 30)   f64 LE
///      - `jac_fd.bin`    — Jacobian A at t₀ from FD, shape (30, 30)   f64 LE
pub fn run_stm() {
    use dual::Dual;
    use dynamics::hou_ode;
    use stm::{
        eval_dynamics_and_jacobian, jacobian_ad, jacobian_fd,
        propagate_stm_rk87_ad, write_A_bin, write_phi_bin, write_phi_t_bin,
        write_x_bin, write_xdot_bin,
    };
    use types::promote_params;

    let ics = ic_read();
    let params = load_params(&ics);

    // promote once — all eps = 0 (params are constants, not differentiated)
    let params_dual = promote_params::<Dual>(&params);

    let ode      = |x: [f64;  30], t: f64| hou_ode(x, t, &params);
    let ode_dual = |x: [Dual; 30], t: f64| hou_ode(x, t, &params_dual);

    // ── Jacobian sanity check at t₀ ───────────────────────────────────────────
    println!("Computing Jacobian at t₀ via AD and FD ...");
    let f0     = hou_ode(ics.x0, ics.t0, &params);
    let jac_fd = jacobian_fd(&ics.x0, ics.t0, &f0, &ode);
    let jac_ad = jacobian_ad(&ics.x0, ics.t0, &ode_dual);

    let max_err = (0..30).flat_map(|i| (0..30).map(move |j|
        (jac_ad[i][j] - jac_fd[i][j]).abs()
    )).fold(0.0_f64, f64::max);
    println!("  Max |AD − FD| = {:.3e}  (FD truncation, expect ~1e-7..1e-8)", max_err);

    std::fs::create_dir_all("output_phi").unwrap();
    write_phi_bin("output_phi", &[jac_ad]).expect("write jac_ad");
    // re-use write_phi_bin for jac_fd by writing to a different file
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(
            std::fs::File::create("output_phi/jac_fd.bin").unwrap());
        for row in jac_fd.iter() {
            for &v in row.iter() { f.write_all(&v.to_le_bytes()).unwrap(); }
        }
    }
    // rename phi_out.bin → jac_ad.bin
    std::fs::rename("output_phi/phi_out.bin", "output_phi/jac_ad.bin").unwrap();

    // ── STM propagation — RK7(8) adaptive ────────────────────────────────────
    println!("Propagating STM: RK7(8) Dormand-Prince adaptive, tol={:.0e} ...",
             ics.tol);
    println!("  [13 stages per step, each with 1 f64 + 30 dual ODE evals]");

    let (ts, xs, phis) = propagate_stm_rk87_ad(
        ics.x0, ics.t0, ics.tf, ics.tol, ode, ode_dual, 1,
    );

    write_phi_bin("output_phi", &phis).expect("write phi");
    write_phi_t_bin("output_phi", &ts).expect("write phi_t");
    write_x_bin("output_phi", &xs).expect("write x");
    println!("  Written {} STM snapshots to output_phi/", phis.len());

    // ── Mode 2: evaluate (ẋ, A) at each recorded epoch ───────────────────────
    // Closures were moved into propagate_stm_rk87_ad; rebuild from still-live refs.
    println!("Evaluating dynamics and exact Jacobian at {} epochs ...", xs.len());
    let ode2      = |x: [f64;  30], t: f64| dynamics::hou_ode(x, t, &params);
    let ode_dual2 = |x: [Dual; 30], t: f64| dynamics::hou_ode(x, t, &params_dual);

    let mut xdots:  Vec<[f64; 30]> = Vec::with_capacity(xs.len());
    let mut jacobs: Vec<stm::Phi>  = Vec::with_capacity(xs.len());
    for (&xi, &ti) in xs.iter().zip(ts.iter()) {
        let (xdot, a) = eval_dynamics_and_jacobian(xi, ti, &ode2, &ode_dual2);
        xdots.push(xdot);
        jacobs.push(a);
    }

    write_xdot_bin("output_phi", &xdots).expect("write xdot");
    write_A_bin("output_phi", &jacobs).expect("write A");
    println!("  Written xdot_out.bin and A_out.bin to output_phi/");
}

// ── run_stm_augmented ─────────────────────────────────────────────────────────

/// Propagate trajectory + STM (Φ_xx) + inertia-parameter sensitivity (Φ_xθ).
///
/// θ = all T_{ijk} entries for the selected body at degrees [min_degree, max_degree].
/// The augmented state is z = [x(30); θ(N)], with θ̇ = 0 (parameters constant).
///
/// # Arguments
/// * `min_degree`, `max_degree` — harmonic degree range (e.g. 2, 2 for degree-2 only)
/// * `which_body`               — 0 = primary T_a,  1 = secondary T_b
///
/// Changing a degree-2 entry also changes the body's inertia tensor in the
/// rotational equations (see [`Params::with_theta`]), so Φ_xθ includes that
/// path.  Degree-1 entries are rejected (see [`check_theta_indices`]).
///
/// # Output — written to `output_phi_aug/`
/// * `phi_out.bin`     — Φ_xx (30×30) STM history,   (nsteps, 30, 30) f64 LE
/// * `phi_xt_out.bin`  — Φ_xθ (30×N) sensitivity,    (nsteps, 30, N)  f64 LE
/// * `phi_t_out.bin`   — times,                       (nsteps,)        f64 LE
/// * `x_out.bin`       — state trajectory,            (nsteps, 30)     f64 LE
/// * `theta_indices.txt` — (i j k) lines for column ordering of Φ_xθ
/// * `stokes_out.txt`    — C/S Stokes coefficients at t₀ (for reference radius r0)
pub fn run_stm_augmented(min_degree: usize, max_degree: usize, which_body: usize) {
    use dual::Dual;
    use dynamics::hou_ode;
    use stm::{propagate_augmented_rk87_ad, write_phi_aug_bin, write_phi_bin,
              write_phi_t_bin, write_x_bin};
    use stokes::{inertia_indices, nijk_to_clm_slm, Normalization};
    use types::promote_params;

    assert!(which_body <= 1, "which_body must be 0 (primary) or 1 (secondary), got {which_body}");
    let ics = ic_read();
    let params = load_params(&ics);

    // ── Select θ indices: all T_{ijk} at the requested degrees ───────────────
    let theta_indices = inertia_indices(min_degree, max_degree);
    check_theta_indices(&theta_indices, params.n).unwrap_or_else(|e| panic!("{e}"));
    let n_theta = theta_indices.len();

    let body = if which_body == 0 { &params.ta } else { &params.tb };
    let theta0: Vec<f64> = theta_indices.iter().map(|&(i, j, k)| body.get(i, j, k)).collect();

    println!("Augmented STM propagation:");
    println!("  Body: {}, degrees [{}, {}], N_theta = {}",
             if which_body == 0 { "primary" } else { "secondary" },
             min_degree, max_degree, n_theta);
    println!("  [Each stage: {} dual ODE evals for augmented Jacobian]", 30 + n_theta);

    // ── Augmented dual ODE: θ = z[30..] patched into the parameters ──────────
    let params_dual_base = promote_params::<Dual>(&params);
    let (idx_a, idx_b) = if which_body == 0 {
        (theta_indices.clone(), Vec::new())
    } else {
        (Vec::new(), theta_indices.clone())
    };
    let aug_ode_dual = move |z: &[Dual], t: f64| {
        augmented_ode(z, t, &params_dual_base, &idx_a, &idx_b)
    };

    // f64 ODE (trajectory only — theta is constant)
    let ode = |x: [f64; 30], t: f64| hou_ode(x, t, &params);

    // ── Propagate ─────────────────────────────────────────────────────────────
    let (ts, xs, phi_augs) = propagate_augmented_rk87_ad(
        ics.x0, theta0, ics.t0, ics.tf, ics.tol,
        ode, aug_ode_dual, 1,
    );
    println!("  {} snapshots, Φ_aug shape ({}, {})", ts.len(), 30+n_theta, 30+n_theta);

    // ── Write output ──────────────────────────────────────────────────────────
    let dir = "output_phi_aug";
    write_phi_aug_bin(dir, &phi_augs).expect("write phi_aug");
    write_phi_t_bin(dir, &ts).expect("write phi_t");
    write_x_bin(dir, &xs).expect("write x");
    // Also write Φ_xx (top-left 30×30 block) as standard phi_out.bin for compatibility
    let phi_xxs: Vec<stm::Phi> = phi_augs.iter().map(|pa| {
        let n_aug = 30 + n_theta;
        let mut pxx = stm::phi_zero();
        for i in 0..30 { for j in 0..30 { pxx[i][j] = pa[i * n_aug + j]; } }
        pxx
    }).collect();
    write_phi_bin(dir, &phi_xxs).expect("write phi_xx");

    // theta index legend
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(format!("{}/theta_indices.txt", dir)).unwrap());
        writeln!(f, "# k  i  j  k_idx  (T_a[i][j][k_idx] entries, degree l=i+j+k)").unwrap();
        for (k, &(ii, jj, kk)) in theta_indices.iter().enumerate() {
            writeln!(f, "{} {} {} {}", k, ii, jj, kk).unwrap();
        }
    }

    // Stokes at t₀
    let (body_ta, body_mass, r0) = if which_body == 0 {
        (&params.ta, params.ta.get(0,0,0), params.refrad1)
    } else {
        (&params.tb, params.tb.get(0,0,0), params.refrad2)
    };
    let cs = nijk_to_clm_slm(body_ta, body_mass, r0, max_degree, Normalization::Unnormalized);
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(format!("{}/stokes_out.txt", dir)).unwrap());
        writeln!(f, "# Unnormalized Stokes coefficients at t0, r0={:.4} km", r0).unwrap();
        writeln!(f, "# l  m  C_lm  S_lm").unwrap();
        for l in 0..=max_degree {
            for mm in 0..=l {
                writeln!(f, "{} {} {:.15e} {:.15e}",
                    l, mm, cs.c[l][mm], cs.s[l][mm]).unwrap();
            }
        }
    }
    println!("  Output written to {}/", dir);
    cs.print();
}

// ── run_stm_augmented_both ────────────────────────────────────────────────────

/// Same as `run_stm_augmented` but estimates inertia integrals for **both** bodies
/// simultaneously.  The augmented state is z = [x(30); θ_a(N_a); θ_b(N_b)].
///
/// * `min_degree_a/max_degree_a` — harmonic degree range for primary   T_a
/// * `min_degree_b/max_degree_b` — harmonic degree range for secondary T_b
///
/// Output (`output_phi_aug/`): same files as `run_stm_augmented`, but Φ_aug is
/// now (30+N_a+N_b)×(30+N_a+N_b).  `theta_indices.txt` has a `body` column
/// ('A' or 'B') so Python can split the Φ_xθ block back into Φ_xθ_a / Φ_xθ_b.
pub fn run_stm_augmented_both(
    min_degree_a: usize, max_degree_a: usize,
    min_degree_b: usize, max_degree_b: usize,
) {
    use dual::Dual;
    use dynamics::hou_ode;
    use stm::{propagate_augmented_rk87_ad, write_phi_aug_bin, write_phi_bin,
              write_phi_t_bin, write_x_bin};
    use stokes::{inertia_indices, nijk_to_clm_slm, Normalization};
    use types::promote_params;

    let ics = ic_read();
    let params = load_params(&ics);

    let theta_indices_a = inertia_indices(min_degree_a, max_degree_a);
    let theta_indices_b = inertia_indices(min_degree_b, max_degree_b);
    for (label, idx) in [("primary", &theta_indices_a), ("secondary", &theta_indices_b)] {
        check_theta_indices(idx, params.n).unwrap_or_else(|e| panic!("{label}: {e}"));
    }
    let n_a = theta_indices_a.len();
    let n_b = theta_indices_b.len();
    let n_theta = n_a + n_b;

    let mut theta0: Vec<f64> = theta_indices_a.iter()
        .map(|&(i,j,k)| params.ta.get(i,j,k)).collect();
    theta0.extend(theta_indices_b.iter().map(|&(i,j,k)| params.tb.get(i,j,k)));

    println!("Augmented STM propagation (both bodies):");
    println!("  Primary   degrees [{}, {}], N_a = {}", min_degree_a, max_degree_a, n_a);
    println!("  Secondary degrees [{}, {}], N_b = {}", min_degree_b, max_degree_b, n_b);
    println!("  N_theta = {},  N_aug = {}", n_theta, 30 + n_theta);

    let params_dual_base = promote_params::<Dual>(&params);
    let (idx_a, idx_b) = (theta_indices_a.clone(), theta_indices_b.clone());
    let aug_ode_dual = move |z: &[Dual], t: f64| {
        augmented_ode(z, t, &params_dual_base, &idx_a, &idx_b)
    };

    let ode = |x: [f64; 30], t: f64| hou_ode(x, t, &params);

    let (ts, xs, phi_augs) = propagate_augmented_rk87_ad(
        ics.x0, theta0, ics.t0, ics.tf, ics.tol, ode, aug_ode_dual, 1,
    );
    println!("  {} snapshots, Φ_aug ({}, {})", ts.len(), 30+n_theta, 30+n_theta);

    let dir = "output_phi_aug";
    write_phi_aug_bin(dir, &phi_augs).expect("write phi_aug");
    write_phi_t_bin(dir, &ts).expect("write phi_t");
    write_x_bin(dir, &xs).expect("write x");
    let phi_xxs: Vec<stm::Phi> = phi_augs.iter().map(|pa| {
        let n_aug = 30 + n_theta;
        let mut pxx = stm::phi_zero();
        for i in 0..30 { for j in 0..30 { pxx[i][j] = pa[i * n_aug + j]; } }
        pxx
    }).collect();
    write_phi_bin(dir, &phi_xxs).expect("write phi_xx");

    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(format!("{}/theta_indices.txt", dir)).unwrap());
        writeln!(f, "# col  body  i  j  k").unwrap();
        for (k, &(ii,jj,kk)) in theta_indices_a.iter().enumerate() {
            writeln!(f, "{}  A  {} {} {}", k, ii, jj, kk).unwrap();
        }
        for (k, &(ii,jj,kk)) in theta_indices_b.iter().enumerate() {
            writeln!(f, "{}  B  {} {} {}", n_a + k, ii, jj, kk).unwrap();
        }
    }

    let cs_a = nijk_to_clm_slm(&params.ta, params.ta.get(0,0,0), params.refrad1,
                                max_degree_a, Normalization::Unnormalized);
    let cs_b = nijk_to_clm_slm(&params.tb, params.tb.get(0,0,0), params.refrad2,
                                max_degree_b, Normalization::Unnormalized);
    {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(format!("{}/stokes_out.txt", dir)).unwrap());
        writeln!(f, "# Stokes at t0, r0_A={:.4} km, r0_B={:.4} km.  Body A then B.",
                 params.refrad1, params.refrad2).unwrap();
        writeln!(f, "# body  l  m  C_lm  S_lm").unwrap();
        for l in 0..=max_degree_a {
            for mm in 0..=l {
                writeln!(f, "A {} {} {:.15e} {:.15e}", l, mm, cs_a.c[l][mm], cs_a.s[l][mm]).unwrap();
            }
        }
        for l in 0..=max_degree_b {
            for mm in 0..=l {
                writeln!(f, "B {} {} {:.15e} {:.15e}", l, mm, cs_b.c[l][mm], cs_b.s[l][mm]).unwrap();
            }
        }
    }
    println!("  Output written to {}/", dir);
}

// ── Python module (only compiled with --features extension-module) ────────────

/// Changes the working directory for the life of the guard and restores the
/// previous one when dropped, also when the run panics, so a `work_dir`
/// argument does not leak into the calling Python process.
#[cfg(feature = "extension-module")]
struct WorkDir {
    previous: Option<std::path::PathBuf>,
}

#[cfg(feature = "extension-module")]
impl WorkDir {
    fn enter(dir: Option<&str>) -> PyResult<Self> {
        let Some(dir) = dir else { return Ok(Self { previous: None }) };
        let previous = std::env::current_dir()?;
        std::env::set_current_dir(dir)
            .map_err(|e| std::io::Error::new(e.kind(), format!("{dir}: {e}")))?;
        Ok(Self { previous: Some(previous) })
    }
}

#[cfg(feature = "extension-module")]
impl Drop for WorkDir {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            let _ = std::env::set_current_dir(previous);
        }
    }
}

/// Run a GUBAS simulation from Python.
///
/// Reads `ic_input.txt` from `work_dir` (defaults to the current working
/// directory) and writes output binary files to the same location.  The
/// working directory is restored afterwards.
///
/// ```python
/// import gubas_rs
/// gubas_rs.run()                   # use Python's current working directory
/// gubas_rs.run("/path/to/run_dir") # run in a specific directory
/// ```
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (work_dir=None))]
fn run(work_dir: Option<&str>) -> PyResult<()> {
    let _cwd = WorkDir::enter(work_dir)?;
    run_simulation();
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (work_dir=None))]
fn run_stm_py(work_dir: Option<&str>) -> PyResult<()> {
    let _cwd = WorkDir::enter(work_dir)?;
    run_stm();
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (min_degree=2, max_degree=2, which_body=0, work_dir=None))]
fn run_stm_augmented_py(
    min_degree: usize,
    max_degree: usize,
    which_body: usize,
    work_dir:   Option<&str>,
) -> PyResult<()> {
    if which_body > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "which_body must be 0 (primary) or 1 (secondary), got {which_body}")));
    }
    let _cwd = WorkDir::enter(work_dir)?;
    run_stm_augmented(min_degree, max_degree, which_body);
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (min_degree_a=2, max_degree_a=2, min_degree_b=2, max_degree_b=2, work_dir=None))]
fn run_stm_augmented_both_py(
    min_degree_a: usize, max_degree_a: usize,
    min_degree_b: usize, max_degree_b: usize,
    work_dir: Option<&str>,
) -> PyResult<()> {
    let _cwd = WorkDir::enter(work_dir)?;
    run_stm_augmented_both(min_degree_a, max_degree_a, min_degree_b, max_degree_b);
    Ok(())
}

// ── DynamicsModel — callable OD interface ─────────────────────────────────────

/// Loaded gravity model for the Didymos binary system.
///
/// Initialises once from `ic_input.txt` in the working directory, then exposes
/// two lightweight call methods:
///
/// ```python
/// import gubas_rs, numpy as np
///
/// model = gubas_rs.DynamicsModel()         # reads ic_input.txt, builds params
///
/// xdot, A_flat = model.eval(x, t)          # ẋ (len 30) + A (len 900, row-major)
/// A = np.array(A_flat).reshape(30, 30)
///
/// dxphi = model.eval_augmented(xphi, t)   # 930-element augmented ODE RHS
/// ```
#[cfg(feature = "extension-module")]
#[pyclass]
pub struct DynamicsModel {
    params:      Params<f64>,
    params_dual: Params<dual::Dual>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl DynamicsModel {
    /// Create a DynamicsModel from `ic_input.txt` and the inertia files it
    /// names, in `work_dir` (default: the current directory).  The working
    /// directory of the process is not changed.
    #[new]
    #[pyo3(signature = (work_dir=None))]
    fn new(work_dir: Option<&str>) -> PyResult<Self> {
        let dir = Path::new(work_dir.unwrap_or("."));
        let params = Params::load(dir, &Initialization::read(dir)?)?;
        let params_dual = types::promote_params::<dual::Dual>(&params);
        Ok(DynamicsModel { params, params_dual })
    }

    /// Gravitational constant G (km³/(kg·s²)).
    #[getter]
    fn g(&self) -> f64 { self.params.g }

    /// Expansion order of the mutual potential.
    #[getter]
    fn order(&self) -> usize { self.params.n }

    /// Primary mass T_a[0,0,0] (kg).
    #[getter]
    fn mass_a(&self) -> f64 { self.params.ta.get(0, 0, 0) }

    /// Secondary mass T_b[0,0,0] (kg).
    #[getter]
    fn mass_b(&self) -> f64 { self.params.tb.get(0, 0, 0) }

    /// Primary inertia tensor in its body frame (kg·km²), as rows.
    #[getter]
    fn inertia_a(&self) -> [[f64; 3]; 3] { self.params.ia }

    /// Secondary inertia tensor in its body frame (kg·km²), as rows.
    #[getter]
    fn inertia_b(&self) -> [[f64; 3]; 3] { self.params.ib }

    /// Evaluate ẋ = f(x, t) and A = ∂f/∂x at a single point via exact AD.
    ///
    /// Args:
    ///   x (list[float]): 30-element state vector.
    ///   t (float): time (s).
    ///
    /// Returns:
    ///   (xdot, A_flat): xdot is len-30, A_flat is len-900 row-major.
    fn eval(&self, x: Vec<f64>, t: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if x.len() != 30 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("x must have 30 elements, got {}", x.len())));
        }
        let xa: [f64; 30] = x.try_into().unwrap();
        let ode      = |xi: [f64;         30], ti: f64| dynamics::hou_ode(xi, ti, &self.params);
        let ode_dual = |xi: [dual::Dual;  30], ti: f64| dynamics::hou_ode(xi, ti, &self.params_dual);
        let (xdot, a) = stm::eval_dynamics_and_jacobian(xa, t, &ode, &ode_dual);
        let a_flat: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();
        Ok((xdot.to_vec(), a_flat))
    }

    /// Augmented ODE RHS for external integrators (scipy, MATLAB, etc.).
    ///
    /// State layout (930 elements): xphi[0:30]=x, xphi[30:930]=Φ row-major.
    /// Returns the same layout: [ẋ; vec(Φ̇ = A·Φ)].
    ///
    /// Example (scipy DOP853)::
    ///
    ///     phi0   = np.eye(30).ravel()
    ///     x0_aug = np.concatenate([x0, phi0])
    ///     sol    = solve_ivp(lambda t, xp: model.eval_augmented(xp.tolist(), t),
    ///                        [t0, tf], x0_aug, method="DOP853", rtol=1e-12)
    fn eval_augmented(&self, xphi: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
        if xphi.len() != 930 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("xphi must have 930 elements, got {}", xphi.len())));
        }
        let xa: [f64; 930] = xphi.try_into().unwrap();
        let ode      = |xi: [f64;         30], ti: f64| dynamics::hou_ode(xi, ti, &self.params);
        let ode_dual = |xi: [dual::Dual;  30], ti: f64| dynamics::hou_ode(xi, ti, &self.params_dual);
        let out = stm::augmented_ode_rhs(xa, t, &ode, &ode_dual);
        Ok(out.to_vec())
    }
}

// ── AugmentedDynamicsModel ───────────────────────────────────────────────────

/// Augmented dynamics model for simultaneous OD of both bodies.
///
/// Exposes the augmented system z = [x(30); θ_a(N_a); θ_b(N_b)] where:
///   - x      : standard 30-element F2BP state (pos, vel, attitude, angular vel)
///   - θ_a    : primary   inertia integrals T_a[i,j,k] at selected degrees
///   - θ_b    : secondary inertia integrals T_b[i,j,k] at selected degrees
///
/// **Augmented Jacobian block structure** (n_aug = 30 + N_a + N_b):
///
/// ```
///   A_aug = | A   B_a  B_b |    A[i,j]  = ∂f_i/∂x_j
///           | 0   0    0   |    B_a[i,k] = ∂f_i/∂θ_a_k
///           | 0   0    0   |    B_b[i,k] = ∂f_i/∂θ_b_k
/// ```
///
/// **Φ_aug block structure:**
///
/// ```
///   Φ_aug = | Φ_xx   Φ_xθ_a   Φ_xθ_b |   ← 30 rows (propagated)
///           | 0      I_Na     0       |   ← N_a rows (trivial: θ̇=0)
///           | 0      0        I_Nb    |   ← N_b rows (trivial: θ̇=0)
/// ```
///
/// # Example
///
/// ```python
/// import gubas_rs, numpy as np
/// from scipy.integrate import solve_ivp
///
/// model = gubas_rs.AugmentedDynamicsModel(
///     min_degree_a=2, max_degree_a=2,
///     min_degree_b=2, max_degree_b=2,
/// )
///
/// # Build initial augmented state
/// z0     = np.concatenate([x0, model.theta_nominal])
/// phi0   = np.eye(model.n_aug).ravel()
/// zphi0  = np.concatenate([z0, phi0])
///
/// # Propagate with scipy (interface B)
/// sol = solve_ivp(
///     lambda t, zp: model.eval_stm(zp.tolist(), t),
///     [t0, tf], zphi0, method="DOP853", rtol=1e-12,
/// )
/// phi_aug = sol.y[model.n_aug:].T.reshape(-1, model.n_aug, model.n_aug)
///
/// # Extract partials (Stokes sensitivity post-processing)
/// phi_xta = phi_aug[:, :30, 30:30+model.n_theta_a]   # (nsteps, 30, N_a)
/// phi_xtb = phi_aug[:, :30, 30+model.n_theta_a:]      # (nsteps, 30, N_b)
/// ```
#[cfg(feature = "extension-module")]
#[pyclass]
pub struct AugmentedDynamicsModel {
    params_dual_base: Params<dual::Dual>,
    theta_indices_a:  Vec<(usize, usize, usize)>,
    theta_indices_b:  Vec<(usize, usize, usize)>,
    pub n_aug:        usize,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl AugmentedDynamicsModel {
    /// Build model from `ic_input.txt` and the inertia files it names, in
    /// `work_dir` (default: the current directory).  The working directory of
    /// the process is not changed.
    ///
    /// State layout: z = [x(30); θ_a(N_a); θ_b(N_b)]
    ///   θ_a = primary   T_a integrals at degrees [min_degree_a, max_degree_a]
    ///   θ_b = secondary T_b integrals at degrees [min_degree_b, max_degree_b]
    ///
    /// Raises ValueError if a degree range includes degree 1 or exceeds the
    /// expansion order (see `check_theta_indices`).
    #[new]
    #[pyo3(signature = (min_degree_a=2, max_degree_a=2, min_degree_b=2, max_degree_b=2, work_dir=None))]
    fn new(
        min_degree_a: usize, max_degree_a: usize,
        min_degree_b: usize, max_degree_b: usize,
        work_dir: Option<&str>,
    ) -> PyResult<Self> {
        let dir = Path::new(work_dir.unwrap_or("."));
        let params = Params::load(dir, &Initialization::read(dir)?)?;

        let theta_indices_a = stokes::inertia_indices(min_degree_a, max_degree_a);
        let theta_indices_b = stokes::inertia_indices(min_degree_b, max_degree_b);
        for (label, idx) in [("primary", &theta_indices_a), ("secondary", &theta_indices_b)] {
            check_theta_indices(idx, params.n).map_err(|e|
                pyo3::exceptions::PyValueError::new_err(format!("{label}: {e}")))?;
        }
        let n_aug = 30 + theta_indices_a.len() + theta_indices_b.len();
        let params_dual_base = types::promote_params::<dual::Dual>(&params);

        Ok(Self { params_dual_base, theta_indices_a, theta_indices_b, n_aug })
    }

    /// Gravitational constant G (km³/(kg·s²)).
    #[getter]
    fn g(&self) -> f64 { self.params_dual_base.g.re }

    /// Expansion order of the mutual potential.
    #[getter]
    fn order(&self) -> usize { self.params_dual_base.n }

    /// Nominal primary mass T_a[0,0,0] (kg).
    #[getter]
    fn mass_a(&self) -> f64 { self.params_dual_base.ta.get(0, 0, 0).re }

    /// Nominal secondary mass T_b[0,0,0] (kg).
    #[getter]
    fn mass_b(&self) -> f64 { self.params_dual_base.tb.get(0, 0, 0).re }

    /// Nominal primary inertia tensor in its body frame (kg·km²), as rows.
    /// [`eval`] and [`eval_stm`] use the tensor implied by the θ in z.
    #[getter]
    fn inertia_a(&self) -> [[f64; 3]; 3] { self.params_dual_base.ia.map(|r| r.map(|v| v.re)) }

    /// Nominal secondary inertia tensor in its body frame (kg·km²), as rows.
    #[getter]
    fn inertia_b(&self) -> [[f64; 3]; 3] { self.params_dual_base.ib.map(|r| r.map(|v| v.re)) }

    /// Total augmented state dimension: 30 + N_a + N_b.
    #[getter]
    fn n_aug(&self) -> usize { self.n_aug }

    /// Number of primary θ_a parameters.
    #[getter]
    fn n_theta_a(&self) -> usize { self.theta_indices_a.len() }

    /// Number of secondary θ_b parameters.
    #[getter]
    fn n_theta_b(&self) -> usize { self.theta_indices_b.len() }

    /// (i,j,k) tuples for primary θ_a — columns 30..30+N_a of z.
    #[getter]
    fn theta_indices_a(&self) -> Vec<(usize,usize,usize)> { self.theta_indices_a.clone() }

    /// (i,j,k) tuples for secondary θ_b — columns 30+N_a..30+N_a+N_b of z.
    #[getter]
    fn theta_indices_b(&self) -> Vec<(usize,usize,usize)> { self.theta_indices_b.clone() }

    /// Nominal θ = [θ_a; θ_b].  Use as θ₀ when building z₀ = [x₀; θ₀].
    #[getter]
    fn theta_nominal(&self) -> Vec<f64> {
        let mut out: Vec<f64> = self.theta_indices_a.iter()
            .map(|&(i,j,k)| self.params_dual_base.ta.get(i,j,k).re).collect();
        out.extend(self.theta_indices_b.iter()
            .map(|&(i,j,k)| self.params_dual_base.tb.get(i,j,k).re));
        out
    }

    /// Nominal T_a values for primary θ_a only.
    #[getter]
    fn theta_nominal_a(&self) -> Vec<f64> {
        self.theta_indices_a.iter()
            .map(|&(i,j,k)| self.params_dual_base.ta.get(i,j,k).re).collect()
    }

    /// Nominal T_b values for secondary θ_b only.
    #[getter]
    fn theta_nominal_b(&self) -> Vec<f64> {
        self.theta_indices_b.iter()
            .map(|&(i,j,k)| self.params_dual_base.tb.get(i,j,k).re).collect()
    }

    /// Point evaluation of augmented dynamics and Jacobian (for EKF / UKF).
    ///
    /// Args:
    ///     z (list[float]): Augmented state of length n_aug = 30 + N_a + N_b.
    ///         Layout: z[:30] = x (F2BP state), z[30:30+N_a] = θ_a,
    ///         z[30+N_a:] = θ_b.
    ///     t (float): Epoch (s).
    ///
    /// Returns:
    ///     (zdot, A_aug_flat) where:
    ///       - zdot (list, len n_aug): augmented ODE RHS — ẋ for first 30,
    ///         0 for θ entries (parameters are constant).
    ///       - A_aug_flat (list, len n_aug²): row-major Jacobian A_aug.
    ///         Reshape to (n_aug, n_aug) with np.array(A_aug_flat).reshape(n_aug, n_aug).
    ///         Use A_aug to propagate covariance:  Ṗ = A·P + P·Aᵀ + Q.
    fn eval(&self, z: Vec<f64>, t: f64) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let n_aug = self.n_aug;
        if z.len() != n_aug {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("z must have {} elements, got {}", n_aug, z.len())));
        }
        let x: [f64; 30] = z[..30].try_into().unwrap();
        let theta = z[30..].to_vec();
        let aug_ode_dual = self.make_aug_dual_ode();
        let (zdot, a_aug) = stm::eval_aug_dynamics_and_jacobian(&x, &theta, t, &aug_ode_dual);
        Ok((zdot, a_aug))
    }

    /// Augmented state + STM ODE RHS for external integrators (scipy, etc.).
    ///
    /// Args:
    ///     z_phi_flat (list[float]): Flat vector of length n_aug + n_aug².
    ///         First n_aug elements: z = [x; θ_a; θ_b].
    ///         Last  n_aug² elements: Φ_aug flattened row-major
    ///         (initialize to np.eye(n_aug).ravel()).
    ///     t (float): Epoch (s).
    ///
    /// Returns:
    ///     list[float] of length n_aug + n_aug²: [ż; vec(Φ̇_aug)].
    ///     Pass directly to scipy.integrate.solve_ivp as the ODE RHS::
    ///
    ///         sol = solve_ivp(
    ///             lambda t, zp: model.eval_stm(zp.tolist(), t),
    ///             [t0, tf], zphi0, method="DOP853", rtol=1e-12,
    ///         )
    fn eval_stm(&self, z_phi_flat: Vec<f64>, t: f64) -> PyResult<Vec<f64>> {
        let expected = self.n_aug + self.n_aug * self.n_aug;
        if z_phi_flat.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("z_phi_flat must have {} elements, got {}",
                        expected, z_phi_flat.len())));
        }
        let aug_ode_dual = self.make_aug_dual_ode();
        Ok(stm::augmented_state_ode_rhs(&z_phi_flat, t, &aug_ode_dual))
    }
}

#[cfg(feature = "extension-module")]
impl AugmentedDynamicsModel {
    fn make_aug_dual_ode(&self)
        -> impl Fn(&[dual::Dual], f64) -> Vec<dual::Dual> + '_
    {
        move |z: &[dual::Dual], t: f64| dynamics::augmented_ode(
            z, t, &self.params_dual_base, &self.theta_indices_a, &self.theta_indices_b)
    }
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn gubas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_stm_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_stm_augmented_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_stm_augmented_both_py, m)?)?;
    m.add_class::<DynamicsModel>()?;
    m.add_class::<AugmentedDynamicsModel>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
