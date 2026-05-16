// stm.rs — generic State Transition Matrix propagator
//
// Computes Φ(t, t₀) = ∂x(t)/∂x₀  (30×30) alongside the state trajectory.
// The ODE closure is fully generic — the integrator is selectable:
//
//   propagate_stm_rk87_ad  (preferred) — Dormand-Prince RK7(8) adaptive,
//                           exact AD Jacobian at every stage (13 per step).
//                           Same step-control algorithm as the trajectory rk87.
//
//   propagate_stm_ad        — fixed-step RK4 + AD (simpler, kept for testing)
//   propagate_stm           — fixed-step RK4 + FD (for comparison / debugging)
//
// Two Jacobian strategies:
//   • AD  (preferred) — exact, via dual numbers; 30 dual ODE calls per sub-step
//   • FD  (verifier)  — forward differences; same call count but O(ε) error
//
// STM update: Φ̇ = A·Φ,  Φ(t₀,t₀) = I₃₀

use crate::dual::Dual;
use std::io::Write;

pub type Phi = [[f64; 30]; 30];

// ── Phi helpers ───────────────────────────────────────────────────────────────

#[inline]
pub fn phi_zero() -> Phi { [[0.0; 30]; 30] }

#[inline]
pub fn phi_eye() -> Phi {
    let mut p = phi_zero();
    for i in 0..30 { p[i][i] = 1.0; }
    p
}

#[inline]
fn phi_scale(s: f64, p: &Phi) -> Phi {
    let mut out = phi_zero();
    for i in 0..30 { for j in 0..30 { out[i][j] = s * p[i][j]; } }
    out
}

#[inline]
fn phi_add(a: &Phi, b: &Phi) -> Phi {
    let mut out = phi_zero();
    for i in 0..30 { for j in 0..30 { out[i][j] = a[i][j] + b[i][j]; } }
    out
}

/// Matrix product A · B  (row i, col j via shared index k).
#[inline]
fn phi_mul(a: &Phi, b: &Phi) -> Phi {
    let mut out = phi_zero();
    for i in 0..30 {
        for k in 0..30 {
            if a[i][k] == 0.0 { continue; }
            for j in 0..30 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

// ── Finite-difference Jacobian ────────────────────────────────────────────────

/// A(x, t) = ∂f/∂x  via forward differences.
///
/// `f0` = f(x, t) pre-computed, so we call `ode` 30 more times (not 31).
pub fn jacobian_fd<F>(x: &[f64; 30], t: f64, f0: &[f64; 30], ode: &F) -> Phi
where
    F: Fn([f64; 30], f64) -> [f64; 30],
{
    let mut a  = phi_zero();
    let mut xp = *x;
    for j in 0..30 {
        let eps = 1.5e-8 * x[j].abs().max(1.0);
        xp[j]  = x[j] + eps;
        let fp = ode(xp, t);
        for i in 0..30 {
            a[i][j] = (fp[i] - f0[i]) / eps;
        }
        xp[j] = x[j];
    }
    a
}

// ── RK4 step for (x, Φ) ──────────────────────────────────────────────────────

/// Advance (x, Φ) by one RK4 step of size h.
///
/// Φ̇ = A(x,t)·Φ  where A is re-evaluated at each of the four sub-steps.
fn step_rk4<F>(
    x:   [f64; 30],
    phi: &Phi,
    t:   f64,
    h:   f64,
    ode: &F,
) -> ([f64; 30], Phi)
where
    F: Fn([f64; 30], f64) -> [f64; 30],
{
    let add_scaled = |base: &[f64; 30], k: &[f64; 30], s: f64| -> [f64; 30] {
        let mut out = *base;
        for i in 0..30 { out[i] += s * k[i]; }
        out
    };

    // k1
    let k1x = ode(x, t);
    let a1  = jacobian_fd(&x, t, &k1x, ode);
    let k1p = phi_mul(&a1, phi);

    // k2
    let x2  = add_scaled(&x,  &k1x, 0.5 * h);
    let p2  = phi_add(phi, &phi_scale(0.5 * h, &k1p));
    let k2x = ode(x2, t + 0.5 * h);
    let a2  = jacobian_fd(&x2, t + 0.5 * h, &k2x, ode);
    let k2p = phi_mul(&a2, &p2);

    // k3
    let x3  = add_scaled(&x,  &k2x, 0.5 * h);
    let p3  = phi_add(phi, &phi_scale(0.5 * h, &k2p));
    let k3x = ode(x3, t + 0.5 * h);
    let a3  = jacobian_fd(&x3, t + 0.5 * h, &k3x, ode);
    let k3p = phi_mul(&a3, &p3);

    // k4
    let x4  = add_scaled(&x,  &k3x, h);
    let p4  = phi_add(phi, &phi_scale(h, &k3p));
    let k4x = ode(x4, t + h);
    let a4  = jacobian_fd(&x4, t + h, &k4x, ode);
    let k4p = phi_mul(&a4, &p4);

    // combine
    let mut xn = x;
    for i in 0..30 {
        xn[i] += h / 6.0 * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]);
    }
    let mut pn = phi_zero();
    for i in 0..30 { for j in 0..30 {
        pn[i][j] = phi[i][j]
            + h / 6.0 * (k1p[i][j] + 2.0 * k2p[i][j] + 2.0 * k3p[i][j] + k4p[i][j]);
    }}

    (xn, pn)
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Propagate state + STM from t₀ to tf with fixed step h.
///
/// * `ode`      — any `Fn([f64;30], f64) -> [f64;30]` closure (wrap `hou_ode`
///                with a captured `&Params` reference, or any other ODE)
/// * `out_freq` — record every `out_freq` steps (1 = every step)
///
/// Returns `(times, states, phis)`.
///
/// # Example
/// ```rust,ignore
/// use gubas_rs::stm::propagate_stm;
/// use gubas_rs::dynamics::hou_ode;
///
/// let ode = |x, t| hou_ode(x, t, &params);
/// let (ts, xs, phis) = propagate_stm(x0, t0, tf, h, ode, 1);
/// ```
pub fn propagate_stm<F>(
    x0:       [f64; 30],
    t0:       f64,
    tf:       f64,
    h:        f64,
    ode:      F,
    out_freq: usize,
) -> (Vec<f64>, Vec<[f64; 30]>, Vec<Phi>)
where
    F: Fn([f64; 30], f64) -> [f64; 30],
{
    let out_freq = out_freq.max(1);
    let nsteps   = ((tf - t0) / h).ceil() as usize;
    let cap      = nsteps / out_freq + 2;

    let mut times  = Vec::with_capacity(cap);
    let mut states = Vec::with_capacity(cap);
    let mut phis   = Vec::with_capacity(cap);

    let mut x   = x0;
    let mut phi = phi_eye();
    let mut t   = t0;

    times.push(t);
    states.push(x);
    phis.push(phi);

    for step in 1..=nsteps {
        let h_actual = h.min(tf - t);
        if h_actual <= 0.0 { break; }

        let (xn, pn) = step_rk4(x, &phi, t, h_actual, &ode);
        x   = xn;
        phi = pn;
        t   = (t0 + step as f64 * h).min(tf);

        if step % out_freq == 0 || (tf - t).abs() < 1e-14 * h {
            times.push(t);
            states.push(x);
            phis.push(phi);
        }
    }

    (times, states, phis)
}

// ── Auto-diff Jacobian ────────────────────────────────────────────────────────

/// A(x, t) = ∂f/∂x  via forward-mode automatic differentiation (exact).
///
/// Column j: seed `x_dual[j].eps = 1`, call ode_dual, read output `.eps`.
/// No step-size to tune; no truncation error beyond f64 rounding.
pub fn jacobian_ad<FD>(x: &[f64; 30], t: f64, ode_dual: &FD) -> Phi
where
    FD: Fn([Dual; 30], f64) -> [Dual; 30],
{
    let mut a = phi_zero();
    let mut xd: [Dual; 30] = x.map(Dual::from_re);
    for j in 0..30 {
        xd[j].eps = 1.0;
        let fd = ode_dual(xd, t);
        for i in 0..30 { a[i][j] = fd[i].eps; }
        xd[j].eps = 0.0;
    }
    a
}

/// Advance (x, Φ) by one RK4 step using the AD Jacobian.
fn step_rk4_ad<F, FD>(
    x:        [f64; 30],
    phi:      &Phi,
    t:        f64,
    h:        f64,
    ode:      &F,
    ode_dual: &FD,
) -> ([f64; 30], Phi)
where
    F:  Fn([f64; 30], f64) -> [f64; 30],
    FD: Fn([Dual; 30], f64) -> [Dual; 30],
{
    let add_scaled = |base: &[f64; 30], k: &[f64; 30], s: f64| -> [f64; 30] {
        let mut out = *base;
        for i in 0..30 { out[i] += s * k[i]; }
        out
    };

    let k1x = ode(x, t);
    let a1  = jacobian_ad(&x, t, ode_dual);
    let k1p = phi_mul(&a1, phi);

    let x2  = add_scaled(&x, &k1x, 0.5 * h);
    let p2  = phi_add(phi, &phi_scale(0.5 * h, &k1p));
    let k2x = ode(x2, t + 0.5 * h);
    let a2  = jacobian_ad(&x2, t + 0.5 * h, ode_dual);
    let k2p = phi_mul(&a2, &p2);

    let x3  = add_scaled(&x, &k2x, 0.5 * h);
    let p3  = phi_add(phi, &phi_scale(0.5 * h, &k2p));
    let k3x = ode(x3, t + 0.5 * h);
    let a3  = jacobian_ad(&x3, t + 0.5 * h, ode_dual);
    let k3p = phi_mul(&a3, &p3);

    let x4  = add_scaled(&x, &k3x, h);
    let p4  = phi_add(phi, &phi_scale(h, &k3p));
    let k4x = ode(x4, t + h);
    let a4  = jacobian_ad(&x4, t + h, ode_dual);
    let k4p = phi_mul(&a4, &p4);

    let mut xn = x;
    for i in 0..30 {
        xn[i] += h / 6.0 * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]);
    }
    let mut pn = phi_zero();
    for i in 0..30 { for j in 0..30 {
        pn[i][j] = phi[i][j]
            + h / 6.0 * (k1p[i][j] + 2.0 * k2p[i][j] + 2.0 * k3p[i][j] + k4p[i][j]);
    }}
    (xn, pn)
}

/// Propagate state + STM using the **exact AD Jacobian**.
///
/// Requires two closures wrapping the same underlying ODE at f64 and Dual
/// precision.  Build them with [`crate::types::promote_params`]:
///
/// ```rust,ignore
/// use gubas_rs::dual::Dual;
/// use gubas_rs::types::promote_params;
///
/// let pd = promote_params::<Dual>(&params);          // one-time clone
/// let ode      = |x, t| hou_ode(x, t, &params);
/// let ode_dual = |x, t| hou_ode(x, t, &pd);
/// let (ts, xs, phis) = propagate_stm_ad(x0, t0, tf, h, ode, ode_dual, 1);
/// ```
pub fn propagate_stm_ad<F, FD>(
    x0:       [f64; 30],
    t0:       f64,
    tf:       f64,
    h:        f64,
    ode:      F,
    ode_dual: FD,
    out_freq: usize,
) -> (Vec<f64>, Vec<[f64; 30]>, Vec<Phi>)
where
    F:  Fn([f64; 30], f64) -> [f64; 30],
    FD: Fn([Dual; 30], f64) -> [Dual; 30],
{
    let out_freq = out_freq.max(1);
    let nsteps   = ((tf - t0) / h).ceil() as usize;
    let cap      = nsteps / out_freq + 2;

    let mut times  = Vec::with_capacity(cap);
    let mut states = Vec::with_capacity(cap);
    let mut phis   = Vec::with_capacity(cap);

    let mut x   = x0;
    let mut phi = phi_eye();
    let mut t   = t0;

    times.push(t); states.push(x); phis.push(phi);

    for step in 1..=nsteps {
        let h_actual = h.min(tf - t);
        if h_actual <= 0.0 { break; }

        let (xn, pn) = step_rk4_ad(x, &phi, t, h_actual, &ode, &ode_dual);
        x   = xn;
        phi = pn;
        t   = (t0 + step as f64 * h).min(tf);

        if step % out_freq == 0 || (tf - t).abs() < 1e-14 * h {
            times.push(t);
            states.push(x);
            phis.push(phi);
        }
    }
    (times, states, phis)
}

// ── RK7(8) Dormand-Prince Butcher tableau ────────────────────────────────────
// (identical coefficients to integrators.rs::rk87 — reproduced here so stm.rs
//  is self-contained and the trajectory integrator stays untouched)

#[rustfmt::skip]
const DP_A: [[f64; 13]; 12] = [
    [1./18., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [1./48., 1./16., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [1./32., 0., 3./32., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [5./16., 0., -75./64., 75./64., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [3./80., 0., 0., 3./16., 3./20., 0., 0., 0., 0., 0., 0., 0., 0.],
    [29443841./614563906., 0., 0., 77736538./692538347.,
     -28693883./1125000000., 23124283./1800000000., 0., 0., 0., 0., 0., 0., 0.],
    [16016141./946692911., 0., 0., 61564180./158732637.,
     22789713./633445777., 545815736./2771057229., -180193667./1043307555.,
     0., 0., 0., 0., 0., 0.],
    [39632708./573591083., 0., 0., -433636366./683701615.,
     -421739975./2616292301., 100302831./723423059., 790204164./839813087.,
     800635310./3783071287., 0., 0., 0., 0., 0.],
    [246121993./1340847787., 0., 0., -37695042795./15268766246.,
     -309121744./1061227803., -12992083./490766935., 6005943493./2108947869.,
     393006217./1396673457., 123872331./1001029789., 0., 0., 0., 0.],
    [-1028468189./846180014., 0., 0., 8478235783./508512852.,
     1311729495./1432422823., -10304129995./1701304382., -48777925059./3047939560.,
     15336726248./1032824649., -45442868181./3398467696., 3065993473./597172653.,
     0., 0., 0.],
    [185892177./718116043., 0., 0., -3185094517./667107341.,
     -477755414./1098053517., -703635378./230739211., 5731566787./1027545527.,
     5232866602./850066563., -4093664535./808688257., 3962137247./1805957418.,
     65686358./487910083., 0., 0.],
    [403863854./491063109., 0., 0., -5068492393./434740067.,
     -411421997./543043805., 652783627./914296604., 11173962825./925320556.,
     -13158990841./6184727034., 3936647629./1978049680., -160528059./685178525.,
     248638103./1413531060., 0., 0.],
];

const DP_C: [f64; 12] = [
    1./18., 1./12., 1./8., 5./16., 3./8., 59./400.,
    93./200., 5490023248./9719169821., 13./20.,
    1201146811./1299019798., 1., 1.,
];

const DP_B8: [f64; 13] = [
    14005451./335480064., 0., 0., 0., 0.,
    -59238493./1068277825., 181606767./758867731., 561292985./797845732.,
    -1041891430./1371343529., 760417239./1151165299., 118820643./751138087.,
    -528747749./2220607170., 1./4.,
];

const DP_B7: [f64; 13] = [
    13451932./455176623., 0., 0., 0., 0.,
    -808719846./976000145., 1757004468./5645159321., 656045339./265891186.,
    -3867574721./1518517206., 465885868./322736535., 53011238./667516719.,
    2./45., 0.,
];

// ── RK7(8) adaptive STM propagator ───────────────────────────────────────────

/// Propagate state + STM with the **Dormand-Prince RK7(8) adaptive integrator**
/// and exact AD Jacobian — the highest-accuracy option available.
///
/// * `tol`      — relative tolerance (same semantics as `rk87`; e.g. 1e-12)
/// * `out_freq` — record every `out_freq` *accepted* steps (1 = every step)
///
/// The Jacobian A(x,t) is evaluated at all 13 Runge-Kutta stages per step.
/// Step size is controlled by the 7th vs 8th order difference in x only
/// (Phi error is not used for step control — standard practice).
///
/// Build the closures exactly as for `propagate_stm_ad`:
/// ```rust,ignore
/// let pd = promote_params::<Dual>(&params);
/// let ode      = |x, t| hou_ode(x, t, &params);
/// let ode_dual = |x, t| hou_ode(x, t, &pd);
/// let (ts, xs, phis) = propagate_stm_rk87_ad(x0, t0, tf, 1e-12, ode, ode_dual, 1);
/// ```
pub fn propagate_stm_rk87_ad<F, FD>(
    x0:       [f64; 30],
    t0:       f64,
    tf:       f64,
    tol:      f64,
    ode:      F,
    ode_dual: FD,
    out_freq: usize,
) -> (Vec<f64>, Vec<[f64; 30]>, Vec<Phi>)
where
    F:  Fn([f64; 30], f64) -> [f64; 30],
    FD: Fn([Dual; 30], f64) -> [Dual; 30],
{
    const EPS: f64 = 5e-16;
    const POW: f64 = 1.0 / 8.0;
    let out_freq = out_freq.max(1);

    let hmax = (tf - t0) / 2.5;
    let mut h = ((tf - t0) / 50.0).min(0.1).min(hmax);

    let mut t   = t0;
    let mut x   = x0;
    let mut phi = phi_eye();

    let mut times  = vec![t];
    let mut states = vec![x];
    let mut phis   = vec![phi];
    let mut n_accepted = 0usize;
    let mut n_rejected = 0usize;

    while t < tf {
        if t + h > tf { h = tf - t; }

        // ── Evaluate all 13 stages ────────────────────────────────────────────
        let mut kx:  Vec<[f64; 30]> = vec![[0.0; 30]; 13];
        let mut kphi: Vec<Phi>      = vec![phi_zero(); 13];

        // Stage 0: at current (x, t)
        kx[0]   = ode(x, t);
        let a0  = jacobian_ad(&x, t, &ode_dual);
        kphi[0] = phi_mul(&a0, &phi);

        // Stages 1..12
        for j in 1..=12_usize {
            // Accumulate x and Φ at this sub-stage
            let mut xj   = x;
            let mut phij = phi;
            for i in 0..j {
                let a = DP_A[j - 1][i];
                if a == 0.0 { continue; }
                for k in 0..30 { xj[k] += h * a * kx[i][k]; }
                for r in 0..30 { for c in 0..30 {
                    phij[r][c] += h * a * kphi[i][r][c];
                }}
            }
            let tj  = t + DP_C[j - 1] * h;
            kx[j]   = ode(xj, tj);
            let aj  = jacobian_ad(&xj, tj, &ode_dual);
            kphi[j] = phi_mul(&aj, &phij);
        }

        // ── 7th and 8th order solutions ───────────────────────────────────────
        let mut x7 = x; let mut x8 = x;
        let mut p7 = phi; let mut p8 = phi;
        for i in 0..13 {
            let b7 = DP_B7[i]; let b8 = DP_B8[i];
            if b7 == 0.0 && b8 == 0.0 { continue; }
            for k in 0..30 {
                x7[k] += h * b7 * kx[i][k];
                x8[k] += h * b8 * kx[i][k];
            }
            for r in 0..30 { for c in 0..30 {
                p7[r][c] += h * b7 * kphi[i][r][c];
                p8[r][c] += h * b8 * kphi[i][r][c];
            }}
        }

        // ── Error estimate (on x only — standard practice) ────────────────────
        let err = (0..30).map(|k| (x8[k] - x7[k]).abs())
                         .fold(0.0_f64, f64::max);
        let tau = tol * (0..30).map(|k| x[k].abs()).fold(0.0_f64, f64::max);

        if err <= tau {
            x   = x7;
            phi = p7;
            t  += h;
            n_accepted += 1;

            if n_accepted % out_freq == 0 || (tf - t).abs() < 1e-14 * h {
                times.push(t);
                states.push(x);
                phis.push(phi);
            }
        } else {
            n_rejected += 1;
        }

        // ── Step-size update ──────────────────────────────────────────────────
        let err_s = if err == 0.0 { 10.0 * EPS } else { err };
        let tau_s = tau.max(tol * EPS);
        h = hmax.min(0.9 * h * (tau_s / err_s).powf(POW));

        if h.abs() <= EPS {
            eprintln!("propagate_stm_rk87_ad: step size at machine precision");
            break;
        }
    }

    eprintln!("  rk87_stm: {} accepted + {} rejected steps", n_accepted, n_rejected);
    (times, states, phis)
}

// ── OD interface — point evaluation ──────────────────────────────────────────

/// Evaluate the dynamics and exact Jacobian at a single (x, t).
///
/// Returns `(ẋ, A)` where:
/// * `ẋ = f(x, t)`          — ODE right-hand side (force model)
/// * `A = ∂f/∂x`            — 30×30 Jacobian matrix
///
/// External OD filters use `A` to:
/// * Propagate the STM:        `Φ̇ = A · Φ`
/// * Propagate the covariance: `Ṗ = A·P + P·Aᵀ + Q`
pub fn eval_dynamics_and_jacobian<F, FD>(
    x:        [f64; 30],
    t:        f64,
    ode:      &F,
    ode_dual: &FD,
) -> ([f64; 30], Phi)
where
    F:  Fn([f64; 30], f64) -> [f64; 30],
    FD: Fn([Dual; 30], f64) -> [Dual; 30],
{
    let xdot = ode(x, t);
    let a    = jacobian_ad(&x, t, ode_dual);
    (xdot, a)
}

/// Augmented-state ODE for external numerical integration.
///
/// State layout (930 elements, little-endian f64):
/// ```text
/// xphi[0  .. 30 ] = x         (30-element state)
/// xphi[30 .. 930] = Φ         (30×30 STM, row-major: xphi[30 + 30·i + j] = Φ[i][j])
/// ```
///
/// Returns `[ẋ; vec(Φ̇)]` in the same layout, so any external integrator can call
/// this function as-is.  Initialize with `xphi = [x₀; vec(I₃₀)]`.
///
/// # Python / scipy usage
/// ```python
/// def aug_ode(t, xphi):
///     return rust_eval_augmented_ode(xphi, t)  # via PyO3 or ctypes
///
/// phi0 = np.eye(30).ravel()
/// sol  = scipy.integrate.solve_ivp(
///     aug_ode, [t0, tf], np.concatenate([x0, phi0]),
///     method="DOP853", rtol=1e-12)
/// phis = sol.y[30:].reshape(30, 30, -1)  # (30, 30, nsteps)
/// ```
pub fn augmented_ode_rhs<F, FD>(
    xphi:     [f64; 930],
    t:        f64,
    ode:      &F,
    ode_dual: &FD,
) -> [f64; 930]
where
    F:  Fn([f64; 30], f64) -> [f64; 30],
    FD: Fn([Dual; 30], f64) -> [Dual; 30],
{
    // Unpack x and Φ
    let x: [f64; 30] = xphi[..30].try_into().unwrap();
    let mut phi = phi_zero();
    for i in 0..30 { for j in 0..30 {
        phi[i][j] = xphi[30 + 30 * i + j];
    }}

    // Compute RHS
    let xdot   = ode(x, t);
    let a      = jacobian_ad(&x, t, ode_dual);
    let phidot = phi_mul(&a, &phi);

    // Pack output in the same [x; Φ] layout
    let mut out = [0.0_f64; 930];
    out[..30].copy_from_slice(&xdot);
    for i in 0..30 { for j in 0..30 {
        out[30 + 30 * i + j] = phidot[i][j];
    }}
    out
}

/// Write state trajectory to `{dir}/x_out.bin` (little-endian f64).
/// Shape: (nsteps, 30).  Read in Python: `np.fromfile(..., "<f8").reshape(-1, 30)`.
pub fn write_x_bin(dir: &str, xs: &[[f64; 30]]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(format!("{}/x_out.bin", dir))?);
    for x in xs {
        for &v in x { f.write_all(&v.to_le_bytes())?; }
    }
    Ok(())
}

/// Write `ẋ` time series to `{dir}/xdot_out.bin` (little-endian f64).
/// Shape: (nsteps, 30).  Read in Python: `np.fromfile(..., "<f8").reshape(-1, 30)`.
pub fn write_xdot_bin(dir: &str, xdots: &[[f64; 30]]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(format!("{}/xdot_out.bin", dir))?);
    for xd in xdots {
        for &v in xd { f.write_all(&v.to_le_bytes())?; }
    }
    Ok(())
}

/// Write Jacobian `A = ∂f/∂x` time series to `{dir}/A_out.bin` (little-endian f64).
/// Shape: (nsteps, 30, 30), row-major.
/// Read in Python: `np.fromfile(..., "<f8").reshape(-1, 30, 30)`.
#[allow(non_snake_case)]
pub fn write_A_bin(dir: &str, As: &[Phi]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(format!("{}/A_out.bin", dir))?);
    for phi in As {
        for row in phi.iter() {
            for &v in row.iter() { f.write_all(&v.to_le_bytes())?; }
        }
    }
    Ok(())
}

// ── Jacobian verification ─────────────────────────────────────────────────────

/// Compute the max |AD − FD| over all 900 Jacobian entries at point (x, t).
///
/// Typical result: ~1e-7 to 1e-9 (FD truncation) when the ODE is smooth.
/// Values much larger than 1e-5 suggest a bug in the dual-number physics.
pub fn verify_jacobian<F, FD>(
    x:        &[f64; 30],
    t:        f64,
    ode:      &F,
    ode_dual: &FD,
) -> f64
where
    F:  Fn([f64; 30], f64) -> [f64; 30],
    FD: Fn([Dual; 30], f64) -> [Dual; 30],
{
    let f0     = ode(*x, t);
    let jac_fd = jacobian_fd(x, t, &f0, ode);
    let jac_ad = jacobian_ad(x, t, ode_dual);
    let mut max_err = 0.0_f64;
    for i in 0..30 { for j in 0..30 {
        let err = (jac_ad[i][j] - jac_fd[i][j]).abs();
        if err > max_err { max_err = err; }
    }}
    max_err
}

// ── Binary output ─────────────────────────────────────────────────────────────

/// Write the STM history to `{dir}/phi_out.bin`  (little-endian f64, row-major).
///
/// Layout per recorded step: 900 f64 values = `Φ[0][0]` … `Φ[29][29]`.
///
/// Read in Python:
/// ```python
/// import numpy as np
/// phis = np.fromfile("output_phi/phi_out.bin", dtype="<f8").reshape(-1, 30, 30)
/// ```
pub fn write_phi_bin(dir: &str, phis: &[Phi]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = format!("{}/phi_out.bin", dir);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
    for phi in phis {
        for row in phi.iter() {
            for &v in row.iter() {
                f.write_all(&v.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

/// Write the time vector to `{dir}/phi_t_out.bin`  (little-endian f64).
///
/// Read in Python:
/// ```python
/// import numpy as np
/// ts = np.fromfile("output_phi/phi_t_out.bin", dtype="<f8")
/// ```
pub fn write_phi_t_bin(dir: &str, times: &[f64]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = format!("{}/phi_t_out.bin", dir);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
    for &t in times {
        f.write_all(&t.to_le_bytes())?;
    }
    Ok(())
}

// ── Augmented STM propagator ─────────────────────────────────────────────────
//
// Propagates the full (30+N)×(30+N) augmented STM Φ_aug alongside the
// trajectory.  State z = [x(30); θ(N)], θ̇ = 0 (constant parameters).
//
// Φ_aug structure (written to phi_aug_out.bin):
//   ┌─────────────────────────────────┐
//   │  Φ_xx (30×30)  │  Φ_xθ (30×N) │
//   │─────────────────│──────────────│
//   │  0     (N×30)  │  I_N  (N×N)  │
//   └─────────────────────────────────┘
// Top rows are propagated; bottom rows are trivially [0|I] (θ̇=0).
//
// Point-evaluation interface (for external OD filters / scipy):
//   eval_aug_dynamics_and_jacobian(x, θ, t, aug_ode_dual) → (ż, A_aug_flat)
//   augmented_state_ode_rhs(z_phi_flat, t, aug_ode_dual)  → full ODE RHS

/// Compute [A (30×30), B (30×N)] from one augmented dual-ODE call per column.
/// `A[i][j]` = ∂f_i/∂x_j,   `B[i][k]` = ∂f_i/∂θ_k.
pub fn jacobian_aug_ad<FD>(
    x:             &[f64; 30],
    theta:         &[f64],       // N parameters
    t:             f64,
    aug_ode_dual:  &FD,
) -> (Phi, Vec<[f64; 30]>)       // (A, B) — B[k] is a 30-vector, the k-th column
where
    FD: Fn(&[Dual], f64) -> Vec<Dual>,
{
    let n_theta = theta.len();
    let n_aug   = 30 + n_theta;

    // Seed point: re-part is [x; theta], eps = 0 everywhere
    let mut zd: Vec<Dual> = Vec::with_capacity(n_aug);
    for &v in x.iter()     { zd.push(Dual::from_re(v)); }
    for &v in theta.iter() { zd.push(Dual::from_re(v)); }

    let mut a   = phi_zero();
    let mut b   = vec![[0.0_f64; 30]; n_theta];

    // Columns 0..30 → A
    for j in 0..30 {
        zd[j].eps = 1.0;
        let fd = aug_ode_dual(&zd, t);
        for i in 0..30 { a[i][j] = fd[i].eps; }
        zd[j].eps = 0.0;
    }
    // Columns 30..30+N → B
    for k in 0..n_theta {
        zd[30 + k].eps = 1.0;
        let fd = aug_ode_dual(&zd, t);
        for i in 0..30 { b[k][i] = fd[i].eps; }
        zd[30 + k].eps = 0.0;
    }
    (a, b)
}

/// Propagate state + full augmented STM Φ_aug with RK7(8) adaptive.
///
/// Φ_aug is (30+N)×(30+N): top rows [Φ_xx | Φ_xθ] are propagated;
/// bottom rows [0 | I_N] are trivial (θ̇ = 0) and restored at output.
///
/// # Returns
/// `(times, x_states, phi_aug_flat_history)` where
/// `phi_aug_flat[step]` has (30+N)² elements, row-major, so in Python:
/// `phi_aug = np.fromfile(...).reshape(-1, 30+N, 30+N)`
pub fn propagate_augmented_rk87_ad<F, FD>(
    x0:           [f64; 30],
    theta0:       Vec<f64>,
    t0:           f64,
    tf:           f64,
    tol:          f64,
    ode:          F,
    aug_ode_dual: FD,
    out_freq:     usize,
) -> (Vec<f64>, Vec<[f64; 30]>, Vec<Vec<f64>>)  // (times, states, phi_aug_flat)
where
    F:  Fn([f64; 30], f64) -> [f64; 30],
    FD: Fn(&[Dual], f64) -> Vec<Dual>,
{
    const EPS: f64 = 5e-16;
    const POW: f64 = 1.0 / 8.0;
    let n_theta  = theta0.len();
    let n_aug    = 30 + n_theta;
    let out_freq = out_freq.max(1);

    let hmax = (tf - t0) / 2.5;
    let mut h = ((tf - t0) / 50.0).min(0.1).min(hmax);

    let mut t      = t0;
    let mut x      = x0;
    let theta      = theta0.clone();
    let mut phi_xx = phi_eye();
    let mut phi_xt = vec![0.0_f64; 30 * n_theta]; // Φ_xθ(t₀) = 0

    let mut times   = vec![t];
    let mut states  = vec![x];
    let mut phi_aug_h = vec![assemble_phi_aug(&phi_xx, &phi_xt, n_theta)];

    let mut n_accepted = 0usize;
    let mut n_rejected = 0usize;

    while t < tf {
        if t + h > tf { h = tf - t; }

        let mut kx   = vec![[0.0_f64; 30]; 13];
        let mut kphi = vec![phi_zero(); 13];
        let mut kpxt = vec![vec![0.0_f64; 30 * n_theta]; 13];

        // Stage 0
        kx[0] = ode(x, t);
        let (a0, b0) = jacobian_aug_ad(&x, &theta, t, &aug_ode_dual);
        kphi[0] = phi_mul(&a0, &phi_xx);
        kpxt[0] = phi_xt_dot(&a0, &b0, &phi_xt, n_theta);

        // Stages 1..12
        for j in 1..=12_usize {
            let mut xj   = x;
            let mut phij = phi_xx;
            let mut pxtj = phi_xt.clone();
            for i in 0..j {
                let a = DP_A[j - 1][i];
                if a == 0.0 { continue; }
                for k in 0..30 { xj[k] += h * a * kx[i][k]; }
                for r in 0..30 { for c in 0..30 { phij[r][c] += h * a * kphi[i][r][c]; }}
                for q in 0..(30 * n_theta) { pxtj[q] += h * a * kpxt[i][q]; }
            }
            let tj = t + DP_C[j - 1] * h;
            kx[j] = ode(xj, tj);
            let (aj, bj) = jacobian_aug_ad(&xj, &theta, tj, &aug_ode_dual);
            kphi[j] = phi_mul(&aj, &phij);
            kpxt[j] = phi_xt_dot(&aj, &bj, &pxtj, n_theta);
        }

        // 7th order solution
        let mut x7   = x;
        let mut p7   = phi_xx;
        let mut pxt7 = phi_xt.clone();
        for i in 0..13 {
            let b7 = DP_B7[i];
            if b7 == 0.0 { continue; }
            for k in 0..30 { x7[k] += h * b7 * kx[i][k]; }
            for r in 0..30 { for c in 0..30 { p7[r][c] += h * b7 * kphi[i][r][c]; }}
            for q in 0..(30 * n_theta) { pxt7[q] += h * b7 * kpxt[i][q]; }
        }

        // 8th order (error estimate on x only)
        let mut x8 = x;
        for i in 0..13 {
            let b8 = DP_B8[i]; if b8 == 0.0 { continue; }
            for k in 0..30 { x8[k] += h * b8 * kx[i][k]; }
        }

        let err = (0..30).map(|k| (x8[k] - x7[k]).abs()).fold(0.0_f64, f64::max);
        let tau = tol * (0..30).map(|k| x[k].abs()).fold(0.0_f64, f64::max);

        if err <= tau {
            x      = x7;
            phi_xx = p7;
            phi_xt = pxt7;
            t     += h;
            n_accepted += 1;
            if n_accepted % out_freq == 0 || (tf - t).abs() < 1e-14 * h {
                times.push(t);
                states.push(x);
                phi_aug_h.push(assemble_phi_aug(&phi_xx, &phi_xt, n_theta));
            }
        } else {
            n_rejected += 1;
        }

        let err_s = if err == 0.0 { 10.0 * EPS } else { err };
        let tau_s = tau.max(tol * EPS);
        h = hmax.min(0.9 * h * (tau_s / err_s).powf(POW));
        if h.abs() <= EPS {
            eprintln!("propagate_augmented_rk87_ad: step size at machine precision");
            break;
        }
    }
    eprintln!("  rk87_aug: {} accepted + {} rejected steps, N_aug={}", n_accepted, n_rejected, n_aug);
    (times, states, phi_aug_h)
}

/// Φ̇_xθ = A · Φ_xθ + B  (flattened, row-major by state index).
fn phi_xt_dot(a: &Phi, b_cols: &[[f64; 30]], phi_xt: &[f64], n_theta: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; 30 * n_theta];
    for i in 0..30 {
        for k in 0..n_theta {
            let apxt: f64 = (0..30).map(|j| a[i][j] * phi_xt[j * n_theta + k]).sum();
            out[i * n_theta + k] = apxt + b_cols[k][i];
        }
    }
    out
}

/// Assemble the full (30+N)×(30+N) Φ_aug from Φ_xx and Φ_xθ.
/// Bottom rows are [0 | I_N] (trivial since θ̇ = 0).
pub fn assemble_phi_aug(phi_xx: &Phi, phi_xt: &[f64], n_theta: usize) -> Vec<f64> {
    let n_aug = 30 + n_theta;
    let mut out = vec![0.0_f64; n_aug * n_aug];
    // Top-left: Φ_xx
    for i in 0..30 { for j in 0..30 { out[i * n_aug + j] = phi_xx[i][j]; } }
    // Top-right: Φ_xθ  (phi_xt[i*n_theta + k] = Φ_xθ[i][k])
    for i in 0..30 { for k in 0..n_theta { out[i * n_aug + 30 + k] = phi_xt[i * n_theta + k]; } }
    // Bottom-right: I_N
    for k in 0..n_theta { out[(30 + k) * n_aug + 30 + k] = 1.0; }
    out
}

// ── Point-evaluation interface for external OD filters ────────────────────────

/// Evaluate augmented dynamics ż and Jacobian A_aug at (z, t).
/// z = [x(30); θ(N)].  Returns:
///   - ż        : Vec of length 30+N   (θ̇ = 0 for last N entries)
///   - A_aug_flat: Vec of length (30+N)²  (row-major)
///     A_aug = | A  B |   `A[i][j]`=∂f_i/∂x_j,   `B[i][k]`=∂f_i/∂θ_k
///             | 0  0 |
pub fn eval_aug_dynamics_and_jacobian<FD>(
    x:            &[f64; 30],
    theta:        &[f64],
    t:            f64,
    aug_ode_dual: &FD,
) -> (Vec<f64>, Vec<f64>)
where
    FD: Fn(&[Dual], f64) -> Vec<Dual>,
{
    let n_theta = theta.len();
    let n_aug   = 30 + n_theta;

    // zdot (re-parts of aug_ode_dual at eps=0)
    let zd: Vec<Dual> = x.iter().chain(theta.iter())
        .map(|&v| Dual::from_re(v)).collect();
    let zdot_dual = aug_ode_dual(&zd, t);
    let zdot: Vec<f64> = zdot_dual.iter().map(|d| d.re).collect();

    // Jacobians A and B
    let (a, b_cols) = jacobian_aug_ad(x, theta, t, aug_ode_dual);

    // Assemble A_aug (n_aug × n_aug, row-major)
    let mut a_aug = vec![0.0_f64; n_aug * n_aug];
    for i in 0..30 {
        for j in 0..30 { a_aug[i * n_aug + j]      = a[i][j]; }
        for k in 0..n_theta { a_aug[i * n_aug + 30 + k] = b_cols[k][i]; }
    }
    (zdot, a_aug)
}

/// Augmented state + STM ODE RHS for external numerical integration (scipy etc.).
///
/// Input `z_phi_flat` has (30+N) + (30+N)² elements:
///   z_phi_flat[..30+N]   = z = [x; θ]
///   z_phi_flat[30+N..]   = Φ_aug flattened row-major, (30+N)×(30+N)
///
/// Returns same layout: [ż; A_aug · Φ_aug].
/// Initialize with z=[x₀; θ₀] and Φ_aug = I_{30+N}.
pub fn augmented_state_ode_rhs<FD>(
    z_phi_flat:   &[f64],
    t:            f64,
    aug_ode_dual: &FD,
) -> Vec<f64>
where
    FD: Fn(&[Dual], f64) -> Vec<Dual>,
{
    // Recover n_aug: n_aug² + n_aug = len → n_aug = (√(1+4·len) - 1)/2
    let len = z_phi_flat.len();
    let n_aug = (((-1.0 + (1.0 + 4.0 * len as f64).sqrt()) / 2.0).round()) as usize;
    let _n_theta = n_aug - 30;

    let z        = &z_phi_flat[..n_aug];
    let phi_flat = &z_phi_flat[n_aug..];
    let x: [f64; 30] = z[..30].try_into().unwrap();
    let theta = &z[30..];

    let (zdot, a_aug) = eval_aug_dynamics_and_jacobian(&x, theta, t, aug_ode_dual);

    // Φ̇_aug = A_aug · Φ_aug  (n_aug × n_aug matmul)
    let mut phi_dot = vec![0.0_f64; n_aug * n_aug];
    for i in 0..n_aug {
        for k in 0..n_aug {
            if a_aug[i * n_aug + k] == 0.0 { continue; }
            for j in 0..n_aug {
                phi_dot[i * n_aug + j] += a_aug[i * n_aug + k] * phi_flat[k * n_aug + j];
            }
        }
    }

    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(&zdot);
    out.extend_from_slice(&phi_dot);
    out
}

// ── Binary I/O for augmented output ──────────────────────────────────────────

/// Write Φ_aug history to `{dir}/phi_aug_out.bin` (little-endian f64).
/// Shape: (nsteps, 30+N, 30+N), row-major.
/// Read in Python: `np.fromfile(..., "<f8").reshape(-1, n_aug, n_aug)`.
pub fn write_phi_aug_bin(dir: &str, phi_augs: &[Vec<f64>]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(format!("{}/phi_aug_out.bin", dir))?);
    for pa in phi_augs {
        for &v in pa { f.write_all(&v.to_le_bytes())?; }
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::Dual;
    use crate::math3::Scalar;

    // Generic linear ODE:  ẋ[0] = x[1],  ẋ[1] = -x[0],  ẋ[i] = x[i] (i≥2)
    // Jacobian is constant:  A[0][1]=1, A[1][0]=-1, A[i][i]=1 (i≥2)
    fn linear_ode<T: Scalar>(x: [T; 30], _t: f64) -> [T; 30] {
        let mut out = [T::zero(); 30];
        out[0] = x[1];
        out[1] = -x[0];
        for i in 2..30 { out[i] = x[i]; }
        out
    }

    fn exact_jac() -> Phi {
        let mut a = phi_zero();
        a[0][1] =  1.0;
        a[1][0] = -1.0;
        for i in 2..30 { a[i][i] = 1.0; }
        a
    }

    #[test]
    fn ad_jacobian_is_exact() {
        let mut x0 = [1.0f64; 30];
        x0[0] = 1.23; x0[1] = -0.77;

        let jac = jacobian_ad(&x0, 0.0, &|x, t| linear_ode::<Dual>(x, t));
        let exact = exact_jac();

        let max_err = (0..30).flat_map(|i| (0..30).map(move |j| (jac[i][j] - exact[i][j]).abs()))
                             .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-14, "AD error = {max_err:.2e}  (should be ~machine eps)");
    }

    #[test]
    fn fd_jacobian_close_to_ad() {
        let mut x0 = [1.0f64; 30];
        x0[0] = 1.23; x0[1] = -0.77;

        let ode      = |x, t| linear_ode::<f64>(x, t);
        let ode_dual = |x, t| linear_ode::<Dual>(x, t);
        let err = verify_jacobian(&x0, 0.0, &ode, &ode_dual);

        // FD truncation is O(h) ≈ 1.5e-8; AD is exact; difference is ~1e-8..1e-7
        assert!(err < 1e-5, "AD vs FD max diff = {err:.2e}  (expected < 1e-5)");
        println!("AD vs FD max |Δ| = {err:.3e}");
    }
}