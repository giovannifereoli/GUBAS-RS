//! Fixed-step RK4 / ABM, adaptive RK7(8), and LGVI integrators.
//!
//! Each integrator writes native-endian `f64` binary files:
//! - `output_t/t_out.bin`    — one `f64` per step (time)
//! - `output_x/x_out.bin`    — 30 `f64`s per step (state, row-major)
//!
//! RK4 and ABM also write perturber positions when the flyby / heliocentric
//! toggles are active (`output_h/` and `output_sun/`).
//!
//! | Function | Method | Step control | Notes |
//! |---|---|---|---|
//! | [`rk4_stack`] | Classical RK4 | fixed | 4 ODE evaluations/step |
//! | [`abm`] | Adams-Bashforth-Moulton 4th order | fixed | 3-step RK4 bootstrap |
//! | [`rk87`] | Dormand-Prince 7(8) | adaptive | inf-norm error control |
//! | [`lgvi_integ`] | Lie Group Variational Integrator | fixed | SO(3)-preserving |

use std::fs::{self, File};
use std::io::{BufWriter, Write};

use crate::dynamics::hou_ode;
use crate::lgvi::{hamiltonian_map, map_potential_partials_lgvi};
use crate::math3::{inf_norm_30, state_add_scaled, state_combine};
use crate::orbit::{kepler, kepler2cart};
use crate::types::Params;

// ── binary I/O helpers ────────────────────────────────────────────────────────

type Bw = BufWriter<File>;

#[inline]
fn wt(f: &mut Bw, t: f64) {
    f.write_all(&t.to_ne_bytes()).expect("write t");
}

#[inline]
fn wx(f: &mut Bw, x: &[f64; 30]) {
    for &v in x { f.write_all(&v.to_ne_bytes()).expect("write x"); }
}

#[inline]
fn w6(f: &mut Bw, xs: &[f64; 6]) {
    for &v in xs { f.write_all(&v.to_ne_bytes()).expect("write 6"); }
}

fn open_bw(path: &str) -> Bw {
    BufWriter::new(File::create(path).expect(path))
}

fn recreate(dirs: &[&str]) {
    for &d in dirs {
        let _ = fs::remove_dir_all(d);
        fs::create_dir_all(d).expect(d);
    }
}

// ── perturber helpers ─────────────────────────────────────────────────────────

fn flyby_pos(p: &Params, t: f64) -> [f64; 6] {
    let f0 = kepler(p.n_hyp, t, p.e_hyp, p.tau_hyp);
    kepler2cart(p.a_hyp, p.e_hyp, p.i_hyp, p.raan_hyp, p.om_hyp, f0, p.g, p.mplanet)
}

fn solar_pos(p: &Params, t: f64) -> [f64; 6] {
    let f0 = kepler(p.n_helio, t, p.e_helio, p.tau_helio);
    kepler2cart(p.a_helio, p.e_helio, p.i_helio, p.raan_helio, p.om_helio, f0, p.g, p.msolar)
}

// ── RK4 (fixed step) ──────────────────────────────────────────────────────────

/// Fixed-step 4th-order Runge-Kutta integrator.
/// Mirrors `void rk4_stack(...)`.
pub fn rk4_stack(t0: f64, tf: f64, x0: [f64; 30], h: f64, params: &Params) {
    recreate(&["output_t", "output_x", "output_h", "output_sun"]);
    let mut tf_  = open_bw("output_t/t_out.bin");
    let mut xf   = open_bw("output_x/x_out.bin");
    let mut hf   = open_bw("output_h/h_out.bin");
    let mut sunf = open_bw("output_sun/sun_out.bin");

    let mut t = t0;
    let mut x = x0;

    // Initial record
    wt(&mut tf_, t);  wx(&mut xf, &x);
    if params.flyby_toggle == 1 { w6(&mut hf, &flyby_pos(params, t)); }
    if params.helio_toggle == 1 { w6(&mut sunf, &solar_pos(params, t)); }

    while t < tf {
        let t_cur = t;
        let k1 = hou_ode(x, t_cur,           params);
        let k2 = hou_ode(state_add_scaled(x, h/2.0, k1), t_cur + h/2.0, params);
        let k3 = hou_ode(state_add_scaled(x, h/2.0, k2), t_cur + h/2.0, params);
        let k4 = hou_ode(state_add_scaled(x, h,     k3), t_cur + h,      params);

        x = state_combine(x, &[(&k1, h/6.0), (&k2, h/3.0), (&k3, h/3.0), (&k4, h/6.0)]);
        t += h;

        wt(&mut tf_, t);  wx(&mut xf, &x);
        if params.flyby_toggle  == 1 { w6(&mut hf,   &flyby_pos(params, t_cur)); }
        if params.helio_toggle  == 1 { w6(&mut sunf,  &solar_pos(params, t_cur)); }
    }
}

// ── Adams-Bashforth-Moulton (fixed step, 4th order) ──────────────────────────

/// 4th-order Adams-Bashforth-Moulton predictor-corrector.
/// First 3 steps are bootstrapped with RK4.  Mirrors `void ABM(...)`.
pub fn abm(t0: f64, tf: f64, x0: [f64; 30], h: f64, params: &Params) {
    recreate(&["output_t", "output_x", "output_h", "output_sun"]);
    let mut tf_  = open_bw("output_t/t_out.bin");
    let mut xf   = open_bw("output_x/x_out.bin");
    let mut hf   = open_bw("output_h/h_out.bin");
    let mut sunf = open_bw("output_sun/sun_out.bin");

    let mut t = t0;
    let mut x = x0;

    wt(&mut tf_, t);  wx(&mut xf, &x);
    if params.flyby_toggle == 1 { w6(&mut hf, &flyby_pos(params, t)); }
    if params.helio_toggle == 1 { w6(&mut sunf, &solar_pos(params, t)); }

    // History ring: (t, x, f) for the three most-recent past steps
    // history[0] = most-recent, history[1] = one older, history[2] = oldest
    let mut hist_f: [[f64; 30]; 3] = [[0.0; 30]; 3];
    let mut hist_t: [f64; 3]       = [0.0; 3];
    let mut steps: usize = 0; // how many steps have been taken

    while t < tf {
        if steps < 3 {
            // Bootstrap with RK4
            let t_cur = t;
            let k1 = hou_ode(x, t_cur,           params);
            let k2 = hou_ode(state_add_scaled(x, h/2.0, k1), t_cur + h/2.0, params);
            let k3 = hou_ode(state_add_scaled(x, h/2.0, k2), t_cur + h/2.0, params);
            let k4 = hou_ode(state_add_scaled(x, h,     k3), t_cur + h,      params);

            // Cache f0 for ABM history (f at current t)
            let f_cur = k1; // k1 = ode(x, t)
            // Shift history
            hist_f[2] = hist_f[1];  hist_t[2] = hist_t[1];
            hist_f[1] = hist_f[0];  hist_t[1] = hist_t[0];
            hist_f[0] = f_cur;      hist_t[0] = t_cur;

            x = state_combine(x, &[(&k1, h/6.0), (&k2, h/3.0), (&k3, h/3.0), (&k4, h/6.0)]);
            t += h;
            steps += 1;

            wt(&mut tf_, t);  wx(&mut xf, &x);
            if params.flyby_toggle  == 1 { w6(&mut hf,   &flyby_pos(params, t_cur)); }
            if params.helio_toggle  == 1 { w6(&mut sunf,  &solar_pos(params, t_cur)); }
        } else {
            // Adams-Bashforth-Moulton predictor-corrector
            let t_cur = t;
            let f0 = hou_ode(x, t_cur, params);

            // Predictor (4-step Adams-Bashforth)
            let y_pred = state_combine(x, &[
                (&f0,        h * 55.0 / 24.0),
                (&hist_f[0], h * -59.0 / 24.0),
                (&hist_f[1], h *  37.0 / 24.0),
                (&hist_f[2], h *  -9.0 / 24.0),
            ]);

            // Corrector (4-step Adams-Moulton)
            let fp = hou_ode(y_pred, t_cur + h, params);
            x = state_combine(x, &[
                (&fp,        h *  9.0 / 24.0),
                (&f0,        h * 19.0 / 24.0),
                (&hist_f[0], h * -5.0 / 24.0),
                (&hist_f[1], h *  1.0 / 24.0),
            ]);
            t += h;
            steps += 1;

            // Shift history (f0 becomes most-recent)
            hist_f[2] = hist_f[1];  hist_t[2] = hist_t[1];
            hist_f[1] = hist_f[0];  hist_t[1] = hist_t[0];
            hist_f[0] = f0;         hist_t[0] = t_cur;

            wt(&mut tf_, t);  wx(&mut xf, &x);
            if params.flyby_toggle  == 1 { w6(&mut hf,   &flyby_pos(params, t_cur)); }
            if params.helio_toggle  == 1 { w6(&mut sunf,  &solar_pos(params, t_cur)); }
        }
    }
}

// ── RK 7(8) Dormand-Prince (adaptive step) ───────────────────────────────────

// Butcher tableau: `a_i_j.col(j)` in the C++ translates to STAGE_A[j][i] here.
// STAGE_A[j][i] = coefficient for stage i when computing stage j+1 (0-indexed j = 0..11).
#[rustfmt::skip]
const STAGE_A: [[f64; 13]; 12] = [
    // j=0 → stage 1
    [1./18., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    // j=1 → stage 2
    [1./48., 1./16., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    // j=2 → stage 3
    [1./32., 0., 3./32., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    // j=3 → stage 4
    [5./16., 0., -75./64., 75./64., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    // j=4 → stage 5
    [3./80., 0., 0., 3./16., 3./20., 0., 0., 0., 0., 0., 0., 0., 0.],
    // j=5 → stage 6
    [29443841./614563906., 0., 0.,
     77736538./692538347., -28693883./1125000000., 23124283./1800000000.,
     0., 0., 0., 0., 0., 0., 0.],
    // j=6 → stage 7
    [16016141./946692911., 0., 0.,
     61564180./158732637., 22789713./633445777., 545815736./2771057229.,
     -180193667./1043307555., 0., 0., 0., 0., 0., 0.],
    // j=7 → stage 8
    [39632708./573591083., 0., 0.,
     -433636366./683701615., -421739975./2616292301., 100302831./723423059.,
     790204164./839813087., 800635310./3783071287., 0., 0., 0., 0., 0.],
    // j=8 → stage 9
    [246121993./1340847787., 0., 0.,
     -37695042795./15268766246., -309121744./1061227803., -12992083./490766935.,
     6005943493./2108947869., 393006217./1396673457., 123872331./1001029789.,
     0., 0., 0., 0.],
    // j=9 → stage 10
    [-1028468189./846180014., 0., 0.,
     8478235783./508512852., 1311729495./1432422823., -10304129995./1701304382.,
     -48777925059./3047939560., 15336726248./1032824649., -45442868181./3398467696.,
     3065993473./597172653., 0., 0., 0.],
    // j=10 → stage 11
    [185892177./718116043., 0., 0.,
     -3185094517./667107341., -477755414./1098053517., -703635378./230739211.,
     5731566787./1027545527., 5232866602./850066563., -4093664535./808688257.,
     3962137247./1805957418., 65686358./487910083., 0., 0.],
    // j=11 → stage 12
    [403863854./491063109., 0., 0.,
     -5068492393./434740067., -411421997./543043805., 652783627./914296604.,
     11173962825./925320556., -13158990841./6184727034., 3936647629./1978049680.,
     -160528059./685178525., 248638103./1413531060., 0., 0.],
];

// c_i nodes (12 entries; loop over j=0..11)
const C_NODES: [f64; 12] = [
    1./18., 1./12., 1./8., 5./16., 3./8., 59./400.,
    93./200., 5490023248./9719169821., 13./20.,
    1201146811./1299019798., 1., 1.,
];

// 8th-order weights (13 entries)
const B8: [f64; 13] = [
    14005451./335480064., 0., 0., 0., 0.,
    -59238493./1068277825., 181606767./758867731., 561292985./797845732.,
    -1041891430./1371343529., 760417239./1151165299., 118820643./751138087.,
    -528747749./2220607170., 1./4.,
];

// 7th-order weights (13 entries — used as the accepted step)
const B7: [f64; 13] = [
    13451932./455176623., 0., 0., 0., 0.,
    -808719846./976000145., 1757004468./5645159321., 656045339./265891186.,
    -3867574721./1518517206., 465885868./322736535., 53011238./667516719.,
    2./45., 0.,
];

/// Dormand-Prince RK 7(8) adaptive integrator.
/// Mirrors `void rk87(...)`.
pub fn rk87(t0: f64, tf: f64, x0: [f64; 30], rel_tol: f64, params: &Params) {
    recreate(&["output_t", "output_x"]);
    let mut tf_ = open_bw("output_t/t_out.bin");
    let mut xf  = open_bw("output_x/x_out.bin");

    let eps   = 5e-16_f64;
    let pow_v = 1.0_f64 / 8.0;
    let hmax  = (tf - t0) / 2.5;
    let mut h = ((tf - t0) / 50.0).min(0.1).min(hmax);

    let mut t = t0;
    let mut x = x0;

    wt(&mut tf_, t);  wx(&mut xf, &x);

    while t < tf {
        if t + h > tf { h = tf - t; }

        // Evaluate all 13 stages
        let mut stages = [[0.0_f64; 30]; 13];
        stages[0] = hou_ode(x, t, params);

        for j in 1..=12_usize {
            let mut x_stage = x;
            for i in 0..j {
                let a = STAGE_A[j-1][i];
                if a != 0.0 {
                    for k in 0..30 { x_stage[k] += h * a * stages[i][k]; }
                }
            }
            stages[j] = hou_ode(x_stage, t + C_NODES[j-1] * h, params);
        }

        // 8th and 7th order solutions
        let mut sol8 = x;
        let mut sol7 = x;
        for i in 0..13 {
            for k in 0..30 {
                sol8[k] += h * B8[i] * stages[i][k];
                sol7[k] += h * B7[i] * stages[i][k];
            }
        }

        let err = {
            let diff: [f64; 30] = {
                let mut d = [0.0_f64; 30];
                for k in 0..30 { d[k] = sol8[k] - sol7[k]; }
                d
            };
            inf_norm_30(&diff).abs()
        };
        let tau = rel_tol * inf_norm_30(&x);

        if err <= tau {
            x = sol7;   // accept 7th-order solution
            t += h;
            wt(&mut tf_, t);  wx(&mut xf, &x);
        }

        let err_safe = if err == 0.0 { 10.0 * eps } else { err };
        let tau_safe = tau.max(rel_tol * eps); // avoid div-by-zero
        h = hmax.min(0.9 * h * (tau_safe / err_safe).powf(pow_v));

        if h.abs() <= eps {
            eprintln!("rk87: step size at machine precision");
        }
    }
}

// ── LGVI (Lie Group Variational Integrator, fixed step) ──────────────────────

/// Symplectic LGVI fixed-step integrator.
/// Mirrors `void LGVI_integ(...)`.
pub fn lgvi_integ(t0: f64, tf: f64, x0: [f64; 30], h: f64, params: &Params) {
    recreate(&["output_t", "output_x"]);
    let mut tf_ = open_bw("output_t/t_out.bin");
    let mut xf  = open_bw("output_x/x_out.bin");

    let mut t = t0;
    let mut x = x0;

    // Initial potential partials
    let c0: crate::math3::Mat3 = [[x[21], x[22], x[23]],
                                   [x[24], x[25], x[26]],
                                   [x[27], x[28], x[29]]];
    let r0: crate::math3::Vec3 = [x[0], x[1], x[2]];
    let (mut du_dr, mut m_grav) = map_potential_partials_lgvi(c0, r0, params);

    wt(&mut tf_, t);  wx(&mut xf, &x);

    while t < tf {
        let (x_new, du_dr_new, m_new) = hamiltonian_map(h, params, x, x0, du_dr, m_grav);
        x       = x_new;
        du_dr   = du_dr_new;
        m_grav  = m_new;
        t      += h;

        wt(&mut tf_, t);  wx(&mut xf, &x);
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coefficients::{a_calc, b_calc, tk_calc};
    use crate::types::{Cube, Params};
    use std::fs;
    use std::io::Read;
    use std::sync::Mutex;

    // All four integrators write to the same output_t / output_x directories.
    // This mutex ensures the tests do not clobber each other when run in parallel.
    static INTEG_LOCK: Mutex<()> = Mutex::new(());

    // ── shared test fixtures ──────────────────────────────────────────────────

    fn monopole_params(ma: f64, mb: f64) -> Params {
        let g = 6.674e-20_f64;
        let n = 0usize;
        let mut ta = Cube::new(n);  ta.set(0, 0, 0, ma);
        let mut tb = Cube::new(n);  tb.set(0, 0, 0, mb);
        let mut p = Params {
            g, m: ma*mb/(ma+mb), nu: mb/(ma+mb),
            ta, tb,
            ia: [1.0, 1.0, 1.0], ib: [1.0, 1.0, 1.0],
            n, tk: tk_calc(n), a: a_calc(n), b: b_calc(n),
            flyby_toggle: 0, helio_toggle: 0, sg_toggle: 0, tt_toggle: 0,
            mplanet: 0.0, a_hyp: -1.0, e_hyp: 1.5,
            i_hyp: 0.0, raan_hyp: 0.0, om_hyp: 0.0, tau_hyp: 0.0, n_hyp: 0.0,
            msolar: 0.0, a_helio: 1.0, e_helio: 0.0,
            i_helio: 0.0, raan_helio: 0.0, om_helio: 0.0, tau_helio: 0.0, n_helio: 0.0,
            sol_rad: 0.0, au_def: 1.496e8, mean_motion: 0.0,
            love1: 0.0, love2: 0.0, refrad1: 1.0, refrad2: 1.0,
            rho_a: 1e12, rho_b: 1e12, eps1: 0.0, eps2: 0.0,
            ida: [[0.0; 3]; 3], idb: [[0.0; 3]; 3],
            msun: 2e30,
        };
        p.compute_lgvi_inertia();
        p
    }

    fn circular_state(a: f64, v: f64) -> [f64; 30] {
        let e = [1.0_f64, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0]; // row-major I₃
        [a, 0.0, 0.0,  0.0, v, 0.0,
         0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
         e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8],
         e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]]
    }

    /// Read every f64 from a binary file (native-endian).
    fn read_f64s(path: &str) -> Vec<f64> {
        let mut bytes = Vec::new();
        fs::File::open(path).unwrap().read_to_end(&mut bytes).unwrap();
        bytes.chunks_exact(8)
             .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
             .collect()
    }

    fn cleanup() {
        for dir in &["output_t", "output_x", "output_h", "output_sun"] {
            let _ = fs::remove_dir_all(dir);
        }
    }

    // ── RK4 ──────────────────────────────────────────────────────────────────

    #[test]
    fn rk4_one_step_file_size_and_non_nan() {
        let _g = INTEG_LOCK.lock().unwrap();
        let ma = 5e11_f64;
        let mb = 2e11_f64;
        let g  = 6.674e-20_f64;
        let a  = 10.0_f64;
        let v  = (g * (ma + mb) / a).sqrt();
        let h  = 1.0_f64;

        rk4_stack(0.0, h, circular_state(a, v), h, &monopole_params(ma, mb));

        let times  = read_f64s("output_t/t_out.bin");
        let states = read_f64s("output_x/x_out.bin");

        // Initial record + 1 step → 2 records
        assert_eq!(times.len(), 2, "expected 2 time records");
        assert_eq!(states.len(), 2 * 30, "expected 60 state values");
        assert!((times[0]).abs() < 1e-14);
        assert!((times[1] - h).abs() < 1e-14);
        for &val in &states { assert!(!val.is_nan(), "NaN in state"); }
        // Orbit moves in +y: r_y of second record should be positive
        assert!(states[30 + 1] > 0.0, "r_y should be positive after 1 step");

        cleanup();
    }

    // ── ABM ──────────────────────────────────────────────────────────────────

    #[test]
    fn abm_four_steps_file_size_and_non_nan() {
        let _g = INTEG_LOCK.lock().unwrap();
        let ma = 5e11_f64;
        let mb = 2e11_f64;
        let g  = 6.674e-20_f64;
        let a  = 10.0_f64;
        let v  = (g * (ma + mb) / a).sqrt();
        let h  = 1.0_f64;

        // 3-step bootstrap + 1 ABM step = 4 steps, 5 records
        abm(0.0, 4.0 * h, circular_state(a, v), h, &monopole_params(ma, mb));

        let times  = read_f64s("output_t/t_out.bin");
        let states = read_f64s("output_x/x_out.bin");

        assert_eq!(times.len(), 5, "expected 5 time records");
        assert_eq!(states.len(), 5 * 30);
        for &val in &states { assert!(!val.is_nan(), "NaN in state"); }
        // r_y of the final record should be positive
        assert!(states[4 * 30 + 1] > 0.0, "r_y should be positive after 4 steps");

        cleanup();
    }

    // ── RK7(8) ───────────────────────────────────────────────────────────────

    #[test]
    fn rk87_adaptive_non_nan_consistent_sizes() {
        let _g = INTEG_LOCK.lock().unwrap();
        let ma = 5e11_f64;
        let mb = 2e11_f64;
        let g  = 6.674e-20_f64;
        let a  = 10.0_f64;
        let v  = (g * (ma + mb) / a).sqrt();

        rk87(0.0, 10.0, circular_state(a, v), 1e-6, &monopole_params(ma, mb));

        let times  = read_f64s("output_t/t_out.bin");
        let states = read_f64s("output_x/x_out.bin");

        assert!(times.len() >= 2, "expected at least initial + 1 accepted step");
        assert_eq!(states.len(), times.len() * 30, "time and state record counts must match");
        for &val in &states { assert!(!val.is_nan(), "NaN in state"); }

        cleanup();
    }

    // ── LGVI ─────────────────────────────────────────────────────────────────

    #[test]
    fn lgvi_one_step_file_size_and_non_nan() {
        let _g = INTEG_LOCK.lock().unwrap();
        let ma = 5e11_f64;
        let mb = 2e11_f64;
        let g  = 6.674e-20_f64;
        let a  = 10.0_f64;
        let v  = (g * (ma + mb) / a).sqrt();
        let h  = 1.0_f64;

        lgvi_integ(0.0, h, circular_state(a, v), h, &monopole_params(ma, mb));

        let times  = read_f64s("output_t/t_out.bin");
        let states = read_f64s("output_x/x_out.bin");

        assert_eq!(times.len(), 2, "expected 2 time records");
        assert_eq!(states.len(), 2 * 30);
        for &val in &states { assert!(!val.is_nan(), "NaN in state"); }

        cleanup();
    }
}
