// integrators.rs — Fixed-step RK4, ABM, adaptive RK7(8), and LGVI integrators
//
// Each integrator writes raw f64 binary to:
//   output_t/t_out.bin  — times (one f64 per step)
//   output_x/x_out.bin  — states (30 f64s per step, row-major)
//
// RK4 and ABM also write perturber positions when the flyby/heliocentric
// toggles are active:
//   output_h/h_out.bin    — flyby body  [rx,ry,rz,vx,vy,vz] per step
//   output_sun/sun_out.bin — solar body [rx,ry,rz,vx,vy,vz] per step
//
// File format is identical to the C++ Armadillo binary output:
// native-endian doubles, states stored row-by-row with the initial
// state always included as the first record.

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
