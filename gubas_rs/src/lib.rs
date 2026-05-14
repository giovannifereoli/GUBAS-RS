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

use coefficients::{a_calc, b_calc, tk_calc};
use inertia::{ell_mass_params_met, load_moi_csv, poly_inertia_met, poly_moi_met, save_moi_csv};
use integrators::{abm, lgvi_integ, rk4_stack, rk87};
use types::{Cube, Initialization, Params};

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

// ── ic_read ───────────────────────────────────────────────────────────────────

pub fn ic_read() -> Initialization {
    let content = std::fs::read_to_string("ic_input.txt")
        .expect("ic_input.txt: cannot open input file");
    let mut lines = content.lines();

    macro_rules! next_line {
        () => {
            lines.next().expect("ic_input.txt: unexpected end of file").trim()
        };
    }
    macro_rules! read_f64 { () => { next_line!().parse::<f64>().expect("f64 parse") }; }
    macro_rules! read_i32 { () => { next_line!().parse::<i32>().expect("i32 parse") }; }
    macro_rules! read_usize { () => { next_line!().parse::<usize>().expect("usize parse") }; }
    macro_rules! read_str  { () => { next_line!().to_string() }; }

    let g        = read_f64!();
    let order    = read_usize!();
    let order_a  = read_usize!();
    let order_b  = read_usize!();
    let a_a      = read_f64!();
    let b_a      = read_f64!();
    let c_a      = read_f64!();
    let a_b      = read_f64!();
    let b_b      = read_f64!();
    let c_b      = read_f64!();
    let a_shape  = read_i32!();
    let b_shape  = read_i32!();
    let rho_a    = read_f64!();
    let rho_b    = read_f64!();
    let t0       = read_f64!();
    let tf       = read_f64!();
    let ta_file  = read_str!();
    let tb_file  = read_str!();
    let ia_file  = read_str!();
    let ib_file  = read_str!();
    let tet_file_a  = read_str!();
    let vert_file_a = read_str!();
    let tet_file_b  = read_str!();
    let vert_file_b = read_str!();
    let mut x0 = [0.0_f64; 30];
    for v in x0.iter_mut() { *v = read_f64!(); }
    let tgen          = read_i32!();
    let integ         = read_i32!();
    let h             = read_f64!();
    let tol           = read_f64!();
    let flyby_toggle  = read_i32!();
    let helio_toggle  = read_i32!();
    let sg_toggle     = read_i32!();
    let tt_toggle     = read_i32!();
    let mplanet       = read_f64!();
    let a_hyp         = read_f64!();
    let e_hyp         = read_f64!();
    let i_hyp         = read_f64!();
    let raan_hyp      = read_f64!();
    let om_hyp        = read_f64!();
    let tau_hyp       = read_f64!();
    let msolar        = read_f64!();
    let a_helio       = read_f64!();
    let e_helio       = read_f64!();
    let i_helio       = read_f64!();
    let raan_helio    = read_f64!();
    let om_helio      = read_f64!();
    let tau_helio     = read_f64!();
    let sol_rad       = read_f64!();
    let au_def        = read_f64!();
    let love1         = read_f64!();
    let love2         = read_f64!();
    let refrad1       = read_f64!();
    let refrad2       = read_f64!();
    let eps1          = read_f64!();
    let eps2          = read_f64!();
    let msun          = read_f64!();

    Initialization {
        g, order, order_a, order_b,
        a_a, b_a, c_a, a_b, b_b, c_b,
        a_shape, b_shape, rho_a, rho_b,
        t0, tf,
        ta_file, tb_file, ia_file, ib_file,
        tet_file_a, vert_file_a, tet_file_b, vert_file_b,
        x0,
        tgen, integ, h, tol,
        flyby_toggle, helio_toggle, sg_toggle, tt_toggle,
        mplanet, a_hyp, e_hyp, i_hyp, raan_hyp, om_hyp, tau_hyp,
        msolar, a_helio, e_helio, i_helio, raan_helio, om_helio, tau_helio,
        sol_rad, au_def,
        love1, love2, refrad1, refrad2, eps1, eps2,
        msun,
    }
}

// ── run_simulation ────────────────────────────────────────────────────────────

/// Run a full GUBAS simulation. Reads `ic_input.txt` from the current working
/// directory and writes binary output to `output_t/t_out.bin` and
/// `output_x/x_out.bin` in the same directory.
pub fn run_simulation() {
    let ics = ic_read();

    let mut order   = ics.order;
    let mut order_a = ics.order_a;
    let mut order_b = ics.order_b;
    if ics.a_shape == 2 { order_a = order_a.max(order); }
    if ics.b_shape == 2 { order_b = order_b.max(order); }
    order = order.max(order_a).max(order_b);

    let (ia, ta, ib, tb);

    if ics.tgen == 1 {
        println!("Generating Inertia Integrals");

        let (ia_v, ta_v) = match ics.a_shape {
            0 => ell_mass_params_met(order, 0, ics.rho_a, ics.a_a, ics.b_a, ics.c_a),
            1 => ell_mass_params_met(order, order_a, ics.rho_a, ics.a_a, ics.b_a, ics.c_a),
            2 => {
                let ta_v = poly_inertia_met(order_a, ics.rho_a,
                                            &ics.tet_file_a, &ics.vert_file_a);
                let ia_v = poly_moi_met(ics.rho_a, &ics.tet_file_a, &ics.vert_file_a);
                (ia_v, ta_v)
            }
            _ => panic!("Bad Primary Shape Input: a_shape = {}", ics.a_shape),
        };

        let (ib_v, tb_v) = match ics.b_shape {
            0 => ell_mass_params_met(order, 0, ics.rho_b, ics.a_b, ics.b_b, ics.c_b),
            1 => ell_mass_params_met(order, order_b, ics.rho_b, ics.a_b, ics.b_b, ics.c_b),
            2 => {
                let tb_v = poly_inertia_met(order_b, ics.rho_b,
                                            &ics.tet_file_b, &ics.vert_file_b);
                let ib_v = poly_moi_met(ics.rho_b, &ics.tet_file_b, &ics.vert_file_b);
                (ib_v, tb_v)
            }
            _ => panic!("Bad Secondary Shape Input: b_shape = {}", ics.b_shape),
        };

        ta_v.save_csv(&format!("TDP_{}.csv", order)).expect("save TDP csv");
        tb_v.save_csv(&format!("TDS_{}.csv", order)).expect("save TDS csv");
        save_moi_csv(ia_v, "IDP.csv").expect("save IDP csv");
        save_moi_csv(ib_v, "IDS.csv").expect("save IDS csv");

        ia = ia_v;  ta = ta_v;
        ib = ib_v;  tb = tb_v;
    } else {
        ta = Cube::load_csv(&format!("TDP_{}.csv", order))
            .expect("TDP csv not found — run with Tgen=1 first");
        tb = Cube::load_csv(&format!("TDS_{}.csv", order))
            .expect("TDS csv not found — run with Tgen=1 first");
        ia = load_moi_csv("IDP.csv").expect("IDP csv not found");
        ib = load_moi_csv("IDS.csv").expect("IDS csv not found");
    }

    let tk = tk_calc(order);
    let a  = a_calc(order);
    let b  = b_calc(order);

    let mc  = ta.get(0, 0, 0);
    let ms  = tb.get(0, 0, 0);
    let m   = mc * ms / (mc + ms);
    let nu  = ms / (mc + ms);
    let mean_motion = (ics.g * (ics.msun + mc + ms)
                       / (ics.sol_rad * ics.au_def / 1000.0).powi(3))
                      .sqrt();
    let n_hyp   = (ics.g * ics.mplanet / ics.a_hyp.abs().powi(3)).sqrt();
    let n_helio = (ics.g * ics.msolar  / ics.a_helio.abs().powi(3)).sqrt();

    let mut params = Params {
        g: ics.g,
        m,
        nu,
        ta,
        tb,
        ia,
        ib,
        n: order,
        tk,
        a,
        b,
        flyby_toggle: ics.flyby_toggle,
        helio_toggle: ics.helio_toggle,
        sg_toggle:    ics.sg_toggle,
        tt_toggle:    ics.tt_toggle,
        mplanet:  ics.mplanet,
        a_hyp:    ics.a_hyp,
        e_hyp:    ics.e_hyp,
        i_hyp:    ics.i_hyp,
        raan_hyp: ics.raan_hyp,
        om_hyp:   ics.om_hyp,
        tau_hyp:  ics.tau_hyp,
        n_hyp,
        msolar:    ics.msolar,
        a_helio:   ics.a_helio,
        e_helio:   ics.e_helio,
        i_helio:   ics.i_helio,
        raan_helio: ics.raan_helio,
        om_helio:  ics.om_helio,
        tau_helio: ics.tau_helio,
        n_helio,
        sol_rad:  ics.sol_rad,
        au_def:   ics.au_def,
        mean_motion,
        love1:    ics.love1,
        love2:    ics.love2,
        refrad1:  ics.refrad1,
        refrad2:  ics.refrad2,
        rho_a:    ics.rho_a,
        rho_b:    ics.rho_b,
        eps1:     ics.eps1,
        eps2:     ics.eps2,
        ida: math3::ZERO_M,
        idb: math3::ZERO_M,
        msun: ics.msun,
    };
    params.compute_lgvi_inertia();

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

    let mut order   = ics.order;
    let mut order_a = ics.order_a;
    let mut order_b = ics.order_b;
    if ics.a_shape == 2 { order_a = order_a.max(order); }
    if ics.b_shape == 2 { order_b = order_b.max(order); }
    order = order.max(order_a).max(order_b);

    let ta = Cube::load_csv(&format!("TDP_{}.csv", order))
        .expect("TDP csv not found — run with Tgen=1 first");
    let tb = Cube::load_csv(&format!("TDS_{}.csv", order))
        .expect("TDS csv not found — run with Tgen=1 first");
    let ia = load_moi_csv("IDP.csv").expect("IDP csv not found");
    let ib = load_moi_csv("IDS.csv").expect("IDS csv not found");

    let tk = tk_calc(order);
    let a  = a_calc(order);
    let b  = b_calc(order);

    let mc  = ta.get(0, 0, 0);
    let ms  = tb.get(0, 0, 0);
    let m   = mc * ms / (mc + ms);
    let nu  = ms / (mc + ms);
    let mean_motion = (ics.g * (ics.msun + mc + ms)
                       / (ics.sol_rad * ics.au_def / 1000.0).powi(3)).sqrt();
    let n_hyp   = (ics.g * ics.mplanet / ics.a_hyp.abs().powi(3)).sqrt();
    let n_helio = (ics.g * ics.msolar  / ics.a_helio.abs().powi(3)).sqrt();

    let mut params = Params {
        g: ics.g, m, nu, ta, tb, ia, ib,
        n: order, tk, a, b,
        flyby_toggle: ics.flyby_toggle,
        helio_toggle: ics.helio_toggle,
        sg_toggle:    ics.sg_toggle,
        tt_toggle:    ics.tt_toggle,
        mplanet:   ics.mplanet,
        a_hyp:     ics.a_hyp, e_hyp: ics.e_hyp, i_hyp: ics.i_hyp,
        raan_hyp:  ics.raan_hyp, om_hyp: ics.om_hyp, tau_hyp: ics.tau_hyp, n_hyp,
        msolar:    ics.msolar,
        a_helio:   ics.a_helio, e_helio: ics.e_helio, i_helio: ics.i_helio,
        raan_helio: ics.raan_helio, om_helio: ics.om_helio, tau_helio: ics.tau_helio, n_helio,
        sol_rad:     ics.sol_rad, au_def: ics.au_def, mean_motion,
        love1:     ics.love1, love2: ics.love2,
        refrad1:   ics.refrad1, refrad2: ics.refrad2,
        rho_a:     ics.rho_a, rho_b: ics.rho_b,
        eps1:      ics.eps1, eps2: ics.eps2,
        ida: math3::ZERO_M, idb: math3::ZERO_M,
        msun: ics.msun,
    };
    params.compute_lgvi_inertia();

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

    let ics = ic_read();

    let mut order   = ics.order;
    let mut order_a = ics.order_a;
    let mut order_b = ics.order_b;
    if ics.a_shape == 2 { order_a = order_a.max(order); }
    if ics.b_shape == 2 { order_b = order_b.max(order); }
    order = order.max(order_a).max(order_b);

    let ta = Cube::load_csv(&format!("TDP_{}.csv", order))
        .expect("TDP csv not found — run with Tgen=1 first");
    let tb = Cube::load_csv(&format!("TDS_{}.csv", order))
        .expect("TDS csv not found — run with Tgen=1 first");
    let ia = load_moi_csv("IDP.csv").expect("IDP csv not found");
    let ib = load_moi_csv("IDS.csv").expect("IDS csv not found");

    let tk = tk_calc(order);
    let a  = a_calc(order);
    let b  = b_calc(order);

    let mc = ta.get(0, 0, 0);
    let ms = tb.get(0, 0, 0);
    let m  = mc * ms / (mc + ms);
    let nu = ms / (mc + ms);
    let mean_motion = (ics.g * (ics.msun + mc + ms)
                       / (ics.sol_rad * ics.au_def / 1000.0).powi(3)).sqrt();
    let n_hyp   = (ics.g * ics.mplanet / ics.a_hyp.abs().powi(3)).sqrt();
    let n_helio = (ics.g * ics.msolar  / ics.a_helio.abs().powi(3)).sqrt();

    let mut params = Params {
        g: ics.g, m, nu, ta, tb, ia, ib,
        n: order, tk, a, b,
        flyby_toggle: ics.flyby_toggle,
        helio_toggle: ics.helio_toggle,
        sg_toggle:    ics.sg_toggle,
        tt_toggle:    ics.tt_toggle,
        mplanet:   ics.mplanet,
        a_hyp:     ics.a_hyp, e_hyp: ics.e_hyp, i_hyp: ics.i_hyp,
        raan_hyp:  ics.raan_hyp, om_hyp: ics.om_hyp, tau_hyp: ics.tau_hyp, n_hyp,
        msolar:    ics.msolar,
        a_helio:   ics.a_helio, e_helio: ics.e_helio, i_helio: ics.i_helio,
        raan_helio: ics.raan_helio, om_helio: ics.om_helio, tau_helio: ics.tau_helio, n_helio,
        sol_rad: ics.sol_rad, au_def: ics.au_def, mean_motion,
        love1: ics.love1, love2: ics.love2,
        refrad1: ics.refrad1, refrad2: ics.refrad2,
        rho_a: ics.rho_a, rho_b: ics.rho_b,
        eps1: ics.eps1, eps2: ics.eps2,
        ida: math3::ZERO_M, idb: math3::ZERO_M,
        msun: ics.msun,
    };
    params.compute_lgvi_inertia();

    // ── Select θ indices: all T_{ijk} at the requested degrees ───────────────
    let theta_indices = inertia_indices(min_degree, max_degree);
    let n_theta = theta_indices.len();

    let theta0: Vec<f64> = theta_indices.iter().map(|&(i, j, k)| {
        if which_body == 0 { params.ta.get(i, j, k) } else { params.tb.get(i, j, k) }
    }).collect();

    println!("Augmented STM propagation:");
    println!("  Body: {}, degrees [{}, {}], N_theta = {}",
             if which_body == 0 { "primary" } else { "secondary" },
             min_degree, max_degree, n_theta);
    println!("  [Each stage: {} dual ODE evals for augmented Jacobian]", 30 + n_theta);

    // ── Promote params to Dual once, stored for cloning in the closure ────────
    let params_dual_base = promote_params::<Dual>(&params);

    // ── Build augmented dual ODE closure ─────────────────────────────────────
    let theta_indices_clone = theta_indices.clone();
    let aug_ode_dual = move |z: &[Dual], t: f64| -> Vec<Dual> {
        // Clone base Dual params and patch the θ entries with values from z[30..]
        let mut pd = params_dual_base.clone();
        for (k, &(ii, jj, kk)) in theta_indices_clone.iter().enumerate() {
            if which_body == 0 {
                pd.ta.set(ii, jj, kk, z[30 + k]);
            } else {
                pd.tb.set(ii, jj, kk, z[30 + k]);
            }
        }
        let mut x = [Dual::from_re(0.0); 30];
        for i in 0..30 { x[i] = z[i]; }
        let xdot = hou_ode(x, t, &pd);
        let mut zdot = vec![Dual::from_re(0.0); 30 + theta_indices_clone.len()];
        for i in 0..30 { zdot[i] = xdot[i]; }
        zdot
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

    // Stokes at t₀ — use max semi-axis of body as reference radius
    let (body_ta, body_mass) = if which_body == 0 {
        (&params.ta, params.ta.get(0,0,0))
    } else {
        (&params.tb, params.tb.get(0,0,0))
    };
    // Reference radius: rough mean radius from MOI (no explicit r0 in params; use 1 km)
    let r0 = 1.0_f64; // km — user should rescale in post-processing
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

// ── Python module (only compiled with --features extension-module) ────────────

/// Run a GUBAS simulation from Python.
///
/// Reads `ic_input.txt` from `work_dir` (defaults to the current working
/// directory) and writes output binary files to the same location.
///
/// ```python
/// import gubas_rs
/// gubas_rs.run()                   # use Python's current working directory
/// gubas_rs.run("/path/to/run_dir") # change into a specific directory first
/// ```
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (work_dir=None))]
fn run(work_dir: Option<&str>) -> PyResult<()> {
    if let Some(dir) = work_dir {
        std::env::set_current_dir(dir)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
    }
    run_simulation();
    Ok(())
}

#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (work_dir=None))]
fn run_stm_py(work_dir: Option<&str>) -> PyResult<()> {
    if let Some(dir) = work_dir {
        std::env::set_current_dir(dir)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
    }
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
    if let Some(dir) = work_dir {
        std::env::set_current_dir(dir)
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
    }
    run_stm_augmented(min_degree, max_degree, which_body);
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
    /// Create a DynamicsModel from `ic_input.txt` in the current (or given) directory.
    #[new]
    #[pyo3(signature = (work_dir=None))]
    fn new(work_dir: Option<&str>) -> PyResult<Self> {
        if let Some(dir) = work_dir {
            std::env::set_current_dir(dir)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        }
        let ics = ic_read();

        let mut order   = ics.order;
        let mut order_a = ics.order_a;
        let mut order_b = ics.order_b;
        if ics.a_shape == 2 { order_a = order_a.max(order); }
        if ics.b_shape == 2 { order_b = order_b.max(order); }
        order = order.max(order_a).max(order_b);

        let ta = Cube::load_csv(&format!("TDP_{}.csv", order))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let tb = Cube::load_csv(&format!("TDS_{}.csv", order))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let ia = load_moi_csv("IDP.csv")
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let ib = load_moi_csv("IDS.csv")
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tk = tk_calc(order);
        let a  = a_calc(order);
        let b  = b_calc(order);

        let mc  = ta.get(0, 0, 0);
        let ms  = tb.get(0, 0, 0);
        let m   = mc * ms / (mc + ms);
        let nu  = ms / (mc + ms);
        let mean_motion = (ics.g * (ics.msun + mc + ms)
                           / (ics.sol_rad * ics.au_def / 1000.0).powi(3)).sqrt();
        let n_hyp   = (ics.g * ics.mplanet / ics.a_hyp.abs().powi(3)).sqrt();
        let n_helio = (ics.g * ics.msolar  / ics.a_helio.abs().powi(3)).sqrt();

        let mut params = Params {
            g: ics.g, m, nu, ta, tb, ia, ib,
            n: order, tk, a, b,
            flyby_toggle: ics.flyby_toggle,
            helio_toggle: ics.helio_toggle,
            sg_toggle:    ics.sg_toggle,
            tt_toggle:    ics.tt_toggle,
            mplanet:      ics.mplanet,
            a_hyp: ics.a_hyp, e_hyp: ics.e_hyp, i_hyp: ics.i_hyp,
            raan_hyp: ics.raan_hyp, om_hyp: ics.om_hyp, tau_hyp: ics.tau_hyp, n_hyp,
            msolar:    ics.msolar,
            a_helio: ics.a_helio, e_helio: ics.e_helio, i_helio: ics.i_helio,
            raan_helio: ics.raan_helio, om_helio: ics.om_helio, tau_helio: ics.tau_helio, n_helio,
            sol_rad: ics.sol_rad, au_def: ics.au_def, mean_motion,
            love1: ics.love1, love2: ics.love2,
            refrad1: ics.refrad1, refrad2: ics.refrad2,
            rho_a: ics.rho_a, rho_b: ics.rho_b,
            eps1: ics.eps1, eps2: ics.eps2,
            ida: math3::ZERO_M, idb: math3::ZERO_M,
            msun: ics.msun,
        };
        params.compute_lgvi_inertia();
        let params_dual = types::promote_params::<dual::Dual>(&params);

        Ok(DynamicsModel { params, params_dual })
    }

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
//
// Callable OD interface for the augmented system z = [x(30); θ(N)].
// Analogous to DynamicsModel but for gravity-parameter estimation.
//
// Python usage:
//   model = gubas_rs.AugmentedDynamicsModel(min_degree=2, max_degree=2)
//   zdot, A_aug = model.eval(z, t)           # point evaluation
//   rhs  = model.eval_stm(z_phi_flat, t)     # full ODE RHS for scipy

#[cfg(feature = "extension-module")]
#[pyclass]
pub struct AugmentedDynamicsModel {
    params_dual_base: Params<dual::Dual>,
    theta_indices:    Vec<(usize, usize, usize)>,
    which_body:       usize,
    pub n_aug:        usize,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl AugmentedDynamicsModel {
    /// Build model from `ic_input.txt` in the current (or given) directory.
    ///
    /// `min_degree`, `max_degree` select which T_{ijk} degrees form θ.
    /// `which_body` : 0 = primary T_a,  1 = secondary T_b.
    #[new]
    #[pyo3(signature = (min_degree=2, max_degree=2, which_body=0, work_dir=None))]
    fn new(
        min_degree: usize, max_degree: usize,
        which_body: usize, work_dir: Option<&str>,
    ) -> PyResult<Self> {
        if let Some(dir) = work_dir {
            std::env::set_current_dir(dir)
                .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
        }
        let ics = ic_read();
        let mut order = ics.order.max(ics.order_a).max(ics.order_b);
        if ics.a_shape == 2 { order = order.max(ics.order_a); }
        if ics.b_shape == 2 { order = order.max(ics.order_b); }

        let ta = Cube::load_csv(&format!("TDP_{}.csv", order))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let tb = Cube::load_csv(&format!("TDS_{}.csv", order))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let ia = load_moi_csv("IDP.csv")
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let ib = load_moi_csv("IDS.csv")
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tk = tk_calc(order); let a = a_calc(order); let b = b_calc(order);
        let mc = ta.get(0,0,0); let ms = tb.get(0,0,0);
        let m  = mc * ms / (mc + ms); let nu = ms / (mc + ms);
        let mean_motion = (ics.g*(ics.msun+mc+ms)/(ics.sol_rad*ics.au_def/1000.0).powi(3)).sqrt();
        let n_hyp   = (ics.g*ics.mplanet/ics.a_hyp.abs().powi(3)).sqrt();
        let n_helio = (ics.g*ics.msolar /ics.a_helio.abs().powi(3)).sqrt();
        let mut params = Params {
            g: ics.g, m, nu, ta, tb, ia, ib, n: order, tk, a, b,
            flyby_toggle:ics.flyby_toggle, helio_toggle:ics.helio_toggle,
            sg_toggle:ics.sg_toggle, tt_toggle:ics.tt_toggle,
            mplanet:ics.mplanet, a_hyp:ics.a_hyp, e_hyp:ics.e_hyp, i_hyp:ics.i_hyp,
            raan_hyp:ics.raan_hyp, om_hyp:ics.om_hyp, tau_hyp:ics.tau_hyp, n_hyp,
            msolar:ics.msolar, a_helio:ics.a_helio, e_helio:ics.e_helio, i_helio:ics.i_helio,
            raan_helio:ics.raan_helio, om_helio:ics.om_helio, tau_helio:ics.tau_helio, n_helio,
            sol_rad:ics.sol_rad, au_def:ics.au_def, mean_motion,
            love1:ics.love1, love2:ics.love2, refrad1:ics.refrad1, refrad2:ics.refrad2,
            rho_a:ics.rho_a, rho_b:ics.rho_b, eps1:ics.eps1, eps2:ics.eps2,
            ida:math3::ZERO_M, idb:math3::ZERO_M, msun:ics.msun,
        };
        params.compute_lgvi_inertia();

        let theta_indices = stokes::inertia_indices(min_degree, max_degree);
        let n_aug = 30 + theta_indices.len();
        let params_dual_base = types::promote_params::<dual::Dual>(&params);

        Ok(Self { params_dual_base, theta_indices, which_body, n_aug })
    }

    /// Augmented state dimension: 30 + N_theta.
    #[getter]
    fn n_aug(&self) -> usize { self.n_aug }

    /// θ indices as list of (i,j,k) tuples.
    #[getter]
    fn theta_indices(&self) -> Vec<(usize,usize,usize)> { self.theta_indices.clone() }

    /// Nominal T_{ijk} values for the selected body and indices.
    /// Use this as θ₀ when setting up the augmented initial state z₀ = [x₀; θ₀].
    #[getter]
    fn theta_nominal(&self) -> Vec<f64> {
        self.theta_indices.iter().map(|&(i, j, k)| {
            let d = if self.which_body == 0 {
                self.params_dual_base.ta.get(i, j, k)
            } else {
                self.params_dual_base.tb.get(i, j, k)
            };
            d.re
        }).collect()
    }

    /// eval(z, t) → (zdot, A_aug_flat)
    ///
    /// z: list of length n_aug = 30+N_theta   (x followed by θ values)
    /// Returns:
    ///   zdot      : list length n_aug         (θ̇ = 0 for last N entries)
    ///   A_aug_flat: list length n_aug²        (row-major)
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

    /// eval_stm(z_phi_flat, t) → full augmented ODE RHS for scipy.
    ///
    /// z_phi_flat: list of length n_aug + n_aug²
    ///   first n_aug elements : z = [x; θ]
    ///   last  n_aug² elements: Φ_aug flattened row-major
    ///
    /// Returns same layout. Initialize with z=[x₀;θ₀] and Φ_aug = I_{n_aug}.
    ///
    /// Example (scipy DOP853)::
    ///
    ///   n = model.n_aug
    ///   phi0 = np.eye(n).ravel()
    ///   y0   = np.concatenate([x0, theta0, phi0])
    ///   sol  = solve_ivp(lambda t,y: model.eval_stm(y.tolist(), t),
    ///                    [t0,tf], y0, method="DOP853", rtol=1e-10,
    ///                    dense_output=True)
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
        let pdb  = self.params_dual_base.clone();
        let tidx = self.theta_indices.clone();
        let wb   = self.which_body;
        move |z: &[dual::Dual], t: f64| -> Vec<dual::Dual> {
            let mut pd = pdb.clone();
            for (k, &(ii,jj,kk)) in tidx.iter().enumerate() {
                if wb == 0 { pd.ta.set(ii, jj, kk, z[30+k]); }
                else       { pd.tb.set(ii, jj, kk, z[30+k]); }
            }
            let mut x = [dual::Dual::from_re(0.0); 30];
            for i in 0..30 { x[i] = z[i]; }
            let xdot = dynamics::hou_ode(x, t, &pd);
            let mut zdot = vec![dual::Dual::from_re(0.0); 30 + tidx.len()];
            for i in 0..30 { zdot[i] = xdot[i]; }
            zdot
        }
    }
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn gubas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_stm_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_stm_augmented_py, m)?)?;
    m.add_class::<DynamicsModel>()?;
    m.add_class::<AugmentedDynamicsModel>()?;
    Ok(())
}
