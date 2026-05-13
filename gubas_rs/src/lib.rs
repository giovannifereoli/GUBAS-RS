#![allow(dead_code)]

pub mod coefficients;
pub mod dynamics;
pub mod inertia;
pub mod integrators;
pub mod lgvi;
pub mod math3;
pub mod orbit;
pub mod potential;
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
#[pymodule]
fn gubas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
