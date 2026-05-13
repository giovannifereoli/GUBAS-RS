// lgvi.rs — Lie Group Variational Integrator (LGVI) helpers
//
// Implements the Hamiltonian map from Lee 2007, used by lgvi_integ in
// integrators.rs.  All intermediate computations are done in dimensionless
// (normalized) variables; physical inputs/outputs use km / kg / s.
//
// Key functions:
//   map_potential_partials_lgvi — du/dr and gravity torque M for LGVI step
//   hamiltonian_map             — one full LGVI time step

use crate::inertia::{dt_dc, inertia_rot};
use crate::math3::*;
use crate::potential::{du_c, du_x};
use crate::types::{Cube, Params};

// ── outer product helper ──────────────────────────────────────────────────────

/// Outer (tensor) product  a ⊗ b  as a 3×3 matrix.
fn outer(a: Vec3, b: Vec3) -> Mat3 {
    [[a[0]*b[0], a[0]*b[1], a[0]*b[2]],
     [a[1]*b[0], a[1]*b[1], a[1]*b[2]],
     [a[2]*b[0], a[2]*b[1], a[2]*b[2]]]
}

// ── map_potential_partials_lgvi ───────────────────────────────────────────────

/// Compute the mutual-potential partial du/dr (Vec3) and the gravity torque M
/// (Vec3) on the secondary body in the A frame, given physical position r and
/// relative rotation C.
///
/// Mirrors `void map_potential_partials(mat* C, mat* r, ...)`.
pub fn map_potential_partials_lgvi(c: Mat3, r: Vec3, p: &Params) -> (Vec3, Vec3) {
    let r_mag = norm(r).max(1e-30);
    let e     = scale_v(1.0 / r_mag, r);

    let tbp = inertia_rot(c, p.n, &p.tb);

    let dts: [[Cube; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| dt_dc(i, j, c, p.n, &p.tb))
    });

    let du_dr: Vec3 = [
        du_x(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, 0, &p.ta, &tbp),
        du_x(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, 1, &p.ta, &tbp),
        du_x(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, 2, &p.ta, &tbp),
    ];

    let du_cols: [Vec3; 3] = std::array::from_fn(|col| {
        [
            du_c(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, &p.ta, &dts[0][col]),
            du_c(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, &p.ta, &dts[1][col]),
            du_c(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, &p.ta, &dts[2][col]),
        ]
    });

    // M = Σ_col  cross(C[:,col], du/dC[:,col])  — torque on secondary (A frame)
    let m_grav = add_v(add_v(
        cross(col(c, 0), du_cols[0]),
        cross(col(c, 1), du_cols[1])),
        cross(col(c, 2), du_cols[2]));

    (du_dr, m_grav)
}

// ── F_exp_calc (Rodrigues / exponential-map implicit solver) ──────────────────

/// Rodrigues implicit solver used as fallback when the Cayley solver diverges.
///
/// `g_scaled` is already multiplied by the step size h (because it is called
/// from within `f_cayley_calc` after g was pre-scaled there, matching the C++
/// calling convention).
///
/// Returns (f, F) where F is the corresponding SO(3) rotation matrix.
fn f_exp_calc_scaled(g_scaled: Vec3, i_mat: Mat3) -> (Vec3, Mat3) {
    let mut f: Vec3 = [0.1; 3];

    for iter in 0..101 {
        let fn_ = norm(f).max(1e-300);
        let i_f = mat_vec(i_mat, f);
        let sin_fn = fn_.sin();
        let cos_fn = fn_.cos();

        // G(f) = sin|f|/|f| · I·f  +  (1−cos|f|)/|f|² · (f × I·f)
        let big_g = add_v(
            scale_v(sin_fn / fn_,                  i_f),
            scale_v((1.0 - cos_fn) / (fn_ * fn_),  cross(f, i_f)),
        );

        let residual = sub_v(g_scaled, big_g);
        if norm(residual) <= 1e-15 { break; }
        if iter == 100 {
            eprintln!("Warning: F_exp_calc did not converge to numerical precision.");
            break;
        }

        // Jacobian  d G / d f
        let a1 = (fn_ * cos_fn - sin_fn) / fn_.powi(3);
        let a2 = sin_fn / fn_;
        let a3 = (fn_ * sin_fn - 2.0 * (1.0 - cos_fn)) / fn_.powi(4);
        let a4 = (1.0 - cos_fn) / (fn_ * fn_);

        let grad_g = add_m(
            add_m(
                add_m(scale_m(a1, outer(i_f, f)),
                      scale_m(a2, i_mat)),
                scale_m(a3, outer(cross(f, i_f), f)),
            ),
            // (1−cos|f|)/|f|² · ( −tilde(I·f)  +  tilde(f)·I )
            scale_m(a4, add_m(
                scale_m(-1.0, tilde(i_f)),
                mat_mul(tilde(f), i_mat),
            )),
        );

        f = add_v(f, mat_vec(inv3(grad_g), residual));
    }

    // Rodrigues rotation matrix
    let fn_ = norm(f).max(1e-300);
    let t   = tilde(f);
    let big_f = add_m(
        add_m(eye(), scale_m(fn_.sin() / fn_,                      t)),
                     scale_m((1.0 - fn_.cos()) / (fn_ * fn_), mat_mul(t, t)),
    );
    (f, big_f)
}

// ── F_cayley_calc ─────────────────────────────────────────────────────────────

/// Cayley implicit solver for the LGVI rotation update.
///
/// Solves  G(f) ≡ g + g×f + (g·f)f − 2·I·f = 0  via Newton's method,
/// then returns the Cayley rotation  F = (I+tilde(f))·(I−tilde(f))⁻¹.
///
/// Falls back to `f_exp_calc_scaled` after 100 iterations, matching the C++.
fn f_cayley_calc(h: f64, g_in: Vec3, i_mat: Mat3) -> (Vec3, Mat3) {
    let mut f: Vec3 = [0.0001; 3];
    let g = scale_v(h, g_in);      // scale once, matches C++: `(*g)=h*(*g)`
    let mut check = 0usize;

    loop {
        // Residual  G = g + cross(g,f) + dot(g,f)·f − 2·I·f
        let big_g = sub_v(
            add_v(add_v(g, cross(g, f)), scale_v(dot(g, f), f)),
            scale_v(2.0, mat_vec(i_mat, f)),
        );

        if norm(big_g) <= 1e-15 {
            // Converged — Cayley rotation: F = (I+tilde(f))·(I−tilde(f))⁻¹
            let big_f = mat_mul(
                add_m(eye(), tilde(f)),
                inv3(sub_m(eye(), tilde(f))),
            );
            return (f, big_f);
        }

        // Jacobian  dG/df = tilde(g) + (g·f)·I + outer(f,g) − 2·I_mat
        let grad_g = sub_m(
            add_m(add_m(tilde(g), scale_m(dot(g, f), eye())), outer(f, g)),
            scale_m(2.0, i_mat),
        );
        f = sub_v(f, mat_vec(inv3(grad_g), big_g));
        check += 1;

        if check > 100 {
            eprintln!("broke Cayley");
            let cay_err = norm(big_g);
            let cay_f   = f;

            // Fallback: exponential solver (g is already h-scaled here)
            let (exp_f, exp_f_mat) = f_exp_calc_scaled(g, i_mat);

            // Cayley residual at exp solution — compare against Cayley's error
            let exp_big_g = sub_v(
                add_v(add_v(g, cross(g, exp_f)), scale_v(dot(g, exp_f), exp_f)),
                scale_v(2.0, mat_vec(i_mat, exp_f)),
            );
            if norm(sub_v(g, exp_big_g)) > cay_err {
                // Exp did not improve; use Cayley f with exp-derived F (mirrors C++)
                return (cay_f, exp_f_mat);
            } else {
                return (exp_f, exp_f_mat);
            }
        }
    }
}

// ── hamiltonian_map ───────────────────────────────────────────────────────────

/// One LGVI Hamiltonian map step.
///
/// Inputs:
///   `h`       — time step (s, physical)
///   `params`  — simulation parameters
///   `x`       — current state [r,v,wc,ws,Cc,C] (30 elements, physical)
///   `x0`      — initial state used for normalization
///   `du_dr_n` — previous potential partial ∂U/∂r (physical, km/kg/s²·kg = force/mass … km²/s²/km)
///   `m_n`     — previous gravity torque on secondary (physical, N·km)
///
/// Returns  (x_new, du_dr_n1, m_n1).
///
/// Mirrors `void hamiltonian_map(...)`.
pub fn hamiltonian_map(
    h:        f64,
    params:   &Params,
    x:        [f64; 30],
    x0:       [f64; 30],
    du_dr_n:  Vec3,
    m_n:      Vec3,
) -> ([f64; 30], Vec3, Vec3) {

    // ── normalisation constants ──────────────────────────────────────────
    let r0_vec: Vec3 = [x0[0], x0[1], x0[2]];
    let nr  = norm(r0_vec);                          // length scale  (km)
    let nm  = params.m;                              // mass scale    (kg)
    let nt  = (params.g * nm / nr.powi(3)).sqrt();   // time⁻¹ scale  (1/s)

    let alpha = nm * nr * nr;                        // nm·nr²
    let beta  = nm * nr * nt * nt;                   // nm·nr·nt² = G·nm²/nr²

    let h_n = h * nt;   // normalised step

    // ── unpack and normalise state ───────────────────────────────────────
    let r : Vec3 = scale_v(1.0 / nr,       [x[0], x[1], x[2]]);
    let v : Vec3 = scale_v(1.0 / (nr*nt),  [x[3], x[4], x[5]]);
    let wc: Vec3 = scale_v(1.0 / nt,       [x[6], x[7], x[8]]);
    let ws: Vec3 = scale_v(1.0 / nt,       [x[9], x[10], x[11]]);
    let cc: Mat3 = [[x[12], x[13], x[14]],
                    [x[15], x[16], x[17]],
                    [x[18], x[19], x[20]]];
    let c : Mat3 = [[x[21], x[22], x[23]],
                    [x[24], x[25], x[26]],
                    [x[27], x[28], x[29]]];

    // Normalised potential partials
    let du_dr_norm: Vec3 = scale_v(1.0 / beta,            du_dr_n);   // ÷ nm·nr·nt²  (force scale)
    let m_norm    : Vec3 = scale_v(1.0 / (alpha * nt*nt), m_n);       // ÷ nm·nr²·nt² (torque scale)

    let ia = params.ia;
    let ib = params.ib;

    // ── inertia tensors (normalised) ─────────────────────────────────────
    // IH  = diag(IA) / (nm·nr²)
    let ih  : Mat3 = scale_m(1.0 / alpha, diag(ia));
    // IBr = C · diag(IB) · Cᵀ / (nm·nr²)
    let ibr : Mat3 = scale_m(
        1.0 / alpha,
        mat_mul(mat_mul(c, diag(ib)), transpose(c)),
    );

    // ── relative-attitude Cayley solve (secondary body B) ────────────────
    // g_B = IBr·ws − (h/2)·M_norm
    let g_b = sub_v(mat_vec(ibr, ws), scale_v(h_n / 2.0, m_norm));
    let (_, fab) = f_cayley_calc(h_n, g_b, ibr);

    // ── inertial-attitude Cayley solve (primary body A) ──────────────────
    // g_A = IH·wc + (h/2)·cross(r, du_dr_norm) + (h/2)·M_norm
    let g_a = add_v(
        add_v(mat_vec(ih, wc),
              scale_v(h_n / 2.0, cross(r, du_dr_norm))),
        scale_v(h_n / 2.0, m_norm),
    );
    let (_, fna) = f_cayley_calc(h_n, g_a, ih);

    // ── propagate position and relative attitude ─────────────────────────
    // r_n1 = Fnaᵀ · (r + h·v − (h²/2)·du_dr_norm)
    let r_n1: Vec3 = mat_vec(
        transpose(fna),
        add_v(add_v(r, scale_v(h_n, v)),
              scale_v(-h_n * h_n / 2.0, du_dr_norm)),
    );
    // C_n1 = Fnaᵀ · Fab · C
    let c_n1: Mat3 = mat_mul(mat_mul(transpose(fna), fab), c);

    // ── evaluate new potential partials at (C_n1, r_n1·nr) ──────────────
    let r_n1_phys: Vec3 = scale_v(nr, r_n1);    // de-normalise for the potential call
    let (du_dr_n1, m_n1) = map_potential_partials_lgvi(c_n1, r_n1_phys, params);

    let du_dr_n1_norm: Vec3 = scale_v(1.0 / beta,            du_dr_n1);
    let m_n1_norm    : Vec3 = scale_v(1.0 / (alpha * nt*nt), m_n1);

    // ── propagate velocity ───────────────────────────────────────────────
    // v_n1 = Fnaᵀ·(v − h/2·du_dr_norm) − h/2·du_dr_n1_norm
    let v_n1: Vec3 = sub_v(
        mat_vec(transpose(fna), sub_v(v, scale_v(h_n / 2.0, du_dr_norm))),
        scale_v(h_n / 2.0, du_dr_n1_norm),
    );

    // ── propagate primary angular velocity ───────────────────────────────
    // wc_n1 = IH⁻¹ · ( Fnaᵀ·(IH·wc + h/2·cross(r,du_dr_norm) + h/2·M_norm)
    //                  + h/2·cross(r_n1,du_dr_n1_norm) + h/2·M_n1_norm )
    // IH⁻¹  = alpha · diag(1/IA)
    let wc_bracket = add_v(
        add_v(mat_vec(ih, wc),
              scale_v(h_n / 2.0, cross(r, du_dr_norm))),
        scale_v(h_n / 2.0, m_norm),
    );
    let wc_rhs = add_v(
        add_v(mat_vec(transpose(fna), wc_bracket),
              scale_v(h_n / 2.0, cross(r_n1, du_dr_n1_norm))),
        scale_v(h_n / 2.0, m_n1_norm),
    );
    let ih_inv: Mat3 = scale_m(alpha, diag([1.0/ia[0], 1.0/ia[1], 1.0/ia[2]]));
    let wc_n1: Vec3  = mat_vec(ih_inv, wc_rhs);

    // ── propagate secondary angular velocity ─────────────────────────────
    // ws_n1 = C_n1·(IB/α)⁻¹·C_n1ᵀ · ( Fnaᵀ·(IBr·ws − h/2·M_norm) − h/2·M_n1_norm )
    // (IB/α)⁻¹ expressed in A-frame = α · C_n1·diag(1/IB)·C_n1ᵀ
    let ws_bracket = sub_v(mat_vec(ibr, ws), scale_v(h_n / 2.0, m_norm));
    let ws_rhs = sub_v(
        mat_vec(transpose(fna), ws_bracket),
        scale_v(h_n / 2.0, m_n1_norm),
    );
    let ib_inv_rot: Mat3 = scale_m(
        alpha,
        mat_mul(mat_mul(c_n1, diag([1.0/ib[0], 1.0/ib[1], 1.0/ib[2]])), transpose(c_n1)),
    );
    let ws_n1: Vec3 = mat_vec(ib_inv_rot, ws_rhs);

    // ── propagate inertial attitude ──────────────────────────────────────
    // Cc_n1 = Cc · Fna
    let cc_n1: Mat3 = mat_mul(cc, fna);

    // ── de-normalise and pack output ─────────────────────────────────────
    let r_out  = r_n1_phys;                   // already physical
    let v_out  = scale_v(nr * nt, v_n1);
    let wc_out = scale_v(nt, wc_n1);
    let ws_out = scale_v(nt, ws_n1);

    let x_out: [f64; 30] = [
        r_out[0],   r_out[1],   r_out[2],
        v_out[0],   v_out[1],   v_out[2],
        wc_out[0],  wc_out[1],  wc_out[2],
        ws_out[0],  ws_out[1],  ws_out[2],
        cc_n1[0][0], cc_n1[0][1], cc_n1[0][2],
        cc_n1[1][0], cc_n1[1][1], cc_n1[1][2],
        cc_n1[2][0], cc_n1[2][1], cc_n1[2][2],
        c_n1[0][0],  c_n1[0][1],  c_n1[0][2],
        c_n1[1][0],  c_n1[1][1],  c_n1[1][2],
        c_n1[2][0],  c_n1[2][1],  c_n1[2][2],
    ];

    (x_out, du_dr_n1, m_n1)
}
