// potential.rs — Mutual gravitational potential and its derivatives
//
// Implements the Hou 2016 mutual potential expansion and all partial
// derivatives needed by the equations of motion.
//
// All functions mirror their C++ counterparts exactly.
// Units throughout: km, kg, s.

use crate::coefficients::t_ind;
use crate::inertia::dt_dc;
use crate::math3::{Mat3, Vec3};
use crate::types::Cube;

// ── u_tilde ───────────────────────────────────────────────────────────────────

/// Compute ũ_n(e, TA, TBp) for expansion order `n`.
///
/// `dim` = mutual potential truncation order (used to size a/b).
/// `e`   = unit position vector (row vec in original code, here just a Vec3).
/// Mirrors `double u_tilde(int dim, int n, ...)`.
pub fn u_tilde(dim: usize, n: usize, tk: &[Vec<f64>],
               a: &[f64], b: &[f64], e: Vec3,
               ta: &Cube, tbp: &Cube) -> f64 {
    let ncols = tk[n].len();
    let mut u = vec![0.0_f64; ncols];

    let mut k = n as i64;
    while k >= 0 {
        let ku = k as usize;
        for i1 in 0..=ku {
          for i2 in 0..=(ku - i1) {
            for i3 in 0..=(ku - i1 - i2) {
              for i4 in 0..=(ku - i1 - i2 - i3) {
                for i5 in 0..=(ku - i1 - i2 - i3 - i4) {
                  let i6 = ku - i1 - i2 - i3 - i4 - i5;
                  let nk = n - ku;
                  for j1 in 0..=nk {
                    for j2 in 0..=(nk - j1) {
                      for j3 in 0..=(nk - j1 - j2) {
                        for j4 in 0..=(nk - j1 - j2 - j3) {
                          for j5 in 0..=(nk - j1 - j2 - j3 - j4) {
                            let j6 = nk - j1 - j2 - j3 - j4 - j5;
                            let a_val = a[t_ind(ku, i1, i2, i3, i4, i5, i6, dim + 1)];
                            let b_val = b[t_ind(nk, j1, j2, j3, j4, j5, j6, dim + 1)];
                            let e_term = e[0].powi((i1 + i4) as i32)
                                       * e[1].powi((i2 + i5) as i32)
                                       * e[2].powi((i3 + i6) as i32);
                            let ta_v = ta.get(i1 + j1, i2 + j2, i3 + j3);
                            let tb_v = tbp.get(i4 + j4, i5 + j5, i6 + j6);
                            u[ku / 2] += a_val * b_val * e_term * ta_v * tb_v;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        u[ku / 2] *= tk[n][ku / 2];
        k -= 2;
    }
    u.iter().sum()
}

// ── de_dx ─────────────────────────────────────────────────────────────────────

/// Partial of the unit position vector component `e[de]` w.r.t. `x[dx]`.
///
/// `e` = unit vector, `r_mag` = magnitude |r|.
/// Mirrors `double de_dx(mat e, double R, int de, int dx)`.
pub fn de_dx(e: Vec3, r_mag: f64, de: usize, dx: usize) -> f64 {
    let x = [e[0] * r_mag, e[1] * r_mag, e[2] * r_mag];
    if de == dx {
        let mut sum_sq = 0.0_f64;
        for i in 0..3 { if i != dx { sum_sq += x[i] * x[i]; } }
        sum_sq / r_mag.powi(3)
    } else {
        -x[de] * x[dx] / r_mag.powi(3)
    }
}

// ── du_dx_tilde ───────────────────────────────────────────────────────────────

/// Partial of ũ_n w.r.t. position component `x[dx]`.
///
/// Mirrors `double du_dx_tilde(int dim, int n, ..., int dx, ...)`.
pub fn du_dx_tilde(dim: usize, n: usize, tk: &[Vec<f64>],
                   a: &[f64], b: &[f64], e: Vec3, r_mag: f64, dx: usize,
                   ta: &Cube, tbp: &Cube) -> f64 {
    let de0 = de_dx(e, r_mag, 0, dx);
    let de1 = de_dx(e, r_mag, 1, dx);
    let de2 = de_dx(e, r_mag, 2, dx);

    let ncols = tk[n].len();
    let mut du = vec![0.0_f64; ncols];

    let mut k = n as i64;
    while k >= 0 {
        let ku = k as usize;
        for i1 in 0..=ku {
          for i2 in 0..=(ku - i1) {
            for i3 in 0..=(ku - i1 - i2) {
              for i4 in 0..=(ku - i1 - i2 - i3) {
                for i5 in 0..=(ku - i1 - i2 - i3 - i4) {
                  let i6 = ku - i1 - i2 - i3 - i4 - i5;
                  let nk = n - ku;
                  for j1 in 0..=nk {
                    for j2 in 0..=(nk - j1) {
                      for j3 in 0..=(nk - j1 - j2) {
                        for j4 in 0..=(nk - j1 - j2 - j3) {
                          for j5 in 0..=(nk - j1 - j2 - j3 - j4) {
                            let j6 = nk - j1 - j2 - j3 - j4 - j5;
                            let p = i1 + i4;
                            let q = i2 + i5;
                            let r = i3 + i6;
                            // chain rule: d/dx[p](e0^p * e1^q * e2^r)
                            let ce = e_partial(e, p, q, r, de0, de1, de2);
                            let a_v = a[t_ind(ku, i1, i2, i3, i4, i5, i6, dim + 1)];
                            let b_v = b[t_ind(nk, j1, j2, j3, j4, j5, j6, dim + 1)];
                            let ta_v = ta.get(i1 + j1, i2 + j2, i3 + j3);
                            let tb_v = tbp.get(i4 + j4, i5 + j5, i6 + j6);
                            du[ku / 2] += a_v * b_v * ce * ta_v * tb_v;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        du[ku / 2] *= tk[n][ku / 2];
        k -= 2;
    }
    du.iter().sum()
}

/// d/dx of  e0^p * e1^q * e2^r  using chain rule and de0,de1,de2.
fn e_partial(e: Vec3, p: usize, q: usize, r: usize,
             de0: f64, de1: f64, de2: f64) -> f64 {
    // Handles the 8 cases by checking which exponents are zero.
    // If an exponent is 0 the corresponding base term disappears in the
    // partial — exactly as in the C++ conditional tree.
    let mut ce = 0.0_f64;
    if p > 0 {
        ce += (p as f64) * e[0].powi(p as i32 - 1) * e[1].powi(q as i32)
              * e[2].powi(r as i32) * de0;
    }
    if q > 0 {
        ce += (q as f64) * e[0].powi(p as i32) * e[1].powi(q as i32 - 1)
              * e[2].powi(r as i32) * de1;
    }
    if r > 0 {
        ce += (r as f64) * e[0].powi(p as i32) * e[1].powi(q as i32)
              * e[2].powi(r as i32 - 1) * de2;
    }
    ce
}

// ── du_dc_tilde ───────────────────────────────────────────────────────────────

/// Partial of ũ_n w.r.t. rotation matrix element C(i,j), given the
/// pre-computed dT cube.
///
/// Mirrors `double du_dc_tilde(int dim, int n, ..., cube* dT)`.
pub fn du_dc_tilde(dim: usize, n: usize, tk: &[Vec<f64>],
                   a: &[f64], b: &[f64], e: Vec3,
                   ta: &Cube, dt: &Cube) -> f64 {
    let ncols = tk[n].len();
    let mut du = vec![0.0_f64; ncols];

    let mut k = n as i64;
    while k >= 0 {
        let ku = k as usize;
        for i1 in 0..=ku {
          for i2 in 0..=(ku - i1) {
            for i3 in 0..=(ku - i1 - i2) {
              for i4 in 0..=(ku - i1 - i2 - i3) {
                for i5 in 0..=(ku - i1 - i2 - i3 - i4) {
                  let i6 = ku - i1 - i2 - i3 - i4 - i5;
                  let nk = n - ku;
                  for j1 in 0..=nk {
                    for j2 in 0..=(nk - j1) {
                      for j3 in 0..=(nk - j1 - j2) {
                        for j4 in 0..=(nk - j1 - j2 - j3) {
                          for j5 in 0..=(nk - j1 - j2 - j3 - j4) {
                            let j6 = nk - j1 - j2 - j3 - j4 - j5;
                            let a_v = a[t_ind(ku, i1, i2, i3, i4, i5, i6, dim + 1)];
                            let b_v = b[t_ind(nk, j1, j2, j3, j4, j5, j6, dim + 1)];
                            let e_term = e[0].powi((i1 + i4) as i32)
                                       * e[1].powi((i2 + i5) as i32)
                                       * e[2].powi((i3 + i6) as i32);
                            let ta_v = ta.get(i1 + j1, i2 + j2, i3 + j3);
                            let dt_v = dt.get(i4 + j4, i5 + j5, i6 + j6);
                            du[ku / 2] += a_v * b_v * e_term * ta_v * dt_v;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        du[ku / 2] *= tk[n][ku / 2];
        k -= 2;
    }
    du.iter().sum()
}

// ── du_x ─────────────────────────────────────────────────────────────────────

/// Partial of the mutual potential energy w.r.t. position component `dx`.
///
/// Returns the *energy* partial (negative of the Hou paper value, i.e. the
/// correct sign for `vd = ... - (1/m)*du_dr`).
/// Mirrors `double du_x(...)`.
pub fn du_x(g: f64, m: usize, tk: &[Vec<f64>],
            a: &[f64], b: &[f64], e: Vec3, r_mag: f64, dx: usize,
            ta: &Cube, tbp: &Cube) -> f64 {
    let x = [e[0] * r_mag, e[1] * r_mag, e[2] * r_mag];
    let mut du = 0.0_f64;
    for n in 0..=m {
        let nf = n as f64;
        du += (-(nf + 1.0) * x[dx] / r_mag.powf(nf + 3.0))
              * u_tilde(m, n, tk, a, b, e, ta, tbp)
            + (1.0 / r_mag.powf(nf + 1.0))
              * du_dx_tilde(m, n, tk, a, b, e, r_mag, dx, ta, tbp);
    }
    -du * g
}

// ── du_c ─────────────────────────────────────────────────────────────────────

/// Partial of the mutual potential energy w.r.t. rotation matrix element
/// C(i,j), given pre-computed dT.
///
/// Mirrors `double du_c(...)`.
pub fn du_c(g: f64, m: usize, tk: &[Vec<f64>],
            a: &[f64], b: &[f64], e: Vec3, r_mag: f64,
            ta: &Cube, dt: &Cube) -> f64 {
    let mut du = 0.0_f64;
    for n in 0..=m {
        du += (1.0 / r_mag.powf(n as f64 + 1.0))
              * du_dc_tilde(m, n, tk, a, b, e, ta, dt);
    }
    -du * g
}

// ── potential ─────────────────────────────────────────────────────────────────

/// Mutual gravitational potential energy (negative sign convention follows
/// Hou 2016 and the C++ code — this is the *energy*, not the force).
///
/// Mirrors `double potential(...)`.
pub fn potential(g: f64, m: usize, tk: &[Vec<f64>],
                 a: &[f64], b: &[f64], e: Vec3, r_mag: f64,
                 ta: &Cube, tbp: &Cube) -> f64 {
    let mut u = 0.0_f64;
    for n in 0..=m {
        u += (1.0 / r_mag.powf(n as f64 + 1.0))
             * u_tilde(m, n, tk, a, b, e, ta, tbp);
    }
    -u * g
}

// ── compute_all_du_dr_and_torques ─────────────────────────────────────────────

/// Convenience function that computes the 9 dT cubes, du/dr (3 components),
/// and all du/dC partials needed for torque computation, for a given state.
///
/// Returns `(du_dr, du_dalpha, du_dbeta, du_dgam)` — all in the A frame —
/// and a side-effect of rotating `tb` into `tbp` which it also returns.
pub fn map_potential_partials_rhs(
    c: Mat3,
    r: Vec3,
    params: &crate::types::Params,
) -> (Cube, Vec3, [Vec3; 3]) {
    let r_mag = crate::math3::norm(r);
    let e = crate::math3::scale_v(1.0 / r_mag, r);

    // Rotate secondary inertia integrals
    let tbp = crate::inertia::inertia_rot(c, params.n, &params.tb);

    // Build 9 partial cubes dT/dC[i][j]
    let mut dt = [[(); 3]; 3].map(|_| std::array::from_fn::<_, 3, _>(|_| Cube::new(params.n)));
    for i in 0..3 {
        for j in 0..3 {
            dt[i][j] = dt_dc(i, j, c, params.n, &params.tb);
        }
    }

    // du/dr (3 components)
    let du_dr: Vec3 = [
        du_x(params.g, params.n, &params.tk, &params.a, &params.b, e, r_mag, 0, &params.ta, &tbp),
        du_x(params.g, params.n, &params.tk, &params.a, &params.b, e, r_mag, 1, &params.ta, &tbp),
        du_x(params.g, params.n, &params.tk, &params.a, &params.b, e, r_mag, 2, &params.ta, &tbp),
    ];

    // du/dC columns (needed for torques on primary and secondary)
    let du_cols: [Vec3; 3] = std::array::from_fn(|col_idx| {
        [
            du_c(params.g, params.n, &params.tk, &params.a, &params.b, e, r_mag,
                 &params.ta, &dt[0][col_idx]),
            du_c(params.g, params.n, &params.tk, &params.a, &params.b, e, r_mag,
                 &params.ta, &dt[1][col_idx]),
            du_c(params.g, params.n, &params.tk, &params.a, &params.b, e, r_mag,
                 &params.ta, &dt[2][col_idx]),
        ]
    });

    (tbp, du_dr, du_cols)
}
