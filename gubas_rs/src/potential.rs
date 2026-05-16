//! Mutual gravitational potential and its partial derivatives (Hou 2016).
//!
//! All functions are generic over [`Scalar`] so that `Dual` types can flow
//! through for STM / parameter-sensitivity computation.  Units: km, kg, s.
//!
//! | Function | Purpose |
//! |---|---|
//! | [`potential`] | Total mutual potential energy |
//! | [`du_x`] | ∂U/∂r — used for translational EOM |
//! | [`du_c`] | ∂U/∂C — one rotation-matrix column, used for torques |

use crate::coefficients::t_ind;
use crate::inertia::dt_dc;
use crate::math3::{Mat3, Scalar, Vec3};
use crate::types::Cube;
use num_traits::NumCast;

// ── u_tilde ───────────────────────────────────────────────────────────────────

/// Compute ũ_n(e, TA, TBp) for expansion order `n`.
///
/// `dim` = mutual potential truncation order (used to size a/b).
/// `e`   = unit position vector (row vec in original code, here just a Vec3).
/// Mirrors `double u_tilde(int dim, int n, ...)`.
pub fn u_tilde<T: Scalar>(dim: usize, n: usize, tk: &[Vec<f64>],
                          a: &[f64], b: &[f64], e: Vec3<T>,
                          ta: &Cube<T>, tbp: &Cube<T>) -> T {
    let to_t = |x: f64| -> T { <T as NumCast>::from(x).unwrap() };
    let ncols = tk[n].len();
    let mut u = vec![T::zero(); ncols];

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
                            u[ku / 2] += to_t(a_val * b_val) * e_term * ta_v * tb_v;
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
        u[ku / 2] = u[ku / 2] * to_t(tk[n][ku / 2]);
        k -= 2;
    }
    u.iter().copied().fold(T::zero(), |acc, x| acc + x)
}

// ── de_dx ─────────────────────────────────────────────────────────────────────

/// Partial of the unit position vector component `e[de]` w.r.t. `x[dx]`.
///
/// `e` = unit vector, `r_mag` = magnitude |r|.
/// Mirrors `double de_dx(mat e, double R, int de, int dx)`.
pub fn de_dx<T: Scalar>(e: Vec3<T>, r_mag: T, de: usize, dx: usize) -> T {
    let x = [e[0] * r_mag, e[1] * r_mag, e[2] * r_mag];
    if de == dx {
        let mut sum_sq = T::zero();
        for i in 0..3 { if i != dx { sum_sq += x[i] * x[i]; } }
        sum_sq / r_mag.powi(3)
    } else {
        -(x[de] * x[dx]) / r_mag.powi(3)
    }
}

// ── du_dx_tilde ───────────────────────────────────────────────────────────────

/// Partial of ũ_n w.r.t. position component `x[dx]`.
///
/// Mirrors `double du_dx_tilde(int dim, int n, ..., int dx, ...)`.
pub fn du_dx_tilde<T: Scalar>(dim: usize, n: usize, tk: &[Vec<f64>],
                              a: &[f64], b: &[f64], e: Vec3<T>, r_mag: T, dx: usize,
                              ta: &Cube<T>, tbp: &Cube<T>) -> T {
    let to_t = |x: f64| -> T { <T as NumCast>::from(x).unwrap() };
    let de0 = de_dx(e, r_mag, 0, dx);
    let de1 = de_dx(e, r_mag, 1, dx);
    let de2 = de_dx(e, r_mag, 2, dx);

    let ncols = tk[n].len();
    let mut du = vec![T::zero(); ncols];

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
                            let ce = e_partial(e, p, q, r, de0, de1, de2);
                            let a_v = a[t_ind(ku, i1, i2, i3, i4, i5, i6, dim + 1)];
                            let b_v = b[t_ind(nk, j1, j2, j3, j4, j5, j6, dim + 1)];
                            let ta_v = ta.get(i1 + j1, i2 + j2, i3 + j3);
                            let tb_v = tbp.get(i4 + j4, i5 + j5, i6 + j6);
                            du[ku / 2] += to_t(a_v * b_v) * ce * ta_v * tb_v;
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
        du[ku / 2] = du[ku / 2] * to_t(tk[n][ku / 2]);
        k -= 2;
    }
    du.iter().copied().fold(T::zero(), |acc, x| acc + x)
}

/// d/dx of  e0^p * e1^q * e2^r  using chain rule and de0,de1,de2.
fn e_partial<T: Scalar>(e: Vec3<T>, p: usize, q: usize, r: usize,
                        de0: T, de1: T, de2: T) -> T {
    let to_t = |x: f64| -> T { <T as NumCast>::from(x).unwrap() };
    let mut ce = T::zero();
    if p > 0 {
        ce += to_t(p as f64) * e[0].powi(p as i32 - 1) * e[1].powi(q as i32)
              * e[2].powi(r as i32) * de0;
    }
    if q > 0 {
        ce += to_t(q as f64) * e[0].powi(p as i32) * e[1].powi(q as i32 - 1)
              * e[2].powi(r as i32) * de1;
    }
    if r > 0 {
        ce += to_t(r as f64) * e[0].powi(p as i32) * e[1].powi(q as i32)
              * e[2].powi(r as i32 - 1) * de2;
    }
    ce
}

// ── du_dc_tilde ───────────────────────────────────────────────────────────────

/// Partial of ũ_n w.r.t. rotation matrix element C(i,j), given the
/// pre-computed dT cube.
///
/// Mirrors `double du_dc_tilde(int dim, int n, ..., cube* dT)`.
pub fn du_dc_tilde<T: Scalar>(dim: usize, n: usize, tk: &[Vec<f64>],
                              a: &[f64], b: &[f64], e: Vec3<T>,
                              ta: &Cube<T>, dt: &Cube<T>) -> T {
    let to_t = |x: f64| -> T { <T as NumCast>::from(x).unwrap() };
    let ncols = tk[n].len();
    let mut du = vec![T::zero(); ncols];

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
                            du[ku / 2] += to_t(a_v * b_v) * e_term * ta_v * dt_v;
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
        du[ku / 2] = du[ku / 2] * to_t(tk[n][ku / 2]);
        k -= 2;
    }
    du.iter().copied().fold(T::zero(), |acc, x| acc + x)
}

// ── du_x ─────────────────────────────────────────────────────────────────────

/// Partial of the mutual potential energy w.r.t. position component `dx`.
///
/// Returns the *energy* partial (negative of the Hou paper value, i.e. the
/// correct sign for `vd = ... - (1/m)*du_dr`).
/// Mirrors `double du_x(...)`.
pub fn du_x<T: Scalar>(g: T, m: usize, tk: &[Vec<f64>],
                       a: &[f64], b: &[f64], e: Vec3<T>, r_mag: T, dx: usize,
                       ta: &Cube<T>, tbp: &Cube<T>) -> T {
    let to_t = |x: f64| -> T { <T as NumCast>::from(x).unwrap() };
    let x = [e[0] * r_mag, e[1] * r_mag, e[2] * r_mag];
    let mut du = T::zero();
    for n in 0..=m {
        let nf = to_t(n as f64);
        du += (-(nf + T::one()) * x[dx] / r_mag.powf(nf + to_t(3.0)))
              * u_tilde(m, n, tk, a, b, e, ta, tbp)
            + (T::one() / r_mag.powf(nf + T::one()))
              * du_dx_tilde(m, n, tk, a, b, e, r_mag, dx, ta, tbp);
    }
    -du * g
}

// ── du_c ─────────────────────────────────────────────────────────────────────

/// Partial of the mutual potential energy w.r.t. rotation matrix element
/// C(i,j), given pre-computed dT.
///
/// Mirrors `double du_c(...)`.
pub fn du_c<T: Scalar>(g: T, m: usize, tk: &[Vec<f64>],
                       a: &[f64], b: &[f64], e: Vec3<T>, r_mag: T,
                       ta: &Cube<T>, dt: &Cube<T>) -> T {
    let to_t = |x: f64| -> T { <T as NumCast>::from(x).unwrap() };
    let mut du = T::zero();
    for n in 0..=m {
        let nf = to_t(n as f64);
        du += (T::one() / r_mag.powf(nf + T::one()))
              * du_dc_tilde(m, n, tk, a, b, e, ta, dt);
    }
    -du * g
}

// ── potential ─────────────────────────────────────────────────────────────────

/// Mutual gravitational potential energy (negative sign convention follows
/// Hou 2016 and the C++ code — this is the *energy*, not the force).
///
/// Mirrors `double potential(...)`.
pub fn potential<T: Scalar>(g: T, m: usize, tk: &[Vec<f64>],
                            a: &[f64], b: &[f64], e: Vec3<T>, r_mag: T,
                            ta: &Cube<T>, tbp: &Cube<T>) -> T {
    let to_t = |x: f64| -> T { <T as NumCast>::from(x).unwrap() };
    let mut u = T::zero();
    for n in 0..=m {
        let nf = to_t(n as f64);
        u += (T::one() / r_mag.powf(nf + T::one()))
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

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coefficients::{a_calc, b_calc, tk_calc};
    use crate::types::Cube;

    fn abs_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {b:.6e}, got {a:.6e}");
    }
    fn rel_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() / b.abs() < tol, "expected {b:.6e}, got {a:.6e}");
    }

    // ── de_dx ────────────────────────────────────────────────────────────────

    #[test]
    fn de_dx_diagonal() {
        // e = [0.6, 0.8, 0], r = 5: ∂e₀/∂x₀ = (x₁²+x₂²)/r³ = 16/125
        let e: crate::math3::Vec3 = [0.6, 0.8, 0.0];
        abs_close(de_dx(e, 5.0, 0, 0), 16.0 / 125.0, 1e-14);
    }

    #[test]
    fn de_dx_off_diagonal() {
        // ∂e₀/∂x₁ = −(x₀·x₁)/r³ = −12/125
        let e: crate::math3::Vec3 = [0.6, 0.8, 0.0];
        abs_close(de_dx(e, 5.0, 0, 1), -12.0 / 125.0, 1e-14);
    }

    #[test]
    fn de_dx_unit_x_diagonal_zero() {
        // e = [1,0,0]: ∂e₀/∂x₀ = (0+0)/r³ = 0
        abs_close(de_dx([1.0, 0.0, 0.0], 3.0, 0, 0), 0.0, 1e-14);
    }

    // ── u_tilde ──────────────────────────────────────────────────────────────

    #[test]
    fn u_tilde_monopole_equals_mass_product() {
        // n=0: ũ₀ = tk[0][0]·a[0]·b[0]·Ma·Mb = Ma·Mb
        let ma = 1e12_f64;
        let mb = 5e11_f64;
        let tk = tk_calc(0);
        let a  = a_calc(0);
        let b  = b_calc(0);
        let mut ta = Cube::new(0);  ta.set(0, 0, 0, ma);
        let mut tb = Cube::new(0);  tb.set(0, 0, 0, mb);
        let e: crate::math3::Vec3 = [1.0, 0.0, 0.0];
        let result = u_tilde(0, 0, &tk, &a, &b, e, &ta, &tb);
        rel_close(result, ma * mb, 1e-14);
    }

    // ── potential ────────────────────────────────────────────────────────────

    #[test]
    fn potential_monopole_newtonian() {
        // Order-0 expansion recovers U = −G·Ma·Mb/r
        let g  = 6.674e-20_f64;
        let ma = 1e12_f64;
        let mb = 5e11_f64;
        let r  = 10.0_f64;
        let tk = tk_calc(0);
        let a  = a_calc(0);
        let b  = b_calc(0);
        let mut ta = Cube::new(0);  ta.set(0, 0, 0, ma);
        let mut tb = Cube::new(0);  tb.set(0, 0, 0, mb);
        let e: crate::math3::Vec3 = [1.0, 0.0, 0.0];
        let u = potential(g, 0, &tk, &a, &b, e, r, &ta, &tb);
        rel_close(u, -g * ma * mb / r, 1e-12);
    }

    // ── du_x ─────────────────────────────────────────────────────────────────

    #[test]
    fn du_x_monopole_radial_force() {
        // Order-0 with e=[1,0,0]: ∂U/∂x₀ = G·Ma·Mb/r²
        let g  = 6.674e-20_f64;
        let ma = 1e12_f64;
        let mb = 5e11_f64;
        let r  = 10.0_f64;
        let tk = tk_calc(0);
        let a  = a_calc(0);
        let b  = b_calc(0);
        let mut ta = Cube::new(0);  ta.set(0, 0, 0, ma);
        let mut tb = Cube::new(0);  tb.set(0, 0, 0, mb);
        let e: crate::math3::Vec3 = [1.0, 0.0, 0.0];
        let du = du_x(g, 0, &tk, &a, &b, e, r, 0, &ta, &tb);
        rel_close(du, g * ma * mb / (r * r), 1e-12);
    }

    #[test]
    fn du_x_monopole_transverse_zero() {
        // For e=[1,0,0], ∂U/∂x₁ = 0 by symmetry
        let g  = 6.674e-20_f64;
        let ma = 1e12_f64;
        let mb = 5e11_f64;
        let tk = tk_calc(0);
        let a  = a_calc(0);
        let b  = b_calc(0);
        let mut ta = Cube::new(0);  ta.set(0, 0, 0, ma);
        let mut tb = Cube::new(0);  tb.set(0, 0, 0, mb);
        let e: crate::math3::Vec3 = [1.0, 0.0, 0.0];
        let du = du_x(g, 0, &tk, &a, &b, e, 10.0, 1, &ta, &tb);
        abs_close(du, 0.0, 1e-40);
    }
}
