// orbit.rs — Keplerian orbit utilities
//
// Mirrors `double kepler(...)` and `vec kepler2cart(...)` from the C++ code.

use crate::math3::{add_v, cross, scale_v};

// ── kepler ────────────────────────────────────────────────────────────────────

/// Solve Kepler's equation for true anomaly `theta` given:
///   `n`   = mean motion (rad/s),
///   `t`   = current time (s),
///   `ecc` = eccentricity,
///   `tau` = time of periapsis passage (s).
///
/// Handles both elliptical (ecc < 1) and hyperbolic (ecc > 1) orbits via
/// Newton's method.  Mirrors `double kepler(double* n_hyp, double t, ...)`.
pub fn kepler(n_val: f64, t: f64, ecc: f64, tau: f64) -> f64 {
    let mean_anom = n_val * (t - tau);
    let tol = 0.001_f64;

    if ecc > 1.0 {
        // Hyperbolic case
        let max_h = (1.0 / ecc).acos();
        let mut h = mean_anom.clamp(-max_h, max_h);
        let mut f = mean_anom - ecc * h.sinh() + h;
        while f.abs() > tol {
            let df = -ecc * h.cosh() + 1.0;
            h -= f / df;
            f = mean_anom - ecc * h.sinh() + h;
        }
        2.0 * (((ecc + 1.0) / (ecc - 1.0)).sqrt() * (h / 2.0).tanh()).atan()
    } else {
        // Elliptical case
        let mut e_anom = mean_anom;
        let mut f = e_anom - mean_anom - ecc * e_anom.sin();
        while f.abs() > tol {
            let df = 1.0 - ecc * e_anom.cos();
            e_anom -= f / df;
            f = e_anom - mean_anom - ecc * e_anom.sin();
        }
        2.0 * (((1.0 + ecc) / (1.0 - ecc)).sqrt() * (e_anom / 2.0).tan()).atan()
    }
}

// ── kepler2cart ───────────────────────────────────────────────────────────────

/// Convert classical Keplerian elements to Cartesian state [r(3), v(3)] in km
/// and km/s.
///
/// Arguments (pointers in C++, plain values here):
///   `a`    = semi-major axis (km),
///   `ecc`  = eccentricity,
///   `inc`  = inclination (rad),
///   `raan` = right ascension of ascending node (rad),
///   `om`   = argument of periapsis (rad),
///   `f0`   = true anomaly (rad),
///   `g`    = gravitational parameter G (km³/kg/s²),
///   `mu_m` = mass of central body (kg).
///
/// Returns `[rx, ry, rz, vx, vy, vz]`.
/// Mirrors `vec kepler2cart(...)`.
pub fn kepler2cart(a: f64, ecc: f64, inc: f64, raan: f64, om: f64,
                   f0: f64, g: f64, mu_m: f64) -> [f64; 6] {
    let mu = g * mu_m;
    // Basis vectors
    let x_hat: [f64; 3] = [1.0, 0.0, 0.0];
    let y_hat: [f64; 3] = [0.0, 1.0, 0.0];
    let z_hat: [f64; 3] = [0.0, 0.0, 1.0];

    let n_omega = add_v(scale_v(raan.cos(), x_hat), scale_v(raan.sin(), y_hat));
    let n_perp = add_v(
        add_v(scale_v(-inc.cos() * raan.sin(), x_hat),
              scale_v( inc.cos() * raan.cos(), y_hat)),
        scale_v(inc.sin(), z_hat),
    );

    let e_hat = add_v(scale_v(om.cos(), n_omega), scale_v(om.sin(), n_perp));
    let e_perp = add_v(scale_v(-om.sin(), n_omega), scale_v(om.cos(), n_perp));

    let p = a * (1.0 - ecc * ecc);
    let r_mag = p / (1.0 + ecc * f0.cos());
    let r_vec = scale_v(r_mag, add_v(scale_v(f0.cos(), e_hat), scale_v(f0.sin(), e_perp)));
    let r_hat = scale_v(1.0 / r_mag, r_vec);

    let h_hat = cross(e_hat, e_perp); // unit angular momentum
    let r_perp_raw = cross(h_hat, r_hat);
    let r_perp_mag = crate::math3::norm(r_perp_raw);
    let r_perp = scale_v(1.0 / r_perp_mag, r_perp_raw);

    let denom = (1.0 + 2.0 * ecc * f0.cos() + ecc * ecc).sqrt();
    let sin_gamma = ecc * f0.sin() / denom;
    let cos_gamma = (1.0 + ecc * f0.cos()) / denom;
    let v_mag = (mu / p * (1.0 + 2.0 * ecc * f0.cos() + ecc * ecc)).sqrt();
    let v_vec = scale_v(v_mag, add_v(scale_v(sin_gamma, r_hat), scale_v(cos_gamma, r_perp)));

    [r_vec[0], r_vec[1], r_vec[2], v_vec[0], v_vec[1], v_vec[2]]
}
