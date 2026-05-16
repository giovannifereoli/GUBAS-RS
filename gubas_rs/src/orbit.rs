//! Keplerian orbit utilities: Kepler's equation solver and element-to-Cartesian conversion.
//!
//! Handles elliptic (`e < 1`) and hyperbolic (`e > 1`) orbits via Newton's method.
//! Used by the ODE and integrators when flyby or heliocentric perturbations are active.

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

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {b:.6e}, got {a:.6e}");
    }

    // Earth-like gravitational parameter for dimensioned tests
    const MU: f64 = 398_600.4418_f64; // km³/s²
    const G: f64  = 6.674e-20_f64;    // km³/(kg·s²)
    const MB: f64 = MU / G;

    // ── kepler ───────────────────────────────────────────────────────────────

    #[test]
    fn kepler_elliptic_at_periapsis() {
        // t = tau → M = 0 → E = 0 → f = 0
        let f = kepler(0.001, 0.0, 0.5, 0.0);
        close(f, 0.0, 0.01);
    }

    #[test]
    fn kepler_circular_quarter_period() {
        // Circular (ecc=0) at t = T/4: M = π/2 → f = π/2
        let n_motion = 2.0 * std::f64::consts::PI;
        let f = kepler(n_motion, 0.25, 0.0, 0.0);
        close(f, std::f64::consts::FRAC_PI_2, 0.01);
    }

    #[test]
    fn kepler_circular_half_period() {
        // At t = T/2: M = π → f = π
        let n_motion = 2.0 * std::f64::consts::PI;
        let f = kepler(n_motion, 0.5, 0.0, 0.0);
        close(f, std::f64::consts::PI, 0.01);
    }

    // ── kepler2cart ──────────────────────────────────────────────────────────

    #[test]
    fn kepler2cart_circular_radius() {
        // |r| = a for circular orbit
        let a = 10.0_f64;
        let s = kepler2cart(a, 0.0, 0.0, 0.0, 0.0, 0.0, G, MB);
        let r = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]).sqrt();
        close(r, a, 1e-10);
    }

    #[test]
    fn kepler2cart_circular_r_dot_v_zero() {
        // Velocity is perpendicular to position for circular orbit
        let a = 10.0_f64;
        let s = kepler2cart(a, 0.0, 0.0, 0.0, 0.0, 0.0, G, MB);
        let rdotv = s[0]*s[3] + s[1]*s[4] + s[2]*s[5];
        close(rdotv, 0.0, 1e-10);
    }

    #[test]
    fn kepler2cart_circular_vis_viva() {
        // |v|² = μ/a for circular orbit
        let a = 10.0_f64;
        let s = kepler2cart(a, 0.0, 0.0, 0.0, 0.0, 0.0, G, MB);
        let v2 = s[3]*s[3] + s[4]*s[4] + s[5]*s[5];
        close(v2, MU / a, 1e-6);
    }

    #[test]
    fn kepler2cart_periapsis_radius() {
        // At f0=0: |r| = a*(1−e)
        let a = 20.0_f64;
        let e = 0.5_f64;
        let s = kepler2cart(a, e, 0.0, 0.0, 0.0, 0.0, G, MB);
        let r = (s[0]*s[0] + s[1]*s[1] + s[2]*s[2]).sqrt();
        close(r, a * (1.0 - e), 1e-10);
    }

    #[test]
    fn kepler2cart_periapsis_r_dot_v_zero() {
        // Velocity perpendicular to position at periapsis
        let a = 20.0_f64;
        let e = 0.5_f64;
        let s = kepler2cart(a, e, 0.0, 0.0, 0.0, 0.0, G, MB);
        let rdotv = s[0]*s[3] + s[1]*s[4] + s[2]*s[5];
        close(rdotv, 0.0, 1e-9);
    }
}
