//! Full Two-Body Problem (F2BP) equations of motion.
//!
//! Provides `hou_ode` — the continuous ODE right-hand side used by all
//! non-symplectic integrators (RK4, RK7/8, ABM).  Generic over [`Scalar`] so
//! `Dual` types can propagate for auto-differentiation.
//!
//! # State vector (30 elements, A frame)
//! | Range | Quantity | Units |
//! |---|---|---|
//! | 0–2   | r   — relative position            | km     |
//! | 3–5   | v   — relative velocity            | km/s   |
//! | 6–8   | ωc  — primary angular velocity     | rad/s  |
//! | 9–11  | ωs  — secondary ang. vel. (A frame)| rad/s  |
//! | 12–20 | Cc  — inertial-to-A, row-major     | —      |
//! | 21–29 | C   — B-to-A, row-major            | —      |

use crate::inertia::{dt_dc, inertia_rot};
use crate::math3::*;
use crate::orbit::{kepler, kepler2cart};
use crate::potential::{du_c, du_x};
use crate::types::{Cube, Params};
use num_traits::NumCast;

// ── helper: unpack state ──────────────────────────────────────────────────────

fn unpack<T: Scalar>(x: &[T; 30]) -> (Vec3<T>, Vec3<T>, Vec3<T>, Vec3<T>, Mat3<T>, Mat3<T>) {
    let r  = [x[0],  x[1],  x[2]];
    let v  = [x[3],  x[4],  x[5]];
    let wc = [x[6],  x[7],  x[8]];
    let ws = [x[9],  x[10], x[11]];
    let cc: Mat3<T> = [[x[12], x[13], x[14]],
                       [x[15], x[16], x[17]],
                       [x[18], x[19], x[20]]];
    let c:  Mat3<T> = [[x[21], x[22], x[23]],
                       [x[24], x[25], x[26]],
                       [x[27], x[28], x[29]]];
    (r, v, wc, ws, cc, c)
}

// ── perturbation forces ───────────────────────────────────────────────────────

/// Legacy Hill-equation solar gravity (circular orbit approximation).
fn hill_solar_grav<T: Scalar>(cc: Mat3<T>, pos: Vec3<T>, vel: Vec3<T>, n_mm: f64) -> Vec3<T> {
    let to_t = |x: f64| -> T { <T as NumCast>::from(x).unwrap() };
    let n2    = n_mm * n_mm;
    let pos_n = mat_vec(cc, pos);
    let vel_n = mat_vec(cc, vel);
    let acc_n = [to_t( 3.0 * n2) * pos_n[0] + to_t(2.0 * n_mm) * vel_n[1],
                 to_t(-2.0 * n_mm) * vel_n[0],
                 to_t(-n2) * pos_n[2]];
    mat_vec(transpose(cc), acc_n)
}

/// Gravitational perturbation from a 3rd spherical body.
fn grav_3bp<T: Scalar>(r_s_n: Vec3<T>, cc: Mat3<T>, pos: Vec3<T>,
                        nu: T, g: T, mplanet: T) -> Vec3<T> {
    let eps    = <T as NumCast>::from(1e-30_f64).unwrap();
    let r_s    = mat_vec(transpose(cc), r_s_n);
    let ra     = sub_v(r_s, scale_v(T::one() - nu, pos));
    let rb     = sub_v(r_s, scale_v(-nu, pos));
    let ra_mag = norm(ra).max(eps);
    let rb_mag = norm(rb).max(eps);
    scale_v(g * mplanet,
            sub_v(scale_v(T::one() / ra_mag.powi(3), ra),
                  scale_v(T::one() / rb_mag.powi(3), rb)))
}

/// Solar gravitational perturbation (heliocentric orbit).
fn solar_accel<T: Scalar>(r_sun_n: Vec3<T>, cc: Mat3<T>, pos: Vec3<T>,
                           nu: T, g: T, msun: T) -> Vec3<T> {
    grav_3bp(r_sun_n, cc, pos, nu, g, msun)
}

/// Tidal torques and orbit perturbation due to internal dissipation.
fn md_tidal_torque<T: Scalar>(
    pos: Vec3<T>, vel: Vec3<T>, w1: Vec3<T>, w2: Vec3<T>,
    cc: Mat3<T>, _c: Mat3<T>,
    params: &Params<T>,
) -> (Vec3<T>, Vec3<T>, Vec3<T>)
{
    let to_t = |x: f64| -> T { <T as NumCast>::from(x).unwrap() };
    let eps  = to_t(1e-30_f64);
    let pi   = to_t(std::f64::consts::PI);

    let r_mag   = norm(pos).max(eps);
    let v_mag   = norm(vel).max(eps);
    let rhat    = scale_v(T::one() / r_mag, mat_vec(cc, pos));
    let _vhat   = scale_v(T::one() / v_mag, mat_vec(cc, vel));
    let rcv     = mat_vec(cc, cross(pos, vel));
    let rcv_mag = norm(rcv).max(eps);
    let _zhat   = scale_v(T::one() / rcv_mag, rcv);
    let wsys    = scale_v(T::one() / (r_mag * r_mag), rcv);

    let w1_n  = mat_vec(cc, w1);
    let w2_n  = mat_vec(cc, w2);
    let phid1 = sub_v(w1_n, wsys);
    let phid2 = sub_v(w2_n, wsys);

    let g1_vec = sub_v(phid1, scale_v(dot(phid1, rhat), rhat));
    let g1_mag = norm(g1_vec).max(eps);
    let g1_hat = scale_v(-T::one() / g1_mag, g1_vec);
    let g2_vec = sub_v(phid2, scale_v(dot(phid2, rhat), rhat));
    let g2_mag = norm(g2_vec).max(eps);
    let g2_hat = scale_v(-T::one() / g2_mag, g2_vec);

    let mc = params.ta.get(0, 0, 0);
    let ms = params.tb.get(0, 0, 0);
    let r6 = r_mag.powi(6);

    // Compute pure-f64 constant factors first, then promote once
    let lf1 = 1.5 * params.love1
        * (3.0 / (4.0 * std::f64::consts::PI * params.rho_a)).powi(2)
        / params.refrad1;
    let lf2 = 1.5 * params.love2
        * (3.0 / (4.0 * std::f64::consts::PI * params.rho_b)).powi(2)
        / params.refrad2;

    let g1 = to_t(lf1 * (2.0 * params.eps1).sin()) * params.g * mc * mc * ms * ms / r6;
    let g2 = to_t(lf2 * (2.0 * params.eps2).sin()) * params.g * mc * mc * ms * ms / r6;

    let del1 = g1 * (to_t(6.0) / (pi * params.g * to_t(params.rho_a))).sqrt() / params.ia[2];
    let del2 = g2 * (to_t(6.0) / (pi * params.g * to_t(params.rho_b))).sqrt() / params.ib[2];

    let tt1 = if norm(phid1) > del1.abs() {
        scale_v(g1, mat_vec(transpose(cc), g1_hat))
    } else {
        let f = norm(phid1)
            * (to_t(6.0) / (pi * params.g * to_t(params.rho_a))).powf(to_t(-0.5))
            * params.ia[2];
        scale_v(f, mat_vec(transpose(cc), g1_hat))
    };
    let tt2 = if norm(phid2) > del2.abs() {
        scale_v(g2, mat_vec(transpose(cc), g2_hat))
    } else {
        let f = norm(phid2)
            * (to_t(6.0) / (pi * params.g * to_t(params.rho_b))).powf(to_t(-0.5))
            * params.ib[2];
        scale_v(f, mat_vec(transpose(cc), g2_hat))
    };

    let tt_orbit = scale_v(T::one() / params.m,
        scale_v(T::one() / (r_mag * r_mag), cross(add_v(tt1, tt2), pos)));
    (tt1, tt2, tt_orbit)
}

// ── hou_ode ───────────────────────────────────────────────────────────────────

/// Continuous equations of motion — the ODE right-hand side used by all
/// non-symplectic integrators (RK4, RK7/8, ABM).
///
/// Generic over `T` so dual-number types can propagate for auto-differentiation.
/// With `T = f64` (default) this is identical to the original f64 behaviour.
pub fn hou_ode<T: Scalar>(x: [T; 30], time: f64, p: &Params<T>) -> [T; 30] {
    let to_t = |v: f64| -> T { <T as NumCast>::from(v).unwrap() };
    let (r, v, wc, ws, cc, c) = unpack(&x);

    let r_mag = norm(r).max(to_t(1e-30_f64));
    let e     = scale_v(T::one() / r_mag, r);

    let ic_inv   = diag([T::one() / p.ia[0], T::one() / p.ia[1], T::one() / p.ia[2]]);
    let is_inv   = diag([T::one() / p.ib[0], T::one() / p.ib[1], T::one() / p.ib[2]]);
    let is_inv_c = mat_mul(mat_mul(c, is_inv), transpose(c));

    let wc_tilde = tilde(wc);
    let ws_s     = mat_vec(transpose(c), ws);
    let ws_tilde = tilde(ws_s);

    let tbp = inertia_rot(c, p.n, &p.tb);

    // ── 3rd-body flyby ────────────────────────────────────────────────────
    let (acc_3bp, _m_sa, _m_sb) = if p.flyby_toggle == 1 {
        let g_f64  = p.g.to_f64().unwrap();
        let f0_hyp = kepler(p.n_hyp, time, p.e_hyp, p.tau_hyp);
        let xs     = kepler2cart(p.a_hyp, p.e_hyp, p.i_hyp, p.raan_hyp, p.om_hyp,
                                 f0_hyp, g_f64, p.mplanet);
        let r_s_n: Vec3<T> = [to_t(xs[0]), to_t(xs[1]), to_t(xs[2])];
        let acc = grav_3bp(r_s_n, cc, r, p.nu, p.g, to_t(p.mplanet));

        let r_sa    = add_v(mat_vec(transpose(cc), r_s_n), scale_v( p.nu, r));
        let r_sb    = add_v(mat_vec(transpose(cc), r_s_n), scale_v(-T::one() + p.nu, r));
        let rsa_mag = norm(r_sa).max(to_t(1e-30_f64));
        let rsb_mag = norm(r_sb).max(to_t(1e-30_f64));
        let e_sa    = scale_v(T::one() / rsa_mag, r_sa);
        let e_sb    = scale_v(T::one() / rsb_mag, r_sb);

        let mut ts_cube: Cube<T> = Cube::new(p.n);
        ts_cube.set(0, 0, 0, to_t(p.mplanet));

        let du_sa: Vec3<T> = [
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sa, rsa_mag, 0, &p.ta, &ts_cube),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sa, rsa_mag, 1, &p.ta, &ts_cube),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sa, rsa_mag, 2, &p.ta, &ts_cube),
        ];
        let du_sb: Vec3<T> = [
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sb, rsb_mag, 0, &tbp, &ts_cube),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sb, rsb_mag, 1, &tbp, &ts_cube),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sb, rsb_mag, 2, &tbp, &ts_cube),
        ];
        let m_sa = cross(r_sa, du_sa);
        let m_sb = cross(r_sb, du_sb);
        (acc, m_sa, m_sb)
    } else {
        (zero_v::<T>(), zero_v::<T>(), zero_v::<T>())
    };

    // ── heliocentric perturbation ─────────────────────────────────────────
    let (acc_solar, _m_suna, _m_sunb) = if p.helio_toggle == 1 {
        let g_f64 = p.g.to_f64().unwrap();
        let f0_h  = kepler(p.n_helio, time, p.e_helio, p.tau_helio);
        let xs    = kepler2cart(p.a_helio, p.e_helio, p.i_helio, p.raan_helio, p.om_helio,
                                f0_h, g_f64, p.msolar);
        let r_sun_n: Vec3<T> = [to_t(xs[0]), to_t(xs[1]), to_t(xs[2])];
        let acc = solar_accel(r_sun_n, cc, r, p.nu, p.g, to_t(p.msolar));

        let r_suna    = add_v(mat_vec(transpose(cc), r_sun_n), scale_v( p.nu, r));
        let r_sunb    = add_v(mat_vec(transpose(cc), r_sun_n), scale_v(-T::one() + p.nu, r));
        let rsuna_mag = norm(r_suna).max(to_t(1e-30_f64));
        let rsunb_mag = norm(r_sunb).max(to_t(1e-30_f64));
        let e_suna    = scale_v(T::one() / rsuna_mag, r_suna);
        let e_sunb    = scale_v(T::one() / rsunb_mag, r_sunb);

        let mut tsun: Cube<T> = Cube::new(p.n);
        tsun.set(0, 0, 0, to_t(p.msolar));

        let du_suna: Vec3<T> = [
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_suna, rsuna_mag, 0, &p.ta, &tsun),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_suna, rsuna_mag, 1, &p.ta, &tsun),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_suna, rsuna_mag, 2, &p.ta, &tsun),
        ];
        let du_sunb: Vec3<T> = [
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sunb, rsunb_mag, 0, &tbp, &tsun),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sunb, rsunb_mag, 1, &tbp, &tsun),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sunb, rsunb_mag, 2, &tbp, &tsun),
        ];
        let m_suna = cross(r_suna, du_suna);
        let m_sunb = cross(r_sunb, du_sunb);
        (acc, m_suna, m_sunb)
    } else {
        (zero_v::<T>(), zero_v::<T>(), zero_v::<T>())
    };

    // ── legacy solar gravity ──────────────────────────────────────────────
    let sg_acc = if p.sg_toggle == 1 {
        hill_solar_grav(cc, r, v, p.mean_motion)
    } else {
        zero_v::<T>()
    };

    // ── tidal torques ─────────────────────────────────────────────────────
    let (tt_1, tt_2, tt_orbit) = if p.tt_toggle == 1 {
        md_tidal_torque(r, v, wc, ws, cc, c, p)
    } else {
        (zero_v::<T>(), zero_v::<T>(), zero_v::<T>())
    };

    // ── mutual potential partials ─────────────────────────────────────────
    let dts: [[Cube<T>; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| dt_dc(i, j, c, p.n, &p.tb))
    });

    let du_dr: Vec3<T> = [
        du_x(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, 0, &p.ta, &tbp),
        du_x(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, 1, &p.ta, &tbp),
        du_x(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, 2, &p.ta, &tbp),
    ];

    let du_cols: [Vec3<T>; 3] = std::array::from_fn(|col| {
        [
            du_c(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, &p.ta, &dts[0][col]),
            du_c(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, &p.ta, &dts[1][col]),
            du_c(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, &p.ta, &dts[2][col]),
        ]
    });

    // ── angular momenta ───────────────────────────────────────────────────
    let la = mat_vec(diag(p.ia), wc);
    let lb = mat_vec(c, mat_vec(diag(p.ib), ws_s));

    // ── torques ───────────────────────────────────────────────────────────
    let mb = scale_v(-T::one(), add_v(add_v(
        cross(col(c, 0), du_cols[0]),
        cross(col(c, 1), du_cols[1])),
        cross(col(c, 2), du_cols[2])));
    let ma = sub_v(cross(r, du_dr), mb);

    // ── equations of motion ───────────────────────────────────────────────
    let rd  = add_v(cross(r, wc), v);
    let vd  = sub_v(
        add_v(add_v(add_v(add_v(cross(v, wc), sg_acc), acc_3bp), acc_solar), tt_orbit),
        scale_v(T::one() / p.m, du_dr),
    );
    let cdc = mat_mul(cc, wc_tilde);
    let cd  = sub_m(mat_mul(c, ws_tilde), mat_mul(wc_tilde, c));
    let cdt = sub_m(mat_mul(scale_m(-T::one(), ws_tilde), transpose(c)),
                    mat_mul(scale_m(-T::one(), transpose(c)), wc_tilde));

    let wcd = mat_vec(ic_inv,
        sub_v(add_v(cross(la, wc), ma), tt_1));
    let wsd = mat_vec(is_inv_c,
        sub_v(sub_v(sub_v(add_v(cross(lb, wc), mb),
                          mat_vec(mat_mul(cd, diag(p.ib)), mat_vec(transpose(c), ws))),
                    mat_vec(mat_mul(c, mat_mul(diag(p.ib), cdt)), ws)),
              tt_2));

    // ── pack output ───────────────────────────────────────────────────────
    [rd[0],  rd[1],  rd[2],
     vd[0],  vd[1],  vd[2],
     wcd[0], wcd[1], wcd[2],
     wsd[0], wsd[1], wsd[2],
     cdc[0][0], cdc[0][1], cdc[0][2],
     cdc[1][0], cdc[1][1], cdc[1][2],
     cdc[2][0], cdc[2][1], cdc[2][2],
     cd[0][0],  cd[0][1],  cd[0][2],
     cd[1][0],  cd[1][1],  cd[1][2],
     cd[2][0],  cd[2][1],  cd[2][2]]
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::hou_ode;
    use crate::coefficients::{a_calc, b_calc, tk_calc};
    use crate::types::{Cube, Params};

    /// Minimal Params for monopole (n=0) with all perturbations disabled.
    fn point_mass_params(ma: f64, mb: f64) -> Params {
        let g     = 6.674e-20_f64;
        let m_red = ma * mb / (ma + mb);
        let nu    = mb / (ma + mb);
        let n     = 0_usize;
        let mut ta = Cube::new(n);  ta.set(0, 0, 0, ma);
        let mut tb = Cube::new(n);  tb.set(0, 0, 0, mb);
        let mut p = Params {
            g, m: m_red, nu,
            ta, tb,
            ia: [1.0, 1.0, 1.0],
            ib: [1.0, 1.0, 1.0],
            n, tk: tk_calc(n), a: a_calc(n), b: b_calc(n),
            flyby_toggle: 0, helio_toggle: 0, sg_toggle: 0, tt_toggle: 0,
            mplanet: 0.0,
            a_hyp: -1.0, e_hyp: 1.5, i_hyp: 0.0, raan_hyp: 0.0,
            om_hyp: 0.0, tau_hyp: 0.0, n_hyp: 0.0,
            msolar: 0.0,
            a_helio: 1.0, e_helio: 0.0, i_helio: 0.0, raan_helio: 0.0,
            om_helio: 0.0, tau_helio: 0.0, n_helio: 0.0,
            sol_rad: 0.0, au_def: 1.496e8, mean_motion: 0.0,
            love1: 0.0, love2: 0.0, refrad1: 1.0, refrad2: 1.0,
            rho_a: 1e12, rho_b: 1e12, eps1: 0.0, eps2: 0.0,
            ida: [[0.0; 3]; 3],
            idb: [[0.0; 3]; 3],
            msun: 2e30,
        };
        p.compute_lgvi_inertia();
        p
    }

    /// Build state with r=[a,0,0], v=[0,v,0], wc=ws=0, Cc=C=I
    fn circular_state(a: f64, v: f64) -> [f64; 30] {
        let i = [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0]; // row-major I₃
        [a, 0.0, 0.0,
         0.0, v, 0.0,
         0.0, 0.0, 0.0,   // wc
         0.0, 0.0, 0.0,   // ws
         i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8],  // Cc
         i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]]  // C
    }

    #[test]
    fn hou_ode_point_mass_centripetal_force() {
        // r=[a,0,0], wc=0 → rd = v, vd[0] = −G(Ma+Mb)/a²
        let ma = 5.0e11_f64;
        let mb = 2.0e11_f64;
        let g  = 6.674e-20_f64;
        let a  = 10.0_f64;
        let mu = g * (ma + mb);
        let v  = (mu / a).sqrt();
        let p  = point_mass_params(ma, mb);
        let xd = hou_ode(circular_state(a, v), 0.0, &p);

        // rd = [0, v, 0]  (no rotating frame: wc = 0)
        assert!((xd[0]).abs() < 1e-12, "rd_x ≠ 0: {}", xd[0]);
        assert!((xd[1] - v).abs() < 1e-10, "rd_y ≠ v_circ: {}", xd[1]);

        // vd[0] = −G(Ma+Mb)/a²  (centripetal, inward)
        let expected = -mu / (a * a);
        assert!((xd[3] - expected).abs() / expected.abs() < 1e-10,
                "vd_x expected {expected:.6e}, got {:.6e}", xd[3]);

        // vd[1] = 0 (no tangential force for aligned monopole)
        assert!(xd[4].abs() < 1e-30, "vd_y ≠ 0: {:.6e}", xd[4]);
    }

    #[test]
    fn hou_ode_point_mass_no_torques() {
        // n=0 expansion: du/dC = 0 → wċ = wṡ = Ċc = Ċ = 0
        let ma = 5.0e11_f64;
        let mb = 2.0e11_f64;
        let g  = 6.674e-20_f64;
        let a  = 10.0_f64;
        let v  = (g * (ma + mb) / a).sqrt();
        let p  = point_mass_params(ma, mb);
        let xd = hou_ode(circular_state(a, v), 0.0, &p);

        // indices 6..12 are angular velocity derivatives (should all be 0)
        for i in 6..12 {
            assert!(xd[i].abs() < 1e-30,
                    "xd[{i}] = {:.6e} should be 0 (no torques for n=0)", xd[i]);
        }
        // indices 12..30 are rotation-matrix derivatives (should all be 0)
        for i in 12..30 {
            assert!(xd[i].abs() < 1e-30,
                    "xd[{i}] = {:.6e} should be 0 (wc = ws = 0)", xd[i]);
        }
    }
}
