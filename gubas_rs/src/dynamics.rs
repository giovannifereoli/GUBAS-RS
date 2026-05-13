// dynamics.rs — Equations of motion and perturbation forces
//
// The ODE used by RK4, RK7/8, and ABM integrators.
// State vector layout (30 elements, A-frame):
//   [0..2]  r     relative position (km)
//   [3..5]  v     relative velocity (km/s)
//   [6..8]  wc    primary angular velocity (rad/s)
//   [9..11] ws    secondary angular velocity in A frame (rad/s)
//   [12..20] Cc   inertial-to-A rotation matrix, row-major
//   [21..29] C    B-to-A rotation matrix, row-major

use crate::inertia::{dt_dc, inertia_rot};
use crate::math3::*;
use crate::orbit::{kepler, kepler2cart};
use crate::potential::du_x;
use crate::potential::du_c;
use crate::types::Params;

// ── helper: unpack state ──────────────────────────────────────────────────────

fn unpack(x: &[f64; 30]) -> (Vec3, Vec3, Vec3, Vec3, Mat3, Mat3) {
    let r  = [x[0],  x[1],  x[2]];
    let v  = [x[3],  x[4],  x[5]];
    let wc = [x[6],  x[7],  x[8]];
    let ws = [x[9],  x[10], x[11]];
    // Rotation matrices are stored row-major in the state vector
    let cc: Mat3 = [[x[12], x[13], x[14]],
                    [x[15], x[16], x[17]],
                    [x[18], x[19], x[20]]];
    let c:  Mat3 = [[x[21], x[22], x[23]],
                    [x[24], x[25], x[26]],
                    [x[27], x[28], x[29]]];
    (r, v, wc, ws, cc, c)
}

// ── perturbation forces ───────────────────────────────────────────────────────

/// Legacy Hill-equation solar gravity (circular orbit approximation).
/// Mirrors `void hill_solar_grav(...)`.
fn hill_solar_grav(cc: Mat3, pos: Vec3, vel: Vec3, n_mm: f64) -> Vec3 {
    // CHp: diagonal acceleration coefficients (3n², 0, -n²)
    // CHv: Coriolis coupling
    let n2 = n_mm * n_mm;
    let pos_n = mat_vec(cc, pos); // position in N frame
    let vel_n = mat_vec(cc, vel); // velocity in N frame
    let acc_n = [3.0 * n2 * pos_n[0] + 2.0 * n_mm * vel_n[1],
                 -2.0 * n_mm * vel_n[0],
                 -n2 * pos_n[2]];
    // Transform back to A frame
    mat_vec(transpose(cc), acc_n)
}

/// Gravitational perturbation from a 3rd spherical body.
/// Mirrors `void grav_3BP(...)`.
fn grav_3bp(r_s_n: Vec3, cc: Mat3, pos: Vec3, nu: f64, g: f64, mplanet: f64) -> Vec3 {
    // Convert R_s to A frame
    let r_s = mat_vec(transpose(cc), r_s_n);
    let r = pos;
    let ra = sub_v(r_s, scale_v(1.0 - nu, r));
    let rb = sub_v(r_s, scale_v(-nu,      r)); // = r_s + nu*r
    // Defensive: add small epsilon to avoid divide-by-zero if bodies coincide
    let ra_mag = norm(ra).max(1e-30);
    let rb_mag = norm(rb).max(1e-30);
    scale_v(g * mplanet,
            sub_v(scale_v(1.0 / ra_mag.powi(3), ra),
                  scale_v(1.0 / rb_mag.powi(3), rb)))
}

/// Solar gravitational perturbation (heliocentric orbit).
/// Same form as grav_3bp but uses msun.
fn solar_accel(r_sun_n: Vec3, cc: Mat3, pos: Vec3, nu: f64, g: f64, msun: f64) -> Vec3 {
    grav_3bp(r_sun_n, cc, pos, nu, g, msun)
}

/// Tidal torques and orbit perturbation due to internal dissipation.
/// Mirrors `void md_tidal_torque(...)`.
fn md_tidal_torque(
    pos: Vec3, vel: Vec3, w1: Vec3, w2: Vec3,
    cc: Mat3, _c: Mat3,
    params: &Params,
) -> (Vec3, Vec3, Vec3) // (tt_1, tt_2, tt_orbit)
{
    let r_mag = norm(pos).max(1e-30);
    let v_mag = norm(vel).max(1e-30);
    let rhat = scale_v(1.0 / r_mag, mat_vec(cc, pos));
    let _vhat = scale_v(1.0 / v_mag, mat_vec(cc, vel));
    let rcv = mat_vec(cc, cross(pos, vel));
    let rcv_mag = norm(rcv).max(1e-30);
    let _zhat = scale_v(1.0 / rcv_mag, rcv);
    let wsys = scale_v(1.0 / (r_mag * r_mag), rcv);

    let w1_n = mat_vec(cc, w1);
    let w2_n = mat_vec(cc, w2);
    let phid1 = sub_v(w1_n, wsys);
    let phid2 = sub_v(w2_n, wsys);

    let g1_vec = sub_v(phid1, scale_v(dot(phid1, rhat), rhat));
    let g1_mag = norm(g1_vec).max(1e-30);
    let g1_hat = scale_v(-1.0 / g1_mag, g1_vec);
    let g2_vec = sub_v(phid2, scale_v(dot(phid2, rhat), rhat));
    let g2_mag = norm(g2_vec).max(1e-30);
    let g2_hat = scale_v(-1.0 / g2_mag, g2_vec);

    let mc = params.ta.get(0, 0, 0);
    let ms = params.tb.get(0, 0, 0);
    let r6 = r_mag.powi(6);
    let pi = std::f64::consts::PI;

    let g1 = 1.5 * params.love1
        * (3.0 / (4.0 * pi * params.rho_a)).powi(2)
        * params.g * mc * mc * ms * ms
        * (2.0 * params.eps1).sin()
        / params.refrad1 / r6;
    let g2 = 1.5 * params.love2
        * (3.0 / (4.0 * pi * params.rho_b)).powi(2)
        * params.g * mc * mc * ms * ms
        * (2.0 * params.eps2).sin()
        / params.refrad2 / r6;

    let del1 = g1 * (6.0 / (pi * params.g * params.rho_a)).sqrt() / params.ia[2];
    let del2 = g2 * (6.0 / (pi * params.g * params.rho_b)).sqrt() / params.ib[2];

    let tt1 = if norm(phid1) > del1.abs() {
        scale_v(g1, mat_vec(transpose(cc), g1_hat))
    } else {
        scale_v(norm(phid1) * (6.0 / (pi * params.g * params.rho_a)).powf(-0.5) * params.ia[2],
                mat_vec(transpose(cc), g1_hat))
    };
    let tt2 = if norm(phid2) > del2.abs() {
        scale_v(g2, mat_vec(transpose(cc), g2_hat))
    } else {
        scale_v(norm(phid2) * (6.0 / (pi * params.g * params.rho_b)).powf(-0.5) * params.ib[2],
                mat_vec(transpose(cc), g2_hat))
    };

    let tt_orbit = scale_v(1.0 / params.m,
        scale_v(1.0 / (r_mag * r_mag), cross(add_v(tt1, tt2), pos)));
    (tt1, tt2, tt_orbit)
}

// ── hou_ode ───────────────────────────────────────────────────────────────────

/// Continuous equations of motion — the ODE right-hand side used by all
/// non-symplectic integrators (RK4, RK7/8, ABM).
///
/// All inputs/outputs are in the A (primary body-fixed) frame, km/kg/s.
/// Mirrors `mat hou_ode(mat x, mat t, parameters inputs)`.
pub fn hou_ode(x: [f64; 30], time: f64, p: &Params) -> [f64; 30] {
    let (r, v, wc, ws, cc, c) = unpack(&x);

    let r_mag = norm(r).max(1e-30);
    let e = scale_v(1.0 / r_mag, r); // unit position vector (A frame)

    // Inverse inertia matrices
    let ic_inv = diag([1.0 / p.ia[0], 1.0 / p.ia[1], 1.0 / p.ia[2]]);
    let is_inv = diag([1.0 / p.ib[0], 1.0 / p.ib[1], 1.0 / p.ib[2]]);
    let is_inv_c = mat_mul(mat_mul(c, is_inv), transpose(c));

    let wc_tilde = tilde(wc);
    let ws_s = mat_vec(transpose(c), ws); // secondary ω in B frame
    let ws_tilde = tilde(ws_s);

    // Rotate secondary inertia integrals into A frame
    let tbp = inertia_rot(c, p.n, &p.tb);

    // ── 3rd body flyby ────────────────────────────────────────────────────
    let (acc_3bp, _m_sa, _m_sb) = if p.flyby_toggle == 1 {
        let f0_hyp = kepler(p.n_hyp, time, p.e_hyp, p.tau_hyp);
        let xs = kepler2cart(p.a_hyp, p.e_hyp, p.i_hyp, p.raan_hyp, p.om_hyp,
                             f0_hyp, p.g, p.mplanet);
        let r_s_n = [xs[0], xs[1], xs[2]];
        let acc = grav_3bp(r_s_n, cc, r, p.nu, p.g, p.mplanet);

        // Torques from 3rd body on each body's inertia integrals
        let r_sa = add_v(mat_vec(transpose(cc), r_s_n), scale_v( p.nu, r));
        let r_sb = add_v(mat_vec(transpose(cc), r_s_n), scale_v(-1.0 + p.nu, r));
        let rsa_mag = norm(r_sa).max(1e-30);
        let rsb_mag = norm(r_sb).max(1e-30);
        let e_sa = scale_v(1.0 / rsa_mag, r_sa);
        let e_sb = scale_v(1.0 / rsb_mag, r_sb);

        // Sphere inertia (TS) used for 3rd body — set to Mplanet * identity order 0
        let mut ts_cube = crate::types::Cube::new(p.n);
        ts_cube.set(0, 0, 0, p.mplanet);

        let du_sa: Vec3 = [
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sa, rsa_mag, 0, &p.ta, &ts_cube),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sa, rsa_mag, 1, &p.ta, &ts_cube),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sa, rsa_mag, 2, &p.ta, &ts_cube),
        ];
        let du_sb: Vec3 = [
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sb, rsb_mag, 0, &tbp, &ts_cube),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sb, rsb_mag, 1, &tbp, &ts_cube),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sb, rsb_mag, 2, &tbp, &ts_cube),
        ];
        let m_sa = cross(r_sa, du_sa);
        let m_sb = cross(r_sb, du_sb);
        (acc, m_sa, m_sb)
    } else {
        (ZERO_V, ZERO_V, ZERO_V)
    };

    // ── heliocentric perturbation ─────────────────────────────────────────
    let (acc_solar, _m_suna, _m_sunb) = if p.helio_toggle == 1 {
        let f0_h = kepler(p.n_helio, time, p.e_helio, p.tau_helio);
        let xs = kepler2cart(p.a_helio, p.e_helio, p.i_helio, p.raan_helio, p.om_helio,
                             f0_h, p.g, p.msolar);
        let r_sun_n = [xs[0], xs[1], xs[2]];
        let acc = solar_accel(r_sun_n, cc, r, p.nu, p.g, p.msolar);

        let r_suna = add_v(mat_vec(transpose(cc), r_sun_n), scale_v( p.nu, r));
        let r_sunb = add_v(mat_vec(transpose(cc), r_sun_n), scale_v(-1.0 + p.nu, r));
        let rsuna_mag = norm(r_suna).max(1e-30);
        let rsunb_mag = norm(r_sunb).max(1e-30);
        let e_suna = scale_v(1.0 / rsuna_mag, r_suna);
        let e_sunb = scale_v(1.0 / rsunb_mag, r_sunb);

        let mut tsun = crate::types::Cube::new(p.n);
        tsun.set(0, 0, 0, p.msolar);

        let du_suna: Vec3 = [
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_suna, rsuna_mag, 0, &p.ta, &tsun),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_suna, rsuna_mag, 1, &p.ta, &tsun),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_suna, rsuna_mag, 2, &p.ta, &tsun),
        ];
        let du_sunb: Vec3 = [
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sunb, rsunb_mag, 0, &tbp, &tsun),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sunb, rsunb_mag, 1, &tbp, &tsun),
            du_x(p.g, p.n, &p.tk, &p.a, &p.b, e_sunb, rsunb_mag, 2, &tbp, &tsun),
        ];
        let m_suna = cross(r_suna, du_suna);
        let m_sunb = cross(r_sunb, du_sunb);
        (acc, m_suna, m_sunb)
    } else {
        (ZERO_V, ZERO_V, ZERO_V)
    };

    // ── legacy solar gravity ──────────────────────────────────────────────
    let sg_acc = if p.sg_toggle == 1 {
        hill_solar_grav(cc, r, v, p.mean_motion)
    } else {
        ZERO_V
    };

    // ── tidal torques ─────────────────────────────────────────────────────
    let (tt_1, tt_2, tt_orbit) = if p.tt_toggle == 1 {
        md_tidal_torque(r, v, wc, ws, cc, c, p)
    } else {
        (ZERO_V, ZERO_V, ZERO_V)
    };

    // ── mutual potential partials ─────────────────────────────────────────
    // Build dT/dC cubes for all 9 elements of C
    let dts: [[crate::types::Cube; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| dt_dc(i, j, c, p.n, &p.tb))
    });

    let du_dr: Vec3 = [
        du_x(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, 0, &p.ta, &tbp),
        du_x(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, 1, &p.ta, &tbp),
        du_x(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, 2, &p.ta, &tbp),
    ];

    // du/dC for each column (alpha, beta, gamma)
    let du_cols: [Vec3; 3] = std::array::from_fn(|col| {
        [
            du_c(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, &p.ta, &dts[0][col]),
            du_c(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, &p.ta, &dts[1][col]),
            du_c(p.g, p.n, &p.tk, &p.a, &p.b, e, r_mag, &p.ta, &dts[2][col]),
        ]
    });

    // ── angular momenta ───────────────────────────────────────────────────
    let la = mat_vec(diag(p.ia), wc);        // primary angular momentum (A)
    let lb = mat_vec(c, mat_vec(diag(p.ib), ws_s)); // secondary ang. mom. (A)

    // ── torques ───────────────────────────────────────────────────────────
    // Mb = -sum_col cross(C[:,col], du/dC[:,col])
    let mb = scale_v(-1.0, add_v(add_v(
        cross(col(c, 0), du_cols[0]),
        cross(col(c, 1), du_cols[1])),
        cross(col(c, 2), du_cols[2])));
    // Ma = cross(r, du/dr) - Mb
    let ma = sub_v(cross(r, du_dr), mb);

    // ── equations of motion (Maciejewski 1995 / Hou 2016, A frame) ───────
    let rd  = add_v(cross(r, wc), v);
    let vd  = sub_v(
        add_v(add_v(add_v(add_v(cross(v, wc), sg_acc), acc_3bp), acc_solar), tt_orbit),
        scale_v(1.0 / p.m, du_dr),
    );
    let cdc = mat_mul(cc, wc_tilde);
    let cd  = sub_m(mat_mul(c, ws_tilde), mat_mul(wc_tilde, c));
    let cdt = sub_m(mat_mul(scale_m(-1.0, ws_tilde), transpose(c)),
                    mat_mul(scale_m(-1.0, transpose(c)), wc_tilde));

    let wcd = mat_vec(ic_inv,
        sub_v(add_v(cross(la, wc), ma), tt_1));
    let wsd = mat_vec(is_inv_c,
        sub_v(sub_v(sub_v(add_v(cross(lb, wc), mb),
                          mat_vec(mat_mul(cd, diag(p.ib)), mat_vec(transpose(c), ws))),
                    mat_vec(mat_mul(c, mat_mul(diag(p.ib), cdt)), ws)),
              tt_2));

    // ── pack output ───────────────────────────────────────────────────────
    [rd[0], rd[1], rd[2],
     vd[0], vd[1], vd[2],
     wcd[0], wcd[1], wcd[2],
     wsd[0], wsd[1], wsd[2],
     cdc[0][0], cdc[0][1], cdc[0][2],
     cdc[1][0], cdc[1][1], cdc[1][2],
     cdc[2][0], cdc[2][1], cdc[2][2],
     cd[0][0],  cd[0][1],  cd[0][2],
     cd[1][0],  cd[1][1],  cd[1][2],
     cd[2][0],  cd[2][1],  cd[2][2]]
}
