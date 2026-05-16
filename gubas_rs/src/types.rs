//! Shared data structures: [`Cube`], [`Params`], and [`Initialization`].
//!
//! - [`Cube<T>`] — 3-D array replacing Armadillo's `cube`.  Flat row-major
//!   storage, indexed `(l, m, n)`.
//! - [`Params<T>`] — all read-only simulation parameters threaded through the
//!   ODE and integrators.  Generic over [`Scalar`] so `Dual` types can replace
//!   `f64` for auto-differentiation.
//! - [`promote_params`] — lift `Params<f64>` → `Params<U>` for dual-number AD.

use crate::math3::{Mat3, Scalar, Vec3};

// ── Cube ─────────────────────────────────────────────────────────────────────

/// A 3-D array of size `(order+1) × (order+1) × (order+1)`.
///
/// Indexed as `cube[l][m][n]` using row-major flattening:
///   flat_idx = l*(order+1)^2 + m*(order+1) + n
///
/// This replaces `arma::cube` from Armadillo.  Defaults to `f64` so existing
/// callers compile without changes.
#[derive(Clone, Debug)]
pub struct Cube<T = f64> {
    pub order: usize,
    data: Vec<T>,
}

impl<T: Scalar> Cube<T> {
    pub fn new(order: usize) -> Self {
        let sz = order + 1;
        Self { order, data: vec![T::zero(); sz * sz * sz] }
    }

    fn idx(&self, l: usize, m: usize, n: usize) -> usize {
        let sz = self.order + 1;
        l * sz * sz + m * sz + n
    }

    #[inline]
    pub fn get(&self, l: usize, m: usize, n: usize) -> T {
        self.data[self.idx(l, m, n)]
    }

    #[inline]
    pub fn set(&mut self, l: usize, m: usize, n: usize, val: T) {
        let i = self.idx(l, m, n);
        self.data[i] = val;
    }

    #[inline]
    pub fn add(&mut self, l: usize, m: usize, n: usize, val: T) {
        let i = self.idx(l, m, n);
        self.data[i] += val;
    }

    pub fn zeros(&mut self) {
        for v in &mut self.data { *v = T::zero(); }
    }

    /// Promote every element to a different scalar type via `NumCast`.
    ///
    /// Used to lift a `Cube<f64>` of constant parameters into `Cube<Dual>`
    /// (with eps = 0) so that dual-number ODE evaluations can flow through.
    pub fn promote<U: Scalar>(&self) -> Cube<U> {
        let mut out = Cube::new(self.order);
        for (i, v) in self.data.iter().enumerate() {
            out.data[i] = <U as num_traits::NumCast>::from(v.to_f64().unwrap()).unwrap();
        }
        out
    }
}

// CSV I/O is only needed for f64 cubes (file format is plain text floats).
impl Cube<f64> {
    /// Save to CSV in the same layout as Armadillo's `save(..., csv_ascii)`.
    ///
    /// The C++ stacks slices vertically: slice 0 (`T[:,:,0]`) then slice 1
    /// (`T[:,:,1]`) … so the CSV has `(order+1)^2` rows and `(order+1)` cols.
    /// Each row `s*(order+1) + l`, col `m` = `T[l][m][s]`.
    pub fn save_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let sz = self.order + 1;
        for s in 0..sz {
            for l in 0..sz {
                let vals: Vec<String> = (0..sz)
                    .map(|m| format!("{:.16e}", self.get(l, m, s)))
                    .collect();
                writeln!(f, "{}", vals.join(","))?;
            }
        }
        Ok(())
    }

    /// Load from the CSV format written by `save_csv` (or by the C++ tool).
    pub fn load_csv(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let rows: Vec<Vec<f64>> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|line| {
                line.split(',')
                    .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        let sz = rows[0].len();
        let order = sz - 1;
        let mut cube = Cube::new(order);
        for s in 0..sz {
            for l in 0..sz {
                let row_idx = s * sz + l;
                for m in 0..sz {
                    cube.set(l, m, s, rows[row_idx][m]);
                }
            }
        }
        Ok(cube)
    }
}

// ── Params ────────────────────────────────────────────────────────────────────

/// All read-only simulation parameters passed to the ODE and integrators.
///
/// Generic over `T` (defaults to `f64`) so that dual-number types can replace
/// f64 for auto-differentiation.  Only the physics fields that appear in
/// differentiated expressions are `T`; toggles and orbit parameters stay `f64`.
///
/// # Units
/// - Length: km
/// - Mass:   kg
/// - Time:   s (angles in radians)
#[derive(Clone)]
pub struct Params<T = f64> {
    // ── Gravity and masses ────────────────────────────────────────────────────
    /// Gravitational constant G (km³ kg⁻¹ s⁻²).
    pub g: T,
    /// Reduced mass μ = Mc·Ms/(Mc+Ms) (kg).
    pub m: T,
    /// Secondary mass fraction ν = Ms/(Mc+Ms) (dimensionless).
    pub nu: T,

    // ── Inertia integrals ─────────────────────────────────────────────────────
    /// T_{ijk} for primary (Didymos A) in body-fixed frame (kg·km^{i+j+k}).
    /// `ta.get(0,0,0)` = total mass Mc.  Not pre-rotated.
    pub ta: Cube<T>,
    /// T_{ijk} for secondary (Dimorphos) in body-fixed frame (kg·km^{i+j+k}).
    /// `tb.get(0,0,0)` = total mass Ms.  Not pre-rotated.
    pub tb: Cube<T>,
    /// Principal moments of inertia [Ixx, Iyy, Izz] for primary (kg·km²).
    pub ia: Vec3<T>,
    /// Principal moments of inertia [Ixx, Iyy, Izz] for secondary (kg·km²).
    pub ib: Vec3<T>,

    // ── Expansion order and combinatorial coefficients ─────────────────────────
    /// Truncation order of the mutual gravity expansion (integer l_max).
    pub n: usize,
    /// T_k combinatorial weights used in the mutual potential sum (Hou 2016).
    pub tk: Vec<Vec<f64>>,
    /// a_k coefficients in the mutual potential expansion.
    pub a: Vec<f64>,
    /// b_k coefficients in the mutual potential expansion.
    pub b: Vec<f64>,

    // ── Physics toggles (0 = off, 1 = on) ─────────────────────────────────────
    /// Enable 3rd-body flyby perturbation (hyperbolic orbit).
    pub flyby_toggle: i32,
    /// Enable heliocentric solar radiation pressure / tide (Keplerian approximation).
    pub helio_toggle: i32,
    /// Enable self-gravity spin–orbit coupling correction.
    pub sg_toggle: i32,
    /// Enable tidal torque dissipation (Kaula-style).
    pub tt_toggle: i32,

    // ── 3rd-body flyby orbital elements ───────────────────────────────────────
    /// Mass of the flyby perturber (kg).
    pub mplanet: f64,
    /// Semi-major axis of the flyby hyperbolic orbit (km, negative for hyperbola).
    pub a_hyp: f64,
    /// Eccentricity of the flyby hyperbolic orbit (> 1).
    pub e_hyp: f64,
    /// Inclination of the flyby hyperbolic orbit (rad).
    pub i_hyp: f64,
    /// RAAN of the flyby orbit (rad).
    pub raan_hyp: f64,
    /// Argument of periapsis of the flyby orbit (rad).
    pub om_hyp: f64,
    /// Time of periapsis passage of the flyby orbit (s).
    pub tau_hyp: f64,
    /// Mean motion of the flyby hyperbolic orbit n = sqrt(G·m_planet/|a|³) (rad/s).
    pub n_hyp: f64,

    // ── Heliocentric orbit parameters ──────────────────────────────────────────
    /// Mass of the heliocentric solar body used for tidal/radiation perturbation (kg).
    pub msolar: f64,
    /// Semi-major axis of the heliocentric orbit (km).
    pub a_helio: f64,
    /// Eccentricity of the heliocentric orbit.
    pub e_helio: f64,
    /// Inclination of the heliocentric orbit (rad).
    pub i_helio: f64,
    /// RAAN of the heliocentric orbit (rad).
    pub raan_helio: f64,
    /// Argument of periapsis of the heliocentric orbit (rad).
    pub om_helio: f64,
    /// Time of periapsis passage of the heliocentric orbit (s).
    pub tau_helio: f64,
    /// Mean motion of the heliocentric orbit (rad/s).
    pub n_helio: f64,

    // ── Simplified circular solar orbit (legacy) ───────────────────────────────
    /// Heliocentric distance (AU, converted internally to km).
    pub sol_rad: f64,
    /// Definition of 1 AU in km (used with sol_rad).
    pub au_def: f64,
    /// Mean motion of the circular solar orbit (rad/s).
    pub mean_motion: f64,

    // ── Tidal torque parameters ────────────────────────────────────────────────
    /// Tidal Love number k₂ for the primary.
    pub love1: f64,
    /// Tidal Love number k₂ for the secondary.
    pub love2: f64,
    /// Reference radius for tidal Love number of primary (km).
    pub refrad1: f64,
    /// Reference radius for tidal Love number of secondary (km).
    pub refrad2: f64,
    /// Bulk density of primary (kg/km³).
    pub rho_a: f64,
    /// Bulk density of secondary (kg/km³).
    pub rho_b: f64,
    /// Tidal phase lag ε₁ for primary (rad).
    pub eps1: f64,
    /// Tidal phase lag ε₂ for secondary (rad).
    pub eps2: f64,

    // ── Modified inertia tensors for LGVI integrator ───────────────────────────
    /// Modified inertia tensor Ĩ_A = 2(½ tr(I_A)·E − I_A) used by LGVI (kg·km²).
    pub ida: Mat3<T>,
    /// Modified inertia tensor Ĩ_B = 2(½ tr(I_B)·E − I_B) used by LGVI (kg·km²).
    pub idb: Mat3<T>,

    // ── Solar mass ─────────────────────────────────────────────────────────────
    /// Mass of the Sun (kg), used to compute mean motion of the binary's heliocentric orbit.
    pub msun: f64,
}

impl<T: Scalar> Params<T> {
    /// Compute the modified inertia tensors needed by the LGVI integrator:
    ///   IdA = 2 * (0.5*tr(diag(IA))*I₃ − diag(IA))
    pub fn compute_lgvi_inertia(&mut self) {
        fn modified<T: Scalar>(i: Vec3<T>) -> Mat3<T> {
            let two  = T::one() + T::one();
            let half = T::one() / two;
            let half_tr = half * (i[0] + i[1] + i[2]);
            let id = crate::math3::diag(i);
            let scaled_eye = crate::math3::scale_m(half_tr, crate::math3::eye());
            crate::math3::scale_m(two, crate::math3::sub_m(scaled_eye, id))
        }
        self.ida = modified(self.ia);
        self.idb = modified(self.ib);
    }
}

// ── promote_params ────────────────────────────────────────────────────────────

/// Lift a `Params<f64>` into `Params<U>` by promoting every physics scalar
/// via `NumCast` (eps = 0 for dual numbers — all params are constants).
///
/// The inertia tensors `ida` / `idb` are recomputed from the promoted `ia`/`ib`.
///
/// # Example
/// ```rust,ignore
/// use gubas_rs::dual::Dual;
/// use gubas_rs::types::promote_params;
///
/// let pd: Params<Dual> = promote_params(&params);
/// let ode_dual = |x: [Dual; 30], t: f64| hou_ode(x, t, &pd);
/// ```
pub fn promote_params<U: Scalar>(p: &Params<f64>) -> Params<U> {
    let lift = |v: f64| -> U { <U as num_traits::NumCast>::from(v).unwrap() };
    let mut out = Params {
        g:  lift(p.g),
        m:  lift(p.m),
        nu: lift(p.nu),
        ta: p.ta.promote::<U>(),
        tb: p.tb.promote::<U>(),
        ia: [lift(p.ia[0]), lift(p.ia[1]), lift(p.ia[2])],
        ib: [lift(p.ib[0]), lift(p.ib[1]), lift(p.ib[2])],
        n:  p.n,
        tk: p.tk.clone(),
        a:  p.a.clone(),
        b:  p.b.clone(),
        flyby_toggle: p.flyby_toggle,
        helio_toggle: p.helio_toggle,
        sg_toggle:    p.sg_toggle,
        tt_toggle:    p.tt_toggle,
        mplanet:   p.mplanet,
        a_hyp:     p.a_hyp,
        e_hyp:     p.e_hyp,
        i_hyp:     p.i_hyp,
        raan_hyp:  p.raan_hyp,
        om_hyp:    p.om_hyp,
        tau_hyp:   p.tau_hyp,
        n_hyp:     p.n_hyp,
        msolar:    p.msolar,
        a_helio:   p.a_helio,
        e_helio:   p.e_helio,
        i_helio:   p.i_helio,
        raan_helio: p.raan_helio,
        om_helio:  p.om_helio,
        tau_helio: p.tau_helio,
        n_helio:   p.n_helio,
        sol_rad:   p.sol_rad,
        au_def:    p.au_def,
        mean_motion: p.mean_motion,
        love1:     p.love1,
        love2:     p.love2,
        refrad1:   p.refrad1,
        refrad2:   p.refrad2,
        rho_a:     p.rho_a,
        rho_b:     p.rho_b,
        eps1:      p.eps1,
        eps2:      p.eps2,
        ida: crate::math3::zero_m::<U>(),
        idb: crate::math3::zero_m::<U>(),
        msun: p.msun,
    };
    out.compute_lgvi_inertia();
    out
}

// ── Initialization ────────────────────────────────────────────────────────────

/// Mirrors `struct initialization` in the C++ code.
/// Values are read from `ic_input.txt` by `ic_read()` in main.rs.
pub struct Initialization {
    pub g: f64,
    pub order: usize,
    pub order_a: usize,
    pub order_b: usize,
    pub a_a: f64, pub b_a: f64, pub c_a: f64,  // primary semi-axes (metres)
    pub a_b: f64, pub b_b: f64, pub c_b: f64,  // secondary semi-axes (metres)
    pub a_shape: i32,  // 0=sphere, 1=ellipsoid, 2=polyhedron
    pub b_shape: i32,
    pub rho_a: f64,    // kg/km³
    pub rho_b: f64,
    pub t0: f64,
    pub tf: f64,
    pub ta_file: String, pub tb_file: String,
    pub ia_file: String, pub ib_file: String,
    pub tet_file_a: String, pub vert_file_a: String,
    pub tet_file_b: String, pub vert_file_b: String,
    pub x0: [f64; 30],
    pub tgen: i32,
    pub integ: i32,
    pub h: f64,
    pub tol: f64,
    pub flyby_toggle: i32,
    pub helio_toggle: i32,
    pub sg_toggle: i32,
    pub tt_toggle: i32,
    pub mplanet: f64,
    pub a_hyp: f64,   pub e_hyp: f64,   pub i_hyp: f64,
    pub raan_hyp: f64, pub om_hyp: f64,  pub tau_hyp: f64,
    pub msolar: f64,
    pub a_helio: f64,  pub e_helio: f64,  pub i_helio: f64,
    pub raan_helio: f64, pub om_helio: f64, pub tau_helio: f64,
    pub sol_rad: f64,
    pub au_def: f64,
    pub love1: f64,   pub love2: f64,
    pub refrad1: f64, pub refrad2: f64,
    pub eps1: f64,    pub eps2: f64,
    pub msun: f64,
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coefficients::{a_calc, b_calc, tk_calc};

    // ── Cube ─────────────────────────────────────────────────────────────────

    #[test]
    fn cube_new_all_zeros() {
        let c = Cube::<f64>::new(2);
        for i in 0..=2 { for j in 0..=2 { for k in 0..=2 {
            assert_eq!(c.get(i, j, k), 0.0);
        }}}
    }

    #[test]
    fn cube_set_get_roundtrip() {
        let mut c = Cube::new(3);
        c.set(1, 2, 3, 42.0);
        assert_eq!(c.get(1, 2, 3), 42.0);
        assert_eq!(c.get(0, 0, 0), 0.0); // others unaffected
    }

    #[test]
    fn cube_add_accumulates() {
        let mut c = Cube::new(2);
        c.set(1, 1, 1, 3.0);
        c.add(1, 1, 1, 7.0);
        assert_eq!(c.get(1, 1, 1), 10.0);
    }

    #[test]
    fn cube_zeros_resets_all() {
        let mut c = Cube::new(2);
        c.set(0, 1, 2, 99.0);
        c.zeros();
        assert_eq!(c.get(0, 1, 2), 0.0);
    }

    #[test]
    fn cube_extremes_addressable() {
        // Can write to and read from the [0,0,0] and [n,n,n] corners
        let n = 4;
        let mut c = Cube::new(n);
        c.set(0, 0, 0, 1.0);
        c.set(n, n, n, 2.0);
        assert_eq!(c.get(0, 0, 0), 1.0);
        assert_eq!(c.get(n, n, n), 2.0);
    }

    #[test]
    fn cube_order_field_matches_constructor() {
        let c = Cube::<f64>::new(5);
        assert_eq!(c.order, 5);
    }

    // ── Params::compute_lgvi_inertia ─────────────────────────────────────────

    fn make_params_with_inertia(ia: [f64; 3], ib: [f64; 3]) -> Params {
        let n = 0usize;
        let mut ta = Cube::new(n);  ta.set(0, 0, 0, 1e12);
        let mut tb = Cube::new(n);  tb.set(0, 0, 0, 5e11);
        let ma = 1e12_f64;
        let mb = 5e11_f64;
        let mut p = Params {
            g: 6.674e-20, m: ma*mb/(ma+mb), nu: mb/(ma+mb),
            ta, tb, ia, ib,
            n, tk: tk_calc(n), a: a_calc(n), b: b_calc(n),
            flyby_toggle: 0, helio_toggle: 0, sg_toggle: 0, tt_toggle: 0,
            mplanet: 0.0, a_hyp: -1.0, e_hyp: 1.5,
            i_hyp: 0.0, raan_hyp: 0.0, om_hyp: 0.0, tau_hyp: 0.0, n_hyp: 0.0,
            msolar: 0.0, a_helio: 1.0, e_helio: 0.0,
            i_helio: 0.0, raan_helio: 0.0, om_helio: 0.0, tau_helio: 0.0, n_helio: 0.0,
            sol_rad: 0.0, au_def: 1.496e8, mean_motion: 0.0,
            love1: 0.0, love2: 0.0, refrad1: 1.0, refrad2: 1.0,
            rho_a: 1e12, rho_b: 1e12, eps1: 0.0, eps2: 0.0,
            ida: [[0.0; 3]; 3], idb: [[0.0; 3]; 3],
            msun: 2e30,
        };
        p.compute_lgvi_inertia();
        p
    }

    #[test]
    fn lgvi_inertia_diagonal_formula() {
        // IdA[i][i] = Σ_{j≠i} IA[j]  =  sum of the other two principal moments
        // ia = [2, 3, 4] → ida = diag([3+4−2, 2+4−3, 2+3−4]) = diag([5, 3, 1])
        let p = make_params_with_inertia([2.0, 3.0, 4.0], [2.0, 3.0, 4.0]);
        let tol = 1e-14;
        assert!((p.ida[0][0] - 5.0).abs() < tol, "ida[0][0]={}", p.ida[0][0]);
        assert!((p.ida[1][1] - 3.0).abs() < tol, "ida[1][1]={}", p.ida[1][1]);
        assert!((p.ida[2][2] - 1.0).abs() < tol, "ida[2][2]={}", p.ida[2][2]);
        // Off-diagonal must be zero
        assert!((p.ida[0][1]).abs() < tol);
        assert!((p.ida[1][2]).abs() < tol);
        assert!((p.ida[0][2]).abs() < tol);
    }

    #[test]
    fn lgvi_inertia_equal_moments() {
        // ia = [I, I, I]: ida[i][i] = 2*(3I/2 − I) = I
        let p = make_params_with_inertia([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
        let tol = 1e-14;
        for i in 0..3 {
            assert!((p.ida[i][i] - 1.0).abs() < tol, "ida[{i}][{i}]={}", p.ida[i][i]);
            for j in 0..3 { if i != j {
                assert!((p.ida[i][j]).abs() < tol, "off-diag ida[{i}][{j}]≠0");
            }}
        }
    }

    #[test]
    fn lgvi_inertia_matches_manual_formula() {
        // IdA = 2*(0.5*tr(IA)*I₃ − diag(IA))
        // Verify with ia = [1, 2, 6]
        let ia: [f64; 3] = [1.0, 2.0, 6.0];
        let p = make_params_with_inertia(ia, ia);
        let tr = ia[0] + ia[1] + ia[2]; // = 9
        for i in 0..3 {
            let expected = tr - 2.0 * ia[i];
            assert!((p.ida[i][i] - expected).abs() < 1e-14,
                    "ida[{i}][{i}]: expected {expected}, got {}", p.ida[i][i]);
        }
    }
}
