//! Shared data structures: [`Cube`], [`Params`], and [`Initialization`].
//!
//! - [`Cube<T>`] — 3-D array replacing Armadillo's `cube`.  Flat row-major
//!   storage, indexed `(l, m, n)`.
//! - [`Params<T>`] — all read-only simulation parameters threaded through the
//!   ODE and integrators.  Generic over [`Scalar`] so `Dual` types can replace
//!   `f64` for auto-differentiation.
//! - [`promote_params`] — lift `Params<f64>` → `Params<U>` for dual-number AD.

use crate::math3::{Mat3, Scalar};

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
        // Without this check an index such as (0, 0, order+1) would alias
        // (0, 1, 0) instead of failing.
        debug_assert!(l <= self.order && m <= self.order && n <= self.order,
                      "Cube index ({l}, {m}, {n}) exceeds order {}", self.order);
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

    /// Second-moment tensor J and inertia tensor I = tr(J)·E − J of the body,
    /// both about the origin of its body frame (kg·km²).
    ///
    /// J_pq = T(e_p + e_q), so J_xx = T_200, J_xy = T_110, and so on.  For
    /// integrals taken about the centre of mass this is the inertia tensor the
    /// rotational equations use.
    pub fn inertia_tensor(&self) -> Mat3<T> {
        second_moment_to_inertia(self.second_moments())
    }

    /// Second-moment tensor J_pq = T(e_p + e_q) (kg·km²).
    pub fn second_moments(&self) -> Mat3<T> {
        let mut j = crate::math3::zero_m::<T>();
        for p in 0..3 {
            for q in 0..3 {
                let mut e = [0usize; 3];
                e[p] += 1;
                e[q] += 1;
                j[p][q] = self.get(e[0], e[1], e[2]);
            }
        }
        j
    }
}

/// I = tr(J)·E − J, the inertia tensor of a second-moment tensor J.  Linear
/// in J, so it maps a change ΔJ to the matching change ΔI.
pub fn second_moment_to_inertia<T: Scalar>(j: Mat3<T>) -> Mat3<T> {
    use crate::math3::{eye, scale_m, sub_m, trace};
    sub_m(scale_m(trace(j), eye()), j)
}

pub(crate) fn invalid_data(path: &str, msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{path}: {msg}"))
}

/// `std::fs::read_to_string`, with the path in the error message.
pub(crate) fn read_text(path: &str) -> std::io::Result<String> {
    std::fs::read_to_string(path)
        .map_err(|e| std::io::Error::new(e.kind(), format!("{path}: {e}")))
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
    ///
    /// Fails with `InvalidData` on an empty file, an entry that is not a
    /// number, or a table that is not `(order+1)^2` rows of `order+1` values.
    pub fn load_csv(path: &str) -> std::io::Result<Self> {
        let content = read_text(path)?;
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for (n, line) in content.lines().enumerate() {
            if line.trim().is_empty() { continue; }
            let row = line.split(',')
                .map(|s| s.trim().parse::<f64>().map_err(|e| invalid_data(
                    path, format!("line {}: cannot parse {:?} ({e})", n + 1, s.trim()))))
                .collect::<std::io::Result<Vec<f64>>>()?;
            rows.push(row);
        }
        let sz = rows.first()
            .map(|r| r.len())
            .ok_or_else(|| invalid_data(path, "no data".into()))?;
        if rows.len() != sz * sz || rows.iter().any(|r| r.len() != sz) {
            return Err(invalid_data(path, format!(
                "expected {} rows of {sz} values for order {}", sz * sz, sz - 1)));
        }
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
    /// Inertia tensor of the primary about its centre of mass, in its body
    /// frame (kg·km²).  Loaded as diag(IDP.csv).  When [`Params::with_theta`]
    /// changes degree-2 entries of `ta` it changes this tensor by the same
    /// amount as tr(J)·E − J, J_pq = T(e_p + e_q).
    pub ia: Mat3<T>,
    /// Inertia tensor of the secondary about its centre of mass, in its body
    /// frame (kg·km²).  Loaded as diag(IDS.csv); updated from `tb` like `ia`.
    pub ib: Mat3<T>,

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
    /// Mean motion of the circular solar orbit (rad/s), from the nominal
    /// masses.  A plain f64, so it is neither differentiated nor updated by
    /// [`Params::with_theta`]; the masses change it by about M/M_sun.
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
    ///   IdA = 2 * (0.5*tr(IA)*I₃ − IA)
    pub fn compute_lgvi_inertia(&mut self) {
        fn modified<T: Scalar>(i: Mat3<T>) -> Mat3<T> {
            use crate::math3::{eye, scale_m, sub_m, trace};
            let two  = T::one() + T::one();
            let half = T::one() / two;
            let scaled_eye = scale_m(half * trace(i), eye());
            scale_m(two, sub_m(scaled_eye, i))
        }
        self.ida = modified(self.ia);
        self.idb = modified(self.ib);
    }

    /// Copy of these parameters with the integrals `idx_a` of the primary set
    /// to `theta_a` and the integrals `idx_b` of the secondary set to
    /// `theta_b`, and with every quantity derived from them updated to match:
    ///
    /// - degree-2 entries change the inertia tensors by ΔI = tr(ΔJ)·E − ΔJ,
    ///   where ΔJ_pq is the change of T(e_p + e_q);
    /// - degree-0 entries change the masses, so `m` and `nu` are recomputed;
    /// - `ida`/`idb` are recomputed from the new inertia tensors.
    ///
    /// Without this the rotational equations would keep the nominal inertia
    /// while the mutual potential used the new integrals.  Degree-1 entries
    /// move the centre of mass away from the body-frame origin, which the
    /// equations of motion assume is the centre of mass; see
    /// [`check_theta_indices`].
    pub fn with_theta(
        &self,
        idx_a: &[(usize, usize, usize)], theta_a: &[T],
        idx_b: &[(usize, usize, usize)], theta_b: &[T],
    ) -> Params<T> {
        fn apply<T: Scalar>(
            t: &mut Cube<T>, inertia: &mut Mat3<T>,
            idx: &[(usize, usize, usize)], theta: &[T],
        ) {
            assert_eq!(idx.len(), theta.len(), "one value per θ index");
            let mut dj = crate::math3::zero_m::<T>();
            for (&(i, j, k), &value) in idx.iter().zip(theta) {
                if i + j + k == 2 {
                    // T(e_p + e_q) is J_pq.
                    let (p, q) = match (i, j, k) {
                        (2, 0, 0) => (0, 0), (0, 2, 0) => (1, 1), (0, 0, 2) => (2, 2),
                        (1, 1, 0) => (0, 1), (1, 0, 1) => (0, 2), (0, 1, 1) => (1, 2),
                        _ => unreachable!("i + j + k == 2"),
                    };
                    let delta = value - t.get(i, j, k);
                    dj[p][q] += delta;
                    if p != q { dj[q][p] += delta; }
                }
                t.set(i, j, k, value);
            }
            *inertia = crate::math3::add_m(*inertia, second_moment_to_inertia(dj));
        }

        let mut out = self.clone();
        apply(&mut out.ta, &mut out.ia, idx_a, theta_a);
        apply(&mut out.tb, &mut out.ib, idx_b, theta_b);
        let mc = out.ta.get(0, 0, 0);
        let ms = out.tb.get(0, 0, 0);
        out.m  = mc * ms / (mc + ms);
        out.nu = ms / (mc + ms);
        out.compute_lgvi_inertia();
        out
    }
}

/// Check that a list of θ indices can be estimated with these equations of
/// motion: every entry must fit in a cube of order `order`, and none may be of
/// degree 1, because a first moment moves the centre of mass off the
/// body-frame origin.
pub fn check_theta_indices(idx: &[(usize, usize, usize)], order: usize) -> Result<(), String> {
    for &(i, j, k) in idx {
        if i + j + k > order {
            return Err(format!(
                "T_{i}{j}{k} has degree {} but the expansion order is {order}", i + j + k));
        }
        if i + j + k == 1 {
            return Err(format!(
                "T_{i}{j}{k} is a first moment; the equations of motion assume the \
                 body-frame origin is the centre of mass, so degree 1 cannot be estimated"));
        }
    }
    Ok(())
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
        ia: p.ia.map(|row| row.map(lift)),
        ib: p.ib.map(|row| row.map(lift)),
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
/// Values are read from `ic_input.txt` by [`Initialization::read`], one per
/// line in the order of the fields below.
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

impl Initialization {
    /// Read `ic_input.txt` from `dir`.  Fails with `InvalidData`, naming the
    /// line, when a value is missing or cannot be parsed.
    pub fn read(dir: &std::path::Path) -> std::io::Result<Self> {
        struct Lines<'a> {
            path:  String,
            lines: std::iter::Enumerate<std::str::Lines<'a>>,
        }
        impl Lines<'_> {
            fn next<V>(&mut self, name: &str) -> std::io::Result<V>
            where V: std::str::FromStr, V::Err: std::fmt::Display {
                let (n, line) = self.lines.next().ok_or_else(|| invalid_data(
                    &self.path, format!("file ends before {name}")))?;
                line.trim().parse().map_err(|e| invalid_data(&self.path, format!(
                    "line {}: cannot read {name} from {:?} ({e})", n + 1, line.trim())))
            }
        }

        let path = dir.join("ic_input.txt").to_string_lossy().into_owned();
        let content = read_text(&path)?;
        let mut r = Lines { path, lines: content.lines().enumerate() };

        let g           = r.next("G")?;
        let order       = r.next("order")?;
        let order_a     = r.next("order_a")?;
        let order_b     = r.next("order_b")?;
        let a_a         = r.next("a_a")?;
        let b_a         = r.next("b_a")?;
        let c_a         = r.next("c_a")?;
        let a_b         = r.next("a_b")?;
        let b_b         = r.next("b_b")?;
        let c_b         = r.next("c_b")?;
        let a_shape     = r.next("a_shape")?;
        let b_shape     = r.next("b_shape")?;
        let rho_a       = r.next("rho_a")?;
        let rho_b       = r.next("rho_b")?;
        let t0          = r.next("t0")?;
        let tf          = r.next("tf")?;
        let ta_file     = r.next("ta_file")?;
        let tb_file     = r.next("tb_file")?;
        let ia_file     = r.next("ia_file")?;
        let ib_file     = r.next("ib_file")?;
        let tet_file_a  = r.next("tet_file_a")?;
        let vert_file_a = r.next("vert_file_a")?;
        let tet_file_b  = r.next("tet_file_b")?;
        let vert_file_b = r.next("vert_file_b")?;
        let mut x0 = [0.0_f64; 30];
        for (i, v) in x0.iter_mut().enumerate() { *v = r.next(&format!("x0[{i}]"))?; }
        let tgen         = r.next("Tgen")?;
        let integ        = r.next("integ")?;
        let h            = r.next("h")?;
        let tol          = r.next("tol")?;
        let flyby_toggle = r.next("flyby_toggle")?;
        let helio_toggle = r.next("helio_toggle")?;
        let sg_toggle    = r.next("sg_toggle")?;
        let tt_toggle    = r.next("tt_toggle")?;
        let mplanet      = r.next("mplanet")?;
        let a_hyp        = r.next("a_hyp")?;
        let e_hyp        = r.next("e_hyp")?;
        let i_hyp        = r.next("i_hyp")?;
        let raan_hyp     = r.next("raan_hyp")?;
        let om_hyp       = r.next("om_hyp")?;
        let tau_hyp      = r.next("tau_hyp")?;
        let msolar       = r.next("msolar")?;
        let a_helio      = r.next("a_helio")?;
        let e_helio      = r.next("e_helio")?;
        let i_helio      = r.next("i_helio")?;
        let raan_helio   = r.next("raan_helio")?;
        let om_helio     = r.next("om_helio")?;
        let tau_helio    = r.next("tau_helio")?;
        let sol_rad      = r.next("sol_rad")?;
        let au_def       = r.next("au_def")?;
        let love1        = r.next("love1")?;
        let love2        = r.next("love2")?;
        let refrad1      = r.next("refrad1")?;
        let refrad2      = r.next("refrad2")?;
        let eps1         = r.next("eps1")?;
        let eps2         = r.next("eps2")?;
        let msun         = r.next("msun")?;

        Ok(Initialization {
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
        })
    }

    /// Expansion orders `(n, n_a, n_b)` actually used.  A polyhedral body is
    /// expanded to at least the mutual-potential order, and the mutual order
    /// is at least each body order.
    pub fn expansion_orders(&self) -> (usize, usize, usize) {
        let mut order_a = self.order_a;
        let mut order_b = self.order_b;
        if self.a_shape == 2 { order_a = order_a.max(self.order); }
        if self.b_shape == 2 { order_b = order_b.max(self.order); }
        (self.order.max(order_a).max(order_b), order_a, order_b)
    }
}

impl Params<f64> {
    /// Load the parameters from the inertia files named in `ics`, relative to
    /// `dir`.  Fails if a file is missing or malformed, or if an
    /// inertia-integral cube is of lower order than the expansion.
    pub fn load(dir: &std::path::Path, ics: &Initialization) -> std::io::Result<Self> {
        use crate::inertia::load_inertia_csv;
        let file = |name: &str| dir.join(name).to_string_lossy().into_owned();
        let (order, _, _) = ics.expansion_orders();
        let ta = Cube::load_csv(&file(&ics.ta_file))?;
        let tb = Cube::load_csv(&file(&ics.tb_file))?;
        for (name, cube) in [(&ics.ta_file, &ta), (&ics.tb_file, &tb)] {
            if cube.order < order {
                return Err(invalid_data(&file(name), format!(
                    "inertia integrals of order {} but the expansion order is {order}; \
                     regenerate them with Tgen = 1", cube.order)));
            }
        }
        let ia = load_inertia_csv(&file(&ics.ia_file))?;
        let ib = load_inertia_csv(&file(&ics.ib_file))?;
        Ok(Self::from_initialization(ics, ta, tb, ia, ib))
    }

    /// Compute the inertia integrals and moments of both bodies from the
    /// shapes in `ics` (polyhedron files relative to `dir`), save them to
    /// the files named in `ics`, and assemble the parameters.
    pub fn generate(dir: &std::path::Path, ics: &Initialization) -> std::io::Result<Self> {
        use crate::inertia::{ell_mass_params_met, poly_inertia_met, poly_moi_met, save_moi_csv};
        use crate::math3::Vec3;
        let file = |name: &str| dir.join(name).to_string_lossy().into_owned();
        let (order, order_a, order_b) = ics.expansion_orders();
        let body = |label: &str, shape: i32, body_order: usize, rho: f64, abc: [f64; 3],
                    tet: &str, vert: &str| -> std::io::Result<(Vec3, Cube)> {
            let [a, b, c] = abc;
            match shape {
                0 => Ok(ell_mass_params_met(order, 0, rho, a, b, c)),
                1 => Ok(ell_mass_params_met(order, body_order, rho, a, b, c)),
                2 => {
                    // A body expanded to a lower order than the mutual
                    // potential has zero integrals above its own order.
                    let own = poly_inertia_met(body_order, rho, &file(tet), &file(vert));
                    let mut t = Cube::new(order);
                    for l in 0..=own.order { for m in 0..=own.order { for n in 0..=own.order {
                        t.set(l, m, n, own.get(l, m, n));
                    }}}
                    Ok((poly_moi_met(rho, &file(tet), &file(vert)), t))
                }
                _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!(
                    "{label} shape flag {shape} is not 0 (sphere), 1 (ellipsoid) or 2 (polyhedron)"))),
            }
        };
        let (ia, ta) = body("primary", ics.a_shape, order_a, ics.rho_a,
                            [ics.a_a, ics.b_a, ics.c_a], &ics.tet_file_a, &ics.vert_file_a)?;
        let (ib, tb) = body("secondary", ics.b_shape, order_b, ics.rho_b,
                            [ics.a_b, ics.b_b, ics.c_b], &ics.tet_file_b, &ics.vert_file_b)?;
        ta.save_csv(&file(&ics.ta_file))?;
        tb.save_csv(&file(&ics.tb_file))?;
        save_moi_csv(ia, &file(&ics.ia_file))?;
        save_moi_csv(ib, &file(&ics.ib_file))?;
        Ok(Self::from_initialization(ics, ta, tb, crate::math3::diag(ia), crate::math3::diag(ib)))
    }

    /// Assemble the parameters from the inputs and the inertia integrals and
    /// tensors of both bodies.  The masses are the `T_000` entries, and the
    /// expansion order is [`Initialization::expansion_orders`].
    pub fn from_initialization(
        ics: &Initialization,
        ta: Cube<f64>, tb: Cube<f64>, ia: Mat3<f64>, ib: Mat3<f64>,
    ) -> Self {
        let (order, _, _) = ics.expansion_orders();
        let mc = ta.get(0, 0, 0);
        let ms = tb.get(0, 0, 0);
        let mean_motion = (ics.g * (ics.msun + mc + ms)
                           / (ics.sol_rad * ics.au_def / 1000.0).powi(3)).sqrt();
        let mut params = Params {
            g: ics.g,
            m: mc * ms / (mc + ms),
            nu: ms / (mc + ms),
            ta, tb, ia, ib,
            n: order,
            tk: crate::coefficients::tk_calc(order),
            a:  crate::coefficients::a_calc(order),
            b:  crate::coefficients::b_calc(order),
            flyby_toggle: ics.flyby_toggle,
            helio_toggle: ics.helio_toggle,
            sg_toggle:    ics.sg_toggle,
            tt_toggle:    ics.tt_toggle,
            mplanet:  ics.mplanet,
            a_hyp:    ics.a_hyp, e_hyp: ics.e_hyp, i_hyp: ics.i_hyp,
            raan_hyp: ics.raan_hyp, om_hyp: ics.om_hyp, tau_hyp: ics.tau_hyp,
            n_hyp: (ics.g * ics.mplanet / ics.a_hyp.abs().powi(3)).sqrt(),
            msolar:   ics.msolar,
            a_helio:  ics.a_helio, e_helio: ics.e_helio, i_helio: ics.i_helio,
            raan_helio: ics.raan_helio, om_helio: ics.om_helio, tau_helio: ics.tau_helio,
            n_helio: (ics.g * ics.msolar / ics.a_helio.abs().powi(3)).sqrt(),
            sol_rad:  ics.sol_rad,
            au_def:   ics.au_def,
            mean_motion,
            love1:    ics.love1,   love2:   ics.love2,
            refrad1:  ics.refrad1, refrad2: ics.refrad2,
            rho_a:    ics.rho_a,   rho_b:   ics.rho_b,
            eps1:     ics.eps1,    eps2:    ics.eps2,
            ida: crate::math3::ZERO_M,
            idb: crate::math3::ZERO_M,
            msun: ics.msun,
        };
        params.compute_lgvi_inertia();
        params
    }
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
            ta, tb, ia: crate::math3::diag(ia), ib: crate::math3::diag(ib),
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

    // ── File-backed fixtures ─────────────────────────────────────────────────

    /// Fresh empty directory under the system temp dir, one per test.
    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("gubas_rs_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The 85 lines of an `ic_input.txt` for two ellipsoids at order 2.
    fn ic_lines() -> Vec<String> {
        let mut l: Vec<String> = [
            "6.67e-20", "2", "2", "2",
            "400", "390", "370", "90", "70", "60",
            "1", "1", "2e12", "2e12", "0", "3600",
            "TA.csv", "TB.csv", "IA.csv", "IB.csv",
            "none", "none", "none", "none",
        ].iter().map(|s| s.to_string()).collect();
        let x0 = [1.2, 0.0, 0.0,  0.0, 1.7e-4, 0.0,  0.0, 0.0, 7.7e-4,  0.0, 0.0, 1.4e-4,
                  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0,
                  1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0];
        l.extend(x0.iter().map(|v| v.to_string()));
        l.extend([
            "1", "2", "10", "1e-12",           // Tgen, integ, h, tol
            "0", "0", "0", "0",                // toggles
            "0", "-1", "1.5", "0", "0", "0", "0",   // flyby
            "0", "1", "0", "0", "0", "0", "0",      // helio
            "1.6", "1.496e11", "0", "0", "1", "1", "0", "0", "1.989e30",
        ].iter().map(|s| s.to_string()));
        l
    }

    fn write_ic(dir: &std::path::Path, lines: &[String]) {
        std::fs::write(dir.join("ic_input.txt"), lines.join("\n") + "\n").unwrap();
    }

    fn ellipsoid_params(name: &str) -> (std::path::PathBuf, Initialization, Params) {
        let dir = scratch_dir(name);
        write_ic(&dir, &ic_lines());
        let ics = Initialization::read(&dir).unwrap();
        let p = Params::generate(&dir, &ics).unwrap();
        (dir, ics, p)
    }

    // ── Initialization::read ─────────────────────────────────────────────────

    #[test]
    fn ic_read_parses_every_field() {
        let dir = scratch_dir("ic_read");
        write_ic(&dir, &ic_lines());
        let ics = Initialization::read(&dir).unwrap();
        assert_eq!(ics.g, 6.67e-20);
        assert_eq!((ics.order, ics.order_a, ics.order_b), (2, 2, 2));
        assert_eq!((ics.a_shape, ics.b_shape), (1, 1));
        assert_eq!((ics.ta_file.as_str(), ics.ib_file.as_str()), ("TA.csv", "IB.csv"));
        assert_eq!(ics.x0[0], 1.2);
        assert_eq!(ics.x0[29], 1.0);
        assert_eq!((ics.tgen, ics.integ), (1, 2));
        assert_eq!(ics.msun, 1.989e30);
    }

    #[test]
    fn ic_read_reports_truncation_and_bad_values() {
        let dir = scratch_dir("ic_bad");
        let lines = ic_lines();
        write_ic(&dir, &lines[..50]);
        let e = Initialization::read(&dir).err().unwrap();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("file ends before x0[26]"), "{e}");

        let mut lines = lines;
        lines[1] = "four".into();
        write_ic(&dir, &lines);
        let e = Initialization::read(&dir).err().unwrap();
        assert!(e.to_string().contains("line 2: cannot read order"), "{e}");

        let e = Initialization::read(&dir.join("missing")).err().unwrap();
        assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
        assert!(e.to_string().contains("ic_input.txt"), "{e}");
    }

    // ── Params::generate / Params::load ──────────────────────────────────────

    #[test]
    fn generate_then_load_roundtrip() {
        let (dir, ics, p) = ellipsoid_params("roundtrip");
        let q = Params::load(&dir, &ics).unwrap();
        assert_eq!(q.n, 2);
        for l in 0..=2 { for m in 0..=2 { for n in 0..=2 {
            assert_eq!(q.ta.get(l, m, n), p.ta.get(l, m, n));
            assert_eq!(q.tb.get(l, m, n), p.tb.get(l, m, n));
        }}}
        assert_eq!((q.ia, q.ib), (p.ia, p.ib));
        assert_eq!((q.m, q.nu), (p.m, p.nu));
    }

    #[test]
    fn load_rejects_cube_below_expansion_order() {
        let (dir, mut ics, _) = ellipsoid_params("low_order");
        ics.order = 4;
        let e = Params::load(&dir, &ics).err().unwrap();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
        assert!(e.to_string().contains("regenerate them with Tgen = 1"), "{e}");
    }

    #[test]
    fn ellipsoid_cube_implies_its_inertia() {
        // I = tr(J)·E − J with J_pq = T(e_p + e_q) reproduces the moments
        // ell_mass_params_met returns.
        let (_, _, p) = ellipsoid_params("ell_inertia");
        for (t, i) in [(&p.ta, p.ia), (&p.tb, p.ib)] {
            let it = t.inertia_tensor();
            for r in 0..3 { for c in 0..3 {
                assert!((it[r][c] - i[r][c]).abs() <= 1e-14 * i[2][2], "I[{r}][{c}]");
            }}
        }
    }

    // ── Cube::load_csv ───────────────────────────────────────────────────────

    #[test]
    fn load_csv_roundtrip_and_errors() {
        let dir = scratch_dir("csv");
        let path = |n: &str| dir.join(n).to_string_lossy().into_owned();
        let mut c = Cube::new(2);
        c.set(2, 0, 0, 1.5);
        c.set(0, 1, 1, -0.25);
        c.save_csv(&path("ok.csv")).unwrap();
        let d = Cube::load_csv(&path("ok.csv")).unwrap();
        assert_eq!((d.order, d.get(2, 0, 0), d.get(0, 1, 1)), (2, 1.5, -0.25));

        let bad = |name: &str, text: &str| {
            std::fs::write(path(name), text).unwrap();
            Cube::load_csv(&path(name)).err().unwrap().to_string()
        };
        assert!(bad("empty.csv", "\n").contains("no data"));
        assert!(bad("nan.csv", "1,2\nx,4\n1,2\n1,2\n").contains("line 2"));
        assert!(bad("shape.csv", "1,2\n3,4\n").contains("expected 4 rows of 2 values"));
        assert_eq!(Cube::load_csv(&path("none.csv")).err().unwrap().kind(),
                   std::io::ErrorKind::NotFound);
    }

    // ── Params::with_theta ───────────────────────────────────────────────────

    #[test]
    fn with_theta_at_nominal_changes_nothing() {
        let (_, _, p) = ellipsoid_params("theta_nominal");
        let idx = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0)];
        let nominal = |t: &Cube| idx.iter().map(|&(i, j, k)| t.get(i, j, k)).collect::<Vec<_>>();
        let q = p.with_theta(&idx, &nominal(&p.ta), &idx, &nominal(&p.tb));
        assert_eq!((q.ia, q.ib, q.ida, q.idb), (p.ia, p.ib, p.ida, p.idb));
        assert_eq!((q.m, q.nu), (p.m, p.nu));
    }

    #[test]
    fn with_theta_updates_inertia_from_second_moments() {
        let (_, _, p) = ellipsoid_params("theta_inertia");
        let d = 1e-3 * p.ta.get(2, 0, 0);
        let tol = 1e-12 * p.ia[2][2];

        // T_200 → J_xx: ΔI = diag(0, d, d).
        let q = p.with_theta(&[(2, 0, 0)], &[p.ta.get(2, 0, 0) + d], &[], &[]);
        let expected = crate::math3::add_m(p.ia, crate::math3::diag([0.0, d, d]));
        for r in 0..3 { for c in 0..3 {
            assert!((q.ia[r][c] - expected[r][c]).abs() <= tol, "ia[{r}][{c}]");
        }}
        assert_eq!(q.ib, p.ib);

        // T_110 → J_xy = J_yx: ΔI_xy = ΔI_yx = −d, diagonal unchanged.
        let q = p.with_theta(&[(1, 1, 0)], &[d], &[], &[]);
        assert!((q.ia[0][1] + d).abs() <= tol && (q.ia[1][0] + d).abs() <= tol);
        for r in 0..3 { assert!((q.ia[r][r] - p.ia[r][r]).abs() <= tol); }

        // ida follows the new tensor.
        let mut r = q.clone();
        r.compute_lgvi_inertia();
        assert_eq!(q.ida, r.ida);
    }

    #[test]
    fn with_theta_updates_masses() {
        let (_, _, p) = ellipsoid_params("theta_mass");
        let ma = p.ta.get(0, 0, 0);
        let mb = 2.0 * p.tb.get(0, 0, 0);
        let q = p.with_theta(&[], &[], &[(0, 0, 0)], &[mb]);
        assert!((q.nu - mb / (ma + mb)).abs() <= 1e-15);
        assert!((q.m - ma * mb / (ma + mb)).abs() <= 1e-15 * q.m);
    }

    #[test]
    fn check_theta_indices_limits() {
        assert!(check_theta_indices(&[(0, 0, 0), (2, 0, 0), (1, 1, 0)], 2).is_ok());
        assert!(check_theta_indices(&[(1, 0, 0)], 2).unwrap_err().contains("first moment"));
        assert!(check_theta_indices(&[(2, 1, 0)], 2).unwrap_err().contains("degree 3"));
    }
}
