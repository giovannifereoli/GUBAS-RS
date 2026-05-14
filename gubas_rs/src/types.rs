// types.rs — shared data structures
//
// `Cube` replaces Armadillo's `cube` type.
// `Params` carries all the read-only simulation parameters that are threaded
// through the integrators and ODE function.
// `Initialization` holds the values read from `ic_input.txt`.
//
// Both `Cube<T>` and `Params<T>` are generic over the `Scalar` trait so that
// dual-number types can flow through the physics for auto-diff (STM / parameter
// sensitivity).  The `T = f64` default keeps all existing callers unchanged.

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
    /// The C++ stacks slices vertically: slice 0 (T[:,:,0]) then slice 1
    /// (T[:,:,1]) … so the CSV has `(order+1)^2` rows and `(order+1)` cols.
    /// Each row `s*(order+1) + l`, col `m` = T[l][m][s].
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
pub struct Params<T = f64> {
    // Gravity & masses
    pub g: T,
    pub m: T,   // reduced mass  Mc*Ms/(Mc+Ms)
    pub nu: T,  // mass ratio  Ms/(Mc+Ms)

    // Inertia integrals (in km, kg)
    pub ta: Cube<T>,  // primary (not rotated)
    pub tb: Cube<T>,  // secondary (not rotated)
    pub ia: Vec3<T>,  // primary principal moments  [Ixx, Iyy, Izz]
    pub ib: Vec3<T>,  // secondary principal moments

    // Expansion coefficients (combinatorial — always f64)
    pub n: usize,
    pub tk: Vec<Vec<f64>>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,

    // Toggles (0 = off, 1 = on)
    pub flyby_toggle: i32,
    pub helio_toggle: i32,
    pub sg_toggle: i32,
    pub tt_toggle: i32,

    // 3rd-body flyby parameters
    pub mplanet: f64,
    pub a_hyp: f64,
    pub e_hyp: f64,
    pub i_hyp: f64,
    pub raan_hyp: f64,
    pub om_hyp: f64,
    pub tau_hyp: f64,
    pub n_hyp: f64,

    // Heliocentric orbit parameters
    pub msolar: f64,
    pub a_helio: f64,
    pub e_helio: f64,
    pub i_helio: f64,
    pub raan_helio: f64,
    pub om_helio: f64,
    pub tau_helio: f64,
    pub n_helio: f64,

    // Legacy circular solar orbit
    pub sol_rad: f64,
    pub au_def: f64,
    pub mean_motion: f64,

    // Tidal torque parameters
    pub love1: f64,
    pub love2: f64,
    pub refrad1: f64,
    pub refrad2: f64,
    pub rho_a: f64,
    pub rho_b: f64,
    pub eps1: f64,
    pub eps2: f64,

    // Modified inertia tensors for LGVI
    pub ida: Mat3<T>,
    pub idb: Mat3<T>,

    // Solar mass (distinct from msolar which is the perturber mass)
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
