// types.rs — shared data structures
//
// `Cube` replaces Armadillo's `cube` type.
// `Params` carries all the read-only simulation parameters that are threaded
// through the integrators and ODE function.
// `Initialization` holds the values read from `ic_input.txt`.

use crate::math3::{Mat3, Vec3};

// ── Cube ─────────────────────────────────────────────────────────────────────

/// A 3-D array of size `(order+1) × (order+1) × (order+1)`.
///
/// Indexed as `T[l][m][n]` using row-major flattening:
///   flat_idx = l*(order+1)^2 + m*(order+1) + n
///
/// This replaces `arma::cube` from Armadillo.
#[derive(Clone, Debug)]
pub struct Cube {
    pub order: usize, // the "q" in the paper — array spans 0..=order
    data: Vec<f64>,
}

impl Cube {
    pub fn new(order: usize) -> Self {
        let sz = order + 1;
        Self { order, data: vec![0.0; sz * sz * sz] }
    }

    fn idx(&self, l: usize, m: usize, n: usize) -> usize {
        let sz = self.order + 1;
        l * sz * sz + m * sz + n
    }

    #[inline]
    pub fn get(&self, l: usize, m: usize, n: usize) -> f64 {
        self.data[self.idx(l, m, n)]
    }

    #[inline]
    pub fn set(&mut self, l: usize, m: usize, n: usize, val: f64) {
        let i = self.idx(l, m, n);
        self.data[i] = val;
    }

    #[inline]
    pub fn add(&mut self, l: usize, m: usize, n: usize, val: f64) {
        let i = self.idx(l, m, n);
        self.data[i] += val;
    }

    pub fn zeros(&mut self) {
        for v in &mut self.data { *v = 0.0; }
    }

    /// Save to CSV in the same layout as Armadillo's `save(..., csv_ascii)`.
    ///
    /// The C++ stacks slices vertically:  slice 0 (T[:,:,0]) then slice 1
    /// (T[:,:,1]) … so the CSV has `(order+1)^2` rows and `(order+1)` cols.
    /// Each row `s*(order+1) + l`, col `m` = T[l][m][s].
    pub fn save_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        let sz = self.order + 1;
        for s in 0..sz {          // slice index (last index in our convention)
            for l in 0..sz {      // row inside slice
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
        let sz = rows[0].len();          // order + 1
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
/// In the C++ code this was `struct parameters` with raw pointers.  Here we
/// own everything by value, which is cleaner and safe.
pub struct Params {
    // Gravity & masses
    pub g: f64,
    pub m: f64,   // reduced mass  Mc*Ms/(Mc+Ms)
    pub nu: f64,  // mass ratio  Ms/(Mc+Ms)

    // Inertia integrals (in km, kg)
    pub ta: Cube,  // primary (not rotated)
    pub tb: Cube,  // secondary (not rotated)
    pub ia: Vec3,  // primary principal moments  [Ixx, Iyy, Izz]
    pub ib: Vec3,  // secondary principal moments

    // Expansion coefficients
    pub n: usize,          // mutual potential truncation order
    pub tk: Vec<Vec<f64>>, // tk[(expansion_order)][(recursion_step)]
    pub a: Vec<f64>,       // a-coefficients (7-D packed into 1-D by t_ind)
    pub b: Vec<f64>,       // b-coefficients

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
    pub ida: Mat3,
    pub idb: Mat3,

    // Solar mass (distinct from msolar which is the perturber mass)
    pub msun: f64,
}

impl Params {
    /// Compute the modified inertia tensors needed by the LGVI integrator:
    ///   IdA = 2*(0.5*trace(diag(IA))*I - diag(IA))
    pub fn compute_lgvi_inertia(&mut self) {
        fn modified(i: Vec3) -> Mat3 {
            let half_tr = 0.5 * (i[0] + i[1] + i[2]);
            let id = crate::math3::diag(i);
            let scaled_eye = crate::math3::scale_m(half_tr, crate::math3::eye());
            crate::math3::scale_m(2.0, crate::math3::sub_m(scaled_eye, id))
        }
        self.ida = modified(self.ia);
        self.idb = modified(self.ib);
    }
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
