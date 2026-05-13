// math3.rs — 3-D vector and 3x3 matrix arithmetic
//
// All types are plain arrays so they are `Copy` — no heap allocation, no
// references.  This makes math code read like math.
//
//   Vec3  = [f64; 3]
//   Mat3  = [[f64; 3]; 3]   (stored ROW-MAJOR: M[row][col])

pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];

pub const ZERO_V: Vec3 = [0.0; 3];
pub const ZERO_M: Mat3 = [[0.0; 3]; 3];
pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// ── scalar on vector ─────────────────────────────────────────────────────────

pub fn scale_v(s: f64, v: Vec3) -> Vec3 {
    [s * v[0], s * v[1], s * v[2]]
}

pub fn add_v(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub_v(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

// ── vector products ───────────────────────────────────────────────────────────

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1],
     a[2] * b[0] - a[0] * b[2],
     a[0] * b[1] - a[1] * b[0]]
}

pub fn norm(v: Vec3) -> f64 {
    dot(v, v).sqrt()
}

/// Infinity norm of a vector (max absolute element) — used for adaptive step
/// control in RK 7(8).
pub fn inf_norm_v(v: Vec3) -> f64 {
    v[0].abs().max(v[1].abs()).max(v[2].abs())
}

/// Infinity norm of a 30-element state vector.
pub fn inf_norm_30(x: &[f64; 30]) -> f64 {
    x.iter().cloned().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

// ── scalar / element-wise on matrices ────────────────────────────────────────

pub fn scale_m(s: f64, m: Mat3) -> Mat3 {
    [[s * m[0][0], s * m[0][1], s * m[0][2]],
     [s * m[1][0], s * m[1][1], s * m[1][2]],
     [s * m[2][0], s * m[2][1], s * m[2][2]]]
}

pub fn add_m(a: Mat3, b: Mat3) -> Mat3 {
    [[a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]],
     [a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]],
     [a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]]]
}

pub fn sub_m(a: Mat3, b: Mat3) -> Mat3 {
    [[a[0][0] - b[0][0], a[0][1] - b[0][1], a[0][2] - b[0][2]],
     [a[1][0] - b[1][0], a[1][1] - b[1][1], a[1][2] - b[1][2]],
     [a[2][0] - b[2][0], a[2][1] - b[2][1], a[2][2] - b[2][2]]]
}

// ── matrix products ───────────────────────────────────────────────────────────

/// Matrix–matrix product  (A * B).
pub fn mat_mul(a: Mat3, b: Mat3) -> Mat3 {
    let mut c = ZERO_M;
    for i in 0..3 {
        for k in 0..3 {
            for j in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Matrix–vector product  A * v  (treating v as a column vector).
pub fn mat_vec(a: Mat3, v: Vec3) -> Vec3 {
    [dot(a[0], v), dot(a[1], v), dot(a[2], v)]
}

pub fn transpose(a: Mat3) -> Mat3 {
    [[a[0][0], a[1][0], a[2][0]],
     [a[0][1], a[1][1], a[2][1]],
     [a[0][2], a[1][2], a[2][2]]]
}

// ── special matrices ──────────────────────────────────────────────────────────

/// Diagonal 3x3 from a vector.
pub fn diag(v: Vec3) -> Mat3 {
    [[v[0], 0.0, 0.0],
     [0.0, v[1], 0.0],
     [0.0, 0.0, v[2]]]
}

/// Skew-symmetric (tilde / cross-product) matrix from a column vector v.
/// Satisfies:  tilde(v) * w == cross(v, w)
pub fn tilde(v: Vec3) -> Mat3 {
    [[ 0.0, -v[2],  v[1]],
     [ v[2],  0.0, -v[0]],
     [-v[1],  v[0],  0.0]]
}

/// 3x3 identity matrix.
pub fn eye() -> Mat3 { IDENTITY }

/// Trace of a 3x3 matrix.
pub fn trace(a: Mat3) -> f64 {
    a[0][0] + a[1][1] + a[2][2]
}

/// Determinant of a 3x3 matrix.
pub fn det3(a: Mat3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Inverse of a 3x3 matrix (panics if singular).
pub fn inv3(a: Mat3) -> Mat3 {
    let d = det3(a);
    let s = 1.0 / d;
    [[(a[1][1] * a[2][2] - a[1][2] * a[2][1]) * s,
      (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * s,
      (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * s],
     [(a[1][2] * a[2][0] - a[1][0] * a[2][2]) * s,
      (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * s,
      (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * s],
     [(a[1][0] * a[2][1] - a[1][1] * a[2][0]) * s,
      (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * s,
      (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * s]]
}

/// Frobenius norm of a 3x3 matrix (used in LGVI convergence checks).
pub fn frob_norm(a: Mat3) -> f64 {
    let mut s = 0.0_f64;
    for row in &a { for &v in row { s += v * v; } }
    s.sqrt()
}

// ── column extraction ─────────────────────────────────────────────────────────

/// Extract column j from a Mat3.
pub fn col(a: Mat3, j: usize) -> Vec3 {
    [a[0][j], a[1][j], a[2][j]]
}

// ── state-vector helpers ──────────────────────────────────────────────────────

/// x + s * dx  for 30-element state vectors.
pub fn state_add_scaled(x: [f64; 30], s: f64, dx: [f64; 30]) -> [f64; 30] {
    let mut r = x;
    for i in 0..30 { r[i] += s * dx[i]; }
    r
}

/// x - y  for 30-element state vectors.
pub fn state_sub(x: [f64; 30], y: [f64; 30]) -> [f64; 30] {
    let mut r = [0.0f64; 30];
    for i in 0..30 { r[i] = x[i] - y[i]; }
    r
}

/// Linear combination of multiple state increments:  x + sum_i s[i]*k[i]
pub fn state_combine(x: [f64; 30], terms: &[(&[f64; 30], f64)]) -> [f64; 30] {
    let mut r = x;
    for &(k, s) in terms {
        for i in 0..30 { r[i] += s * k[i]; }
    }
    r
}
