// math3.rs — Generic 3-D vector and 3×3 matrix arithmetic
//
// Vec3<T> and Mat3<T> are generic over any Scalar type (f64 by default).
// The default keeps all existing callers working without changes.
//
// The Scalar trait is a blanket impl over num_traits::Float + Copy + AddAssign,
// so f64 satisfies it automatically.  Dual-number types (added in the STM step)
// will also satisfy it once we add num-dual.

use std::ops::AddAssign;
use num_traits::Float;

// ── Scalar trait ──────────────────────────────────────────────────────────────

/// Any numeric type usable as a Vec3 / Mat3 element.
///
/// Blanket-implemented for f64 and for dual-number types once num-dual is added.
/// The Float supertrait provides sqrt, sin, cos, powi, powf, abs, max, zero,
/// one, and all four arithmetic operators.  AddAssign is needed for accumulation
/// loops (e.g. mat_mul, frob_norm).
pub trait Scalar: Float + Copy + AddAssign {}
impl<T: Float + Copy + AddAssign> Scalar for T {}

// ── Type aliases ──────────────────────────────────────────────────────────────

/// Column-vector of 3 elements.  Defaults to f64 so existing code compiles.
pub type Vec3<T = f64> = [T; 3];

/// Row-major 3×3 matrix.  M[row][col].  Defaults to f64.
pub type Mat3<T = f64> = [[T; 3]; 3];

// ── f64 constants (kept for non-generic callers) ──────────────────────────────

pub const ZERO_V: Vec3 = [0.0; 3];
pub const ZERO_M: Mat3 = [[0.0; 3]; 3];
pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// ── generic zero / identity constructors ─────────────────────────────────────

#[inline] pub fn zero_v<T: Scalar>() -> Vec3<T> { [T::zero(); 3] }
#[inline] pub fn zero_m<T: Scalar>() -> Mat3<T> { [[T::zero(); 3]; 3] }

// ── scalar × vector ──────────────────────────────────────────────────────────

#[inline]
pub fn scale_v<T: Scalar>(s: T, v: Vec3<T>) -> Vec3<T> {
    [s * v[0], s * v[1], s * v[2]]
}

#[inline]
pub fn add_v<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub_v<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

// ── vector products ───────────────────────────────────────────────────────────

#[inline]
pub fn dot<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross<T: Scalar>(a: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
    [a[1] * b[2] - a[2] * b[1],
     a[2] * b[0] - a[0] * b[2],
     a[0] * b[1] - a[1] * b[0]]
}

#[inline]
pub fn norm<T: Scalar>(v: Vec3<T>) -> T {
    dot(v, v).sqrt()
}

// ── scalar × matrix ───────────────────────────────────────────────────────────

#[inline]
pub fn scale_m<T: Scalar>(s: T, m: Mat3<T>) -> Mat3<T> {
    [[s*m[0][0], s*m[0][1], s*m[0][2]],
     [s*m[1][0], s*m[1][1], s*m[1][2]],
     [s*m[2][0], s*m[2][1], s*m[2][2]]]
}

#[inline]
pub fn add_m<T: Scalar>(a: Mat3<T>, b: Mat3<T>) -> Mat3<T> {
    [[a[0][0]+b[0][0], a[0][1]+b[0][1], a[0][2]+b[0][2]],
     [a[1][0]+b[1][0], a[1][1]+b[1][1], a[1][2]+b[1][2]],
     [a[2][0]+b[2][0], a[2][1]+b[2][1], a[2][2]+b[2][2]]]
}

#[inline]
pub fn sub_m<T: Scalar>(a: Mat3<T>, b: Mat3<T>) -> Mat3<T> {
    [[a[0][0]-b[0][0], a[0][1]-b[0][1], a[0][2]-b[0][2]],
     [a[1][0]-b[1][0], a[1][1]-b[1][1], a[1][2]-b[1][2]],
     [a[2][0]-b[2][0], a[2][1]-b[2][1], a[2][2]-b[2][2]]]
}

// ── matrix products ───────────────────────────────────────────────────────────

/// Matrix–matrix product A * B.
pub fn mat_mul<T: Scalar>(a: Mat3<T>, b: Mat3<T>) -> Mat3<T> {
    let mut c = zero_m::<T>();
    for i in 0..3 {
        for k in 0..3 {
            for j in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Matrix–vector product A * v (v treated as a column vector).
#[inline]
pub fn mat_vec<T: Scalar>(a: Mat3<T>, v: Vec3<T>) -> Vec3<T> {
    [dot(a[0], v), dot(a[1], v), dot(a[2], v)]
}

#[inline]
pub fn transpose<T: Scalar>(a: Mat3<T>) -> Mat3<T> {
    [[a[0][0], a[1][0], a[2][0]],
     [a[0][1], a[1][1], a[2][1]],
     [a[0][2], a[1][2], a[2][2]]]
}

// ── special matrices ──────────────────────────────────────────────────────────

/// 3×3 identity matrix.
#[inline]
pub fn eye<T: Scalar>() -> Mat3<T> {
    let z = T::zero();
    let o = T::one();
    [[o, z, z], [z, o, z], [z, z, o]]
}

/// Diagonal matrix from a vector.
#[inline]
pub fn diag<T: Scalar>(v: Vec3<T>) -> Mat3<T> {
    let z = T::zero();
    [[v[0], z, z], [z, v[1], z], [z, z, v[2]]]
}

/// Skew-symmetric (cross-product) matrix: tilde(v) * w == cross(v, w).
#[inline]
pub fn tilde<T: Scalar>(v: Vec3<T>) -> Mat3<T> {
    let z = T::zero();
    [[ z,    -v[2],  v[1]],
     [ v[2],  z,    -v[0]],
     [-v[1],  v[0],  z   ]]
}

/// Trace of a 3×3 matrix.
#[inline]
pub fn trace<T: Scalar>(a: Mat3<T>) -> T {
    a[0][0] + a[1][1] + a[2][2]
}

/// Determinant of a 3×3 matrix.
#[inline]
pub fn det3<T: Scalar>(a: Mat3<T>) -> T {
    a[0][0] * (a[1][1]*a[2][2] - a[1][2]*a[2][1])
  - a[0][1] * (a[1][0]*a[2][2] - a[1][2]*a[2][0])
  + a[0][2] * (a[1][0]*a[2][1] - a[1][1]*a[2][0])
}

/// Inverse of a 3×3 matrix.
pub fn inv3<T: Scalar>(a: Mat3<T>) -> Mat3<T> {
    let d = det3(a);
    let s = T::one() / d;
    [[(a[1][1]*a[2][2] - a[1][2]*a[2][1]) * s,
      (a[0][2]*a[2][1] - a[0][1]*a[2][2]) * s,
      (a[0][1]*a[1][2] - a[0][2]*a[1][1]) * s],
     [(a[1][2]*a[2][0] - a[1][0]*a[2][2]) * s,
      (a[0][0]*a[2][2] - a[0][2]*a[2][0]) * s,
      (a[0][2]*a[1][0] - a[0][0]*a[1][2]) * s],
     [(a[1][0]*a[2][1] - a[1][1]*a[2][0]) * s,
      (a[0][1]*a[2][0] - a[0][0]*a[2][1]) * s,
      (a[0][0]*a[1][1] - a[0][1]*a[1][0]) * s]]
}

/// Frobenius norm of a 3×3 matrix.
pub fn frob_norm<T: Scalar>(a: Mat3<T>) -> T {
    let mut s = T::zero();
    for row in &a { for &v in row { s += v * v; } }
    s.sqrt()
}

/// Infinity norm of a Vec3 (max |element|).
#[inline]
pub fn inf_norm_v<T: Scalar>(v: Vec3<T>) -> T {
    v[0].abs().max(v[1].abs()).max(v[2].abs())
}

// ── column extraction ─────────────────────────────────────────────────────────

#[inline]
pub fn col<T: Scalar>(a: Mat3<T>, j: usize) -> Vec3<T> {
    [a[0][j], a[1][j], a[2][j]]
}

// ── f64-only helpers (integrator state vectors, not part of AD path) ──────────

/// Infinity norm of a 30-element f64 state vector (used in RK7/8 step control).
pub fn inf_norm_30(x: &[f64; 30]) -> f64 {
    x.iter().cloned().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

/// x + s*dx  for 30-element f64 state vectors.
pub fn state_add_scaled(x: [f64; 30], s: f64, dx: [f64; 30]) -> [f64; 30] {
    let mut r = x;
    for i in 0..30 { r[i] += s * dx[i]; }
    r
}

/// x - y  for 30-element f64 state vectors.
pub fn state_sub(x: [f64; 30], y: [f64; 30]) -> [f64; 30] {
    let mut r = [0.0f64; 30];
    for i in 0..30 { r[i] = x[i] - y[i]; }
    r
}

/// x + Σᵢ sᵢ*kᵢ  for 30-element f64 state vectors.
pub fn state_combine(x: [f64; 30], terms: &[(&[f64; 30], f64)]) -> [f64; 30] {
    let mut r = x;
    for &(k, s) in terms {
        for i in 0..30 { r[i] += s * k[i]; }
    }
    r
}
