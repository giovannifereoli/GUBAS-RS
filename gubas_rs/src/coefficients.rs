// coefficients.rs — Hou 2016 expansion coefficients
//
// Computes tk, a, and b coefficient arrays used by the mutual potential
// and its derivatives.  All logic mirrors the C++ functions exactly.

/// Double factorial helper (matches `double factorial(double x)` in C++).
pub fn factorial(x: f64) -> f64 {
    let mut f = 1.0_f64;
    let mut v = x;
    while v > 1.0 { f *= v; v -= 1.0; }
    f
}

/// Integer factorial used in inertia integrals (avoids float rounding).
pub fn ifact(n: usize) -> f64 {
    (1..=n).fold(1.0_f64, |acc, k| acc * k as f64)
}

// ── t_ind ─────────────────────────────────────────────────────────────────────

/// Map 7 coefficient indices (k, i1..i6) and the dimension `dim = n+1` to a
/// single flat index into the 1-D `a` or `b` Vec.
///
/// The C++ `sp_mat` used to store these is effectively just a 1-D array
/// accessed through this index function.
#[allow(clippy::too_many_arguments)]
pub fn t_ind(a: usize, b: usize, c: usize, d: usize, e: usize, f: usize,
             g: usize, dim: usize) -> usize {
    // For all practical expansion orders the sum a+..+g << dim^7, so the
    // first branch of the C++ conditional is never taken.  We implement both
    // branches for correctness.
    let sum = a + b + c + d + e + f + g;
    let dim7 = dim.pow(7);
    if sum > dim7 {
        (dim - a) * dim.pow(6)
            + (dim - b) * dim.pow(5)
            + (dim - c) * dim.pow(4)
            + (dim - d) * dim.pow(3)
            + (dim - e) * dim.pow(2)
            + (dim - f) * dim
            + g
    } else {
        a * dim.pow(6)
            + b * dim.pow(5)
            + c * dim.pow(4)
            + d * dim.pow(3)
            + e * dim.pow(2)
            + f * dim
            + g
    }
}

/// Size needed for the a/b coefficient Vec given truncation order `n`.
pub fn coeff_vec_len(n: usize) -> usize {
    t_ind(n, n, n, n, n, n, n, n + 1) + 1
}

// ── tk_calc ───────────────────────────────────────────────────────────────────

/// Compute the tk expansion coefficients up to truncation order `m`.
///
/// Returns a 2-D Vec `tk[n][i]` where `n` is the expansion order (0..=m)
/// and `i` is the recursion index.  Mirrors `void tk_calc(int m, mat* t)`.
pub fn tk_calc(m: usize) -> Vec<Vec<f64>> {
    let ncols = m / 2 + 2;
    let mut t = vec![vec![0.0_f64; ncols]; m + 1];
    for n in 0..=m {
        let nf = n as f64;
        if n % 2 == 1 {
            // odd
            t[n][0] = (-1.0_f64).powf((nf - 1.0) / 2.0) * factorial(nf)
                / (2.0_f64.powf(nf - 1.0) * factorial((nf - 1.0) / 2.0).powi(2));
        } else {
            // even
            t[n][0] = (-1.0_f64).powf(nf / 2.0) * factorial(nf)
                / (2.0_f64.powf(nf) * factorial(nf / 2.0).powi(2));
        }
        let mut k = (n % 2) as f64; // starts at 0 (even n) or 1 (odd n)
        let mut i = 1usize;
        while k <= nf {
            t[n][i] = -(nf - k) * (nf + k + 1.0) * t[n][i - 1]
                / ((k + 2.0) * (k + 1.0));
            k += 2.0;
            i += 1;
        }
    }
    t
}

// ── a_calc ────────────────────────────────────────────────────────────────────

/// Compute the `a` expansion coefficients for truncation order `n`.
///
/// The 7-D coefficient array is stored in a flat Vec indexed by `t_ind`.
/// Mirrors `void a_calc(int n, sp_mat* a)`.
pub fn a_calc(n: usize) -> Vec<f64> {
    let len = coeff_vec_len(n);
    let dim = n + 1;
    let mut a = vec![0.0_f64; len];
    a[0] = 1.0; // a[0][0][0][0][0][0][0] = 1
    if n == 0 { return a; }

    // Order 1 initial values (from Hou paper)
    a[t_ind(1, 1, 0, 0, 0, 0, 0, dim)] = 1.0;
    a[t_ind(1, 0, 1, 0, 0, 0, 0, dim)] = 1.0;
    a[t_ind(1, 0, 0, 1, 0, 0, 0, dim)] = 1.0;
    a[t_ind(1, 0, 0, 0, 1, 0, 0, dim)] = -1.0;
    a[t_ind(1, 0, 0, 0, 0, 1, 0, dim)] = -1.0;
    a[t_ind(1, 0, 0, 0, 0, 0, 1, dim)] = -1.0;

    if n < 2 { return a; }

    for k in 2..=n {
        for i1 in 0..=k {
          for i2 in 0..=(k - i1) {
            for i3 in 0..=(k - i1 - i2) {
              for i4 in 0..=(k - i1 - i2 - i3) {
                for i5 in 0..=(k - i1 - i2 - i3 - i4) {
                  let i6 = k - i1 - i2 - i3 - i4 - i5;
                  let idx = t_ind(k, i1, i2, i3, i4, i5, i6, dim);
                  if i1 > 0 { a[idx] += a[t_ind(k-1, i1-1, i2, i3, i4, i5, i6, dim)]; }
                  if i2 > 0 { a[idx] += a[t_ind(k-1, i1, i2-1, i3, i4, i5, i6, dim)]; }
                  if i3 > 0 { a[idx] += a[t_ind(k-1, i1, i2, i3-1, i4, i5, i6, dim)]; }
                  if i4 > 0 { a[idx] -= a[t_ind(k-1, i1, i2, i3, i4-1, i5, i6, dim)]; }
                  if i5 > 0 { a[idx] -= a[t_ind(k-1, i1, i2, i3, i4, i5-1, i6, dim)]; }
                  if i6 > 0 { a[idx] -= a[t_ind(k-1, i1, i2, i3, i4, i5, i6-1, dim)]; }
                }
              }
            }
          }
        }
    }
    a
}

// ── b_calc ────────────────────────────────────────────────────────────────────

/// Compute the `b` expansion coefficients for truncation order `n`.
///
/// Mirrors `void b_calc(int n, sp_mat* b)`.
pub fn b_calc(n: usize) -> Vec<f64> {
    let len = coeff_vec_len(n);
    let dim = n + 1;
    let mut b = vec![0.0_f64; len];
    b[0] = 1.0; // b[0][0][0][0][0][0][0] = 1
    if n < 2 { return b; }

    // Order 2 initial values (from Hou paper)
    b[t_ind(2, 2, 0, 0, 0, 0, 0, dim)] = 1.0;
    b[t_ind(2, 0, 2, 0, 0, 0, 0, dim)] = 1.0;
    b[t_ind(2, 0, 0, 2, 0, 0, 0, dim)] = 1.0;
    b[t_ind(2, 0, 0, 0, 2, 0, 0, dim)] = 1.0;
    b[t_ind(2, 0, 0, 0, 0, 2, 0, dim)] = 1.0;
    b[t_ind(2, 0, 0, 0, 0, 0, 2, dim)] = 1.0;
    b[t_ind(2, 1, 0, 0, 1, 0, 0, dim)] = -2.0;
    b[t_ind(2, 0, 1, 0, 0, 1, 0, dim)] = -2.0;
    b[t_ind(2, 0, 0, 1, 0, 0, 1, dim)] = -2.0;

    // Recursion: loop from n down to 0
    for k in (0..=n).rev() {
        let nk = n - k; // the "n-k" in the Hou paper for b
        for j1 in 0..=nk {
          for j2 in 0..=(nk - j1) {
            for j3 in 0..=(nk - j1 - j2) {
              for j4 in 0..=(nk - j1 - j2 - j3) {
                for j5 in 0..=(nk - j1 - j2 - j3 - j4) {
                  let j6 = nk - j1 - j2 - j3 - j4 - j5;
                  if nk > 2 {
                    let idx = t_ind(nk, j1, j2, j3, j4, j5, j6, dim);
                    let nk2 = nk - 2;
                    if j1 > 0 && j4 > 0 { b[idx] += -2.0 * b[t_ind(nk2, j1-1, j2, j3, j4-1, j5, j6, dim)]; }
                    if j2 > 0 && j5 > 0 { b[idx] += -2.0 * b[t_ind(nk2, j1, j2-1, j3, j4, j5-1, j6, dim)]; }
                    if j3 > 0 && j6 > 0 { b[idx] += -2.0 * b[t_ind(nk2, j1, j2, j3-1, j4, j5, j6-1, dim)]; }
                    if j1 > 1 { b[idx] += b[t_ind(nk2, j1-2, j2, j3, j4, j5, j6, dim)]; }
                    if j2 > 1 { b[idx] += b[t_ind(nk2, j1, j2-2, j3, j4, j5, j6, dim)]; }
                    if j3 > 1 { b[idx] += b[t_ind(nk2, j1, j2, j3-2, j4, j5, j6, dim)]; }
                    if j4 > 1 { b[idx] += b[t_ind(nk2, j1, j2, j3, j4-2, j5, j6, dim)]; }
                    if j5 > 1 { b[idx] += b[t_ind(nk2, j1, j2, j3, j4, j5-2, j6, dim)]; }
                    if j6 > 1 { b[idx] += b[t_ind(nk2, j1, j2, j3, j4, j5, j6-2, dim)]; }
                  }
                }
              }
            }
          }
        }
    }
    b
}
