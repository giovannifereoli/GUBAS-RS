//! Expansion coefficients for the Hou 2016 mutual gravitational potential.
//!
//! Computes the `tk`, `a`, and `b` coefficient arrays required by the mutual
//! potential series.  All recursions mirror the original C++ implementation.
//!
//! # Coefficient layout
//! The 7-D `a` / `b` arrays are stored flat via [`t_ind`]:
//! ```text
//! flat_idx = a·dim⁶ + b·dim⁵ + c·dim⁴ + d·dim³ + e·dim² + f·dim + g
//! ```
//! where `dim = n + 1` and `n` is the truncation order.

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

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-14, "expected {b:.16}, got {a:.16}");
    }

    // ── factorial ────────────────────────────────────────────────────────────

    #[test] fn factorial_zero() { close(factorial(0.0), 1.0); }
    #[test] fn factorial_one()  { close(factorial(1.0), 1.0); }
    #[test] fn factorial_five() { close(factorial(5.0), 120.0); }

    // ── ifact ────────────────────────────────────────────────────────────────

    #[test] fn ifact_zero() { close(ifact(0), 1.0); }
    #[test] fn ifact_one()  { close(ifact(1), 1.0); }
    #[test] fn ifact_five() { close(ifact(5), 120.0); }

    // ── t_ind ────────────────────────────────────────────────────────────────

    #[test]
    fn t_ind_all_zero() {
        assert_eq!(t_ind(0, 0, 0, 0, 0, 0, 0, 5), 0);
    }

    #[test]
    fn t_ind_unit_last() {
        // g=1, dim=3 → flat index = 1
        assert_eq!(t_ind(0, 0, 0, 0, 0, 0, 1, 3), 1);
    }

    #[test]
    fn t_ind_unit_f() {
        // f=1, g=0, dim=3 → 1*3 = 3
        assert_eq!(t_ind(0, 0, 0, 0, 0, 1, 0, 3), 3);
    }

    #[test]
    fn t_ind_unit_a() {
        // a=1, all others=0, dim=4 → 4^6 = 4096
        assert_eq!(t_ind(1, 0, 0, 0, 0, 0, 0, 4), 4_usize.pow(6));
    }

    // ── coeff_vec_len ────────────────────────────────────────────────────────

    #[test]
    fn coeff_vec_len_zero() {
        // t_ind(0,..,0, dim=1) = 0 → length = 1
        assert_eq!(coeff_vec_len(0), 1);
    }

    #[test]
    fn coeff_vec_len_two() {
        // t_ind(2,..,2, dim=3) = 2*(3^6+3^5+3^4+3^3+3^2+3+1) = 2*1093 = 2186 → length=2187
        assert_eq!(coeff_vec_len(2), 2187);
    }

    // ── tk_calc ──────────────────────────────────────────────────────────────

    #[test]
    fn tk_calc_n0_t00_is_one() {
        // n=0 even: t[0][0] = (−1)^0 · 0! / (2^0 · (0!)²) = 1
        close(tk_calc(0)[0][0], 1.0);
    }

    #[test]
    fn tk_calc_n1_t10_is_one() {
        // n=1 odd: t[1][0] = (−1)^0 · 1! / (2^0 · (0!)²) = 1
        close(tk_calc(2)[1][0], 1.0);
    }

    #[test]
    fn tk_calc_n2_t20_is_minus_half() {
        // n=2 even: t[2][0] = (−1)^1 · 2! / (2^2 · (1!)²) = −2/4 = −0.5
        close(tk_calc(2)[2][0], -0.5);
    }

    #[test]
    fn tk_calc_dimensions() {
        let m = 4;
        let tk = tk_calc(m);
        assert_eq!(tk.len(), m + 1);
        let ncols = m / 2 + 2;
        for row in &tk { assert_eq!(row.len(), ncols); }
    }

    // ── a_calc ───────────────────────────────────────────────────────────────

    #[test]
    fn a_calc_n0_monopole() {
        let a = a_calc(0);
        assert_eq!(a.len(), 1);
        close(a[0], 1.0);
    }

    #[test]
    fn a_calc_n1_seeds() {
        // Order-1 seed values from Hou 2016 Table 1
        let dim = 2; // n+1
        let a = a_calc(1);
        close(a[t_ind(1, 1, 0, 0, 0, 0, 0, dim)],  1.0);
        close(a[t_ind(1, 0, 1, 0, 0, 0, 0, dim)],  1.0);
        close(a[t_ind(1, 0, 0, 1, 0, 0, 0, dim)],  1.0);
        close(a[t_ind(1, 0, 0, 0, 1, 0, 0, dim)], -1.0);
        close(a[t_ind(1, 0, 0, 0, 0, 1, 0, dim)], -1.0);
        close(a[t_ind(1, 0, 0, 0, 0, 0, 1, dim)], -1.0);
    }

    // ── b_calc ───────────────────────────────────────────────────────────────

    #[test]
    fn b_calc_n0_monopole() {
        let b = b_calc(0);
        assert_eq!(b.len(), 1);
        close(b[0], 1.0);
    }

    #[test]
    fn b_calc_n2_seeds() {
        // Order-2 seed values from Hou 2016 (B tensor structure)
        let dim = 3; // n+1
        let b = b_calc(2);
        close(b[t_ind(2, 2, 0, 0, 0, 0, 0, dim)],  1.0);
        close(b[t_ind(2, 0, 2, 0, 0, 0, 0, dim)],  1.0);
        close(b[t_ind(2, 0, 0, 2, 0, 0, 0, dim)],  1.0);
        close(b[t_ind(2, 0, 0, 0, 2, 0, 0, dim)],  1.0);
        close(b[t_ind(2, 0, 0, 0, 0, 2, 0, dim)],  1.0);
        close(b[t_ind(2, 0, 0, 0, 0, 0, 2, dim)],  1.0);
        close(b[t_ind(2, 1, 0, 0, 1, 0, 0, dim)], -2.0);
        close(b[t_ind(2, 0, 1, 0, 0, 1, 0, dim)], -2.0);
        close(b[t_ind(2, 0, 0, 1, 0, 0, 1, dim)], -2.0);
    }
}
