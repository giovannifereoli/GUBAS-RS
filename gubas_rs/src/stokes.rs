// stokes.rs — Stokes coefficients ↔ inertia integrals (Tricarico 2008)
//
// Forward direction: T_{ijk} → C_{lm}, S_{lm}
// Reference: P. Tricarico (2008) "Global Gravity Inversion of Bodies with
//            Arbitrary Shape", Eqs. 16–17 (unnormalized).
//
// Conventions:
//   N_{ijk} = T_{ijk} / (M · r₀^l)   where l = i+j+k
//   M = T_{000}  (total mass),  r₀ = reference radius (same units as T)
//
// Normalization:
//   Unnormalized:    C_lm  (Tricarico Eqs. 16–17 directly)
//   FullyNormalized: C̄_lm = √((2l+1)(2−δ₀ₘ)(l−m)!/(l+m)!) · C_lm

use crate::types::Cube;

// ── Normalization ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Normalization {
    Unnormalized,
    FullyNormalized,
}

// ── StokesCoeffs ─────────────────────────────────────────────────────────────

/// Stokes coefficients indexed as `c[l][m]` and `s[l][m]`,
/// for l = 0..=max_degree, m = 0..=l.  `s[l][0]` = 0 by convention.
#[derive(Debug, Clone)]
pub struct StokesCoeffs {
    pub c:          Vec<Vec<f64>>,
    pub s:          Vec<Vec<f64>>,
    pub max_degree: usize,
}

impl StokesCoeffs {
    pub fn new(max_degree: usize) -> Self {
        let c: Vec<Vec<f64>> = (0..=max_degree).map(|l| vec![0.0; l + 1]).collect();
        let s: Vec<Vec<f64>> = (0..=max_degree).map(|l| vec![0.0; l + 1]).collect();
        Self { c, s, max_degree }
    }

    /// Print a compact table: l, m, C_lm, S_lm.
    pub fn print(&self) {
        println!("{:>3} {:>3}  {:>18}  {:>18}", "l", "m", "C_lm", "S_lm");
        println!("{}", "-".repeat(50));
        for l in 0..=self.max_degree {
            for m in 0..=l {
                println!("{:>3} {:>3}  {:>18.10e}  {:>18.10e}",
                    l, m, self.c[l][m], self.s[l][m]);
            }
        }
    }
}

// ── Combinatorics helpers ─────────────────────────────────────────────────────

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0_f64, |acc, k| acc * k as f64)
}

fn binom(n: usize, k: usize) -> f64 {
    if k > n { return 0.0; }
    factorial(n) / (factorial(k) * factorial(n - k))
}

/// Rising factorial (Pochhammer): (a)_m = a(a+1)…(a+m−1).  Returns 1 if m=0.
fn pochhammer(a: i64, m: usize) -> f64 {
    (0..m).fold(1.0_f64, |acc, k| acc * (a + k as i64) as f64)
}

fn norm_factor(l: usize, m: usize, norm: Normalization) -> f64 {
    match norm {
        Normalization::Unnormalized => 1.0,
        Normalization::FullyNormalized => {
            let delta = if m == 0 { 1.0 } else { 0.0 };
            ((2.0 * l as f64 + 1.0) * (2.0 - delta)
             * factorial(l - m) / factorial(l + m)).sqrt()
        }
    }
}

// ── Forward conversion ────────────────────────────────────────────────────────

/// Convert GUBAS inertia integrals to Stokes gravity coefficients.
///
/// # Arguments
/// * `ta`         — T_{ijk} cube (kg · km^l, indexed as ta.get(i,j,k))
/// * `mass`       — total mass M = ta.get(0,0,0)  (kg)
/// * `r0`         — reference radius (km)
/// * `max_degree` — highest harmonic degree (must be ≤ ta.order)
/// * `norm`       — `Unnormalized` or `FullyNormalized`
pub fn nijk_to_clm_slm(
    ta:         &Cube<f64>,
    mass:       f64,
    r0:         f64,
    max_degree: usize,
    norm:       Normalization,
) -> StokesCoeffs {
    let mut out = StokesCoeffs::new(max_degree);

    for l in 0..=max_degree {
        let r0_l  = r0.powi(l as i32);
        let pre_l = (0.5_f64).powi(l as i32) * norm_factor(l, 0, norm); // will override m-dependent
        let _ = pre_l; // computed per (l,m) below

        for m in 0..=l {
            let delta = if m == 0 { 1.0 } else { 0.0 };
            let factor = (0.5_f64).powi(l as i32)
                * (2.0 - delta) * factorial(l - m) / factorial(l + m)
                * norm_factor(l, m, norm);

            // ── C_lm (Tricarico Eq. 16) ─────────────────────────────────────
            let mut c_sum = 0.0;
            for p in 0..=(l / 2) {
                let poch = pochhammer(
                    (l as i64) - (m as i64) - 2 * (p as i64) + 1, m);
                if poch == 0.0 { continue; }
                let cp_l  = binom(l, p);
                let cp_2l = binom(2 * l - 2 * p, l);
                let coeff_lp = cp_l * cp_2l * poch;

                for q in 0..=(m / 2) {
                    let cm_2q = binom(m, 2 * q);
                    if cm_2q == 0.0 { continue; }
                    let sign_pq = if (p + q) % 2 == 0 { 1.0 } else { -1.0 };

                    for nux in 0..=p {
                        for nuy in 0..=(p - nux) {
                            let nuz = p - nux - nuy;
                            let ix = (m as i64) - 2*(q as i64) + 2*(nux as i64);
                            let iy =              2*(q as i64) + 2*(nuy as i64);
                            let iz = (l as i64) - (m as i64)
                                   - 2*(nux as i64) - 2*(nuy as i64);
                            if ix < 0 || iz < 0 { continue; }
                            let (ix, iy, iz) =
                                (ix as usize, iy as usize, iz as usize);
                            if ix > ta.order || iy > ta.order || iz > ta.order {
                                continue;
                            }
                            let multi = factorial(p)
                                / (factorial(nux) * factorial(nuy) * factorial(nuz));
                            let n_ijk = ta.get(ix, iy, iz) / (mass * r0_l);
                            c_sum += sign_pq * coeff_lp * cm_2q * multi * n_ijk;
                        }
                    }
                }
            }
            out.c[l][m] = factor * c_sum;

            // ── S_lm (Tricarico Eq. 17, only for m ≥ 1) ─────────────────────
            if m == 0 { continue; }
            let mut s_sum = 0.0;
            for p in 0..=(l / 2) {
                let poch = pochhammer(
                    (l as i64) - (m as i64) - 2 * (p as i64) + 1, m);
                if poch == 0.0 { continue; }
                let cp_l  = binom(l, p);
                let cp_2l = binom(2 * l - 2 * p, l);
                let coeff_lp = cp_l * cp_2l * poch;

                for q in 0..=(m.saturating_sub(1) / 2) {
                    let cm_2q1 = binom(m, 2 * q + 1);
                    if cm_2q1 == 0.0 { continue; }
                    let sign_pq = if (p + q) % 2 == 0 { 1.0 } else { -1.0 };

                    for nux in 0..=p {
                        for nuy in 0..=(p - nux) {
                            let nuz = p - nux - nuy;
                            let ix = (m as i64) - 2*(q as i64) - 1 + 2*(nux as i64);
                            let iy =              2*(q as i64) + 1 + 2*(nuy as i64);
                            let iz = (l as i64) - (m as i64)
                                   - 2*(nux as i64) - 2*(nuy as i64);
                            if ix < 0 || iz < 0 { continue; }
                            let (ix, iy, iz) =
                                (ix as usize, iy as usize, iz as usize);
                            if ix > ta.order || iy > ta.order || iz > ta.order {
                                continue;
                            }
                            let multi = factorial(p)
                                / (factorial(nux) * factorial(nuy) * factorial(nuz));
                            let n_ijk = ta.get(ix, iy, iz) / (mass * r0_l);
                            s_sum += sign_pq * coeff_lp * cm_2q1 * multi * n_ijk;
                        }
                    }
                }
            }
            out.s[l][m] = factor * s_sum;
        }
    }
    out
}

// ── Sensitivity matrix ────────────────────────────────────────────────────────

/// Return all (i, j, k) index triples with `min_degree ≤ i+j+k ≤ max_degree`.
/// Ordered by (l ascending, then lexicographic).
pub fn inertia_indices(min_degree: usize, max_degree: usize)
    -> Vec<(usize, usize, usize)>
{
    let mut out = Vec::new();
    for l in min_degree..=max_degree {
        for i in 0..=l {
            for j in 0..=(l - i) {
                let k = l - i - j;
                out.push((i, j, k));
            }
        }
    }
    out
}

/// Build the linear sensitivity matrix M of shape (N_cs × N_theta) such that
/// `C_flat = M * T_theta_flat` (both unnormalized or normalized consistently).
///
/// Row ordering (C/S interleaved per degree, starting from min_degree):
///   l=min_degree: C_{l,0}, C_{l,1}, S_{l,1}, C_{l,2}, S_{l,2}, …, C_{l,l}, S_{l,l}
///   l=min_degree+1: …
///
/// Column ordering follows `theta_indices` (the order returned by `inertia_indices`
/// or as passed by the caller).
///
/// # Arguments
/// * `theta_indices` — list of (i,j,k) for the selected T_a entries
/// * `mass`, `r0`   — normalization constants
/// * `min_degree`, `max_degree` — harmonic degree range for the rows
/// * `norm`         — normalization convention
pub fn stokes_matrix(
    theta_indices: &[(usize, usize, usize)],
    mass:         f64,
    r0:           f64,
    min_degree:   usize,
    max_degree:   usize,
    norm:         Normalization,
) -> Vec<Vec<f64>> {
    let n_theta = theta_indices.len();

    // Count rows: 2l+1 per degree (C_{l,0} + pairs (C_{l,m}, S_{l,m}) for m=1..l)
    let n_cs: usize = (min_degree..=max_degree).map(|l| 2*l + 1).sum();
    let mut mat = vec![vec![0.0_f64; n_theta]; n_cs];

    // For each T_{ijk_k} = 1, rest = 0 → compute ΔC/S
    let mut unit_cube = Cube::<f64>::new(max_degree);
    for (k, &(ii, jj, kk)) in theta_indices.iter().enumerate() {
        unit_cube.zeros();
        unit_cube.set(ii, jj, kk, 1.0);

        // We set mass=1 and r0=1 for the unit-impulse cube; the actual
        // normalization factors (mass, r0) cancel because C_{lm} = Σ α * T/(M r0^l)
        // and we set T_{ijk_k}=1, M=1, r0=1 → gives ∂C_{lm}/∂(T_{ijk_k}/(M r0^l))
        // Then divide by (M r0^l) at the right degree.
        let degree_k = ii + jj + kk;
        let r0_l     = r0.powi(degree_k as i32);
        let cs = nijk_to_clm_slm(&unit_cube, 1.0, 1.0, max_degree, norm);

        let mut row = 0_usize;
        for l in min_degree..=max_degree {
            // C_{l,0}
            mat[row][k] = cs.c[l][0] * mass * r0_l;
            row += 1;
            for m in 1..=l {
                mat[row][k]     = cs.c[l][m] * mass * r0_l;
                mat[row + 1][k] = cs.s[l][m] * mass * r0_l;
                row += 2;
            }
        }
    }
    mat
}

/// Ordered list of (l, m, is_sin) for the rows of `stokes_matrix`.
/// `is_sin = false` → C_{lm}, `is_sin = true` → S_{lm}.
pub fn stokes_row_labels(min_degree: usize, max_degree: usize)
    -> Vec<(usize, usize, bool)>
{
    let mut out = Vec::new();
    for l in min_degree..=max_degree {
        out.push((l, 0, false)); // C_{l,0}
        for m in 1..=l {
            out.push((l, m, false)); // C_{l,m}
            out.push((l, m, true));  // S_{l,m}
        }
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Cube;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {b:.6e}, got {a:.6e} (diff={:.2e})", (a-b).abs());
    }

    // ── inertia_indices ───────────────────────────────────────────────────────

    #[test]
    fn inertia_indices_degree2_count() {
        // Monomials of total degree 2: (2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2) → 6
        let idx = inertia_indices(2, 2);
        assert_eq!(idx.len(), 6);
        // All must satisfy i+j+k = 2
        for (i, j, k) in &idx { assert_eq!(i + j + k, 2); }
    }

    #[test]
    fn inertia_indices_cumulative_count() {
        // l=0: 1,  l=1: 3,  l=2: 6 → total 0..=2: 10
        let idx = inertia_indices(0, 2);
        assert_eq!(idx.len(), 10);
    }

    #[test]
    fn inertia_indices_includes_pure_axes() {
        let idx = inertia_indices(2, 2);
        assert!(idx.contains(&(2, 0, 0)));
        assert!(idx.contains(&(0, 2, 0)));
        assert!(idx.contains(&(0, 0, 2)));
        assert!(idx.contains(&(1, 1, 0)));
    }

    // ── stokes_row_labels ─────────────────────────────────────────────────────

    #[test]
    fn stokes_row_labels_degree2() {
        let labels = stokes_row_labels(2, 2);
        // C20, C21, S21, C22, S22  → 5 rows
        assert_eq!(labels.len(), 5);
        assert_eq!(labels[0], (2, 0, false)); // C20
        assert_eq!(labels[1], (2, 1, false)); // C21
        assert_eq!(labels[2], (2, 1, true));  // S21
        assert_eq!(labels[3], (2, 2, false)); // C22
        assert_eq!(labels[4], (2, 2, true));  // S22
    }

    #[test]
    fn stokes_row_labels_range_count() {
        // For l=2..=4: 5 + 7 + 9 = 21 rows
        let labels = stokes_row_labels(2, 4);
        assert_eq!(labels.len(), 21);
    }

    // ── nijk_to_clm_slm — point-mass / zeroth degree ─────────────────────────

    #[test]
    fn c00_is_one_for_any_body() {
        // C_{00} = T_{000} / (mass * r0^0) = 1 by normalization
        let mass = 1.23e12;
        let mut ta = Cube::<f64>::new(4);
        ta.set(0, 0, 0, mass);
        ta.set(2, 0, 0, mass * 0.1);   // some non-zero higher moments
        ta.set(0, 2, 0, mass * 0.08);
        ta.set(0, 0, 2, mass * 0.06);
        let cs = nijk_to_clm_slm(&ta, mass, 1.0, 2, Normalization::Unnormalized);
        close(cs.c[0][0], 1.0, 1e-12);
    }

    // ── nijk_to_clm_slm — sphere (all C_{lm}=0 for l≥1) ─────────────────────

    #[test]
    fn sphere_degree2_all_zero() {
        // For a sphere: T_{200}=T_{020}=T_{002}=M*r²/5, all cross-terms=0
        // → C_{20}=C_{21}=C_{22}=S_{21}=S_{22}=0
        let mass = 1.0;
        let r = 2.0; // arbitrary radius (km), r0 = r for self-consistency
        let mut ta = Cube::<f64>::new(4);
        ta.set(0, 0, 0, mass);
        ta.set(2, 0, 0, mass * r * r / 5.0);
        ta.set(0, 2, 0, mass * r * r / 5.0);
        ta.set(0, 0, 2, mass * r * r / 5.0);
        let cs = nijk_to_clm_slm(&ta, mass, r, 2, Normalization::Unnormalized);
        close(cs.c[2][0], 0.0, 1e-15);
        close(cs.c[2][1], 0.0, 1e-15);
        close(cs.c[2][2], 0.0, 1e-15);
        close(cs.s[2][1], 0.0, 1e-15);
        close(cs.s[2][2], 0.0, 1e-15);
    }

    // ── nijk_to_clm_slm — oblate spheroid analytic C20 ───────────────────────

    #[test]
    fn oblate_c20_analytic() {
        // Oblate spheroid: equatorial semi-axes a=b=2 km, polar c=1 km.
        // Using r0 = 1 km, mass = 1 kg (arbitrary — cancels in N_{ijk}).
        //
        // T_{200}=T_{020}=M*a²/5=M*4/5,  T_{002}=M*c²/5=M/5.
        //
        // Tricarico Eq.16 for C_{20}:
        //   C_{20} = (1/4)(4*N_{002} - 2*N_{020} - 2*N_{200})
        //          = (1/4)(4*(1/5) - 2*(4/5) - 2*(4/5))  [r0=1, mass cancels]
        //          = (1/4)(-12/5) = -3/5
        let mass = 1.0;
        let r0   = 1.0;
        let a = 2.0; let c = 1.0;
        let mut ta = Cube::<f64>::new(4);
        ta.set(0, 0, 0, mass);
        ta.set(2, 0, 0, mass * a * a / 5.0);
        ta.set(0, 2, 0, mass * a * a / 5.0);
        ta.set(0, 0, 2, mass * c * c / 5.0);
        let cs = nijk_to_clm_slm(&ta, mass, r0, 2, Normalization::Unnormalized);
        close(cs.c[2][0], -3.0 / 5.0, 1e-14);
        // Oblate with a=b → C22 = 0 (equatorial symmetry)
        close(cs.c[2][2], 0.0, 1e-14);
        close(cs.s[2][2], 0.0, 1e-14);
    }

    #[test]
    fn triaxial_c22_nonzero() {
        // Triaxial body: a≠b (equatorial asymmetry) → C22 ≠ 0.
        // C_{22} = 0.25 * 12 * (N_{200} - N_{020})  (derived in Tricarico Eq.16 for m=2)
        //        = 3 * (N_{200} - N_{020})
        // With r0=1: N_{ijk} = T_{ijk}/mass.
        let mass = 1.0;
        let a = 3.0; let b = 2.0; let c = 1.0;
        let mut ta = Cube::<f64>::new(4);
        ta.set(0, 0, 0, mass);
        ta.set(2, 0, 0, mass * a * a / 5.0);
        ta.set(0, 2, 0, mass * b * b / 5.0);
        ta.set(0, 0, 2, mass * c * c / 5.0);
        let cs = nijk_to_clm_slm(&ta, mass, 1.0, 2, Normalization::Unnormalized);
        // C_22 = (2-0)*(0!/4!) * (1/4)*12*(a²-b²)/5 = (a²-b²)/20
        let expected_c22 = (a * a - b * b) / 20.0;
        close(cs.c[2][2], expected_c22, 1e-14);
        // S22 = 0 for a body aligned with principal axes
        close(cs.s[2][2], 0.0, 1e-14);
    }

    // ── stokes_matrix — shape ─────────────────────────────────────────────────

    #[test]
    fn stokes_matrix_shape_degree2() {
        let theta = inertia_indices(2, 2);
        let mat = stokes_matrix(&theta, 1.0, 1.0, 2, 2, Normalization::Unnormalized);
        // 5 rows (C20, C21, S21, C22, S22),  6 columns (monomials of degree 2)
        assert_eq!(mat.len(), 5);
        assert_eq!(mat[0].len(), 6);
    }

    #[test]
    fn stokes_matrix_shape_degree2_to_4() {
        let theta = inertia_indices(2, 4);
        // n_theta = 6 + 10 + 15 = 31
        assert_eq!(theta.len(), 31);
        let mat = stokes_matrix(&theta, 1.0, 1.0, 2, 4, Normalization::Unnormalized);
        // n_cs = 5 + 7 + 9 = 21
        assert_eq!(mat.len(), 21);
        for row in &mat { assert_eq!(row.len(), 31); }
    }

    // ── Consistency: stokes_matrix row == nijk_to_clm_slm column ─────────────

    #[test]
    fn matrix_times_nvec_equals_direct_cs() {
        // With mass=1, r0=1: M * N_flat should reproduce C/S from nijk_to_clm_slm.
        // We test an oblate spheroid (all cross-terms zero so N is sparse).
        let mass = 1.0;
        let a = 2.0; let c = 1.0;
        let mut ta = Cube::<f64>::new(4);
        ta.set(0, 0, 0, mass);
        ta.set(2, 0, 0, mass * a * a / 5.0);
        ta.set(0, 2, 0, mass * a * a / 5.0);
        ta.set(0, 0, 2, mass * c * c / 5.0);

        let theta = inertia_indices(2, 2);
        let mat = stokes_matrix(&theta, mass, 1.0, 2, 2, Normalization::Unnormalized);
        let cs  = nijk_to_clm_slm(&ta, mass, 1.0, 2, Normalization::Unnormalized);

        // Build N vector (theta is degree-2 indices only)
        let n_vec: Vec<f64> = theta.iter()
            .map(|&(i, j, k)| ta.get(i, j, k) / mass)   // N_{ijk} = T/(mass * r0^2), r0=1
            .collect();

        // Ordered rows: C20, C21, S21, C22, S22
        let c_from_mat: Vec<f64> = mat.iter().map(|row|
            row.iter().zip(n_vec.iter()).map(|(m, n)| m * n).sum()
        ).collect();

        let labels = stokes_row_labels(2, 2);
        let c_direct: Vec<f64> = labels.iter().map(|&(l, m, is_sin)| {
            if is_sin { cs.s[l][m] } else { cs.c[l][m] }
        }).collect();

        for (i, (&cf, &cd)) in c_from_mat.iter().zip(c_direct.iter()).enumerate() {
            assert!((cf - cd).abs() < 1e-13,
                "row {i}: M*N={cf:.6e}, direct={cd:.6e}");
        }
    }
}