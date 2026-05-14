// inertia.rs — Inertia integral and moment-of-inertia functions
//
// All functions mirror their C++ counterparts in hou_cpp.cpp.
// Units: km for length, kg for mass (as in the original code).

use crate::coefficients::ifact;
use crate::math3::{Mat3, Scalar, Vec3, det3};
use crate::types::Cube;
use num_traits::NumCast;

// ── Q_ijk ─────────────────────────────────────────────────────────────────────

/// Q parameter as defined by Hou 2016.
pub fn q_ijk(i: f64, j: f64, k: f64) -> f64 {
    ifact(i as usize) * ifact(j as usize) * ifact(k as usize)
        / ifact((i + j + k + 3.0) as usize)
}

// ── tet_sums ──────────────────────────────────────────────────────────────────

/// Summation over a tetrahedron with one vertex at the origin.
/// Vertices are at coordinates (x1,y1,z1), (x2,y2,z2), (x3,y3,z3) in km.
///
/// Mirrors `double tet_sums(...)` in C++.
#[allow(clippy::too_many_arguments)]
pub fn tet_sums(l: f64, m: f64, n: f64,
                x1: f64, x2: f64, x3: f64,
                y1: f64, y2: f64, y3: f64,
                z1: f64, z2: f64, z3: f64) -> f64 {
    let mut s = 0.0_f64;
    let li = l as usize;
    let mi = m as usize;
    let ni = n as usize;
    let lf = ifact(li);
    let mf = ifact(mi);
    let nf = ifact(ni);
    for i1 in 0..=li {
        for j1 in 0..=(li - i1) {
            let c_l = lf / (ifact(i1) * ifact(j1) * ifact(li - i1 - j1));
            for i2 in 0..=mi {
                for j2 in 0..=(mi - i2) {
                    let c_m = mf / (ifact(i2) * ifact(j2) * ifact(mi - i2 - j2));
                    for i3 in 0..=ni {
                        for j3 in 0..=(ni - i3) {
                            let c_n = nf / (ifact(i3) * ifact(j3) * ifact(ni - i3 - j3));
                            s += c_l * c_m * c_n
                                * x1.powi(i1 as i32) * x2.powi(j1 as i32)
                                    * x3.powi((li - i1 - j1) as i32)
                                * y1.powi(i2 as i32) * y2.powi(j2 as i32)
                                    * y3.powi((mi - i2 - j2) as i32)
                                * z1.powi(i3 as i32) * z2.powi(j3 as i32)
                                    * z3.powi((ni - i3 - j3) as i32)
                                * q_ijk((i1 + i2 + i3) as f64,
                                        (j1 + j2 + j3) as f64,
                                        (li + mi + ni - i1 - i2 - i3 - j1 - j2 - j3) as f64);
                        }
                    }
                }
            }
        }
    }
    s
}

// ── CSV helpers for tet / vert files ─────────────────────────────────────────

fn read_csv_matrix(path: &str) -> Vec<Vec<f64>> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path, e))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            line.split(',')
                .map(|s| s.trim().parse::<f64>().unwrap_or(f64::NAN))
                .collect::<Vec<f64>>()
        })
        .collect()
}

/// Load a tetrahedra file and return 0-indexed face data.
/// Drops all-NaN columns (replicates the Python logic).
fn load_tet(path: &str) -> Vec<Vec<usize>> {
    let raw = read_csv_matrix(path);
    raw.iter()
        .map(|row| row.iter().filter(|v| !v.is_nan()).map(|&v| (v - 1.0) as usize).collect())
        .collect()
}

fn load_vert(path: &str, scale_to_km: bool) -> Vec<Vec<f64>> {
    let raw = read_csv_matrix(path);
    if scale_to_km {
        raw.iter().map(|row| row.iter().map(|&v| v / 1000.0).collect()).collect()
    } else {
        raw
    }
}

// ── poly_inertia_met ──────────────────────────────────────────────────────────

/// Compute inertia integrals from polyhedron files given in **metres**.
/// Converts to km internally.  Returns a Cube of size (q+1)^3.
///
/// Mirrors `void poly_inertia_met(int q, ...)`.
pub fn poly_inertia_met(q: usize, rho: f64, tet_file: &str, vert_file: &str) -> Cube {
    let tet = load_tet(tet_file);
    let vert = load_vert(vert_file, true); // metres → km
    poly_inertia_impl(q, rho, &tet, &vert)
}

/// Compute inertia integrals from polyhedron files given in **km**.
///
/// Mirrors `void poly_inertia(int q, ...)`.
pub fn poly_inertia(q: usize, rho: f64, tet_file: &str, vert_file: &str) -> Cube {
    let tet = load_tet(tet_file);
    let vert = load_vert(vert_file, false);
    poly_inertia_impl(q, rho, &tet, &vert)
}

fn poly_inertia_impl(q: usize, rho: f64, tet: &[Vec<usize>], vert: &[Vec<f64>]) -> Cube {
    let mut t = Cube::new(q);
    for l in 0..=q {
        for m in 0..=(q - l) {
            for n in 0..=(q - l - m) {
                let mut val = 0.0_f64;
                for face in tet {
                    // vert row has 4 columns: [index, x, y, z]
                    let x1 = &vert[face[0]][1..=3];
                    let x2 = &vert[face[1]][1..=3];
                    let x3 = &vert[face[2]][1..=3];
                    // determinant of 3x3 formed by the three vertices
                    let det_mat: Mat3 = [
                        [x1[0], x2[0], x3[0]],
                        [x1[1], x2[1], x3[1]],
                        [x1[2], x2[2], x3[2]],
                    ];
                    let ta_abs = det3(det_mat).abs();
                    val += rho * ta_abs
                        * tet_sums(l as f64, m as f64, n as f64,
                                   x1[0], x2[0], x3[0],
                                   x1[1], x2[1], x3[1],
                                   x1[2], x2[2], x3[2]);
                }
                t.set(l, m, n, val);
            }
        }
    }
    t
}

// ── poly_moi_met ──────────────────────────────────────────────────────────────

/// Moments of inertia [Ixx, Iyy, Izz] from polyhedron files in metres.
///
/// Mirrors `void poly_moi_met(...)`.
pub fn poly_moi_met(rho: f64, tet_file: &str, vert_file: &str) -> Vec3 {
    let tet = load_tet(tet_file);
    let vert = load_vert(vert_file, true); // metres → km
    poly_moi_impl(rho, &tet, &vert)
}

/// Moments of inertia from polyhedron files in km.
pub fn poly_moi(rho: f64, tet_file: &str, vert_file: &str) -> Vec3 {
    let tet = load_tet(tet_file);
    let vert = load_vert(vert_file, false);
    poly_moi_impl(rho, &tet, &vert)
}

fn poly_moi_impl(rho: f64, tet: &[Vec<usize>], vert: &[Vec<f64>]) -> Vec3 {
    let mut moi = [0.0_f64; 3];
    for face in tet {
        let p1: [f64; 3] = [0.0; 3]; // vertex at origin
        let p2: [f64; 3] = [vert[face[0]][1], vert[face[0]][2], vert[face[0]][3]];
        let p3: [f64; 3] = [vert[face[1]][1], vert[face[1]][2], vert[face[1]][3]];
        let p4: [f64; 3] = [vert[face[2]][1], vert[face[2]][2], vert[face[2]][3]];
        let det_mat: Mat3 = [
            [p2[0], p3[0], p4[0]],
            [p2[1], p3[1], p4[1]],
            [p2[2], p3[2], p4[2]],
        ];
        let v_vol = rho * det3(det_mat).abs() / 6.0;
        // Ixx: integral of (y^2 + z^2) dm
        moi[0] += v_vol * (p1[1].powi(2) + p1[1]*p2[1] + p2[1].powi(2)
                           + p1[1]*p3[1] + p2[1]*p3[1] + p3[1].powi(2)
                           + p1[2].powi(2) + p1[2]*p2[2] + p2[2].powi(2)
                           + p1[2]*p3[2] + p2[2]*p3[2] + p3[2].powi(2)
                           + p1[1]*p4[1] + p2[1]*p4[1] + p3[1]*p4[1] + p4[1].powi(2)
                           + p1[2]*p4[2] + p2[2]*p4[2] + p3[2]*p4[2] + p4[2].powi(2)) / 10.0;
        // Iyy: integral of (x^2 + z^2) dm
        moi[1] += v_vol * (p1[0].powi(2) + p1[0]*p2[0] + p2[0].powi(2)
                           + p1[0]*p3[0] + p2[0]*p3[0] + p3[0].powi(2)
                           + p1[2].powi(2) + p1[2]*p2[2] + p2[2].powi(2)
                           + p1[2]*p3[2] + p2[2]*p3[2] + p3[2].powi(2)
                           + p1[0]*p4[0] + p2[0]*p4[0] + p3[0]*p4[0] + p4[0].powi(2)
                           + p1[2]*p4[2] + p2[2]*p4[2] + p3[2]*p4[2] + p4[2].powi(2)) / 10.0;
        // Izz: integral of (x^2 + y^2) dm
        moi[2] += v_vol * (p1[1].powi(2) + p1[1]*p2[1] + p2[1].powi(2)
                           + p1[1]*p3[1] + p2[1]*p3[1] + p3[1].powi(2)
                           + p1[0].powi(2) + p1[0]*p2[0] + p2[0].powi(2)
                           + p1[0]*p3[0] + p2[0]*p3[0] + p3[0].powi(2)
                           + p1[1]*p4[1] + p2[1]*p4[1] + p3[1]*p4[1] + p4[1].powi(2)
                           + p1[0]*p4[0] + p2[0]*p4[0] + p3[0]*p4[0] + p4[0].powi(2)) / 10.0;
    }
    moi
}

// ── ell_mass_params_met ───────────────────────────────────────────────────────

/// Ellipsoidal inertia integrals and moments, with semi-axes in **metres**.
///
/// Returns `(I, T)` where `I = [Ixx, Iyy, Izz]` and `T` is a Cube of size
/// `(order+1)^3`.  Mirrors `void ell_mass_params_met(...)`.
pub fn ell_mass_params_met(order: usize, order_body: usize,
                           rho: f64, a: f64, b: f64, c: f64)
                           -> (Vec3, Cube) {
    let a = a / 1000.0; // metres → km
    let b = b / 1000.0;
    let c = c / 1000.0;
    let mass = 4.0 * rho * std::f64::consts::PI * a * b * c / 3.0;
    let mut t = Cube::new(order);
    t.set(0, 0, 0, mass);
    let ia = [mass * (b * b + c * c) / 5.0,
              mass * (a * a + c * c) / 5.0,
              mass * (b * b + a * a) / 5.0];
    if order_body > 0 {
        t.set(2, 0, 0, mass * a * a / 5.0);
        t.set(0, 2, 0, mass * b * b / 5.0);
        t.set(0, 0, 2, mass * c * c / 5.0);
        if order_body > 3 {
            t.set(4, 0, 0, 3.0 * mass * a.powi(4) / 35.0);
            t.set(0, 4, 0, 3.0 * mass * b.powi(4) / 35.0);
            t.set(0, 0, 4, 3.0 * mass * c.powi(4) / 35.0);
            t.set(2, 2, 0, mass * a * a * b * b / 35.0);
            t.set(0, 2, 2, mass * c * c * b * b / 35.0);
            t.set(2, 0, 2, mass * a * a * c * c / 35.0);
        }
    }
    (ia, t)
}

// ── inertia_rot ───────────────────────────────────────────────────────────────

/// Rotate inertia integrals from body-fixed frame to A frame using rotation
/// matrix C (maps from B to A).  Returns a new Cube `Tp`.
///
/// Generic over `T` so dual-number types can flow through for auto-diff.
/// Mirrors `void inertia_rot(mat C, int q, cube* T, cube* Tp)`.
pub fn inertia_rot<T: Scalar>(c: Mat3<T>, q: usize, t: &Cube<T>) -> Cube<T> {
    let mut tp = Cube::new(q);
    for l in 0..=q {
        for m in 0..=(q - l) {
            for n in 0..=(q - l - m) {
                let mut val = T::zero();
                for i1 in 0..=l {
                    for j1 in 0..=(l - i1) {
                        let cl = ifact(l) / (ifact(i1) * ifact(j1) * ifact(l - i1 - j1));
                        for i2 in 0..=m {
                            for j2 in 0..=(m - i2) {
                                let cm = ifact(m) / (ifact(i2) * ifact(j2) * ifact(m - i2 - j2));
                                for i3 in 0..=n {
                                    for j3 in 0..=(n - i3) {
                                        let cn = ifact(n) / (ifact(i3) * ifact(j3) * ifact(n - i3 - j3));
                                        let sum_i = i1 + i2 + i3;
                                        let sum_j = j1 + j2 + j3;
                                        let sum_k = l + m + n - sum_i - sum_j;
                                        if sum_i <= q && sum_j <= q && sum_k <= q {
                                            let coeff = <T as NumCast>::from(cl * cm * cn).unwrap();
                                            val += coeff
                                                * c[0][0].powi(i1 as i32) * c[0][1].powi(j1 as i32)
                                                  * c[0][2].powi((l - i1 - j1) as i32)
                                                * c[1][0].powi(i2 as i32) * c[1][1].powi(j2 as i32)
                                                  * c[1][2].powi((m - i2 - j2) as i32)
                                                * c[2][0].powi(i3 as i32) * c[2][1].powi(j3 as i32)
                                                  * c[2][2].powi((n - i3 - j3) as i32)
                                                * t.get(sum_i, sum_j, sum_k);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                tp.set(l, m, n, val);
            }
        }
    }
    tp
}

// ── dT_dc ─────────────────────────────────────────────────────────────────────

/// Partial of the rotated inertia integral Cube with respect to rotation
/// matrix element C(row, col).
///
/// Generic over `T` for auto-diff.
/// Mirrors `void dT_dc(int i, int j, mat C, int q, cube* TA, cube* dT)`.
pub fn dt_dc<T: Scalar>(row: usize, col: usize, c: Mat3<T>, q: usize, t: &Cube<T>) -> Cube<T> {
    let mut dt = Cube::new(q);
    for l in 0..=q {
        for m in 0..=(q - l) {
            for n in 0..=(q - l - m) {
                let mut val = T::zero();
                for i1 in 0..=l {
                    for j1 in 0..=(l - i1) {
                        let cl = ifact(l) / (ifact(i1) * ifact(j1) * ifact(l - i1 - j1));
                        for i2 in 0..=m {
                            for j2 in 0..=(m - i2) {
                                let cm = ifact(m) / (ifact(i2) * ifact(j2) * ifact(m - i2 - j2));
                                for i3 in 0..=n {
                                    for j3 in 0..=(n - i3) {
                                        let cn = ifact(n) / (ifact(i3) * ifact(j3) * ifact(n - i3 - j3));
                                        let si = i1 + i2 + i3;
                                        let sj = j1 + j2 + j3;
                                        let sk = l + m + n - si - sj;
                                        if si <= q && sj <= q && sk <= q {
                                            let cv = rotation_partial_coeff(
                                                row, col, c,
                                                i1, j1, l - i1 - j1,
                                                i2, j2, m - i2 - j2,
                                                i3, j3, n - i3 - j3,
                                            );
                                            let coeff = <T as NumCast>::from(cl * cm * cn).unwrap();
                                            val += coeff * cv * t.get(si, sj, sk);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                dt.set(l, m, n, val);
            }
        }
    }
    dt
}

/// Compute `exp * C[row][col]^(exp-1) * product-of-all-other-C-elements`.
///
/// This is the power-rule partial for the monomial of rotation matrix elements
/// that appears in each term of `inertia_rot`.  Computing it without division
/// by `C[row][col]` keeps the function correct for dual-number types and when
/// a matrix element is exactly zero.
fn rotation_partial_coeff<T: Scalar>(row: usize, col: usize, c: Mat3<T>,
                                     i1: usize, j1: usize, k1: usize,
                                     i2: usize, j2: usize, k2: usize,
                                     i3: usize, j3: usize, k3: usize) -> T {
    let exp = match (row, col) {
        (0, 0) => i1, (0, 1) => j1, (0, 2) => k1,
        (1, 0) => i2, (1, 1) => j2, (1, 2) => k2,
        (2, 0) => i3, (2, 1) => j3, (2, 2) => k3,
        _ => unreachable!(),
    };
    if exp == 0 { return T::zero(); }
    // Reduce the exponent of c[row][col] by 1 (power rule), leave others unchanged.
    let ei1 = if (row, col) == (0, 0) { i1 - 1 } else { i1 };
    let ej1 = if (row, col) == (0, 1) { j1 - 1 } else { j1 };
    let ek1 = if (row, col) == (0, 2) { k1 - 1 } else { k1 };
    let ei2 = if (row, col) == (1, 0) { i2 - 1 } else { i2 };
    let ej2 = if (row, col) == (1, 1) { j2 - 1 } else { j2 };
    let ek2 = if (row, col) == (1, 2) { k2 - 1 } else { k2 };
    let ei3 = if (row, col) == (2, 0) { i3 - 1 } else { i3 };
    let ej3 = if (row, col) == (2, 1) { j3 - 1 } else { j3 };
    let ek3 = if (row, col) == (2, 2) { k3 - 1 } else { k3 };
    let prod = c[0][0].powi(ei1 as i32) * c[0][1].powi(ej1 as i32) * c[0][2].powi(ek1 as i32)
             * c[1][0].powi(ei2 as i32) * c[1][1].powi(ej2 as i32) * c[1][2].powi(ek2 as i32)
             * c[2][0].powi(ei3 as i32) * c[2][1].powi(ej3 as i32) * c[2][2].powi(ek3 as i32);
    <T as NumCast>::from(exp as f64).unwrap() * prod
}

// ── CSV helpers for saving IA / IB ───────────────────────────────────────────

/// Save a moment-of-inertia vector [Ixx, Iyy, Izz] as a single CSV row.
pub fn save_moi_csv(ia: Vec3, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{:.16e},{:.16e},{:.16e}", ia[0], ia[1], ia[2])
}

/// Load a moment-of-inertia vector from CSV.
pub fn load_moi_csv(path: &str) -> std::io::Result<Vec3> {
    let s = std::fs::read_to_string(path)?;
    let v: Vec<f64> = s.split(|c| c == ',' || c == '\n')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse().unwrap_or(0.0))
        .collect();
    Ok([v[0], v[1], v[2]])
}
