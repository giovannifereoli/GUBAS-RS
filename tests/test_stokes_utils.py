"""
test_stokes_utils.py
====================
Pytest suite for stokes_utils.py.

Covers:
  - cs_labels            : count, ordering, format
  - stokes_matrix        : shape, analytic C20/C22 values, normalization ratio
  - stokes_pseudoinverse : M * M+ ≈ I, shape, rank, null space dimension
  - convert_phi_xt_to_cs : output shape, chain-rule projection identity
  - _clm_contrib / _slm_contrib internals : degree guard, S_{l,0}=0

No Rust extension required — only numpy and stokes_utils.py.
sys.path is patched by tests/conftest.py.
"""

import numpy as np
import pytest

from stokes_utils import (
    cs_labels,
    inertia_labels,
    stokes_matrix,
    stokes_pseudoinverse,
    convert_phi_xt_to_cs,
    _clm_contrib,
    _slm_contrib,
)


# ── local helpers ──────────────────────────────────────────────────────────────

def _theta_indices(min_degree, max_degree):
    """All (i,j,k) with min_degree <= i+j+k <= max_degree, lex order."""
    idx = []
    for l in range(min_degree, max_degree + 1):
        for i in range(l + 1):
            for j in range(l - i + 1):
                idx.append((i, j, l - i - j))
    return idx


def _sphere_T(mass=1.0, r=1.0):
    """T_{ijk} dict for a uniform sphere of radius r, mass m."""
    return {
        (0, 0, 0): mass,
        (2, 0, 0): mass * r**2 / 5,
        (0, 2, 0): mass * r**2 / 5,
        (0, 0, 2): mass * r**2 / 5,
        (1, 1, 0): 0.0, (1, 0, 1): 0.0, (0, 1, 1): 0.0,
    }


def _oblate_T(a=2.0, c=1.0, mass=1.0):
    """T_{ijk} dict for oblate spheroid with equatorial radius a, polar radius c."""
    return {
        (0, 0, 0): mass,
        (2, 0, 0): mass * a**2 / 5,
        (0, 2, 0): mass * a**2 / 5,
        (0, 0, 2): mass * c**2 / 5,
        (1, 1, 0): 0.0, (1, 0, 1): 0.0, (0, 1, 1): 0.0,
    }


def _N_vec(T_dict, indices, mass=1.0, r0=1.0):
    """Dimensionless N_{ijk} = T_{ijk} / (mass * r0^{i+j+k})."""
    return np.array(
        [T_dict.get(ijk, 0.0) / (mass * r0 ** sum(ijk)) for ijk in indices]
    )


# ── cs_labels ─────────────────────────────────────────────────────────────────

class TestCsLabels:

    def test_count_degree2(self):
        assert len(cs_labels(2, 2)) == 5          # C20, C21, S21, C22, S22

    def test_count_degree2_to_4(self):
        assert len(cs_labels(2, 4)) == 21         # 5+7+9

    def test_exact_ordering_degree2(self):
        assert cs_labels(2, 2) == ["C20", "C21", "S21", "C22", "S22"]

    def test_first_label_is_c_l0(self):
        for l in [2, 3, 4]:
            labels = cs_labels(l, l + 1)
            assert labels[0] == f"C{l}0"

    def test_s_immediately_follows_c(self):
        labels = cs_labels(2, 3)
        idx_c21 = labels.index("C21")
        idx_s21 = labels.index("S21")
        assert idx_s21 == idx_c21 + 1


# ── inertia_labels ────────────────────────────────────────────────────────────

class TestInertiaLabels:

    def test_format(self):
        idx = [(2, 0, 0), (1, 1, 0), (0, 0, 2)]
        assert inertia_labels(idx) == ["T200", "T110", "T002"]


# ── stokes_matrix ─────────────────────────────────────────────────────────────

class TestStokesMatrix:

    def test_shape_degree2(self):
        M = stokes_matrix(_theta_indices(2, 2), 2, 2)
        assert M.shape == (5, 6)   # 5 coefficients, 6 monomials

    def test_shape_degree2_to_4(self):
        M = stokes_matrix(_theta_indices(2, 4), 2, 4)
        assert M.shape == (21, 31)   # 5+7+9=21, 6+10+15=31

    def test_sphere_all_cs_zero(self):
        # Sphere: N_{200}=N_{020}=N_{002}=r²/5, cross-terms=0 → C/S=0
        idx = _theta_indices(2, 2)
        N = _N_vec(_sphere_T(r=1.0), idx, mass=1.0, r0=1.0)
        cs = stokes_matrix(idx, 2, 2) @ N
        np.testing.assert_allclose(cs, 0.0, atol=1e-14)

    def test_oblate_c20_analytic(self):
        # C_{20} = (1/4)(4N_{002} - 2N_{020} - 2N_{200})
        #        = (1/4)(4/5 - 8/5 - 8/5) = -3/5  for a=2, c=1, r0=1, mass=1
        idx = _theta_indices(2, 2)
        N = _N_vec(_oblate_T(a=2.0, c=1.0, mass=1.0), idx)
        cs = stokes_matrix(idx, 2, 2) @ N
        np.testing.assert_allclose(cs[0], -3.0 / 5.0, atol=1e-14)

    def test_triaxial_c22_analytic(self):
        # C_{22} = (2-0)*(0!/4!) * (1/4)*12*(a²-b²)/5 = (a²-b²)/20
        # For a=3, b=2: (9-4)/20 = 0.25
        idx = _theta_indices(2, 2)
        T_d = {(0,0,0):1.0, (2,0,0):9/5, (0,2,0):4/5, (0,0,2):1/5,
               (1,1,0):0, (1,0,1):0, (0,1,1):0}
        N = _N_vec(T_d, idx)
        cs = stokes_matrix(idx, 2, 2) @ N
        c22_idx = cs_labels(2, 2).index("C22")
        np.testing.assert_allclose(cs[c22_idx], (9 - 4) / 20.0, atol=1e-13)

    def test_s_lm_zero_for_aligned_body(self):
        # No cross-terms → all S_{lm} = 0
        idx = _theta_indices(2, 2)
        N = _N_vec(_oblate_T(), idx)
        cs = stokes_matrix(idx, 2, 2) @ N
        for i, lbl in enumerate(cs_labels(2, 2)):
            if lbl.startswith("S"):
                np.testing.assert_allclose(cs[i], 0.0, atol=1e-14, err_msg=lbl)

    def test_normalized_c20_factor(self):
        # C̄_{20} = sqrt(5) * C_{20}  (geodesy full normalization, m=0: factor=sqrt(5))
        import math
        idx = _theta_indices(2, 2)
        N = _N_vec(_oblate_T(a=2.0, c=1.0), idx)
        c20_un = (stokes_matrix(idx, 2, 2, normalized=False) @ N)[0]
        c20_no = (stokes_matrix(idx, 2, 2, normalized=True)  @ N)[0]
        np.testing.assert_allclose(c20_no, math.sqrt(5) * c20_un, rtol=1e-13)


# ── stokes_pseudoinverse ──────────────────────────────────────────────────────

class TestStokesPseudoinverse:

    def test_mmp_is_identity_degree2(self):
        idx = _theta_indices(2, 2)
        M  = stokes_matrix(idx, 2, 2)
        Mp = stokes_pseudoinverse(M)
        np.testing.assert_allclose(M @ Mp, np.eye(5), atol=1e-12)

    def test_mmp_is_identity_degree3(self):
        idx = _theta_indices(2, 3)
        M  = stokes_matrix(idx, 2, 3)
        Mp = stokes_pseudoinverse(M)
        np.testing.assert_allclose(M @ Mp, np.eye(12), atol=1e-11)

    def test_pseudoinverse_shape(self):
        idx = _theta_indices(2, 4)
        M  = stokes_matrix(idx, 2, 4)
        Mp = stokes_pseudoinverse(M)
        assert Mp.shape == (M.shape[1], M.shape[0])

    def test_full_row_rank(self):
        idx = _theta_indices(2, 3)
        M  = stokes_matrix(idx, 2, 3)
        sv = np.linalg.svd(M, compute_uv=False)
        assert np.all(sv > 1e-10), f"Near-zero singular values: {sv}"

    def test_null_space_dim_degree2(self):
        # n_theta=6, n_cs=5 → null space = 1 (trace-like combination)
        idx = _theta_indices(2, 2)
        M   = stokes_matrix(idx, 2, 2)
        rank = np.linalg.matrix_rank(M)
        assert rank == 5
        assert M.shape[1] - rank == 1


# ── convert_phi_xt_to_cs ─────────────────────────────────────────────────────

class TestConvertPhiXtToCs:

    def test_output_shape_3d(self):
        idx = _theta_indices(2, 2)
        phi = np.random.default_rng(42).standard_normal((10, 30, len(idx)))
        phi_cs, _, _ = convert_phi_xt_to_cs(phi, idx, 2, 2)
        assert phi_cs.shape == (10, 30, 5)

    def test_output_shape_multi_degree(self):
        idx = _theta_indices(2, 3)
        phi = np.random.default_rng(1).standard_normal((20, 30, len(idx)))
        phi_cs, _, _ = convert_phi_xt_to_cs(phi, idx, 2, 3)
        assert phi_cs.shape == (20, 30, 12)   # 5+7 = 12 C/S coefficients

    def test_output_shape_2d(self):
        # Single snapshot: (30, N_theta) → (30, N_cs)
        idx = _theta_indices(2, 2)
        phi = np.random.default_rng(7).standard_normal((30, len(idx)))
        phi_cs, _, _ = convert_phi_xt_to_cs(phi, idx, 2, 2)
        assert phi_cs.shape == (30, 5)

    def test_chain_rule_projection(self):
        # Phi_xCS = Phi_xT @ M+.
        # Re-multiplying: Phi_xCS @ M = Phi_xT @ (M+ @ M) = Phi_xT @ P
        # where P = M+ @ M is the orthogonal projector onto M's row space.
        idx = _theta_indices(2, 2)
        rng = np.random.default_rng(0)
        phi = rng.standard_normal((5, 30, len(idx)))
        phi_cs, M, Mp = convert_phi_xt_to_cs(phi, idx, 2, 2)
        phi_reconstructed = phi_cs @ M          # (5, 30, 6)
        phi_projected     = phi @ (Mp @ M)      # (5, 30, 6)
        np.testing.assert_allclose(phi_reconstructed, phi_projected, atol=1e-12)


# ── Tricarico coefficient internals ───────────────────────────────────────────

class TestContribInternals:

    def test_s_l0_always_zero(self):
        # S_{l,0} = 0 by definition (Tricarico Eq. 17 requires m ≥ 1)
        for l in range(1, 5):
            for i in range(l + 1):
                for j in range(l - i + 1):
                    assert _slm_contrib(l, 0, i, j, l - i - j) == 0.0

    def test_clm_wrong_degree_zero(self):
        # T_{ijk} with i+j+k ≠ l cannot contribute to C_{lm}
        assert _clm_contrib(2, 0, 3, 0, 0) == 0.0   # degree 3 ≠ 2
        assert _clm_contrib(2, 0, 0, 0, 1) == 0.0   # degree 1 ≠ 2
        assert _clm_contrib(3, 1, 1, 0, 0) == 0.0   # degree 2 ≠ 3