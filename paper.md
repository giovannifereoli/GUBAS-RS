---
title: 'GUBAS-RS: A Rust library for high-fidelity binary asteroid dynamics and orbit determination sensitivity'
tags:
  - Rust
  - Python
  - astronomy
  - binary asteroids
  - orbital mechanics
  - orbit determination
  - automatic differentiation
  - state transition matrix
authors:
  - name: Giovanni Fereoli
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: Department of Aerospace Engineering Sciences, University of Colorado Boulder, United States
    index: 1
date: 16 May 2026
bibliography: paper.bib

---

# Summary

Binary asteroid systems — two rocky bodies gravitationally bound to each other — encode
unique information about the formation, composition, and interior structure of small Solar
System bodies [@Margot:2015].  Their mutual orbit measures both masses, while the
coupled spin-orbit evolution constrains the internal mass distribution of each body.
Modelling these systems accurately requires the **Full Two-Body Problem (F2BP)**, in which
each body is treated as an extended, non-spherical mass with an arbitrary gravity field
[@Hou:2016].

Orbit determination (OD) from spacecraft radiometric tracking data additionally requires
not only the trajectory but also the **State Transition Matrix (STM)**, the derivative of
the state with respect to initial conditions and gravity-field parameters.  Computing an
exact STM for the F2BP is non-trivial because the mutual potential is a multi-body
series expansion in inertia integrals, making finite-difference Jacobians both slow and
numerically unreliable.

GUBAS-RS is a Rust library — with a Python extension via PyO3/maturin — that solves
both problems.  It implements the Hou 2016 F2BP formulation and combines it with
forward-mode automatic differentiation (AD) to deliver machine-precision Jacobians at
the cost of a single additional ODE evaluation per state component.  The resulting
STM and augmented parameter-sensitivity matrix are propagated alongside the trajectory,
producing a complete, self-consistent OD observation-model package for binary asteroid
gravity field estimation.

# Statement of need

Interest in binary asteroid dynamics has intensified since the DART kinetic impactor
mission altered the orbit of Dimorphos around Didymos in September 2022 [@Daly:2023],
and the ESA Hera mission is now en route to characterise both bodies' gravity fields from
radiometric and altimetric measurements [@Michel:2022].  Gravity field recovery from
such data requires propagating not only the trajectory but also the sensitivity of that
trajectory to each gravity parameter — the partial derivative matrix $\partial\mathbf{x}/\partial\boldsymbol{\theta}$ — integrated self-consistently with the same dynamical model.

Existing open tools address parts of this problem:

- The original GUBAS (C++) computes F2BP trajectories to high fidelity but does not
  provide STM or parameter-sensitivity machinery [@Davis:2020].
- General astrodynamics packages such as OREKIT [@Maisonobe:2010] support STM
  propagation but implement point-mass or simple oblate-spheroid gravity models for
  the secondary, not the full mutual potential.
- General-purpose AD libraries (e.g., `autodiff` for C++) can compute Jacobians but
  are not integrated with the F2BP ODE or an OD-ready Python interface.

GUBAS-RS fills this gap by providing:

1. **High-fidelity F2BP dynamics** — mutual gravitational potential, forces, and torques
   evaluated as a Hou 2016 series in inertia integrals $T_{ijk}$, valid for any body
   shape.
2. **Exact STM** — forward-mode dual-number AD replaces finite differences; no step-size
   tuning, no truncation error.
3. **Augmented parameter sensitivity** — the augmented STM propagates
   $\partial\mathbf{x}/\partial T_{ijk}$ simultaneously for *both* bodies, enabling
   direct use in batch least-squares or Kalman filters.
4. **Stokes coefficient conversion** — a linear map converts $\partial\mathbf{x}/\partial T_{ijk}$
   to $\partial\mathbf{x}/\partial C_{lm}, \partial\mathbf{x}/\partial S_{lm}$
   [@Tricarico:2008], delivering partials in the spherical harmonic basis used by
   standard geodesy tools.
5. **Python interface** — compiled via PyO3/maturin so any Python OD framework (scipy,
   FilterPy, custom EKF) can call the Rust core without subprocess overhead.

The primary target audience is (i) planetary scientists modelling post-DART binary
dynamics, (ii) astrodynamicists building OD pipelines for the Hera mission or future
small-body rendezvous missions, and (iii) researchers who need a self-consistent,
open-source F2BP simulator with exact sensitivity analysis.

# State of the field

Several tools address binary asteroid or general rigid-body dynamics:

The original GUBAS code [@Davis:2020] implements the same Hou 2016 F2BP potential and
served as the reference implementation for this work.  GUBAS-RS extends it with AD-based
Jacobians, STM propagation, and a Python API while preserving identical dynamics and
output conventions.  The Rust port also eliminates the dependency on the Armadillo C++
linear algebra library.

`pkdgrav` and similar N-body codes simulate aggregate rubble-pile bodies but model
inter-particle gravity, not the mutual potential of two macroscopic bodies; they are not
suited for precision OD sensitivity analysis.  OREKIT [@Maisonobe:2010] and GMAT are
general-purpose astrodynamics frameworks that support STM propagation but assume simple
gravity models for the attractor.

No existing open tool combines (1) a full Hou 2016 mutual potential, (2) an exact AD
Jacobian, and (3) a ready-to-use Python OD interface.  GUBAS-RS was built to fill that
specific niche, motivated directly by DART/Hera science requirements.

# Software design

The library is structured around a generic scalar type `T` that is instantiated either
as `f64` (for trajectory propagation) or as `Dual` (for AD-based Jacobian computation).
A single `hou_ode(x, t, &params)` function implements the complete F2BP ODE for both
scalar types; passing `Dual` state automatically propagates the dual (derivative) part
through every arithmetic operation.

The key modules are:

| Module | Role |
|---|---|
| `dynamics` | F2BP ODE right-hand side — 30-element state $[\mathbf{r}, \mathbf{v}, \boldsymbol{\omega}_c, \boldsymbol{\omega}_s, C_c, C]$ |
| `dual` | Forward-mode dual number: $a + b\varepsilon$, $\varepsilon^2 = 0$ |
| `stm` | STM and augmented-STM propagators ($\partial\mathbf{x}/\partial\mathbf{x}_0$ and $\partial\mathbf{x}/\partial\boldsymbol{\theta}$) |
| `stokes` | $N_{ijk} \to C_{lm}/S_{lm}$ Stokes matrix (Tricarico 2008) |
| `potential` | Mutual gravitational potential and partial derivatives |
| `coefficients` | Hou 2016 expansion coefficients ($t_k$, $a_k$, $b_k$) |
| `inertia` | Inertia integrals $T_{ijk}$ from ellipsoid or polyhedron model |
| `integrators` | RK4, Adams–Bashforth–Moulton, adaptive RK7(8), LGVI [@Lee:2007] |
| `orbit` | Kepler's equation solver, orbital elements $\to$ Cartesian |
| `math3` | 3-vector / 3×3-matrix primitives |
| `types` | `Cube<T>`, `Params<T>` — generic simulation data structures |

The Python interface wraps two main classes.  `DynamicsModel` exposes
`eval(x, t) -> (ẋ, A)` and `eval_augmented(xΦ, t) -> [ẋ; Φ̇]` for single-body STM
use.  `AugmentedDynamicsModel` extends this to the both-body augmented state
$\mathbf{z} = [\mathbf{x}; \boldsymbol{\theta}_a; \boldsymbol{\theta}_b]$ and
exposes `eval_stm(zΦ, t)` compatible with `scipy.integrate.solve_ivp`.

# Mathematics

## Mutual gravitational potential

Following @Hou:2016, the mutual gravitational potential of two rigid bodies $A$ and $B$
with inertia integrals $T^A_{ijk}$ and $T^B_{ijk}$ is

$$U = -G \sum_{n=0}^{N} \sum_{2p+q+r+s+t=n} a_{pqrst}\, b_{pqrst}\;
      e_1^q e_2^r e_3^s \cdot r^{-(2p+q+r+s+t+1)}\;
      T^A_{i_1 i_2 i_3}\, T^B_{j_1 j_2 j_3},$$

where $\mathbf{e} = \mathbf{r}/r$ is the unit separation vector expressed in the
primary body frame, $a_{pqrst}$ and $b_{pqrst}$ are scalar expansion coefficients
computed once at initialisation, and $N$ is the truncation order.

The inertia integrals are defined as

$$T^A_{ijk} = \int_{\mathcal{B}_A} x^i y^j z^k \, \mathrm{d}m,$$

evaluated analytically for ellipsoids or numerically via tetrahedral quadrature for
arbitrary polyhedra.

## Dual-number automatic differentiation

A dual number is $d = a + b\varepsilon$ with $\varepsilon^2 = 0$.  Elementary
arithmetic propagates the dual part exactly:

$$f(d) = f(a) + f'(a)\,b\,\varepsilon.$$

To compute the $j$-th column of the Jacobian $\partial\mathbf{f}/\partial\mathbf{x}$,
the $j$-th state component is seeded with $\varepsilon = 1$ and all others with
$\varepsilon = 0$.  The resulting dual part of each output component is the exact
partial derivative, with no finite-difference truncation error and no
step-size selection.

## State Transition Matrix

The STM $\Phi(t, t_0) = \partial\mathbf{x}(t)/\partial\mathbf{x}(t_0)$ satisfies

$$\dot{\Phi} = A(t)\,\Phi, \qquad \Phi(t_0, t_0) = I_{30},$$

where $A(t) = \partial\mathbf{f}/\partial\mathbf{x}$ is the exact dual-number Jacobian
evaluated at the current trajectory point.

The augmented parameter sensitivity matrix $\Phi_{x\theta}(t) = \partial\mathbf{x}(t)/\partial\boldsymbol{\theta}$
is propagated simultaneously by augmenting the state to
$\mathbf{z} = [\mathbf{x}; \boldsymbol{\theta}]$ with $\dot{\boldsymbol{\theta}} = \mathbf{0}$,
so that the full augmented STM $\Phi_\text{aug}$ satisfies the same matrix ODE with
a larger Jacobian $A_\text{aug}$ that includes the $\partial\mathbf{f}/\partial\boldsymbol{\theta}$
columns computed by the same dual-number sweep.

## Stokes coefficient conversion

The linear map from inertia integrals to spherical harmonic Stokes coefficients
[@Tricarico:2008] at a reference radius $r_0$ is

$$\begin{pmatrix} C_{lm} \\ S_{lm} \end{pmatrix} = M\,\boldsymbol{T},$$

where $M$ is the $(2l+1) \times n_{ijk}$ Stokes matrix.  Because the number of
independent $T_{ijk}$ entries exceeds the number of observable $C_{lm}/S_{lm}$
coefficients at the same degree, $M$ is wide and the right pseudoinverse
$M^+ = M^\top(MM^\top)^{-1}$ is used to convert sensitivity partials:

$$\frac{\partial\mathbf{x}}{\partial\mathbf{C}/\mathbf{S}} =
  \frac{\partial\mathbf{x}}{\partial\boldsymbol{T}} \cdot M^+.$$

# Automated tests

The test suite verifies every module against analytic reference values with no relaxed
tolerances.  The Rust suite (108 unit tests) is run with `cargo test`; the Python suite
(24 tests via pytest) covers the `stokes_utils.py` utilities and requires no compiled
extension.

Key analytic checks include: dual-number chain and product rules to machine precision;
$\mathbf{i} \times \mathbf{j} = \mathbf{k}$ and $\det(I) = 1$ for `math3`; sphere
$C_{2m} = 0$ and oblate $C_{20} = -3/5$ for `stokes`; point-mass potential
$U = -GM_a M_b/r$ and centripetal force $G(M_a+M_b)/r^2$ for `potential` and
`dynamics`; vis-viva identity and $\mathbf{r}\cdot\mathbf{v}=0$ at periapsis for
`orbit`; and non-NaN output of correct size for all four integrators.

# AI usage disclosure

No generative AI tools were used in the writing of this manuscript.  AI-assisted tools
(Claude Code) were used during software development for code generation and testing.

# Acknowledgements

The author thanks Alex B. Davis and Alex J. Meyer for the original C++ GUBAS codebase
and the associated user documentation, which served as the direct reference for the
dynamics implementation.  The author thanks Daniel J. Scheeres for guidance on the
F2BP formulation and the orbit determination application context.

# References
