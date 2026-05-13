# GUBAS — General Use Binary Asteroid Simulator

GUBAS models the coupled translational and rotational dynamics of binary asteroid systems using the Hou 2016 Full Two-Body Problem (F2BP) formulation.  Each body is represented by its inertia integrals up to a user-defined expansion order; the mutual gravitational potential and resulting forces and torques are evaluated as a truncated series expansion.

This repository contains a **Rust port** of the original C++ integrator, plus the Python pre/post-processing utilities.  The Rust core is free of external dependencies (no Armadillo), compiles to a single portable binary, and can also be loaded directly as a Python extension module via [maturin](https://github.com/PyO3/maturin).

---

## Repository layout

```
gubas_RUST/
├── gubas_rs/               Rust crate (integrator core)
│   ├── Cargo.toml
│   ├── pyproject.toml      maturin build config
│   └── src/
│       ├── lib.rs          library root + PyO3 module
│       ├── main.rs         hou_cpp_final binary entry point
│       └── *.rs            physics modules
├── gubas/                  Python package (config, post-processing)
│   ├── __init__.py
│   ├── config.py           hou_config.cfg reader
│   ├── icfile.py           ic_input.txt writer
│   ├── inertia.py          Python inertia integral functions
│   ├── coefficients.py     Hou tk / a / b expansion coefficients
│   ├── potential.py        mutual potential and partials
│   ├── benchmark.py        Fahnestock-format file reader
│   └── postprocess.py      binary output → CSV post-processor
├── example/
│   ├── hou_config.cfg      Didymos example configuration
│   ├── Didymos_A_facet.csv primary shape tetrahedra
│   ├── Didymos_A_vert_met.csv primary shape vertices (metres)
│   └── run_example.py      self-contained example script
├── hou_shell_cfg.py        main driver script
├── hou_config.cfg          default configuration file
└── General_Use_Binary_Asteroid_Simulator_Tool_User_Guide.pdf
```

---

## Quick start

### 1. Build the Rust binary

```bash
cd gubas_rs
cargo build --release
# binary: gubas_rs/target/release/hou_cpp_final
```

### 2. Run a simulation

```bash
# copy or symlink the binary next to your config file, then:
python hou_shell_cfg.py hou_config.cfg
```

Or run the self-contained example:

```bash
cd example
python run_example.py
```

### 3. Build as a Python extension (maturin)

```bash
pip install maturin
cd gubas_rs
maturin develop          # installs into the current virtualenv
# or
maturin build --release  # produces a .whl wheel
```

After installation, `hou_shell_cfg.py` automatically uses `gubas_rs.run()` instead of the subprocess call.

---

## Integrators

| Flag | Name      | Type          | Notes                              |
|------|-----------|---------------|------------------------------------|
| 1    | RK4       | Fixed step    | General purpose                    |
| 2    | LGVI      | Fixed step    | Symplectic, conserves SO(3); fastest for pure F2BP |
| 3    | RK 7(8)   | Adaptive step | Dormand-Prince; use for accuracy-critical runs |
| 4    | ABM       | Fixed step    | Adams-Bashforth-Moulton 4th order; ~2× faster than RK4 |

---

## Configuration

Edit `hou_config.cfg` (or copy it and pass the new filename to `hou_shell_cfg.py`).  Key sections:

- **Gravity Parameter** — G in m³/(kg·s²)
- **Initial Conditions** — relative position/velocity (m, m/s), angular velocities (rad/s), rotation matrices or Euler angles
- **Integration Settings** — start/end time (s), integrator flag, time step (s), tolerance
- **Output Settings** — output frequency, case name, post-processing toggle
- **Body Model Definitions** — shape flags (0=sphere, 1=ellipsoid, 2=polyhedron), semi-axes (m), density (g/cm³), mesh files
- **Mutual Gravity Expansion Parameters** — expansion orders N, NA, NB
- **Additional Forces and Perturbations** — flyby, heliocentric, solar gravity, tidal torques

---

## Output files

The integrator writes native-endian `float64` binary files:

| File                    | Contents                          |
|-------------------------|-----------------------------------|
| `output_t/t_out.bin`    | Time vector (one f64 per step)    |
| `output_x/x_out.bin`    | State matrix (30 f64s per step)   |
| `output_h/h_out.bin`    | Flyby body state (6 f64s/step)    |
| `output_sun/sun_out.bin`| Solar body state (6 f64s/step)    |

Post-processing converts these to labelled CSV files (`LagrangianStateOut_*.csv`, `Energy+AngMom_*.csv`, etc.).

State vector ordering: `[r(3), v(3), ωc(3), ωs(3), Cc(9), C(9)]` in units of km, km/s, rad/s, dimensionless.

---

## Dependencies

**Rust**: standard library only (no external crates except `pyo3` for the Python extension).

**Python**: `numpy`, `scipy` (only for Fahnestock file reading in `benchmark.py`).  `matplotlib` is optional and only used by `run_example.py`.

```bash
pip install numpy scipy          # required
pip install matplotlib           # optional, for run_example.py plots
pip install maturin              # only if building the Python extension
```

---

## Citation

If you use GUBAS in a publication, please cite:

> Alex B. Davis and Daniel J. Scheeres,
> *Doubly Synchronous Binary Asteroid Mass Parameter Observability*,
> Icarus, Vol. 341 (2020), https://doi.org/10.1016/j.icarus.2019.113439

If using the 3rd-body flyby or heliocentric perturbation features, also cite:

> Alex J. Meyer and Daniel J. Scheeres,
> *The effect of planetary flybys on singly synchronous binary asteroids*,
> Icarus, Vol. 367 (2021), https://doi.org/10.1016/j.icarus.2021.114554

---

## Errata

User Guide Eq. 9: the provided definition of the inertia integrals is mass-normalised; the code and subsequent equations use the non-normalised form.

---

*Original C++ code by Alex B. Davis & Alex J. Meyer.  Rust port by Giovanni Fereoli.*
