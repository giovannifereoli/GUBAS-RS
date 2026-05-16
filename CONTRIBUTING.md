# Contributing to GUBAS-RS

Thank you for your interest in contributing to GUBAS-RS! This document explains how to report issues, propose features, and submit code changes.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Reporting Bugs](#reporting-bugs)
3. [Requesting Features](#requesting-features)
4. [Development Setup](#development-setup)
5. [Submitting a Pull Request](#submitting-a-pull-request)
6. [Coding Style](#coding-style)
7. [Running Tests](#running-tests)
8. [Seeking Support](#seeking-support)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) v2.1.

---

## Reporting Bugs

Open a [GitHub Issue](../../issues/new) and include:

- A **minimal reproducible example** — the shortest Rust snippet or Python script that triggers the bug.
- The **expected** vs. **actual** output.
- Your Rust toolchain version (`rustc --version`, `cargo --version`) and OS.
- If the bug involves the Python bindings, also include `python --version` and `maturin --version`.

Label the issue `bug`.

---

## Requesting Features

Open a [GitHub Issue](../../issues/new) with the label `enhancement` and describe:

- The **use-case** — what scientific or engineering problem does this solve?
- A sketch of the **desired API** (function signatures, return types, units).
- Whether you are willing to implement it or would like guidance.

---

## Development Setup

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| Rust (stable toolchain) | ≥ 1.75 |
| Python | ≥ 3.8 |
| maturin | ≥ 1.0 |
| numpy | any recent |
| scipy | any recent |
| pytest | any recent |

### Build

```bash
# Clone the repo
git clone https://github.com/<org>/gubas_RUST.git
cd gubas_RUST/gubas_rs

# Build the Rust library
cargo build

# Build and install the Python extension into the current venv
cd ..
maturin develop --release
```

### Generate documentation

```bash
cd gubas_rs
cargo doc --no-deps --open
```

---

## Submitting a Pull Request

1. **Fork** the repository and create a branch from `master`:
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/short-description
   ```

2. **Make your changes.** Keep commits focused — one logical change per commit.

3. **Add tests** for any new functionality (see [Running Tests](#running-tests)).

4. **Ensure all tests pass** before opening the PR.

5. **Open a Pull Request** against `master`. Fill in the PR template:
   - What problem does this solve?
   - How was it tested?
   - Any breaking changes?

A maintainer will review within a reasonable time. Please be patient and responsive to feedback.

---

## Coding Style

- Follow standard Rust idioms (`cargo fmt`, `cargo clippy`).
- Run `cargo fmt` before committing:
  ```bash
  cargo fmt --all
  cargo clippy -- -D warnings
  ```
- Keep public APIs documented with `///` doc comments. Module-level docs use `//!`.
- Physical units must be stated explicitly in doc comments (SI unless noted).
- Do not introduce `unwrap()` or `expect()` in library code paths that a caller cannot control; return `Result` or `Option` instead.

---

## Running Tests

### Rust unit tests

```bash
cd gubas_rs
cargo test
```

To run a specific module's tests:

```bash
cargo test --lib dual::tests
cargo test --lib stokes::tests
```

### Python integration tests

```bash
cd example
python -m pytest -v
```

Expected: all tests pass. The pytest suite checks the Python bindings against independent analytical results (monopole potential, Stokes conversion, orbit propagation).

### Full check before a PR

```bash
cd gubas_rs
cargo fmt --all -- --check
cargo clippy -- -D warnings
cargo test
cd ../example
python -m pytest -v
```

---

## Seeking Support

- **Bug or question about the code** → open a [GitHub Issue](../../issues).
- **General usage question** → open a [GitHub Discussion](../../discussions).
- **Private inquiry** → contact the maintainer at <giovafere@gmail.com>.
