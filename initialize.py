"""GUBAS initializer — run this once after cloning the repo.

What it does:
  1. Checks that Rust / cargo is installed.
  2. Installs Python dependencies (numpy, scipy, matplotlib, maturin).
  3. Builds the standalone Rust binary  →  gubas_rs/target/release/hou_cpp_final
  4. Optionally builds and installs the maturin Python extension  →  import gubas_rs

Usage:
    python initialize.py            # binary only
    python initialize.py --maturin  # binary + Python extension
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
CRATE_DIR  = os.path.join(REPO_ROOT, "gubas_rs")
BINARY_OUT = os.path.join(CRATE_DIR, "target", "release", "hou_cpp_final")


def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def check_cargo():
    try:
        result = subprocess.run(["cargo", "--version"], capture_output=True, text=True)
        print(f"  cargo found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("\nERROR: cargo not found. Install Rust from https://rustup.rs/ and retry.\n")
        sys.exit(1)


def install_python_deps():
    run([sys.executable, "-m", "pip", "install", "-r",
         os.path.join(REPO_ROOT, "requirements.txt")])


def build_binary():
    run(["cargo", "build", "--release"], cwd=CRATE_DIR)
    print(f"\n  Binary built: {BINARY_OUT}")


def build_maturin_extension():
    run([sys.executable, "-m", "maturin", "develop", "--release"], cwd=CRATE_DIR)
    print("\n  Python extension installed — you can now 'import gubas_rs'")


def main():
    parser = argparse.ArgumentParser(description="GUBAS initializer")
    parser.add_argument(
        "--maturin", action="store_true",
        help="also build and install the gubas_rs Python extension via maturin",
    )
    args = parser.parse_args()

    print("\n=== GUBAS initializer ===\n")

    print("[1/3] Checking for Rust / cargo ...")
    check_cargo()

    print("\n[2/3] Installing Python dependencies ...")
    install_python_deps()

    print("\n[3/3] Building Rust binary (release) ...")
    build_binary()

    if args.maturin:
        print("\n[+]   Building maturin Python extension ...")
        build_maturin_extension()

    print("\n=== Done ===")
    print(f"\nBinary:  {BINARY_OUT}")
    if args.maturin:
        print("Module:  import gubas_rs")
    print("\nRun the example:")
    print("  cd example && python run_example.py\n")


if __name__ == "__main__":
    main()
