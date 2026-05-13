"""GUBAS initializer — run this once after cloning the repo.

What it does:
  1. Checks that Rust / cargo is installed.
  2. Creates a local Python virtual environment in .venv.
  3. Installs Python dependencies inside .venv.
  4. Builds the standalone Rust binary  →  gubas_rs/target/release/hou_cpp_final
  5. Optionally builds and installs the maturin Python extension  →  import gubas_rs

Usage:
    python initialize.py            # binary only
    python initialize.py --maturin  # binary + Python extension
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CRATE_DIR = os.path.join(REPO_ROOT, "gubas_rs")
BINARY_OUT = os.path.join(CRATE_DIR, "target", "release", "hou_cpp_final")
VENV_DIR = os.path.join(REPO_ROOT, ".venv")


def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def venv_python():
    if os.name == "nt":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def create_venv():
    py = venv_python()

    if not os.path.exists(py):
        print(f"  Creating virtual environment: {VENV_DIR}")
        run([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print(f"  Virtual environment already exists: {VENV_DIR}")

    return py


def check_cargo():
    try:
        result = subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  cargo found: {result.stdout.strip()}")
    except FileNotFoundError:
        print(
            "\nERROR: cargo not found. Install Rust from https://rustup.rs/ and retry.\n"
        )
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("\nERROR: cargo exists but failed to run correctly.\n")
        sys.exit(1)


def install_python_deps():
    py = create_venv()

    requirements = os.path.join(REPO_ROOT, "requirements.txt")

    if not os.path.exists(requirements):
        print(f"\nERROR: requirements.txt not found at:\n  {requirements}\n")
        sys.exit(1)

    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "-r", requirements])

    return py


def build_binary():
    if not os.path.isdir(CRATE_DIR):
        print(f"\nERROR: Rust crate directory not found:\n  {CRATE_DIR}\n")
        sys.exit(1)

    run(["cargo", "build", "--release"], cwd=CRATE_DIR)
    print(f"\n  Binary built: {BINARY_OUT}")


def build_maturin_extension(py):
    run([py, "-m", "maturin", "develop", "--release"], cwd=CRATE_DIR)
    print("\n  Python extension installed — you can now 'import gubas_rs'")


def main():
    parser = argparse.ArgumentParser(description="GUBAS initializer")
    parser.add_argument(
        "--maturin",
        action="store_true",
        help="also build and install the gubas_rs Python extension via maturin",
    )
    args = parser.parse_args()

    print("\n=== GUBAS initializer ===\n")

    print("[1/3] Checking for Rust / cargo ...")
    check_cargo()

    print("\n[2/3] Creating venv and installing Python dependencies ...")
    py = install_python_deps()

    print("\n[3/3] Building Rust binary (release) ...")
    build_binary()

    if args.maturin:
        print("\n[+] Building maturin Python extension ...")
        build_maturin_extension(py)

    print("\n=== Done ===")
    print(f"\nBinary:  {BINARY_OUT}")
    print(f"Python:  {py}")

    if args.maturin:
        print("Module:  import gubas_rs")

    print("\nRun the example:")
    print("  source .venv/bin/activate")
    print("  cd example")
    print("  python run_example.py\n")


if __name__ == "__main__":
    main()
