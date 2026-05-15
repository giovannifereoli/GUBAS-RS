"""
conftest.py — pytest configuration for GUBAS-RS Python tests.

Adds the `example/` directory to sys.path so that pure-Python utilities
(stokes_utils.py) can be imported without installing a package.
The tests in this directory do NOT require the compiled Rust extension
(gubas_rs) to be installed — they test stokes_utils.py in isolation.
"""

import sys
import os

# Resolve `example/` relative to the repo root (one level above `tests/`)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_example_dir = os.path.join(_repo_root, "example")

if _example_dir not in sys.path:
    sys.path.insert(0, _example_dir)