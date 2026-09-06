"""Pytest config for wide_pulse validation tests.

GPU tests (``test_01``..``test_05`` and ``test_gradient_sim_2boundary``) import
``pycuda.autoinit`` at module load. When pycuda is unavailable, ignore them at
collection time so the non-GPU waveform-design tests still run.
"""
import importlib.util

_HAS_PYCUDA = importlib.util.find_spec("pycuda") is not None

collect_ignore_glob = []
if not _HAS_PYCUDA:
    collect_ignore_glob = [
        "test_01_*.py",
        "test_02_*.py",
        "test_03_*.py",
        "test_04_*.py",
        "test_05_*.py",
        "test_gradient_sim_2boundary_*.py",
    ]
