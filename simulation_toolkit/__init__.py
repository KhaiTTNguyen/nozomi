"""
Geometry Simulation Toolkit

A comprehensive toolkit for substrate generation and physics simulation.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main classes for easy access
# from .substrate_generator import initialization_2d
from .simulation_engine import diffsim3d
from .config import Config

# Make main functionality easily accessible
__all__ = [
    "initialization_2d",
    "diffsim3d", 
    "Config",
    "__version__",
]

# Package-level configuration
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Package root directory
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))