import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from simulation_toolkit.config import Config
from simulation_toolkit.substrate_generator.core import SubstrateGenerator
from simulation_toolkit.simulation_engine.core import SimulationEngine

@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "geometry": {
            "width": 100.0,
            "height": 100.0,
            "fiber_density": 0.1
        },
        "physics": {
            "temperature": 300.0,
            "pressure": 1.0
        }
    }

@pytest.fixture
def substrate_generator():
    """Substrate generator instance."""
    return SubstrateGenerator()

@pytest.fixture
def simulation_engine():
    """Simulation engine instance."""
    return SimulationEngine()

@pytest.fixture
def sample_geometry_data():
    """Sample geometry data for testing."""
    return np.random.rand(100, 100)