import pytest
import numpy as np
from simulation_toolkit.simulation_engine.physics import PhysicsEngine

class TestPhysicsEngine:
    
    @pytest.fixture
    def physics_engine(self):
        return PhysicsEngine()
    
    def test_temperature_field_calculation(self, physics_engine):
        """Test temperature field calculation."""
        geometry = np.ones((10, 10))
        temp_field = physics_engine.calculate_temperature_field(geometry)
        
        assert temp_field.shape == geometry.shape
        assert np.all(temp_field >= 0)  # Temperature should be positive
    
    @pytest.mark.parametrize("boundary_condition", [
        "dirichlet",
        "neumann", 
        "mixed"
    ])
    def test_boundary_conditions(self, physics_engine, boundary_condition):
        """Test different boundary conditions."""
        geometry = np.ones((5, 5))
        result = physics_engine.apply_boundary_condition(geometry, boundary_condition)
        assert result is not None
    
    def test_conservation_laws(self, physics_engine):
        """Test physics conservation laws."""
        initial_energy = 100.0
        geometry = np.ones((10, 10))
        
        final_energy = physics_engine.simulate_step(geometry, initial_energy)
        
        # Energy should be conserved (within tolerance)
        np.testing.assert_allclose(initial_energy, final_energy, rtol=1e-10)