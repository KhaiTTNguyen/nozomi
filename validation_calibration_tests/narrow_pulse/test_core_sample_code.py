import pytest
import numpy as np
from simulation_toolkit.substrate_generator.core import SubstrateGenerator

class TestSubstrateGenerator:
    
    def test_initialization(self):
        """Test substrate generator initialization."""
        generator = SubstrateGenerator()
        assert generator is not None
        assert hasattr(generator, 'config')
    
    @pytest.mark.parametrize("width,height,expected_area", [
        (10, 10, 100),
        (20, 15, 300),
        (5, 8, 40),
    ])
    def test_geometry_area_calculation(self, substrate_generator, width, height, expected_area):
        """Test geometry area calculation with different dimensions."""
        area = substrate_generator.calculate_area(width, height)
        assert area == expected_area
    
    def test_generate_substrate_with_valid_config(self, substrate_generator, sample_config):
        """Test substrate generation with valid configuration."""
        result = substrate_generator.generate(sample_config)
        assert result is not None
        assert 'geometry' in result
        assert 'fibers' in result
    
    def test_generate_substrate_invalid_config(self, substrate_generator):
        """Test substrate generation with invalid configuration."""
        invalid_config = {"invalid": "config"}
        with pytest.raises(ValueError, match="Invalid configuration"):
            substrate_generator.generate(invalid_config)
    
    @pytest.mark.slow
    def test_large_substrate_generation(self, substrate_generator):
        """Test generation of large substrates (marked as slow)."""
        large_config = {
            "geometry": {"width": 1000, "height": 1000, "fiber_density": 0.5}
        }
        result = substrate_generator.generate(large_config)
        assert result is not None