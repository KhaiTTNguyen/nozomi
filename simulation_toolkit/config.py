"""Global configuration management"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Global configuration manager"""
    
    def __init__(self):
        self.package_root = Path(__file__).parent
        self.project_root = self.package_root.parent
        self.config_dir = self.package_root / "configs"
        self.examples_dir = self.project_root / "simulation_toolkit/defaults/"
        # Default settings
        # self.defaults = {
        #     "substrate_generator": {
        #         "mean_diameter": 2.0,
        #         "sigma_diameter": 0.5,
        #         "num_fibers": 500,
        #         "target_volume_fraction": 0.7,
        #         "orientation_shape_parameter": 20,
        #         "repeats": 5,
        #         "dist_shape": 0.1,
        #         "space_buffer_starts_ends": 0.23,
        #         "spheres_spacing": 0.5,
        #         "space_buffer_repulse": 0.001,
        #         "w_overlap":10,
        #         "w_curve": 3, 
        #         "w_length": 3,
        #         "bead_spacing_mean": 5.70,
        #         "bead_spacing_stdv": 2.88,
        #     },
        #     "simulation": {
        #         "time_steps": 1000,
        #         "dt": 0.001,
        #         "solver": "rk4"
        #     },
        #     "gpu": {
        #         "max_gpus": 5,
        #         "auto_detect": True
        #     }
        # }
    
    # def get(self, key: str, default: Any = None) -> Any:
    #     """Get configuration value using dot notation"""
    #     keys = key.split('.')
    #     value = self.defaults
        
    #     for k in keys:
    #         if isinstance(value, dict) and k in value:
    #             value = value[k]
    #         else:
    #             return default
        
    #     return value
    
    # IN USE: load config from file instead of defaults
    def load_config_file(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        
        # If relative path, try "simulation_toolkit/defaults/" first
        if not config_path.is_absolute():
            default_config = self.examples_dir / config_path
            if default_config.exists():
                config_path = default_config
        
        with open(config_path, 'r') as f:
            return json.load(f)

# Global config instance
config = Config()