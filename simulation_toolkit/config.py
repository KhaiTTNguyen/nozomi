import json
from pathlib import Path
from typing import Dict

class Config:
    """Global configuration manager"""
    
    def __init__(self):
        self.package_root = Path(__file__).parent
        self.project_root = self.package_root.parent
    
    # load substrate / simulation config from file
    def load_config_file(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            return json.load(f)

config = Config()