"""Main CLI entry point - Updated from previous version"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import torch
# Add package to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulation_toolkit.cli.substrate_main import substrate_main
# from simulation_toolkit.geometry_generator import meshing
# from simulation_toolkit.geometry_generator import optimization

from simulation_toolkit.simulation_engine import diffsim3d
from simulation_toolkit.utils.gpu_manager import GPUManager
from simulation_toolkit.utils.logging_utils import get_logger
from simulation_toolkit.config import config

logger = get_logger(__name__)

class GeometryToolkitCLI:
    """Main CLI class combining previous functionality"""
    
    def __init__(self):
        self.gpu_manager = GPUManager()
    
    def run_substrate_generation(self, params: dict, gpu_id: int = 0, experiment_name: str = "default_experiment"):
        """Run substrate generation"""
        '''TODO: do init, meshing, optimization as needed'''
        for experiment in range(0, params['repeats']):
            torch.autograd.set_detect_anomaly(True)
            substrate_main(params, experiment_name)

        # generator = initialization_2d(gpu_id=gpu_id)
        # result = generator.generate(params)
        logger.info(f"Substrate generation completed: {result}")
        # return result
    
    def run_batch_processing_for_folder(self, folder_path: str, max_gpus: int = 5, 
                           user_gpu_list: Optional[List[int]] = None):
        """Run batch processing for multiple configurations"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            logger.error(f"Folder {folder_path} does not exist")
            sys.exit(1)
        
        json_files = list(folder_path.glob("*.json"))
        if not json_files:
            logger.error(f"No JSON files found in {folder_path}")
            sys.exit(1)
        
        logger.info(f"Found {len(json_files)} configuration files")
        
        # Allocate GPUs
        gpu_list = self.gpu_manager.allocate_gpus(len(json_files), max_gpus, user_gpu_list)
        
        # Process each configuration
        for i, json_file in enumerate(json_files):
            gpu_id = gpu_list[i % len(gpu_list)]
            config_data = config.load_config_file(json_file)
            
            logger.info(f"Processing {json_file.name} on GPU {gpu_id}")
            
            if "substrates" in config_data:
                for substrate in config_data["substrates"]:
                    params = substrate["parameters"]
                    repeats = params.get("repeats", 1)
                    
                    for repeat in range(repeats):
                        logger.info(f"  Repeat {repeat + 1}/{repeats}")
                        self.run_substrate_generation(params, gpu_id)
            else:
                # Single substrate config
                repeats = config_data.get("repeats", 1)
                for repeat in range(repeats):
                    logger.info(f"  Repeat {repeat + 1}/{repeats}")
                    self.run_substrate_generation(config_data, gpu_id)

def parse_command_line_params(args: List[str]) -> dict:
    """Parse command line parameters in key=value format"""
    params = {}
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value
    return params

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description='Geometry Simulation Toolkit')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Substrate generation command
    substrate_parser = subparsers.add_parser('substrate', help='Generate substrates')
    # substrate_parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    substrate_parser.add_argument('--config', type=str, 
                            default='./simulation_toolkit/defaults/default-susbtrate.json',  # Add default here
                            help='Path to JSON configuration file (default: config/default_config.json)')
    substrate_parser.add_argument('--folder', type=str, help='Path to folder containing multiple JSON configs')
    substrate_parser.add_argument('--max-gpus', type=int, default=5, help='Maximum number of GPUs to use')
    substrate_parser.add_argument('--gpu-list', type=int, nargs='+', help='Specific GPU IDs to use')
    substrate_parser.add_argument('params', nargs='*', help='Parameters in key=value format')
    
    # Simulation command (placeholder)
    sim_parser = subparsers.add_parser('simulation', help='Run simulations')
    sim_parser.add_argument('--config', type=str, help='Path to simulation configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = GeometryToolkitCLI()
    
    if args.command == 'substrate':
        if args.folder:
            # TODO:
            cli.run_batch_processing_for_folder(args.folder, args.max_gpus, args.gpu_list)
        elif args.config:
            config_data = config.load_config_file(args.config)

            print('config_data', config_data)
            cmd_params = parse_command_line_params(args.params)
            
            # Merge config with command line overrides
            experiment_name = config_data['experiment_name']
            params = config_data["parameters"].copy()
            params.update(cmd_params)
            
            gpu_id = cmd_params.get('gpu', 0)
            # ===== START substrate generation =====
            cli.run_substrate_generation(params, gpu_id, experiment_name)
            exit()
            # ===== END substrate generation & save experiment config file =====
            shutil.copy(args.config, 
                os.path.join(config.OUTPUT_FOLDER_PATH, experiment_name, os.path.basename(args.config)))

        else:
            # TODO:Direct parameters
            params = parse_command_line_params(args.params)
            gpu_id = params.get('gpu', 0)
            cli.run_substrate_generation(params, gpu_id)
    
    elif args.command == 'simulation':
        logger.info("Simulation functionality coming soon!")

if __name__ == "__main__":
    main()