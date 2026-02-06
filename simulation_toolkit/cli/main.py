"""Main CLI entry point - Updated from previous version"""

import argparse
import sys
from pathlib import Path
from typing import List
import torch
# Add package to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import shutil
import os
import simulation_toolkit.toolkit_params as config_params
from simulation_toolkit.cli.substrate_main import substrate_main
from simulation_toolkit.cli.simulation_main import simulation_main

from simulation_toolkit.simulation_engine import diffsim3d
from simulation_toolkit.config import config

class GeometryToolkitCLI:
    """Main CLI class combining previous functionality"""
    
    def __init__(self):
        return 
    
    def run_substrate_generation(self, params: dict, experiment_name: str = "default_experiment"):
        """Run substrate generation"""
        for experiment in range(0, params['repeats']):
            torch.autograd.set_detect_anomaly(True)
            substrate_main(params, experiment_name)

        # generator = initialization_2d(gpu_id=gpu_id)
        # result = generator.generate(params)
        print(f"Substrate generation completed for {experiment_name}")
    
    def run_simulation(self, full_sim_params):
        """Run substrate generation"""    
        main_folder = full_sim_params['substrates']
        if not os.path.isdir(main_folder):
            print(f"Error: {main_folder} is not a valid directory")
            return False
        
        # Get all subfolders in the main folder
        subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

        for subfolder in subfolders:
            subfolder_path = os.path.join(main_folder, subfolder)
            data_folder = os.path.join(subfolder_path, "data")
            
            # Check if data folder exists
            if os.path.isdir(data_folder):
                # Get substrate files in the data folder
                substrate_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
                
                for substrate_file in substrate_files:
                    substrate_path = os.path.join(data_folder, substrate_file)
                    torch.autograd.set_detect_anomaly(True)
                    simulation_main(full_sim_params, substrate_path)
            else:
                print(f"Warning: Data folder not found in {subfolder_path}")

        print(f"Simulation completed for {os.path.basename(full_sim_params['substrates'])}")
    
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
    
    # ============= Substrate generation command =============
    substrate_parser = subparsers.add_parser('substrate', help='Generate substrates')
    substrate_parser.add_argument('--config', type=str, 
                            default='./experiment/setup/substrate/default/default-substrate.json',  # Add default here
                            help='Path to JSON configuration file (default: ./experiment/setup/substrate/default/default-substrate.json)')
    substrate_parser.add_argument('--gpu', type=int, nargs='+', help='Specific GPU IDs to use')
    
    # ============= Simulation command =============
    sim_parser = subparsers.add_parser('simulation', help='Run simulations')
    sim_parser.add_argument('--substrates', type=str, help='Path to substrates folder')
    sim_parser.add_argument('--gpu', type=int, nargs='+', help='Specific GPU IDs to use')
    sim_parser.add_argument('--compartment', type=str, help='Compartment to simulate (e.g., intra, extra)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = GeometryToolkitCLI()
    
    if args.command == 'substrate':
        if args.config:
                # Set GPU before importing CUDA libraries
            # print('str(args.gpu', str(args.gpu[0]))
            # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu[0])
            # exit()
            config_data = config.load_config_file(args.config)
            
            # Merge config with command line overrides
            experiment_name = config_data['experiment_name']
            print('experiment_name', experiment_name)
        
            params = config_data["parameters"].copy()
            
            # gpu_id = cmd_params.get('gpu', 0)
            # ===== START substrate generation =====
            cli.run_substrate_generation(params, experiment_name)
            # ===== END substrate generation & save experiment config file =====
            shutil.copy(args.config, 
                os.path.join(config_params.OUTPUT_FOLDER_PATH, experiment_name, str(config_params.EXP_DATE_TIME) + "_" + os.path.basename(args.config)))
    
    elif args.command == 'simulation':
        # Set GPU before importing CUDA libraries
        sim_config = config.load_config_file(config_params.SIM_CONFIG_FILE)
        cmd_params = vars(args)

        # Merge config with command line overrides
        full_sim_params = sim_config.copy()
        full_sim_params.update(cmd_params)
        # ===== START simulation =====
        cli.run_simulation(full_sim_params)
        # ===== END simulation =====
        shutil.copy(config_params.SIM_CONFIG_FILE, 
            os.path.join(full_sim_params['substrates'], str(config_params.EXP_DATE_TIME) + "_" + os.path.basename(config_params.SIM_CONFIG_FILE)))
    
if __name__ == "__main__":
    main()