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
import json
import simulation_toolkit.toolkit_params as config_params
from simulation_toolkit.cli.substrate_main import substrate_main
from simulation_toolkit.cli.simulation_main import simulation_main
import simulation_toolkit.utils.common_utils as common_util
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
            
        print(f"Substrate generation completed for {experiment_name}")
    
    def run_simulation(self, full_sim_params):
        """Run substrate generation"""    
        main_folder = full_sim_params['substrates']
        if not os.path.isdir(main_folder):
            print(f"Error: {main_folder} is not a valid directory")
            return False
        
        # Get all subfolders in the main folder
        subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

        # Count total substrate files for progress tracking
        total_substrates = 0
        for subfolder in subfolders:
            subfolder_path = os.path.join(main_folder, subfolder)
            data_folder = os.path.join(subfolder_path, "data")
            if os.path.isdir(data_folder):
                substrate_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
                total_substrates += len(substrate_files)

        current_sim = 0
        for subfolder in subfolders:
            subfolder_path = os.path.join(main_folder, subfolder)
            data_folder = os.path.join(subfolder_path, "data")
            
            # Check if data folder exists
            if os.path.isdir(data_folder):
                # Get substrate files in the data folder
                substrate_files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
                
                for substrate_file in substrate_files:
                    current_sim += 1
                    print(f"=====Starting simulation {current_sim}/{total_substrates}=====")
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
    sim_parser.add_argument('--sim_time', type=float, help='Total diffusion time')
    sim_parser.add_argument('--compartment', type=str, help='Compartment to simulate (e.g., intra, extra)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = GeometryToolkitCLI()
    
    if args.command == 'substrate':
        config_data = config.load_config_file(args.config)
        params = config_data["parameters"].copy()
        experiment_name = common_util.build_experiment_name_from_params(params)  # Validate required parameters for naming
        print(f"Generated experiment name: {experiment_name}")
        config_data['experiment_name'] = experiment_name
        # ===== START substrate generation =====
        cli.run_substrate_generation(params, experiment_name)
        # ===== END substrate generation & save experiment config file =====
        shutil.copy(args.config, 
            os.path.join(config_params.OUTPUT_FOLDER_PATH, experiment_name, str(config_params.EXP_DATE_TIME) + "_" + os.path.basename(args.config)))
        
    elif args.command == 'simulation':
        sim_config = config.load_config_file(config_params.SIM_CONFIG_FILE)
        cmd_params = vars(args)
        # Merge config with command line params
        sim_config.update(cmd_params)
        # ===== START simulation =====
        config_params.EXP_DATE_TIME = str(common_util.get_date_time())
        cli.run_simulation(sim_config)
        # ===== END simulation =====
        
        # Save the complete simulation configuration as JSON
        config_filename = str(config_params.EXP_DATE_TIME) + "_" + os.path.basename(config_params.SIM_CONFIG_FILE)
        config_save_path = os.path.join(sim_config['substrates'], config_filename)
        with open(config_save_path, 'w') as f:
            json.dump(sim_config, f, indent=4)
    
if __name__ == "__main__":
    main()