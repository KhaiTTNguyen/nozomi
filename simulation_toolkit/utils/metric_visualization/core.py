import os
import re
from collections import defaultdict
from pathlib import Path


def extract_k_value_from_filename(filename):
    """
    Extract K value from filename using regex.
    
    Args:
        filename: String containing the filename
        
    Returns:
        K value as string (e.g., '10', '20', '200') or None if not found
    """
    # Look for pattern like _K10_ or _K20_ or _K200_
    match = re.search(r'_K(\d+)_', filename)
    if match:
        return match.group(1)
    return None


def extract_diameter_from_filename(filename):
    """
    Extract diameter value from filename.
    
    Args:
        filename: String containing the filename
        
    Returns:
        Diameter value as string or None if not found
    """
    # Look for pattern like _d1.68_ or _d4.5_
    match = re.search(r'_d([\d.]+)_', filename)
    if match:
        return match.group(1)
    return None


def extract_compartment_from_filename(filename):
    """
    Extract compartment (intra/extra) from filename.
    
    Args:
        filename: String containing the filename
        
    Returns:
        'intra' or 'extra' as string, or None if not found
    """
    if '_intra_' in filename:
        return 'intra'
    elif '_extra_' in filename:
        return 'extra'
    return None


def gather_experiment_files(base_folder, metric='adc'):
    """
    Gather all experiment files organized by K value, diameter, and compartment.
    
    Args:
        base_folder: Path to the base experiment visualization folder
        
    Returns:
        Dictionary organized as:
        {
            'K10': {
                'intra': {
                    '1.68': [file_paths],
                    '2.58': [file_paths],
                    ...
                },
                'extra': {
                    '1.68': [file_paths],
                    ...
                }
            },
            'K20': {...},
            ...
        }
    """
    base_path = Path(base_folder)
    data_path = base_path / "data"
    
    if not data_path.exists():
        raise ValueError(f"Data folder not found: {data_path}")
    
    # Structure: K_value -> compartment -> diameter -> [file_paths]
    organized_files = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    if metric == 'adc':
        metric = 'ADCdata'
    elif metric == 'kurtosis':
        metric = 'kurtosis_data'
    # Walk through all subdirectories
    for experiment_dir in data_path.iterdir():
        if not experiment_dir.is_dir():
            continue
            
        print(f"Processing experiment: {experiment_dir.name}")
        
        # Look for ADCdata and kurtosis_data folders in the experiment directory
        for root, dirs, files in os.walk(experiment_dir):
            folder_name = os.path.basename(root)
            if metric in folder_name: # or 'kurtosis_data' in folder_name:
                # print(f"  Found data folder: {root}")
                
                # Process all .pkl files in this data folder
                for filename in files:
                    if filename.endswith('.pkl'):
                        file_path = os.path.join(root, filename)
                        
                        # Extract metadata from filename
                        k_value = extract_k_value_from_filename(filename)
                        diameter = extract_diameter_from_filename(filename)
                        compartment = extract_compartment_from_filename(filename)
                        
                        if k_value and diameter and compartment:
                            organized_files[f'K{k_value}'][compartment][diameter].append(file_path)
                            # print(f"    Added: K={k_value}, {compartment}, d={diameter} from {folder_name}")
                        else:
                            print(f"    Skipped (missing metadata): {filename}")
    
    # Convert defaultdict to regular dict for cleaner output
    result = {}
    for k_value, k_data in organized_files.items():
        result[k_value] = {}
        for compartment, comp_data in k_data.items():
            result[k_value][compartment] = dict(comp_data)
    
    return result


def print_organized_files(organized_files):
    """
    Print the organized file structure for verification.
    
    Args:
        organized_files: Dictionary from gather_experiment_files()
    """
    print("\n" + "="*60)
    print("ORGANIZED EXPERIMENT FILES")
    print("="*60)
    
    for k_value, k_data in sorted(organized_files.items()):
        print(f"\n{k_value}:")
        
        for compartment, comp_data in sorted(k_data.items()):
            print(f"  {compartment}:")
            
            for diameter, file_list in sorted(comp_data.items(), key=lambda x: float(x[0])):
                print(f"    d={diameter}μm: {len(file_list)} files")
                
                # Print first few file paths as examples
                for i, file_path in enumerate(file_list[:2]):  # Show first 2 files
                    print(f"      {i+1}. {file_path}")
                
                if len(file_list) > 2:
                    print(f"      ... and {len(file_list)-2} more files")


def get_files_by_criteria(organized_files, diameter=None, k_value=None, compartment=None):
    """
    Get files matching specific criteria.
    
    Args:
        organized_files: Dictionary from gather_experiment_files()
        k_value: K value to filter by (e.g., 'K10', 'K20')
        compartment: Compartment to filter by ('intra', 'extra')
        diameter: Diameter to filter by (e.g., '1.68', '2.58')
        
    Returns:
        List of file paths matching criteria
    """
    matching_files = []
    
    for k_val, k_data in organized_files.items():
        if k_value and k_val != k_value:
            continue
            
        for comp, comp_data in k_data.items():
            if compartment and comp != compartment:
                continue
                
            for diam, file_list in comp_data.items():
                if diameter and diam != diameter:
                    continue
                    
                matching_files.extend(file_list)
    
    return matching_files


# def main():
#     """
#     Main function to demonstrate the file gathering functionality.
#     """
#     # Define base folder path
#     base_folder = "./experiment/visualization"
    
#     try:
#         # Gather all experiment files
#         print("Gathering experiment files...")
#         organized_files = gather_experiment_files(base_folder)
        
#         # Print organized structure
#         # print_organized_files(organized_files)
        
#         # Example: Get all K10 intra files
#         k10_intra_files = get_files_by_criteria(organized_files, k_value='K10', compartment='intra')
#         print(f"\n\nExample - K10 intra files: {len(k10_intra_files)} total")
        
#         # Example: Get all files for diameter 2.58
#         d258_files = get_files_by_criteria(organized_files, diameter='2.58')
#         print(f"Example - d=2.58μm files: {len(d258_files)} total")
        
#         d258_k10_intra_files = get_files_by_criteria(organized_files, diameter='2.58', k_value='K10')
#         print(f"\n\nExample - K10 intra files: {len(d258_k10_intra_files)} total")
#         # print(d258_k10_intra_files)
#         return organized_files
        
#     except Exception as e:
#         print(f"Error: {e}")
#         return None


# if __name__ == "__main__":
#     organized_files = main()