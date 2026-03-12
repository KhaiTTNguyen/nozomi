from datetime import datetime
import pickle
import torch
import numpy as np
import simulation_toolkit.toolkit_params as params

def get_date_time():        
    return str(datetime.now().strftime("%Y-%m-%d_%H-%M"))

def load_data_pickle(file_name):
    # Open the Pickle file for reading in binary mode ('rb')
    with open(file_name, 'rb') as file:
        # Unpickle the data
        loaded_data = pickle.load(file)
    return loaded_data

def import_array_geometry_full_path(file_name):
    optimized_fibers, L = load_data_pickle(file_name)
    return optimized_fibers.cpu().numpy(), L

def save_data_array_to_pickle(file_name, optimized_fibers, L):
    with open(file_name, 'wb') as f:
        pickle.dump([optimized_fibers, L], f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done saving data file')
    return

def map_matrix_to_list_numpy(spheres_xyz_r_fid):
    unique_values = torch.unique(spheres_xyz_r_fid[:, -1])
    current_fiber_list = []
    for value in unique_values:
        current_fiber_list.append(spheres_xyz_r_fid[spheres_xyz_r_fid[:, -1] == value].cpu().numpy())
    return current_fiber_list

def map_matrix_to_list_torch(spheres_xyz_r_fid):
    unique_values = torch.unique(spheres_xyz_r_fid[:, -1])
    current_fiber_list = []
    for value in unique_values:
        current_fiber_list.append(spheres_xyz_r_fid[spheres_xyz_r_fid[:, -1] == value])
    return current_fiber_list

def split_matrix_to_list(A):
    # Initialize the list to hold the submatrices
    listA = []     
    # Initialize the start index
    start_idx = 0
    # Loop through the rows and identify the split points
    for i in range(0, len(A)):
        if A[i, 2] == params.BOX_LENGTH/2 :
            # Add the submatrix to the list
            listA.append(A[start_idx:i+1])
            # Update the start index
            start_idx = i+1   
    # Convert each submatrix to numpy array
    listA = [np.array(submatrix) for submatrix in listA]
    return listA

def build_experiment_name_from_params(params):
    """Generate experiment name using required substrate parameters."""
    required_keys = [
        'target_volume_fraction',
        'mean_diameter',
        'orientation_shape_parameter',
        'bead_alpha_mean',
        'num_fibers'
    ]
    missing = [key for key in required_keys if key not in params]
    if missing:
        missing_str = ', '.join(missing)
        raise KeyError(f"Missing parameters for experiment naming: {missing_str}")
    vf = str(params['target_volume_fraction'])
    mean_d = str(params['mean_diameter'])
    bead_amp = str(params['bead_alpha_mean'])
    orientation = str(params['orientation_shape_parameter'])
    num_fibers = str(params['num_fibers'])

    return f"bead{bead_amp}_d{mean_d}_OD{orientation}_initVF{vf}_{num_fibers}axons"

