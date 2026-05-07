from datetime import datetime
import pickle
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
import simulation_toolkit.toolkit_params as params


@dataclass
class LoadedSubstrate:
    box_length: float
    outer_fibers: np.ndarray
    inner_fibers: Optional[np.ndarray] = None
    is_myelinated: bool = False
    g_ratio: Optional[float] = None
    inner_sphere_spacing_ratio: Optional[float] = None

def get_date_time():        
    return str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def load_data_pickle(file_name):
    # Open the Pickle file for reading in binary mode ('rb')
    with open(file_name, 'rb') as file:
        # Unpickle the data
        loaded_data = pickle.load(file)
    return loaded_data

def import_array_geometry_full_path(file_name):
    substrate = load_substrate_geometry(file_name)
    return substrate.outer_fibers, substrate.box_length


def _geometry_to_numpy(geometry):
    if isinstance(geometry, torch.Tensor):
        return geometry.detach().cpu().numpy()
    return np.asarray(geometry)


def fiber_id_column(spheres_xyz_r_fid):
    if spheres_xyz_r_fid.shape[1] < 5:
        raise ValueError("Fiber geometry must have at least x, y, z, radius, fiber_id columns")
    return 4


def fiber_id_values(spheres_xyz_r_fid):
    return spheres_xyz_r_fid[:, fiber_id_column(spheres_xyz_r_fid)]


def load_substrate_geometry(file_name):
    loaded_data = load_data_pickle(file_name)
    if isinstance(loaded_data, dict):
        outer_fibers = _geometry_to_numpy(loaded_data["outer_fibers"]).astype(np.float32, copy=False)
        inner_fibers = loaded_data.get("inner_fibers")
        if inner_fibers is not None:
            inner_fibers = _geometry_to_numpy(inner_fibers).astype(np.float32, copy=False)
        return LoadedSubstrate(
            box_length=float(loaded_data["box_length"]),
            outer_fibers=outer_fibers,
            inner_fibers=inner_fibers,
            is_myelinated=inner_fibers is not None,
            g_ratio=loaded_data.get("g_ratio"),
            inner_sphere_spacing_ratio=loaded_data.get("inner_sphere_spacing_ratio"),
        )

    optimized_fibers, L = loaded_data
    return LoadedSubstrate(
        box_length=float(L),
        outer_fibers=_geometry_to_numpy(optimized_fibers).astype(np.float32, copy=False),
    )

def save_data_array_to_pickle(file_name, optimized_fibers, L):
    with open(file_name, 'wb') as f:
        pickle.dump([optimized_fibers, L], f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done saving data file')
    return


def save_myelinated_substrate_to_pickle(file_name, outer_fibers, inner_fibers, L, g_ratio, inner_sphere_spacing_ratio):
    payload = {
        "version": 2,
        "box_length": float(L),
        "outer_fibers": _geometry_to_numpy(outer_fibers).astype(np.float32, copy=False),
        "inner_fibers": _geometry_to_numpy(inner_fibers).astype(np.float32, copy=False),
        "g_ratio": float(g_ratio),
        "inner_sphere_spacing_ratio": float(inner_sphere_spacing_ratio),
    }
    with open(file_name, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done saving myelinated substrate data file')
    return

def map_matrix_to_list_numpy(spheres_xyz_r_fid):
    fid_col = fiber_id_column(spheres_xyz_r_fid)
    unique_values = torch.unique(spheres_xyz_r_fid[:, fid_col])
    current_fiber_list = []
    for value in unique_values:
        current_fiber_list.append(spheres_xyz_r_fid[spheres_xyz_r_fid[:, fid_col] == value].cpu().numpy())
    return current_fiber_list

def map_matrix_to_list_torch(spheres_xyz_r_fid):
    fid_col = fiber_id_column(spheres_xyz_r_fid)
    unique_values = torch.unique(spheres_xyz_r_fid[:, fid_col])
    current_fiber_list = []
    for value in unique_values:
        current_fiber_list.append(spheres_xyz_r_fid[spheres_xyz_r_fid[:, fid_col] == value])
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
        'mean_diameter',
        'orientation_shape_parameter',
        'bead_alpha_mean',
        'num_fibers'
    ]
    # Accept either final_volume_fraction or target_volume_fraction
    has_vf = 'final_volume_fraction' in params or 'target_volume_fraction' in params
    missing = [key for key in required_keys if key not in params]
    if not has_vf:
        missing.append('target_volume_fraction or final_volume_fraction')
    if missing:
        missing_str = ', '.join(missing)
        raise KeyError(f"Missing parameters for experiment naming: {missing_str}")
    vf = str(params.get('final_volume_fraction', params.get('target_volume_fraction')))
    mean_d = str(params['mean_diameter'])
    bead_amp = str(params['bead_alpha_mean'])
    orientation = str(params['orientation_shape_parameter'])
    num_fibers = str(params['num_fibers'])

    return f"bead{bead_amp}_d{mean_d}_OD{orientation}_initVF{vf}_{num_fibers}axons"

