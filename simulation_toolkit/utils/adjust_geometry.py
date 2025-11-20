# load data from Pickle format
import numpy as np
import os
import matplotlib.animation as animation
from scipy.interpolate import interp1d, CubicSpline
from scipy.special import sph_harm
from scipy import linalg
from matplotlib.colors import LightSource
from scipy.stats import lognorm, genextreme
import matplotlib.ticker as ticker
from matplotlib.pyplot import cm

def createBoundaryZ_straight_cylinder(fiberlist_xyz_r_fid, Lz):
    for start_fiber in fiberlist_xyz_r_fid:
        start_fiber[:,-1] = start_fiber[:,-1].astype(int) # change fiber_id to (int)
    fiber_list_z_bounded = []
    dictionary = {}
    
    # Create the dictionary
    for end_fiber in fiberlist_xyz_r_fid:
        dictionary[(end_fiber[-1][0], end_fiber[-1][1])] = end_fiber # grab (x,y) coord
    # Iterate through list A and populate the result
    for start_fiber in fiberlist_xyz_r_fid:
        if (start_fiber[0][0], start_fiber[0][1]) in dictionary: # grab (x,y) coord
            end_fiber=dictionary[(start_fiber[0][0], start_fiber[0][1])]
            fiber_list_z_bounded.append(create_pad(end_fiber, start_fiber, Lz))
        else:
            fiber_list_z_bounded.append(start_fiber)
    
    return fiber_list_z_bounded

def createBoundaryZ(fiberlist_xyz_r_fid, Lz):
    for start_fiber in fiberlist_xyz_r_fid:
        start_fiber[:,-1] = start_fiber[:,-1].astype(int) # change fiber_id to (int)
    fiber_list_z_bounded = []
    dictionary = {}
    
    # Create the dictionary
    for end_fiber in fiberlist_xyz_r_fid:
        dictionary[(end_fiber[-1][0], end_fiber[-1][1])] = end_fiber # grab (x,y) coord
    # Iterate through list A and populate the result
    for start_fiber in fiberlist_xyz_r_fid:
        if (start_fiber[0][0], start_fiber[0][1]) in dictionary: # grab (x,y) coord
            end_fiber=dictionary[(start_fiber[0][0], start_fiber[0][1])]
            fiber_list_z_bounded.append(create_pad(end_fiber, start_fiber, Lz))
        else:
            fiber_list_z_bounded.append(start_fiber)
    
    return fiber_list_z_bounded

def create_pad(end_fiber, start_fiber, Lz):
    pad=60
    # pad = 20
    start_fiber_swp = end_fiber
    # start_fiber_swp[:,-1] = start_fiber[:,-1]
    fbr_z_bnded = np.zeros((start_fiber.shape[0]+pad, start_fiber.shape[-1]))
    sx, sy, sz, sr, fid = start_fiber[:,0], start_fiber[:,1], start_fiber[:,2], start_fiber[:,3], start_fiber[:,4]
    swx, swy, swz, swr, swfid = start_fiber_swp[:,0], start_fiber_swp[:,1], start_fiber_swp[:,2], start_fiber_swp[:,3], start_fiber_swp[:,4]
    sx = np.concatenate((swx[-int(pad/2):],    sx, swx[0:int(pad/2)]    ), axis=0)  
    sy = np.concatenate((swy[-int(pad/2):],    sy, swy[0:int(pad/2)]    ), axis=0)
    sz = np.concatenate((swz[-int(pad/2):]-Lz, sz, swz[0:int(pad/2)]+Lz ), axis=0)
    sr = np.concatenate((swr[-int(pad/2):],    sr, swr[0:int(pad/2)]    ), axis=0)  
    fid = np.concatenate((fid[-int(pad/2):],    fid, fid[0:int(pad/2)]    ), axis=0)  
    fbr_z_bnded[:,0], fbr_z_bnded[:,1], fbr_z_bnded[:,2], fbr_z_bnded[:,3], fbr_z_bnded[:,4] = sx, sy, sz, sr, fid
    return fbr_z_bnded

# def unpack_fiber_list_to_matrix_forms(fiber_list):
#         inbox_xa = np.array([x for fiber in fiber_list for x in fiber.x])
#         inbox_ya = np.array([y for fiber in fiber_list for y in fiber.y])
#         inbox_za = np.array([z for fiber in fiber_list for z in fiber.z])
#         inbox_ra = np.array([r for fiber in fiber_list for r in fiber.radius])
#         fiber_id = np.array([fiber_id for fiber in fiber_list for fiber_id in fiber.fiber_id])
#         sphere_id = np.array([sphere_id for fiber in fiber_list for sphere_id in fiber.sphere_id])
#         return inbox_xa, inbox_ya, inbox_za, inbox_ra, fiber_id, sphere_id


'deprecated?'
def get_fiber_starts_ends(wrapped_fiber_id):
        current = -1
        starts = np.array([])
        ends = np.array([])
        for i in range(len(wrapped_fiber_id)):
            if wrapped_fiber_id[i]!=current:
                starts=np.append(starts, np.array([i]))
                current = wrapped_fiber_id[i]
                if i>0:
                    ends = np.append(ends, np.array([i-1]))
            if i==len(wrapped_fiber_id)-1:
                ends = np.append(ends, np.array([i]))
        return starts.astype(int), ends.astype(int)

def shiftFOVtoPlusMinusHalfLx(fiber_list, Lx, Ly, Lz):
    for fiber in fiber_list:
        fiber.x, fiber.y, fiber.z \
              = fiber.x-Lx/2, fiber.y-Ly/2, fiber.z-Lz/2
    return fiber_list      

def split_matrix_to_list(A, box_length):
    ''''''
    # Initialize the list to hold the submatrices
    listA = []     
    # Initialize the start index
    start_idx = 0
    # Loop through the rows and identify the split points
    for i in range(0, len(A)):
        if A[i, 2] == box_length/2 :
            # Add the submatrix to the list
            listA.append(A[start_idx:i+1])
            # Update the start index
            start_idx = i+1   
    # Convert each submatrix to numpy array
    listA = [np.array(submatrix) for submatrix in listA]
    return listA

def map_matrix_to_list_numpy(spheres_xyz_r_fid):
    unique_values = np.unique(spheres_xyz_r_fid[:, -1])
    current_fiber_list = []
    for value in unique_values:
        current_fiber_list.append(spheres_xyz_r_fid[spheres_xyz_r_fid[:, -1] == value])
    return current_fiber_list

def adjust_time_step(current_time, max_dt):
    if 0<=current_time and current_time <= 0.001:
        return 0.001  # Increase current_time step
    elif 0.001<current_time and current_time <= 0.01:
        return 0.001  # Increase current_time step
    elif 0.01<current_time and current_time <= 0.1:
        return 0.001  # Increase current_time step
    elif 0.1<current_time and current_time <=1. :
        return 0.001  # Increase time step
    elif 1.<current_time and current_time <=10.:
        return 0.001  # Increase time step
    elif 10.<current_time:
        return 0.001  # Increase time step
    elif 100.<current_time:
        return 0.001  # Increase time step
