import numpy as np

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