import torch
import torch.multiprocessing as mp

@torch.no_grad()
def detect_in_sphere(pbc_spheres_xyz, r, avf_nodes, L, fid):
    return detect_in_sphere_gpu(pbc_spheres_xyz, r, avf_nodes, L, fiber_id=fid)

def detect_in_sphere_gpu(pbc_spheres_xyz, r, avf_nodes, L, fiber_id):
    '''in_sphere_mask NxM matrix'''
    device = pbc_spheres_xyz.device
    # Partition the 3D cube into smaller blocks
    block_size = torch.tensor([10.0], device=device)  # Adjust this value to control the block size (relative to box length so, 5/10)
    grid_size = torch.ceil(torch.tensor([1.0, 1.0, 1.0], device=device) * torch.tensor([L, L, L], device=device) / block_size).long()
    # -------------- Assign spheres to blocks --------------
    block_assignments  , max_sph_per_seg = set_segments(pbc_spheres_xyz, r, L, grid_size[0], grid_size[1], grid_size[2])
    block_assignments_n = set_segments_nodes(avf_nodes, L, grid_size[0], grid_size[1], grid_size[2])
    
    num_nodes_in_spheres = torch.tensor(0, device=device)
    nsegx, nsegy, nsegz = grid_size[0],grid_size[1],grid_size[2]
    for p in torch.arange(nsegz):
        for m in torch.arange(nsegy):
            for n in torch.arange(nsegx):
                seg_idx = max_sph_per_seg*(p*nsegx*nsegy + m*nsegx + n)
                seg_idx_n = p*nsegx*nsegy + m*nsegx + n
                sphere_indices_in_positions =   block_assignments[seg_idx  :seg_idx+max_sph_per_seg]
                nodes_indices_in_positions  =   block_assignments_n[seg_idx_n.item()]
                if nodes_indices_in_positions.shape[0]>0:
                    block_positions = pbc_spheres_xyz[sphere_indices_in_positions]
                    block_radii     = r[sphere_indices_in_positions]
                    
                    nodes_positions = avf_nodes[nodes_indices_in_positions]
                    # #################################In here only#######################################
                    in_sphere_mask = intersect_block(block_positions,nodes_positions, block_radii)
                    node_mask = torch.sum(in_sphere_mask, dim=0) 
                    num_nodes_in_spheres += torch.count_nonzero(node_mask)
    return num_nodes_in_spheres

def intersect_block(x, x_n, rx):
    ''' distance betwen any 2 spheres of different fiber & sum of 2 axon radii '''
    dm = torch.linalg.norm(x.unsqueeze(1) - x_n.unsqueeze(0), dim=-1)
    # Broadcast the expanded tensor to the shape of dm
    rm = rx.unsqueeze(1).expand(dm.shape)
    in_sphere_mask = dm - rm < 0
    del rm, dm
    return in_sphere_mask

def set_segments_nodes(pos, L, nsegx, nsegy, nsegz):
    Lx, Ly, Lz = L,L,L
    '''return segments = Mx1 vector storing sph ids for each segment. 
    M = k*max_sphere_per_seg, where 'k' is the number of segments. '''
    jump_tol = 0. # um
    dLx = Lx/nsegx
    dLy = Ly/nsegy
    dLz = Lz/nsegz
    segments_dict = {}
    count_nodes=0
    for p in torch.arange(nsegz):
        for m in torch.arange(nsegy):
            for n in torch.arange(nsegx):
                # how far is each sphere from the 6 boundaries of this segment
                inx_min = n*dLx- pos[:,0] -Lx/2
                inx_max = pos[:,0] - ((n+1)*dLx- Lx/2)
                iny_min = m*dLy- pos[:,1] - Ly/2
                iny_max = pos[:,1] - ((m+1)*dLy- Ly/2)
                inz_min = p*dLz- pos[:,2] - Lz/2
                inz_max = pos[:,2] - ((p+1)*dLz- Lz/2)

                in_segment = torch.maximum(inx_min,inx_max)
                in_segment = torch.maximum(in_segment,iny_min)
                in_segment = torch.maximum(in_segment,iny_max)
                in_segment = torch.maximum(in_segment,inz_min)
                in_segment = torch.maximum(in_segment,inz_max)
                seg_idx = p*nsegx*nsegy + m*nsegx + n
                '''get indices of items < 0'''
                count_nodes += torch.nonzero(in_segment < 0).flatten().shape[0]
                segments_dict[seg_idx.item()] = torch.nonzero(in_segment < 0).flatten()
    return segments_dict
#---------------------------------------------------
def set_segments(pos, r, L, nsegx, nsegy, nsegz):
    Lx, Ly, Lz = L,L,L
    '''return segments = Mx1 vector storing sph ids for each segment. 
    M = k*max_sphere_per_seg, where 'k' is the number of segments. '''
    jump_tol = 0
    dLx = Lx/nsegx
    dLy = Ly/nsegy
    dLz = Lz/nsegz
    
    # what is the maximum number of spheres in each segment?
    max_sph_per_seg = torch.tensor([0], device=pos.device)
    for p in torch.arange(nsegz):
        for m in torch.arange(nsegy):
            for n in torch.arange(nsegx):
                # how far is each sphere from the 6 boundaries of this segment?
                inx_min = n*dLx-r - pos[:,0] -Lx/2  # A& intersect if Aleft < Bright && Aright > Bleft %% Intersect if this_ops < 0 
                inx_max = pos[:,0] - ((n+1)*dLx+r - Lx/2)   #%% Intersect if this_ops < 0
                iny_min = m*dLy-r - pos[:,1] -Ly/2          #%% Intersect if this_ops < 0
                iny_max = pos[:,1] - ((m+1)*dLy+r - Ly/2)   #%% Intersect if this_ops < 0
                inz_min = p*dLz-r - pos[:,2] -Lz/2          #%% Intersect if this_ops < 0
                inz_max = pos[:,2] - ((p+1)*dLz+r - Lz/2)   #%% Intersect if this_ops < 0

                in_segment = torch.maximum(inx_min,inx_max)  # if still negative, sphere intesects segment in x-axis
                in_segment = torch.maximum(in_segment,iny_min)
                in_segment = torch.maximum(in_segment,iny_max)
                in_segment = torch.maximum(in_segment,inz_min)
                in_segment = torch.maximum(in_segment,inz_max)
                # negative value of in_segment = inside/overlapping the segment from this boundary
                sph_this_seg = torch.count_nonzero(in_segment < jump_tol) # include into the segment spheres whose in_min/in_max has a 'jump_tol' distance from the segment boundary
                max_sph_per_seg = max(max_sph_per_seg,sph_this_seg)
    
    spheres_per_segment = max_sph_per_seg

    # make a list of spheres in each segment
    segments = torch.empty([nsegx*nsegy*nsegz*spheres_per_segment,],dtype=torch.long, device=pos.device)
    for p in torch.arange(nsegz):
        for m in torch.arange(nsegy):
            for n in torch.arange(nsegx):
                # how far is each sphere from the 6 boundaries of this segment?
                inx_min = n*dLx-r - pos[:,0] -Lx/2
                inx_max = pos[:,0] - ((n+1)*dLx+r - Lx/2)
                iny_min = m*dLy-r - pos[:,1] - Ly/2
                iny_max = pos[:,1] - ((m+1)*dLy+r - Ly/2)
                inz_min = p*dLz-r - pos[:,2] - Lz/2
                inz_max = pos[:,2] - ((p+1)*dLz+r - Lz/2)

                in_segment = torch.maximum(inx_min,inx_max)
                in_segment = torch.maximum(in_segment,iny_min)
                in_segment = torch.maximum(in_segment,iny_max)
                in_segment = torch.maximum(in_segment,inz_min)
                in_segment = torch.maximum(in_segment,inz_max)
                # negative value of in_segment = inside/overlapping the segment from this boundary
                seg_idx = p*nsegx*nsegy + m*nsegx + n
                seg_idx = spheres_per_segment*(p*nsegx*nsegy + m*nsegx + n)
                segments[seg_idx:seg_idx+spheres_per_segment] = torch.argsort(in_segment)[:spheres_per_segment]  # sort the most negative in front, the grab the first [0:spheres_per_segment] spheres

    return segments, spheres_per_segment