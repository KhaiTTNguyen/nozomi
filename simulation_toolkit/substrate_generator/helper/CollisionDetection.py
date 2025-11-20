import torch, math, time
from itertools import cycle
import torch.multiprocessing as mp
# from linr impor

MATRIX_SEG_LENGTH = 1000

@torch.no_grad()
def detect_collision(sphere_positions, sphere_radii, fiber_id, Lx, Ly, Lz, buff, gpus_ids=None):
    # if sphere_positions.shape[0] < 4e6: # for now running just single GPU multisegments
    return collide_spheres_gpu(sphere_positions, sphere_radii, fiber_id, Lx, Ly, Lz, buff)
    # else:
    #     return collide_spheres_multigpu(sphere_positions, sphere_radii, fiber_id, Lx, Ly, Lz, buff, gpus_ids)
def collide_spheres_gpu(sphere_positions, sphere_radii, fiber_id, Lx, Ly, Lz, buff):
    """
    Detect collisions between N sphere objects on the GPU.
    
    Args:
        sphere_positions (torch.Tensor): Tensor of shape (N, 3) containing the (x, y, z) positions of the spheres.
        sphere_radii (torch.Tensor): Tensor of shape (N,) containing the radii of the spheres.
        device (str, optional): The device to use, either "cuda:0" or "cpu". Defaults to "cuda:0".
    
    Returns:
        torch.Tensor: Tensor of shape (M, 2) containing the indices of the colliding sphere pairs.
    """
    device = sphere_positions.device
    
    # Partition the 3D cube into smaller blocks 
    sub_cube_ratio = torch.tensor(10, device=device)

    block_size = Lx.clone().detach()/sub_cube_ratio
    # block_size = torch.tensor([10.0], device=device)  # PREVIOUSLY USED Adjust this value to control the block size (relative to box length so, 5/10)
    # block_size = torch.tensor([5.0], device=device)  # Adjust this value to control the block size (relative to box length so, 5/10)
    grid_size = torch.ceil(torch.tensor([1.0, 1.0, 1.0], device=device) * torch.tensor([Lx, Ly, Lz], device=device) / block_size).long()
    # print('grid_size', grid_size)

    # -------------- Assign spheres to blocks --------------
    # print('set_segments')
    block_assignments, max_sph_per_seg = set_segments(sphere_positions, sphere_radii, Lx, Ly, Lz, grid_size[0], grid_size[1], grid_size[2])
    # print('Done set_segments')
    
    # block_assignments, max_sph_per_seg = block_assignments.to(device), max_sph_per_seg.to(device)

    collision_pairs = torch.tensor([], device=device)
    nsegx, nsegy, nsegz = grid_size[0],grid_size[1],grid_size[2]
    for p in torch.arange(nsegz):
        for m in torch.arange(nsegy):
            for n in torch.arange(nsegx):
                seg_idx = max_sph_per_seg*(p*nsegx*nsegy + m*nsegx + n)
                sphere_indices_in_positions = block_assignments[seg_idx:seg_idx+max_sph_per_seg]
                
                block_positions = sphere_positions[sphere_indices_in_positions]
                block_radii     = sphere_radii[sphere_indices_in_positions]
                block_fid       = fiber_id[sphere_indices_in_positions]
                # #################################In here only#######################################
                # Perform collision detection for spheres within the block
                ret = torch.nonzero(intersect_block(block_positions,block_radii,block_fid, buff)) #\
                
                '''Fix indices'''
                temp = torch.stack([sphere_indices_in_positions[ret[:,0]],
                                        sphere_indices_in_positions[ret[:,1]]]).T
                # #################################In here only#######################################
                temp, idx = temp.sort(dim=1) # forcepairs to be [a,b] where a<b, dim=1 sorts along columns
                collision_pairs = torch.cat((collision_pairs, temp))
                collision_pairs = torch.unique(collision_pairs, dim=0) #dim=0 check for elements along rows
    collision_pairs = torch.unique(collision_pairs, dim=0) #dim=0 check for elements along rows
    return collision_pairs.long()
def intersect_block(x, rx, fx, buff):
    fm = (fx.view(-1, 1) - fx.view(1, -1)).bool() # fiber_id
    rm = rx.view(-1, 1) + rx.view(1, -1)

    ''' distance betwen any 2 spheres of different fiber & sum of 2 axon radii '''
    dm = torch.linalg.norm(x.unsqueeze(1) - x.unsqueeze(0), dim=-1)
    mask = rm + buff - dm > 0

    del rm, dm
    ret = torch.logical_and(mask, fm)
    del mask, fm
    # torch.cuda.empty_cache() still important when run out of memory
    return ret
def set_segments(pos, r,  Lx, Ly, Lz, nsegx, nsegy, nsegz):
    '''return segments = Mx1 vector storing sph ids for each segment. 
    M = k*max_sphere_per_seg, where 'k' is the number of segments. '''
    jump_tol = 1 # um, tolerance to include a sphere inside a boundary...?
    # dLx, dLy, dLz = block_size, block_size, block_size
    dLx = Lx/nsegx
    dLy = Ly/nsegy
    dLz = Lz/nsegz
    
    # what is the maximum number of spheres in each segment?
    max_sph_per_seg = torch.tensor([0], device=pos.device)
    for p in torch.arange(nsegz):
        for m in torch.arange(nsegy):
            for n in torch.arange(nsegx):
                # how far is each sphere from the 6 boundaries of this segment?
                inx_min = n*dLx-r - pos[:,0] -Lx/2  # A& intersect if Aleft < Bright && Aright > Bleft
                inx_max = pos[:,0] - ((n+1)*dLx+r - Lx/2)
                iny_min = m*dLy-r - pos[:,1] -Ly/2
                iny_max = pos[:,1] - ((m+1)*dLy+r - Ly/2)
                inz_min = p*dLz-r - pos[:,2] -Lz/2
                inz_max = pos[:,2] - ((p+1)*dLz+r - Lz/2)

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
    # segments = torch.empty([nsegx*nsegy*nsegz*spheres_per_segment,])
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

                seg_idx = spheres_per_segment*(p*nsegx*nsegy + m*nsegx + n)
                segments[seg_idx:seg_idx+spheres_per_segment] = torch.argsort(in_segment)[:spheres_per_segment]  # sort the most negative in front, the grab the first [0:spheres_per_segment] spheres

    return segments, spheres_per_segment

def collide_spheres_multigpu(sphere_positions, sphere_radii, fiber_id, Lx, Ly, Lz, buff, gpus_ids):
    # Multi-GPU code goes here
    pass
