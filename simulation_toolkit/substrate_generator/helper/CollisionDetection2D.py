import torch, math, time
from itertools import cycle
import torch.multiprocessing as mp


MATRIX_SEG_LENGTH = 1000

@torch.no_grad()
def detect_collision(pos_, ra_, fid_, Lx, Ly, buff, device):
    # if sphere_positions.shape[0] < 4e6: # for now running just single GPU multisegments
    buff=torch.tensor([buff])
    return collide_spheres_gpu(pos_, ra_, fid_, Lx, Ly, buff, device)

def collide_spheres_gpu(pos_, ra_, fid_, Lx, Ly, buff, device):
    sphere_positions = pos_.to(device)
    sphere_radii = ra_.to(device)
    fiber_id = fid_.to(device)
    buff = buff.to(device)

    # Partition the 3D cube into smaller blocks
    block_size = torch.tensor([5.0], device=device)  # Adjust this value to control the block size (relative to box length so, 5/10)
    grid_size = torch.ceil(torch.tensor([1.0, 1.0], device=device) * torch.tensor([Lx, Ly], device=device) / block_size).long().to(device)
    # print('grid_size', grid_size)

    # -------------- Assign spheres to blocks --------------
    block_assignments, max_sph_per_seg = set_segments(sphere_positions, sphere_radii, Lx, Ly, grid_size[0], grid_size[1])
    # print(f'max_sph_per_seg: {max_sph_per_seg}')
    # print('block_assignments', (block_assignments))
    block_assignments, max_sph_per_seg = block_assignments.to(device), max_sph_per_seg.to(device)

    collision_pairs = torch.tensor([], device=device)
    nsegx, nsegy = grid_size[0],grid_size[1]
    for n in torch.arange(nsegx):
        for m in torch.arange(nsegy):
            seg_idx = max_sph_per_seg*(m*nsegx + n)
            sphere_indices_in_positions = block_assignments[seg_idx:seg_idx+max_sph_per_seg]

            block_positions = sphere_positions[sphere_indices_in_positions]
            block_radii     = sphere_radii[sphere_indices_in_positions]
            # #################################In here only#######################################
            # Perform collision detection for spheres within the block
            ret = torch.nonzero(intersect_block(block_positions,block_radii, buff)) #\

            '''Fix indices'''
            temp = torch.stack([sphere_indices_in_positions[ret[:,0]],
                                    sphere_indices_in_positions[ret[:,1]]]).T
            # #################################In here only#######################################
            temp, idx = temp.sort(dim=1) # forcepairs to be [a,b] where a<b, dim=1 sorts along columns
            collision_pairs = torch.cat((collision_pairs, temp))
            collision_pairs = torch.unique(collision_pairs, dim=0) #dim=0 check for elements along rows
    collision_pairs = torch.unique(collision_pairs, dim=0) #dim=0 check for elements along rows
    return collision_pairs.long()

def intersect_block(x, rx, buff):
    rm = rx.view(-1, 1) + rx.view(1, -1)

    ''' distance betwen any 2 spheres of different fiber & sum of 2 axon radii '''
    dm = torch.linalg.norm(x.unsqueeze(1) - x.unsqueeze(0), dim=-1)
    mask = rm + buff - dm > 0
    mask = torch.tril(mask, diagonal=-1)
    del rm, dm
    # torch.cuda.empty_cache()
    return mask


def set_segments(pos, r,  Lx, Ly, nsegx, nsegy):
    # how many segements are we using?
    #
    jump_tol = 1 # um, tolerance to include a sphere inside a boundary...?
    # dLx, dLy, dLz = block_size, block_size, block_size
    dLx = Lx/nsegx
    dLy = Ly/nsegy
    # print('circle_centers', pos[:,0], min(pos[:,0]))
    # print('circle_centers', pos[:,1], min(pos[:,1]))
    # what is the maximum number of spheres in each segment?
    max_sph_per_seg = 0
    for n in torch.arange(nsegx):
        for m in torch.arange(nsegy):
            # how far is each sphere from the 6 boundaries of this segment?
            inx_min = n*dLx-r - pos[:,0] -Lx/2  # A& intersect if Aleft < Bright && Aright > Bleft
            inx_max = pos[:,0] - ((n+1)*dLx+r - Lx/2)
            iny_min = m*dLy-r - pos[:,1] -Ly/2
            iny_max = pos[:,1] - ((m+1)*dLy+r - Ly/2)

            in_segment = torch.maximum(inx_min,inx_max)  # if still negative, sphere intesects segment in x-axis
            in_segment = torch.maximum(in_segment,iny_min)
            in_segment = torch.maximum(in_segment,iny_max)
            # negative value of in_segment = inside/overlapping the segment from this boundary
            sph_this_seg = torch.count_nonzero(in_segment < jump_tol) # include into the segment spheres whose in_min/in_max has a 'jump_tol' distance from the segment boundary

            max_sph_per_seg = max(max_sph_per_seg,sph_this_seg)
    spheres_per_segment = max_sph_per_seg

    # make a list of spheres in each segment
    segments = torch.zeros([nsegx*nsegy*spheres_per_segment,],dtype=torch.long)
    for n in torch.arange(nsegx):
        for m in torch.arange(nsegy):
            # how far is each sphere from the 6 boundaries of this segment?
            inx_min = n*dLx-r - pos[:,0] -Lx/2
            inx_max = pos[:,0] - ((n+1)*dLx+r - Lx/2)
            iny_min = m*dLy-r - pos[:,1] - Ly/2
            iny_max = pos[:,1] - ((m+1)*dLy+r - Ly/2)

            in_segment = torch.maximum(inx_min,inx_max)
            in_segment = torch.maximum(in_segment,iny_min)
            in_segment = torch.maximum(in_segment,iny_max)
            # negative value of in_segment = inside/overlapping the segment from this boundary

            seg_idx = spheres_per_segment*(m*nsegx + n)
            segments[seg_idx:seg_idx+spheres_per_segment] = torch.argsort(in_segment)[:spheres_per_segment]  # sort the most negative in front, the grab the first [0:spheres_per_segment] spheres

    return segments, spheres_per_segment

def collide_spheres_multigpu(sphere_positions, sphere_radii, fiber_id, Lx, Ly, Lz, buff, gpus_ids):
    # Multi-GPU code goes here
    pass
