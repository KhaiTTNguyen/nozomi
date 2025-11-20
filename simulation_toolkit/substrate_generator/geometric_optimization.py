import simulation_toolkit.substrate_generator.helper.CollisionDetection as CD
import simulation_toolkit.substrate_generator.helper.CollisionDetectionNode as CDN
import simulation_toolkit.utils.common_utils as util
# from simulation_toolkit.utils.common_utils import fiber_3D_plot
import math
import torch
from torch.optim import LBFGS
from functools import partial
import numpy as np
import simulation_toolkit.defaults.params as config_params

class GeometricOptimization(object):
    """

    A class for GeometricOptimization
    """
    def __init__(self, device, meshing, spheres_spacing_ratio,  w_overlap, w_curve, w_length, space_buffer_repulse,\
            mean_d_underlying, sigma_d_underlying, bead_spacing_mean=5.70, bead_spacing_stdv=2.88):
        self.device=device
        self.date_time = meshing.date_time
        self.spheres_spacing_ratio = spheres_spacing_ratio  
        self.w_overlap, self.w_curve, self.w_length = w_overlap, w_curve, w_length 
        self.space_buffer_repulse = torch.tensor(space_buffer_repulse, device=self.device)
        self.mean_d_underlying, self.sigma_d_underlying = mean_d_underlying, sigma_d_underlying
        self.bead_spacing_mean, self.bead_spacing_stdv = bead_spacing_mean, bead_spacing_stdv
        
        # self.interpolated_cylinder_fiber_list = self.interpolate_fibers_with_spheres_start_end(cylinder=True)
        self.interpolated_fiber_list = meshing.interpolated_fiber_list
        '''UNCOMMENT THIS PART AFTER TESTING'''
        self.optimized = False
        self.optimizer = None
        self.optimized_fibers = self.repulse_3d_with_optimizer(num_iteration=90)

    def get_volume_fraction(self, fibers=None):
        L, N = config_params.BOX_LENGTH,  config_params.NUM_NODES_FOR_IVF_CALCULATION 
        avf_nodes = torch.rand(N, 3, device=self.device)*L - L/2
        if fibers!=None :
            spheres_xyz_r_fid = fibers[:,0:6]
        else:
            curr_fibers = self.interpolated_fiber_list
            spheres_xyz_r_fid = curr_fibers[:,0:6]
        filler_m = torch.zeros(spheres_xyz_r_fid.shape[0], device=self.device)
        x,y,z,r,_, fid = self.torch_optimizer_wrapPBC(spheres_xyz_r_fid, filler_m)
        pbc_spheres_xyz = torch.hstack((x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)))
        
        # segment space, assign spheres to segment, assign nodes to segment, calc intersect.
        num_nodes_in_spheres = CDN.detect_in_sphere(pbc_spheres_xyz, r, avf_nodes, L, fid)
        avf = num_nodes_in_spheres / N
        print('-----avf-----', np.round(avf.item(),3))
        return round(avf.item(),2)
    
    def torch_create_starts_ends_mask(self, starts_ends, positions_length):
        starts, ends = starts_ends[0], starts_ends[1]
        # Check if both tensors have the same length
        if len(starts) != len(ends):    raise ValueError("Tensors starts and ends must have the same length.")
        # Merge the tensors by interleaving elements
        starts_ends_indices = torch.cat((starts.unsqueeze(1), ends.unsqueeze(1)), dim=1).reshape(-1).long()
        mask = torch.zeros(positions_length, dtype=torch.bool, device=self.device)
        mask[starts_ends_indices] = True
        return mask
    
    def torch_get_fiber_starts_ends(self, wrapped_fiber_id):
        unique_ids, first_indices = wrapped_fiber_id.unique(return_counts=True)
        cumsum_end =  torch.cumsum(first_indices,0)
        ends = cumsum_end - 1
        cumsum_all = torch.hstack((torch.tensor([0],device=self.device), cumsum_end))
        starts = cumsum_all[:-1]
        return starts, ends
    
    # @profile
    def check_num_overlaps(self, positions_with_grad, torch_ra, mask_starts_ends, fiber_id):
        detached_positions = positions_with_grad.detach()        
        xyzr_fid = torch.hstack([detached_positions, torch_ra.unsqueeze(1), fiber_id.unsqueeze(1)])
        xa_, ya_, za_, ra_, mask_, fid_ = self.torch_optimizer_wrapPBC(xyzr_fid, mask_starts_ends, tol=config_params.BOX_LENGTH/5)
        pos_ = torch.stack([xa_, ya_, za_]).T
        Lx, Ly, Lz, buff = config_params.BOX_LENGTH, config_params.BOX_LENGTH, config_params.BOX_LENGTH, torch.tensor([0.000], device=self.device) #self.space_buffer_repulse #torch.tensor(2,device=self.device) #torch.tensor([0.0005], device=self.device)
        # print('detect collision')
        collision_set = CD.detect_collision(pos_, ra_, fid_, Lx, Ly, Lz, buff)
        # print('Done wrapping')
        num_overlap = collision_set.shape[0]
        #-----ovelap_cost--------
        # ovlap_cost = self.get_steps_with_optimizer(pos_, mask_, ra_, collision_set)
        return num_overlap, xa_, ya_, za_, ra_, fid_ 
    
    '''UNCOMMENT THIS PART AFTER TESTING'''
    def repulse_3d_with_optimizer(self, num_iteration=90):
        torch.autograd.set_detect_anomaly(True)
        # @profile
        def apply_mask(c_pos, c_mask):
            expanded_mask = c_mask[:, None].expand(c_pos.shape)
            masked_positions = torch.where(expanded_mask, c_pos.detach(), c_pos)
            return masked_positions
        # @profile
        def objective_function(c_pos, c_mask, c_ra, wrapped_fiber_id, wo,wc,wl):
            # print('apply mask')
            m_pos = apply_mask(c_pos, c_mask)  # '''detach startend points from computation map'''
            # Compute the loss based on both fixed and variable entries             
            # print('calc overlap cost')
            overlap_cost = self.overlap_cost_function(m_pos, c_mask, c_ra, wrapped_fiber_id)
            # print('calc length cost')
            length_cost = self.length_cost_function(m_pos, c_mask, c_ra)
            # print('calc curve cost')
            curvature_cost = self.curvature_cost_function(m_pos, c_mask, c_ra)
            # w_overlap, wlength, w_curve = 6, 0.05, 0.6 #GOOD FOR 20 fibers?  20, 0.05, 0.6 GOOD for 30 fibers?
            total_cost = wo*overlap_cost + wc*curvature_cost + wl*length_cost
            return total_cost
        '''closure function for LBFGS'''
        # @profile
        def closure(wo_wc_wl):
            (wo, wc, wl) = wo_wc_wl # unpack tuple
            self.optimizer.zero_grad()
            xyzr_fid = torch.hstack([positions_with_grad, t_ra_original.unsqueeze(1), f_id.unsqueeze(1)])
            c_xa, c_ya, c_za, c_ra, c_mask, c_fid = self.torch_optimizer_wrapPBC(xyzr_fid, mask_starts_ends, tol=config_params.BOX_LENGTH/5)
            c_pos = torch.stack([c_xa, c_ya, c_za]).T.contiguous()
            loss = objective_function(c_pos, c_mask, c_ra, c_fid, wo=wo,wc=wc,wl=wl)
            loss.backward(retain_graph=True)
            return loss  
        
        print('''-----------------------Starting overlap removal-----------------------''' )       
        # filler_m = torch.empty(self.interpolated_fiber_list.shape[0],device=self.device)
        # x,y,z,r,_,f_id = self.torch_optimizer_wrapPBC(self.interpolated_fiber_list, filler_m)
        # pbc_spheres_xyz_r_fid = torch.hstack((x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1), r.unsqueeze(1), f_id.unsqueeze(1)))
        # self.plot_fibers(pbc_spheres_xyz_r_fid, optimized=False, POV='top_down')
        # self.plot_fibers(pbc_spheres_xyz_r_fid, optimized=False, POV='horizontal_90')
        # self.plot_fibers(pbc_spheres_xyz_r_fid, optimized=False, POV='horizontal_0')
        # self.plot_fibers(pbc_spheres_xyz_r_fid, optimized=False)
        
        # self.check_start_ends_intact(pbc_spheres_xyz_r_fid)
        
        # pbc_spheres_xyz_r_fid ------------interpolated_fiber_list!!!!!!!!!!
        init_xyz_r_fid = self.interpolated_fiber_list[:,0:6]
        t_xa, t_ya, t_za, t_ra_original, f_id = init_xyz_r_fid[:, 0], init_xyz_r_fid[:, 1], init_xyz_r_fid[:, 2], init_xyz_r_fid[:,3], init_xyz_r_fid[:,4]  
        positions_with_grad = torch.stack([t_xa, t_ya, t_za]).T.contiguous()
        mask_starts_ends = self.torch_create_starts_ends_mask(self.torch_get_fiber_starts_ends(f_id), len(positions_with_grad)) # startpoints donot move
        positions_with_grad.requires_grad_(True)
        # self.optimizer = LBFGS([positions_with_grad], lr=1)
        self.optimizer = LBFGS([positions_with_grad], lr=1., line_search_fn='strong_wolfe')
        # self.optimizer = SGD([positions_with_grad], lr=0.01, momentum=0.9) 
        # self.optimizer = Adagrad([positions_with_grad], lr=1, lr_decay=0.01) 
        num_overlap = 0
        from collections import deque
        m_cost, m_overlaps = deque(maxlen=10), deque(maxlen=10)
        
        xa_, ya_, za_, ra_, fid_, original_fid_ = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        for num_iter in range(0,num_iteration):         
            num_overlap, xa_, ya_, za_, ra_, fid_ = \
                self.check_num_overlaps(positions_with_grad, t_ra_original, mask_starts_ends, f_id)
            print('------------- Iteration:',num_iter,'. Overlaps: ', str(int(num_overlap)), '---------------')
            m_overlaps.append(num_overlap)
            # TEMPORARILY TURNED OFF
            # if num_iter%200==0: # and num_iter!=0:
            #     pbc_spheres_xyz_r_fid = torch.hstack((xa_.unsqueeze(1),ya_.unsqueeze(1),za_.unsqueeze(1), ra_.unsqueeze(1), fid_.unsqueeze(1)))
            #     fiber_3D_plot.plot_fibers(pbc_spheres_xyz_r_fid, color=None, optimized=False, num_iter=num_iter)        
            if num_overlap == 0:
                self.optimized = True       
                break
            '''works for 250 2.5+0.6'''
            if num_iter>85:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1000000,self.w_curve, self.w_length)))            
            elif num_iter>80:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*600000,self.w_curve, self.w_length)))            
            elif num_iter>75:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*300000,self.w_curve, self.w_length)))
            elif num_iter>70:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*100000,self.w_curve, self.w_length)))
            elif num_iter>65:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*60000,self.w_curve, self.w_length)))
            elif num_iter>60:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*30000,self.w_curve, self.w_length)))
            elif num_iter>55:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10000,self.w_curve, self.w_length)))
                # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*2500,self.w_curve, self.w_length)))
            elif num_iter>50:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*6000,self.w_curve, self.w_length)))
                # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*2000,self.w_curve, self.w_length)))
            elif num_iter>45:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*3000,self.w_curve, self.w_length)))
                # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1500,self.w_curve, self.w_length)))
            elif num_iter>40:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1000,self.w_curve, self.w_length)))
            elif num_iter>35:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*500,self.w_curve, self.w_length)))
            elif num_iter>30:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*300,self.w_curve, self.w_length)))
            elif num_iter>25:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*150,self.w_curve, self.w_length)))
            elif num_iter>20:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*70,self.w_curve, self.w_length)))
            elif num_iter>15:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*20,self.w_curve, self.w_length)))
            elif num_iter>10:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10,self.w_curve, self.w_length)))
            elif num_iter>5:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*5,self.w_curve, self.w_length)))
            else:
                self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap,self.w_curve, self.w_length)))

            # '''WoRKS for 250 2.5+0.6'''
            if num_overlap == 0:
                self.optimized = True       
                break
            if num_iter > 95:
                self.optimized = False     
                break
            #==========================================================
            # if num_iter>150:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*600000,self.w_curve, self.w_length)))
            # elif num_iter>140:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*300000,self.w_curve, self.w_length)))
            # elif num_iter>130:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*100000,self.w_curve, self.w_length)))
            # elif num_iter>120:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*60000,self.w_curve, self.w_length)))
            # elif num_iter>110:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*30000,self.w_curve, self.w_length)))
            #     # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*2500,self.w_curve, self.w_length)))
            # elif num_iter>100:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10000,self.w_curve, self.w_length)))
            #     # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*2000,self.w_curve, self.w_length)))
            # elif num_iter>90:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*5000,self.w_curve, self.w_length)))
            #     # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1500,self.w_curve, self.w_length)))
            # elif num_iter>80:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1000,self.w_curve, self.w_length)))
            # elif num_iter>70:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*500,self.w_curve, self.w_length)))
            # elif num_iter>60:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*300,self.w_curve, self.w_length)))
            # elif num_iter>50:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*150,self.w_curve, self.w_length)))
            # elif num_iter>40:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*70,self.w_curve, self.w_length)))
            # elif num_iter>30:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*20,self.w_curve, self.w_length)))
            # elif num_iter>20:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10,self.w_curve, self.w_length)))
            # elif num_iter>10:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*5,self.w_curve, self.w_length)))
            # else:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap,self.w_curve, self.w_length)))
            
            # # '''WoRKS for 250 2.5+0.6'''
            # if num_overlap == 0:
            #     self.optimized = True       
            #     break
            # if num_iter > 165:
            #     self.optimized = False     
            #     break
            ##=============================================
            # if num_iter>370:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10100,self.w_curve, self.w_length)))
            # elif num_iter>320:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10000,self.w_curve, self.w_length)))
            #     # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*2500,self.w_curve, self.w_length)))
            # elif num_iter>280:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*5000,self.w_curve, self.w_length)))
            #     # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*2000,self.w_curve, self.w_length)))
            # elif num_iter>240:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*2000,self.w_curve, self.w_length)))
            #     # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1500,self.w_curve, self.w_length)))
            # elif num_iter>210:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1000,self.w_curve, self.w_length)))
            # elif num_iter>180:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*500,self.w_curve, self.w_length)))
            # elif num_iter>160:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*300,self.w_curve, self.w_length)))
            # elif num_iter>130:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*140,self.w_curve, self.w_length)))
            # elif num_iter>100:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*70,self.w_curve, self.w_length)))
            # elif num_iter>80:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*20,self.w_curve, self.w_length)))
            # elif num_iter>50:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10,self.w_curve, self.w_length)))
            # elif num_iter>20:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*5,self.w_curve, self.w_length)))
            # else:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap,self.w_curve, self.w_length)))
            
            # trying for beading
            # if num_iter>370:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*15000,self.w_curve, self.w_length*5)))
            # elif num_iter>320:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10000,self.w_curve, self.w_length*5)))
            #     # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*2500,self.w_curve, self.w_length)))
            # elif num_iter>270: ### key step
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*5000,self.w_curve, self.w_length*5)))
            #     # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*2000,self.w_curve, self.w_length)))
            # elif num_iter>240:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*3000,self.w_curve, self.w_length*5)))
            #     # self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1500,self.w_curve, self.w_length)))
            # elif num_iter>210:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1600,self.w_curve, self.w_length*5)))
            # elif num_iter>180:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*800,self.w_curve, self.w_length*5)))
            # elif num_iter>160: ### key step
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*400,self.w_curve, self.w_length*5)))
            # elif num_iter>130: 
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*200,self.w_curve, self.w_length*5)))
            # elif num_iter>100:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*100,self.w_curve, self.w_length*5)))
            # elif num_iter>80:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*50,self.w_curve, self.w_length*5)))
            # elif num_iter>50:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*20,self.w_curve, self.w_length*5)))
            # elif num_iter>20:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10,self.w_curve, self.w_length*2.5)))
            # else:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap,self.w_curve, self.w_length)))
            


            # if num_iter>60:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*3000,self.w_curve, self.w_length)))
            # elif num_iter>50:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*1000,self.w_curve, self.w_length)))
            # elif num_iter>40:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*500,self.w_curve, self.w_length)))
            # elif num_iter>30:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*100,self.w_curve, self.w_length)))
            # elif num_iter>20:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*20,self.w_curve, self.w_length)))
            # elif num_iter>10:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*10,self.w_curve, self.w_length)))
            # elif num_iter>5:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap*5,self.w_curve, self.w_length)))
            # else:
            #     self.optimizer.step(partial(closure, self.normalize_weights(self.w_overlap,self.w_curve, self.w_length)))
            
            # else:    
            #     self.optimizer.step(closure)    
    
        xyzr_fid = torch.hstack([positions_with_grad, t_ra_original.unsqueeze(1), f_id.unsqueeze(1)])
        xa_, ya_, za_, ra_, mask_se, fid_ = self.torch_optimizer_wrapPBC(xyzr_fid, mask_starts_ends, tol=config_params.BOX_LENGTH/5)
        pbc_spheres_xyz_r_fid = torch.hstack((xa_.unsqueeze(1),ya_.unsqueeze(1),za_.unsqueeze(1), ra_.unsqueeze(1), fid_.unsqueeze(1)))
        final_xyz             = torch.hstack((xyzr_fid[:,0].unsqueeze(1),xyzr_fid[:,1].unsqueeze(1),xyzr_fid[:,2].unsqueeze(1)))
        print('GOT HERE AFTER OPTIMIZATION')
        if num_overlap == 0:
            num_overlap, x_, y_, z_, r_, fd_ = \
                    self.check_num_overlaps(final_xyz, xyzr_fid[:,3], mask_starts_ends, xyzr_fid[:,-1])
            self.check_start_ends_intact(pbc_spheres_xyz_r_fid)            
            print('NUM OVERLAP FINAL', num_overlap)
            print('Done optimization')
            config_params.VOLUME_FRACTION = self.get_volume_fraction(fibers=pbc_spheres_xyz_r_fid)
        if num_overlap == 0: 
            self.optimized = True        
        return pbc_spheres_xyz_r_fid.detach().cpu()
    
    def normalize_weights(self, a, b, c):
        sum = a+b+c
        return a/sum, b/sum, c/sum

    # @profile
    def overlap_cost_function(self, positions, mask_starts_ends, torch_wrapped_ra, fid_):
        Lx, Ly, Lz, detect_buff = config_params.BOX_LENGTH, config_params.BOX_LENGTH, config_params.BOX_LENGTH, self.space_buffer_repulse
        collision_set = CD.detect_collision(positions, torch_wrapped_ra, fid_, Lx, Ly, Lz, detect_buff)
        overlap_cost = self.get_steps_with_optimizer(positions, mask_starts_ends, torch_wrapped_ra, collision_set)
        return overlap_cost
    
    def get_steps_with_optimizer(self, positions, mask_starts_ends, torch_wrapped_ra, collision_set_np):        
        radii_ = torch_wrapped_ra[:, None]
        left = collision_set_np[:, 0] # indices in positions
        right = collision_set_np[:, 1] # indices in positions
        position_deltas = positions[left] - positions[right]
        dm = torch.linalg.norm(position_deltas, dim=1, keepdims=True)
        radii_left = radii_[left]
        radii_right = radii_[right]
        rm = radii_left + radii_right
        space_buffer = self.space_buffer_repulse
        collision_depths = torch.clamp(rm + space_buffer - dm, min=0, max=None) # depth >= 0 
        # print('len(positions[0])', positions.shape)
        pos_length = torch.tensor(math.sqrt(len(positions[0])), device=self.device)
        overlap_cost =  torch.sum(torch.square((collision_depths/rm))*radii_left*radii_right)/ pos_length # / number of spheres? or # of overlaps?
        return overlap_cost
    
    
    # @profile
    def length_cost_function(self, positions, mask_starts_ends,torch_wrapped_ra):
        mask_indices = torch.nonzero(mask_starts_ends).flatten()
        start_indices = mask_indices[::2]
        end_indices = mask_indices[1::2]
        num_spaces_in_between_endpoints = end_indices - start_indices
        # ----------------- Calculate the distances between consecutive rows in the positions_matrix
        row_distances = torch.norm((positions[1:] - positions[:-1]), dim=1)
        start_points, end_points = positions[start_indices], positions[end_indices]
        start_end_distances = torch.norm(end_points - start_points, dim=1) / num_spaces_in_between_endpoints
        new_row_distances = row_distances.clone()
        # new_row_distances = torch.zeros_like(row_distances, device=self.device) # creat on deice
        count = 0
        for start_idx, end_idx in zip(start_indices, end_indices):
            new_row_distances[start_idx: end_idx] = row_distances[start_idx: end_idx] - start_end_distances[count]
            count=count+1
        '''create new tensor to scale distances'''
        scaled_row_distances = 2*torch.square(new_row_distances)/(torch_wrapped_ra[:-1] + torch_wrapped_ra[1:])
        # scaled_row_distances = torch.square(new_row_distances)\
            # *torch.square((torch_wrapped_ra[:-1] + torch_wrapped_ra[1:])/2.)
        space_mask = torch.logical_not(torch.logical_and(mask_starts_ends[:-1], mask_starts_ends[1:]))
        remaining_distances = scaled_row_distances[space_mask]
    
        return torch.sum(remaining_distances)
    
    def curvature_cost_function(self, positions, mask_starts_ends, wrapped_ra):
        '''
        chop off start/end points
        mask should be used for both cosine and radius
        '''
        fmask = self.filter_start_ends(mask_starts_ends)
        f2mask = torch.roll(fmask, 1)
        f2mask[0] = False
        f3mask = torch.roll(f2mask, 1)
        f3mask[0] = False

        a1, r1= positions[fmask][:-2],  wrapped_ra[fmask][:-2]
        a2, r2= positions[f2mask][:-1], wrapped_ra[f2mask][:-1]
        a3, r3= positions[f3mask],      wrapped_ra[f3mask]
        vec1, vec2 = a2-a1, a3-a2
        element_wise_mul = vec1 * vec2 # Element-wise multiplication
        dot_product = torch.sum(element_wise_mul, dim=1) # # Sum along axis 1 (to get the dot product of corresponding rows)
        cosine = dot_product/(torch.norm(vec1, dim=1)*torch.norm(vec2, dim=1))
        return torch.sum(torch.square(1-cosine)/(r1+r2+r3))
    
    # @profile
    def filter_start_ends(self, a):
        b = torch.ones_like(a)
        t0=torch.tensor([0], device=self.device)
        a = torch.cat((t0,a,t0))
        
        start_indices = (a[:-2] == 0) & (a[1:-1] == 1) & (a[2:] == 1)
        b[torch.roll(start_indices,-1)] = 0
        b[start_indices] = 0
        b[torch.roll(start_indices,1)] = 1
        return b 
    
    def torch_optimizer_wrapPBC(self, xyzr_fid, mask_starts_ends, tol=0):
        inbox_xa, inbox_ya, inbox_za, inbox_ra, fiber_id = xyzr_fid[:, 0], xyzr_fid[:, 1], xyzr_fid[:, 2], xyzr_fid[:,3], xyzr_fid[:,4]
        
        '''wrap along x-axis'''
        mx = inbox_xa - inbox_ra < -config_params.BOX_LENGTH/2 + tol
        torch_coord_indices = torch.argwhere(torch.isin(fiber_id , fiber_id[mx])).ravel().to(inbox_xa.device)
        mxL = config_params.BOX_LENGTH - inbox_xa - inbox_ra< config_params.BOX_LENGTH/2 + tol
        torch_coord_indices_L = torch.argwhere(torch.isin(fiber_id , fiber_id[mxL])).ravel().to(inbox_xa.device)

        inbox_xa = torch.cat((inbox_xa, torch.take(inbox_xa,torch_coord_indices)+ config_params.BOX_LENGTH))
        inbox_ya = torch.cat((inbox_ya, torch.take(inbox_ya,torch_coord_indices)))
        inbox_za = torch.cat((inbox_za, torch.take(inbox_za,torch_coord_indices)))
        inbox_ra = torch.cat((inbox_ra, torch.take(inbox_ra,torch_coord_indices)))
        mask_starts_ends = torch.cat((mask_starts_ends, torch.take(mask_starts_ends, torch_coord_indices)))
        fiber_id = torch.cat((fiber_id, torch.take(fiber_id,torch_coord_indices)))
        
        inbox_xa = torch.cat((inbox_xa, torch.take(inbox_xa,torch_coord_indices_L) - config_params.BOX_LENGTH))
        inbox_ya = torch.cat((inbox_ya, torch.take(inbox_ya,torch_coord_indices_L)))
        inbox_za = torch.cat((inbox_za, torch.take(inbox_za,torch_coord_indices_L)))
        inbox_ra = torch.cat((inbox_ra, torch.take(inbox_ra,torch_coord_indices_L)))
        mask_starts_ends = torch.cat((mask_starts_ends, torch.take(mask_starts_ends, torch_coord_indices_L)))
        fiber_id = torch.cat((fiber_id, torch.take(fiber_id,torch_coord_indices_L)))

        '''wrap along y-axis'''
        my = inbox_ya - inbox_ra < -config_params.BOX_LENGTH/2 + tol
        torch_coord_indices_y = torch.argwhere(torch.isin(fiber_id , fiber_id[my])).ravel()
        myL = config_params.BOX_LENGTH - inbox_ya - inbox_ra< config_params.BOX_LENGTH/2 + tol
        torch_coord_indices_y_L = torch.argwhere(torch.isin(fiber_id , fiber_id[myL])).ravel()
        
        inbox_xa = torch.cat((inbox_xa, torch.take(inbox_xa,torch_coord_indices_y)))
        inbox_ya = torch.cat((inbox_ya, torch.take(inbox_ya,torch_coord_indices_y)+ config_params.BOX_LENGTH))
        inbox_za = torch.cat((inbox_za, torch.take(inbox_za,torch_coord_indices_y)))
        inbox_ra = torch.cat((inbox_ra, torch.take(inbox_ra,torch_coord_indices_y)))
        mask_starts_ends = torch.cat((mask_starts_ends, torch.take(mask_starts_ends, torch_coord_indices_y)))
        fiber_id = torch.cat((fiber_id, torch.take(fiber_id,torch_coord_indices_y)))
        
        inbox_xa = torch.cat((inbox_xa, torch.take(inbox_xa,torch_coord_indices_y_L)))
        inbox_ya = torch.cat((inbox_ya, torch.take(inbox_ya,torch_coord_indices_y_L)- config_params.BOX_LENGTH))
        inbox_za = torch.cat((inbox_za, torch.take(inbox_za,torch_coord_indices_y_L)))
        inbox_ra = torch.cat((inbox_ra, torch.take(inbox_ra,torch_coord_indices_y_L)))
        mask_starts_ends = torch.cat((mask_starts_ends, torch.take(mask_starts_ends, torch_coord_indices_y_L)))
        fiber_id = torch.cat((fiber_id, torch.take(fiber_id,torch_coord_indices_y_L)))
        
        return inbox_xa, inbox_ya, inbox_za, inbox_ra, \
            mask_starts_ends, fiber_id 
    
    def check_start_ends_intact(self, pbc_spheres_xyz_r_fid):
        fiber_list_xyz_r_fid = util.map_matrix_to_list_torch(pbc_spheres_xyz_r_fid)
        intactFlag = True
        for fiber in fiber_list_xyz_r_fid:
            intactFlag = intactFlag and (torch.equal(fiber[0][2],-config_params.BOX_LENGTH.to(fiber.device)/2)) 
            intactFlag = intactFlag and (torch.equal(fiber[-1][2], config_params.BOX_LENGTH.to(fiber.device)/2))
        if intactFlag:    
            print('INTACT')
        else:
            print('NOT INTACT')
            raise
        return intactFlag
    