import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os.path
import numpy as np
from scipy.stats import genextreme
from scipy.optimize import minimize
import simulation_toolkit.substrate_generator.helper.CollisionDetection2D as CD
import simulation_toolkit.toolkit_params as config_params

class Init2D(object):
    """
    A class for 2D initliaization of 2 ends of cube
    Optimization for IVF by remove overlaps between disks
    init: radii + buff/2
    check_overlap: buff/5
    cost_fucntion: buff
    """
    def __init__(self, device, date_time, orientation_shape_parameter=200, target_volume_fraction=0.6, 
                 num_fibers=500, dist_shape=0, mean_diameter=1,sigma_radii=0.5,space_buffer=0.0, box_length_init=0, output_folder=None):
        self.device=device
        self.output_folder=output_folder
        self.volume_fraction = 0
        self.space_buffer = space_buffer
        self.date_time=date_time
        self.num_fibers = num_fibers
        self.dist_shape = dist_shape
        self.mean_diameter = mean_diameter
        self.sigma_radii = sigma_radii
        self.target_volume_fraction = target_volume_fraction    
        self.orientation_shape_parameter = orientation_shape_parameter
        self.box_length_init = torch.tensor(box_length_init)
        self.radii, self.fiber_id, self.box_length, self.initial_positions, \
        self.mean_d_underlying, self.sigma_d_underlying\
        = self.initialize_circle_radius_GEV_and_positions_halfLx()

        self.initial_positions = self.create_opt_2D_packing_with_auto_restart()
        self.start_points, self.end_points = self.get_original_starts_ends_with_no_wrapping()
        
    def save_optimized_positions_to_file():
        return
    
    def objective(self, params, expected_mean, expected_stdv):
        c, loc, scale = params
        if scale <= 0:
            return 1e10
        try:
            mean_actual = genextreme.mean(c, loc=loc, scale=scale)
            std_actual = genextreme.std(c, loc=loc, scale=scale)
            return (mean_actual - expected_mean)**2 + (std_actual - expected_stdv)**2
        except:
            return 1e10
    

    def initialize_circle_radius_GEV_and_positions_halfLx(self):
            # loc, scale = self.mean_diameter/2, self.sigma_radii/2
            # mean_r_underlying, sigma_r_underlying, skew, kurt = genextreme.stats(c=self.dist_shape, loc=loc, scale=scale, moments='mvsk')
            # mean_underlying_normal_distribution = np.log(u**2/np.sqrt(u**2+v**2)) #  mean of underlying normal distribution
            # sigma_underlying_normal_distribution = np.sqrt(np.log(v**2/u**2 + 1)) #  standard deviation of underlying normal distribution
            # lognormal_params = torch.tensor([mean_underlying_normal_distribution,sigma_underlying_normal_distribution])
                    
            # fix num axons & AVF --> LxLy
            vf = self.target_volume_fraction  # Expected volume fraction
            # log_mean, log_std = lognormal_params[0], lognormal_params[1]  # Parameters of the lognormal distribution

            # diameter_dist = torch.distributions.log_normal.LogNormal(loc=log_mean, scale=log_std)
            # radii_0 = diameter_dist.sample((int(self.num_fibers),))/2.
            gev_fitted_diameter = minimize(self.objective, [self.dist_shape, self.mean_diameter, self.sigma_radii], args=(self.mean_diameter, self.sigma_radii), method='Nelder-Mead')
            c_opt, loc_opt, scale_opt = gev_fitted_diameter.x
            diameter_np = genextreme.rvs(c=c_opt, loc=loc_opt, scale=scale_opt, size=self.num_fibers)
            diameter_np = np.clip(diameter_np, a_min=0.15, a_max=None)
            # print(f"GEV parameters: c={c_opt}, loc={loc_opt}, scale={scale_opt}")
            print(f"Generated diameters - Min: {diameter_np.min()}, Max: {diameter_np.max()}")
            
            radii_0 = torch.from_numpy(diameter_np/2)
            fid_0 = torch.arange(radii_0.shape[0])
            radii = torch.stack([radii_0, radii_0], dim=1).flatten() # INTERLEAVE radi0 values to form pairs
            fiber_id = torch.stack([fid_0, fid_0], dim=1).flatten()

            # self.plot_histogram_GEV(radii*2)
            # space_buffer = torch.tensor(self.space_buffer, device=self.device)
            if self.box_length_init==0:
                total_area = torch.sum(torch.pi * (radii+self.space_buffer/2)**2)
                box_length = torch.round(torch.sqrt(total_area / vf))
            else:
                box_length=self.box_length_init
            print('box_length', box_length.item())

            init_range = box_length  #'''DO I NEED THIS?'''
            '''init startpoints from radius'''
            start_points = torch.rand(radii_0.shape[0], 2)*init_range - init_range/2 # start points only, close to center, and shift FOV to halfLx
            watson_dist = torch.distributions.von_mises.VonMises(0, self.orientation_shape_parameter)
            target_angles = watson_dist.sample((int(start_points.shape[0]),)) # draw target point from dist
            d_start_target = box_length*torch.tan(target_angles)
            azithmuth = torch.distributions.von_mises.VonMises(0, 0.1).sample((int(start_points.shape[0]),))
            
            target_points = torch.stack([d_start_target*torch.cos(azithmuth)+start_points[:,0], 
                                        d_start_target*torch.sin(azithmuth)+start_points[:,1]]).T

            circle_centers_x = torch.stack([start_points[:,0], target_points[:,0]], dim=1).flatten() # INTERLEAVE start-targets values to form pairs
            circle_centers_y = torch.stack([start_points[:,1], target_points[:,1]], dim=1).flatten() # INTERLEAVE start-targets values to form pairs
            circle_centers = torch.stack([circle_centers_x, circle_centers_y]).T
            circle_centers = torch.cat((circle_centers, radii.unsqueeze(1)), dim=1)
            circle_centers = torch.cat((circle_centers, fiber_id.unsqueeze(1)), dim=1)

            mean_d_underlying, sigma_d_underlying = diameter_np.mean()*2, diameter_np.std()*2
            config_params.BOX_LENGTH = box_length
            return radii, fiber_id, box_length, circle_centers,\
                mean_d_underlying, sigma_d_underlying

    def overlap_cost_function(self, c_pos, c_ra, f_id, box_length):
        Lx, Ly, buff = box_length, box_length, self.space_buffer
        collision_set = CD.detect_collision(c_pos, c_ra, f_id, Lx, Ly, buff, self.device)
        overlap_cost = self.get_steps_with_optimizer(c_pos, c_ra, collision_set, buff)
        return overlap_cost
    
    def get_steps_with_optimizer(self, positions, torch_wrapped_ra, collision_set_np, space_buffer):
        radii_ = torch_wrapped_ra[:, None]
        left = collision_set_np[:, 0] # indices in positions
        right = collision_set_np[:, 1] # indices in positions
        position_deltas = positions[left] - positions[right]
        dm = torch.linalg.norm(position_deltas, dim=1, keepdims=True)
        radii_left = radii_[left]
        radii_right = radii_[right]
        rm = radii_left + radii_right
        space_buffer = torch.tensor([space_buffer]).to(self.device)
        collision_depths = torch.clamp(rm + space_buffer - dm, min=0, max=None) # depth >= 0
        overlap_cost =  torch.sum(torch.square(collision_depths)) #/ pos_length # / number of spheres? or # of overlaps?
        return overlap_cost
    
    def torch_optimizer_wrapPBC_disk(self, pos_grad, L, tol=0):
        '''pos_grad has x,y,r,fid'''
        inbox_xa, inbox_ya, inbox_ra, fiber_id = pos_grad[:, 0], pos_grad[:, 1], pos_grad[:, 2], pos_grad[:,3]
        box_length_x, box_length_y = L, L
        '''wrap along x-axis'''
        mx = inbox_xa - inbox_ra < -box_length_x/2 + tol
        torch_coord_indices = torch.argwhere(torch.isin(fiber_id , fiber_id[mx])).ravel().to(inbox_xa.device)
        mxL = box_length_x - inbox_xa - inbox_ra< box_length_x/2 + tol
        torch_coord_indices_L = torch.argwhere(torch.isin(fiber_id , fiber_id[mxL])).ravel().to(inbox_xa.device)

        inbox_xa = torch.cat((inbox_xa, torch.take(inbox_xa,torch_coord_indices)+ box_length_x))
        inbox_ya = torch.cat((inbox_ya, torch.take(inbox_ya,torch_coord_indices)))
        inbox_ra = torch.cat((inbox_ra, torch.take(inbox_ra,torch_coord_indices)))
        fiber_id = torch.cat((fiber_id, torch.take(fiber_id,torch_coord_indices)))

        inbox_xa = torch.cat((inbox_xa, torch.take(inbox_xa,torch_coord_indices_L) - box_length_x))
        inbox_ya = torch.cat((inbox_ya, torch.take(inbox_ya,torch_coord_indices_L)))
        inbox_ra = torch.cat((inbox_ra, torch.take(inbox_ra,torch_coord_indices_L)))
        fiber_id = torch.cat((fiber_id, torch.take(fiber_id,torch_coord_indices_L)))

        '''wrap along y-axis'''
        my = inbox_ya - inbox_ra < -box_length_y/2 + tol
        torch_coord_indices_y = torch.argwhere(torch.isin(fiber_id , fiber_id[my])).ravel()
        myL = box_length_y - inbox_ya - inbox_ra< box_length_y/2 + tol
        torch_coord_indices_y_L = torch.argwhere(torch.isin(fiber_id , fiber_id[myL])).ravel()

        inbox_xa = torch.cat((inbox_xa, torch.take(inbox_xa,torch_coord_indices_y)))
        inbox_ya = torch.cat((inbox_ya, torch.take(inbox_ya,torch_coord_indices_y)+ box_length_y))
        inbox_ra = torch.cat((inbox_ra, torch.take(inbox_ra,torch_coord_indices_y)))
        fiber_id = torch.cat((fiber_id, torch.take(fiber_id,torch_coord_indices_y)))

        inbox_xa = torch.cat((inbox_xa, torch.take(inbox_xa,torch_coord_indices_y_L)))
        inbox_ya = torch.cat((inbox_ya, torch.take(inbox_ya,torch_coord_indices_y_L)- box_length_y))
        inbox_ra = torch.cat((inbox_ra, torch.take(inbox_ra,torch_coord_indices_y_L)))
        fiber_id = torch.cat((fiber_id, torch.take(fiber_id,torch_coord_indices_y_L)))
        return inbox_xa, inbox_ya, inbox_ra, fiber_id

    def check_num_overlaps(self, positions_with_grad, box_length):
        dtached_pos = positions_with_grad.detach()
        xa_, ya_, ra_, fid_ = self.torch_optimizer_wrapPBC_disk(dtached_pos, box_length, tol=0)
        pos_ = torch.stack([xa_, ya_]).T
        Lx, Ly, buff, device = box_length, box_length, self.space_buffer/5, self.device
        collision_set = CD.detect_collision(pos_, ra_, fid_, Lx, Ly, buff, device)
        num_overlap = collision_set.shape[0]
        sphere_id = torch.arange(pos_.shape[0])
        return num_overlap, xa_, ya_, ra_, fid_, sphere_id
    
    def create_opt_2D_packing_with_auto_restart(self):
            # K_watson=self.orientation_shape_parameter # for Stein variational 
            # watson_K10 = torch.distributions.von_mises.VonMises(0, K_watson)
            # K_rbf = RBF()

            initial_positions = self.initial_positions.to(self.device).contiguous()
            initp_0 = initial_positions.detach().clone()
            initial_positions.requires_grad = True
            self.optimizer = optim.LBFGS([initial_positions], lr=0.1, line_search_fn='strong_wolfe')
            num_overlap, num_iter = 1000, 0

            def closure():
                self.optimizer.zero_grad()
                c_xa, c_ya, c_ra, f_id = self.torch_optimizer_wrapPBC_disk(initial_positions, self.box_length, tol=0)
                c_pos = torch.stack([c_xa, c_ya]).T.contiguous()
                c_ra=c_ra.detach()
                f_id=f_id.detach()
                overlap_loss = self.overlap_cost_function(c_pos, c_ra, f_id, self.box_length)
                overlap_loss.backward()
                return overlap_loss

            while num_overlap>0:
                num_overlap, xa_, ya_, ra_, fid_, sphere_id = self.check_num_overlaps(initial_positions, self.box_length)
                print('num_iter', num_iter, 'overlap ', num_overlap)
                if num_overlap == 0:
                    xa_, ya_, ra_, fid_ = self.torch_optimizer_wrapPBC_disk(initial_positions, self.box_length, tol=0)
                    print('Final num_overlap:', num_overlap)
                    break
                if num_overlap > 0 and num_iter == 10:
                    print(f"Overlaps detected at iteration {num_iter}. Restarting initialization...")
                    self.radii, self.fiber_id, self.box_length, self.initial_positions, \
                        self.mean_d_underlying, self.sigma_d_underlying = self.initialize_circle_radius_GEV_and_positions_halfLx()
                    initial_positions = self.initial_positions.to(self.device).contiguous()
                    initp_0 = initial_positions.detach().clone()
                    initial_positions.requires_grad = True
                    self.optimizer = optim.LBFGS([initial_positions], lr=0.1, line_search_fn='strong_wolfe')
                    num_overlap, num_iter = 1000, 0
                    continue
                c_pos = torch.stack([xa_, ya_]).T.contiguous()
                ### TEMPORARILY turned off
                # if num_iter==0:
                    # print('recorded new init_p at 0th iter')
                    # initp_0 = c_pos.detach().clone()
                    # self.plot_circles(c_pos.cpu().detach().numpy(), ra_.cpu().detach().numpy(), fid_, self.box_length, sphere_id, num_iter, converged=False)
        
                # if num_iter%20==0:
                #     print('------------- num iteration', num_iter, '---------------')
                #     print('number of sphere overlaps: ', str(int(num_overlap)))
                #     self.get_2D_volume_fraction(c_pos.cpu().detach().numpy(), ra_.cpu().detach().numpy(), self.box_length)
                #     self.plot_circles(c_pos.cpu().detach().numpy(), ra_.cpu().detach().numpy(), fid_, self.box_length, sphere_id, num_iter)

                self.optimizer.step(closure)
                num_iter+=1
            opt_pos = torch.stack([xa_, ya_]).T.contiguous()
            ### TEMPORARILY turned off
            self.get_2D_volume_fraction(opt_pos.cpu().detach().numpy(), ra_.cpu().detach().numpy(), self.box_length)
            self.plot_circles(opt_pos.cpu().detach().numpy(), ra_.cpu().detach().numpy(), fid_, self.box_length, sphere_id, num_iter, converged=True)
            # angles_0, angles_opt = self.calc_angles(initp_0, self.box_length), self.calc_angles(opt_pos, self.box_length)
            # self.plot_angle_distribution(angles_0, angles_opt)
            # self.plot_angle_distribution_normalized(angles_0, angles_opt)
            print("-----------------Done packing 2D-----------------")
            return initial_positions.detach() 
    
    '''deprecating'''
    def get_original_starts_ends_with_no_wrapping_arXiv(self):
        
        xy_0 = self.initial_positions[::2] # paired start points
        xy_L = self.initial_positions[1::2] # paired end points
        L = self.box_length.to(self.device)
        z_0, z_L = torch.full((xy_0.shape[0], 1), -L/2, device=self.device), torch.full((xy_L.shape[0], 1), L/2, device=self.device)
        
        circle_centers_0 = torch.cat([xy_0[:, :2], z_0, xy_0[:, 2:]], dim=1)
        circle_centers_L = torch.cat([xy_L[:, :2], z_L, xy_L[:, 2:]], dim=1)
        circle_centers_0_L = torch.cat([xy_L[:, :2], z_0, xy_L[:, 2:]], dim=1)
        circle_centers_L_0 = torch.cat([xy_0[:, :2], z_L, xy_0[:, 2:]], dim=1)
        
        circle_centers_0 = torch.cat((circle_centers_0, circle_centers_0_L), dim=0)
        circle_centers_L = torch.cat((circle_centers_L, circle_centers_L_0), dim=0)
        
        # print('circle_centers_0 \n', circle_centers_0)
        # print('circle_centers_L \n', circle_centers_L)
        # exit()
        return circle_centers_0, circle_centers_L
    
    def get_original_starts_ends_with_no_wrapping(self):
        
        xy_0 = self.initial_positions[::2] # paired start points
        xy_L = self.initial_positions[1::2] # paired end points
        L = self.box_length.to(self.device)
        z_0, z_L = torch.full((xy_0.shape[0], 1), -L/2, device=self.device), torch.full((xy_L.shape[0], 1), L/2, device=self.device)
        
        circle_centers_0 = torch.cat([xy_0[:, :2], z_0, xy_0[:, 2:]], dim=1)
        circle_centers_L = torch.cat([xy_L[:, :2], z_L, xy_L[:, 2:]], dim=1)
        circle_centers_0_L = torch.cat([xy_L[:, :2], z_0, xy_L[:, 2:]], dim=1)
        circle_centers_L_0 = torch.cat([xy_0[:, :2], z_L, xy_0[:, 2:]], dim=1)
        
        start_points = torch.stack([circle_centers_0, circle_centers_0_L], dim=1).flatten(0, 1)
        end_points = torch.stack([circle_centers_L, circle_centers_L_0], dim=1).flatten(0, 1)
        '''
        Here each fiber will have exactly 2 consecutive start points and 2 consecutive end points,
        start_points  = [fiber 1 startpoint 1, fiber 1 startpoint 2, fiber 2 startpoint 1, fiber 2 startpoint 2,....
        end_points    = [fiber 1 endpoint 1  , fiber 1 endpoint  2 , fiber 2 endpoint  1 , fiber 2 endpoint  2 ,....]
        '''
        return start_points, end_points
    
    
    def get_2D_volume_fraction(self, pos, ra, box_length):
        number_of_avf_nodes=int(5e4)
        nodes_x = np.random.uniform(-box_length/2, box_length/2, size=number_of_avf_nodes)
        nodes_y = np.random.uniform(-box_length/2, box_length/2, size=number_of_avf_nodes)
        dx = nodes_x - np.array(pos[:,0]).reshape(-1,1)
        dy = nodes_y - np.array(pos[:,1]).reshape(-1,1)
        dm = np.sqrt(np.power(dx,2) + np.power(dy,2))
        rm = np.broadcast_to(np.expand_dims(np.array(ra), 1),dm.shape)
        in_sphere_mask = (dm - rm)<0
        node_mask = np.sum(in_sphere_mask, axis=0)
        num_nodes_in_spheres = np.count_nonzero(node_mask)
        avf = num_nodes_in_spheres / number_of_avf_nodes
        self.volume_fraction = avf
        print('--- 2D volume fraction ---: ', avf)

    def plot_circles(self, centers, radii, ids, box_length, sphere_id, num_iter, converged=False):
        centers = np.squeeze(centers)
        fig, ax = plt.subplots()
        plt.title('2D packing of starts and ends',fontsize=17, pad=20)
        ax.tick_params(axis='both', which='major', labelsize=13)
        # Create a list of unique ids
        unique_ids = list(set(ids))
        # Define a color map for the lines
        cmap = plt.get_cmap('tab20')
        colors = {'1.0': 'blue', '1.68': 'red', '2.58': 'green', '3.5': 'purple', '4.5': 'brown'}
        color_i = colors[str(config_params.MEAN_DIAMETER)]
        for i in range(len(centers)):
            circle1 = plt.Circle((centers[i, 0], centers[i, 1]), radii[i], color=color_i, fill=False)
            ax.add_patch(circle1)
            
            # if i % 2 == 0 and i + 1 < len(centers) and ids[i] == ids[i+1]:
            #     color = cmap(unique_ids.index(ids[i]) / len(unique_ids))
            #     ax.plot([centers[i, 0], centers[i+1, 0]], [centers[i, 1], centers[i+1, 1]], color=color, linewidth=0.5, alpha=0.3)
        L_h = box_length/2
        x0 = [-L_h, L_h, L_h, -L_h, -L_h]
        y0 = [-L_h, -L_h, L_h, L_h, -L_h]
        ax.set_xlabel('x (µm)', fontsize=15, labelpad=3)
        ax.set_ylabel('y (µm)',fontsize=15, labelpad=3)
        # Create the plot
        ax.plot(x0, y0, color='black')
        ax.set_aspect('equal', adjustable='box')
        folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/init2D"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if converged==True:
            plt.savefig(folder_path+"/2D_converged_packing_PBC_start_end_buff"+str(self.space_buffer)+"_"+self.date_time+"_avf"+str(self.volume_fraction)+".png", dpi=500)
            plt.close(fig)
        else:    
            plt.savefig(folder_path+"/2D_packing_PBC_start_end_buff"+str(self.space_buffer)+"_"+self.date_time+"_"+str(num_iter)+".png", dpi=500)
            plt.close(fig)


    # --------------- Log Normal ---------------
    def initialize_circle_radius_and_positions_halfLx(self):
        u, v = self.mean_diameter, self.sigma_radii
        mean_underlying_normal_distribution = np.log(u**2/np.sqrt(u**2+v**2)) #  mean of underlying normal distribution
        sigma_underlying_normal_distribution = np.sqrt(np.log(v**2/u**2 + 1)) #  standard deviation of underlying normal distribution
        lognormal_params = torch.tensor([mean_underlying_normal_distribution,sigma_underlying_normal_distribution])
                
        # fix num axons & AVF --> LxLy
        vf = self.target_volume_fraction  # Expected volume fraction
        log_mean, log_std = lognormal_params[0], lognormal_params[1]  # Parameters of the lognormal distribution

        diameter_dist = torch.distributions.log_normal.LogNormal(loc=log_mean, scale=log_std)
        radii_0 = diameter_dist.sample((int(self.num_fibers),))/2.
        fid_0 = torch.arange(radii_0.shape[0])
        radii = torch.stack([radii_0, radii_0], dim=1).flatten() # INTERLEAVE radi0 values to form pairs
        fiber_id = torch.stack([fid_0, fid_0], dim=1).flatten()

        self.plot_histogram(radii*2, log_mean, log_std)
        space_buffer = torch.tensor(self.space_buffer, device=self.device)
        total_area = torch.sum(torch.pi * (radii+self.space_buffer/2)**2)
        box_length = torch.round(torch.sqrt(total_area / vf))
        print('box_length', box_length)

        init_range = box_length  #'''DO I NEED THIS?'''
        '''init startpoints from radius'''
        start_points = torch.rand(radii_0.shape[0], 2)*init_range - init_range/2 # start points only, close to center, and shift FOV to halfLx
        watson_dist = torch.distributions.von_mises.VonMises(0, self.orientation_shape_parameter)
        target_angles = watson_dist.sample((int(start_points.shape[0]),)) # draw target point from dist
        d_start_target = box_length*torch.tan(target_angles)
        azithmuth = torch.distributions.von_mises.VonMises(0, 0.1).sample((int(start_points.shape[0]),))
        
        target_points = torch.stack([d_start_target*torch.cos(azithmuth)+start_points[:,0], 
                                    d_start_target*torch.sin(azithmuth)+start_points[:,1]]).T

        circle_centers_x = torch.stack([start_points[:,0], target_points[:,0]], dim=1).flatten() # INTERLEAVE start-targets values to form pairs
        circle_centers_y = torch.stack([start_points[:,1], target_points[:,1]], dim=1).flatten() # INTERLEAVE start-targets values to form pairs
        circle_centers = torch.stack([circle_centers_x, circle_centers_y]).T
        circle_centers = torch.cat((circle_centers, radii.unsqueeze(1)), dim=1)
        circle_centers = torch.cat((circle_centers, fiber_id.unsqueeze(1)), dim=1)

        mean_d_underlying, sigma_d_underlying = mean_underlying_normal_distribution, sigma_underlying_normal_distribution
        return radii, fiber_id, box_length, circle_centers,\
            mean_d_underlying, sigma_d_underlying
    
    def plot_angle_distribution_normalized(self, angles_0, angles_opt):
        X10 = angles_0.cpu().detach()
        X200 = angles_opt.cpu().detach()

        # Prep data histogram - squeeze between -pi/2 to pi/2
        data1 = torch.atan(torch.tan(X10[:, 0])).cpu().numpy()
        data2 = torch.atan(torch.tan(X200[:, 0])).cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the first histogram
        ax1.hist(data1, bins=50, density=True, label='density')
        ax1.set_title('Histogram 1',fontsize=17, pad=20)
        ax1.tick_params(axis='both', which='major', labelsize=13)
        ax1.set_xlabel('Pre-opt', fontsize=15, labelpad=5)
        ax1.set_ylabel('Density', fontsize=15, labelpad=5)
        # ax1.set_xlim(-np.pi/4, np.pi/4)

        # Plot the second histogram
        ax2.hist(data2, bins=50, density=True, label='density')
        ax2.tick_params(axis='both', which='major', labelsize=13)
        ax2.set_title('Histogram 2', fontsize=17, pad=20)
        ax2.set_xlabel('Optimized', fontsize=15, labelpad=5)
        ax2.set_ylabel('Density',fontsize=15, labelpad=5)
        # ax2.set_xlim(-np.pi/2, np.pi/2)

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.5)
        folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/init2D"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(folder_path+"/Angle_Histogram_2D_PBC_start_end_"+str(self.num_fibers)+"_"+self.date_time+"_"+".png", dpi=500)
        plt.close(fig)

    def calc_angles(self, init_p, box_length):
        distance_pair =  torch.linalg.norm((init_p[::2] - init_p[1::2]), dim=1)
        angles = torch.atan(distance_pair/box_length).unsqueeze(1)
        return angles
    
    ''' plot after initializing start and start copies'''
    def plot_PBC(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        #-------------------- plot fibers ----------------------
        ''' for start node - target node
        connect 
        plot spheres
        '''
        for start_sphere, target_sphere in zip(self.start_points, self.end_points):
            # print('start_sphere, target_sphere', start_sphere, target_sphere)
            start_node=start_sphere.cpu().numpy()
            target_node=target_sphere.cpu().numpy()
            fiber_x = [start_node[0], target_node[0]]
            fiber_y = [start_node[1], target_node[1]]
            fiber_z = [start_node[2], target_node[2]]
            fiber_matrix=np.vstack([start_node, target_node])
            ax.plot3D(fiber_x, fiber_y, fiber_z, color='r', lw='0.5')
            self.plot_spheres(ax=ax, fiber_matrix=fiber_matrix, color='g')
        # ------------------- plot box edges ----------------------
        # NodeNetwork.plot_box(ax)            
        ax.set_xlim(-self.box_length/2, self.box_length/2)
        ax.set_ylim(-self.box_length/2, self.box_length/2)
        ax.set_zlim(-self.box_length/2, self.box_length/2)
        ax.set_xlabel("x (µm)",fontsize=15, labelpad=10)
        ax.set_ylabel("y (µm)",fontsize=15, labelpad=10)
        ax.set_zlabel("z (µm)",fontsize=15, labelpad=10)
        plt.title("3D plot of start/end spheres", fontsize=17, pad=20) 
        ax.tick_params(axis='both', which='major', labelsize=13)
        folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/init2D"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(folder_path+"/PBC_start_end_"+str(self.num_fibers)+"_"+self.date_time+".png", dpi=500)
    
    def plot_spheres(self, ax, fiber_matrix, color):
        color_sphere=0
        u = np.linspace(0, 2 * np.pi, 8) # used to be  np.linspace(0, 2 * np.pi, 8)
        v = np.linspace(0, np.pi, 8)
        for sphere_idx, node in enumerate(fiber_matrix):
            sphere_x = node[3] * np.outer(np.cos(u), np.sin(v)) + node[0]
            sphere_y = node[3] * np.outer(np.sin(u), np.sin(v)) + node[1]
            sphere_z = node[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + node[2]
            color_sphere = color
            ax.plot_surface(sphere_x, sphere_y, sphere_z, color=color_sphere, alpha=.2)
