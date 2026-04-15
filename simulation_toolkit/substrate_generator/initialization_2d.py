import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os.path
import numpy as np
from scipy.stats import genextreme
from scipy.optimize import minimize
import simulation_toolkit.substrate_generator.helper.watson_distribution as wd
import simulation_toolkit.substrate_generator.helper.CollisionDetection2D as CD
import simulation_toolkit.toolkit_params as config_params

class Init2D(object):
    """
    A class for 2D initialization of start & end points 
    on top/bottom faces of simulation cube.
    
    Optimization for volume fraction 
    by removing overlaps between disks in 2D.
    """
    def __init__(self, device, date_time, orientation_shape_parameter=200, target_volume_fraction=0.6, 
                 num_fibers=500, dist_shape=0, mean_diameter=1,sigma_radii=0.5,space_buffer=0.0, box_length_init=0):
        self.device=device
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
        ''' Calculate box length from target volume fraction,
        number of fibers, and diameter distribution parameters'''
        vf = self.target_volume_fraction  # Expected volume fraction
        gev_fitted_diameter = minimize(self.objective, [self.dist_shape, self.mean_diameter, self.sigma_radii], args=(self.mean_diameter, self.sigma_radii), method='Nelder-Mead')
        c_opt, loc_opt, scale_opt = gev_fitted_diameter.x
        diameter_np = genextreme.rvs(c=c_opt, loc=loc_opt, scale=scale_opt, size=self.num_fibers)
        diameter_np = np.clip(diameter_np, a_min=0.1 * self.mean_diameter, a_max=None)
        
        radii_0 = torch.from_numpy(diameter_np/2)
        fid_0 = torch.arange(radii_0.shape[0])
        radii = torch.stack([radii_0, radii_0], dim=1).flatten() # INTERLEAVE radi0 values to form pairs
        fiber_id = torch.stack([fid_0, fid_0], dim=1).flatten()

        if self.box_length_init==0:
            total_area = torch.sum(torch.pi * (radii+self.space_buffer/2)**2)
            box_length = torch.round(torch.sqrt(total_area / vf))
        else:
            box_length=self.box_length_init
        print('Box_length', box_length.item())

        '''init startpoints from radius'''
        start_points = torch.rand(radii_0.shape[0], 2)*box_length - box_length/2 # start points only, close to center, and shift FOV to halfLx
        
        # ---- Get end points from Watson Distribution ------
        # Create a Watson distribution instance
        mu = np.array([0, 0, 1])  # Mean direction (z-axis)
        watson_distribution = wd.WatsonDistribution(mu, self.orientation_shape_parameter)
        # Generate samples
        direction_vectors  = torch.from_numpy(watson_distribution.sample(int(start_points.shape[0])))
        # Visualize results
        watson_distribution.visualize_watson_samples(direction_vectors, mu, self.orientation_shape_parameter)
        
        dir_x, dir_y, dir_z = direction_vectors[:,0], direction_vectors[:,1], direction_vectors[:,2]
        d_start_target = box_length*torch.sqrt(dir_x**2+dir_y**2)/dir_z
        
        target_points = torch.stack([d_start_target*dir_x/torch.sqrt(dir_x**2+dir_y**2)+start_points[:,0], 
                                    d_start_target*dir_y/torch.sqrt(dir_x**2+dir_y**2)+start_points[:,1]]).T

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
        print('------ Start packing 2D initialization ------')
        while num_overlap>0:
            num_overlap, xa_, ya_, ra_, fid_, sphere_id = self.check_num_overlaps(initial_positions, self.box_length)
            print('Iteration ', num_iter, ' . Overlaps ', num_overlap)
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
            
            self.optimizer.step(closure)
            num_iter+=1
        opt_pos = torch.stack([xa_, ya_]).T.contiguous()
        self.get_2D_volume_fraction(opt_pos.cpu().detach().numpy(), ra_.cpu().detach().numpy(), self.box_length)
        self.plot_circles(opt_pos.cpu().detach().numpy(), ra_.cpu().detach().numpy(), fid_, self.box_length, sphere_id, num_iter, converged=True)
        print("------ Done packing 2D initialization ------")
        return initial_positions.detach() 

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
        # print('--- 2D volume fraction ---: ', avf)

    def plot_circles(self, centers, radii, ids, box_length, sphere_id, num_iter, converged=False):
        centers = np.squeeze(centers)
        fig, ax = plt.subplots()
        plt.title('2D packing of starts and ends',fontsize=17, pad=20)
        ax.tick_params(axis='both', which='major', labelsize=13)
        
        # Dynamic color assignment based on MEAN_DIAMETER
        # Normalize MEAN_DIAMETER from range [0.1, 8] to [0, 1]
        diameter_range = (0.1, 8.0)
        normalized_diameter = (config_params.MEAN_DIAMETER - diameter_range[0]) / (diameter_range[1] - diameter_range[0])
        normalized_diameter = np.clip(normalized_diameter, 0, 1)  # Ensure it's within [0,1]
        
        # Use a colormap that works well on white background
        cmap = plt.get_cmap('tab10')  # Discrete colors, good contrast
        # Alternative: use 'Set1', 'Dark2', or 'viridis' for different color schemes
        color_i = cmap(normalized_diameter)
        
        for i in range(len(centers)):
            circle1 = plt.Circle((centers[i, 0], centers[i, 1]), radii[i], color=color_i, fill=False)
            ax.add_patch(circle1)
            
        L_h = box_length/2
        x0 = [-L_h, L_h, L_h, -L_h, -L_h]
        y0 = [-L_h, -L_h, L_h, L_h, -L_h]
        ax.set_xlabel('x (µm)', fontsize=15, labelpad=3)
        ax.set_ylabel('y (µm)',fontsize=15, labelpad=3)
        ax.plot(x0, y0, color='black')
        ax.set_aspect('equal', adjustable='box')
        folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/init2D"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if converged==True:
            plt.savefig(folder_path+"/2D_converged_initialization"+"_"+self.date_time+".png", dpi=500)
            plt.close(fig)
        else:    
            plt.savefig(folder_path+"/2D_initialization"+"_"+self.date_time+"_"+str(num_iter)+".png", dpi=500)
            plt.close(fig)

    def plot_PBC(self):
        ''' plot after initializing start and end points'''
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        #-------------------- plot fibers start & end spheres ----------------------
        for start_sphere, target_sphere in zip(self.start_points, self.end_points):
            start_node=start_sphere.cpu().numpy()
            target_node=target_sphere.cpu().numpy()
            fiber_x = [start_node[0], target_node[0]]
            fiber_y = [start_node[1], target_node[1]]
            fiber_z = [start_node[2], target_node[2]]
            fiber_matrix=np.vstack([start_node, target_node])
            ax.plot3D(fiber_x, fiber_y, fiber_z, color='r', lw='0.5')
            self.plot_spheres(ax=ax, fiber_matrix=fiber_matrix, color='g')
        # ------------------- plot box edges ----------------------         
        ax.set_xlim(-self.box_length/2, self.box_length/2)
        ax.set_ylim(-self.box_length/2, self.box_length/2)
        ax.set_zlim(-self.box_length/2, self.box_length/2)
        ax.set_xlabel("x (µm)",fontsize=15, labelpad=10)
        ax.set_ylabel("y (µm)",fontsize=15, labelpad=10)
        ax.set_zlabel("z (µm)",fontsize=15, labelpad=10)
        plt.title("3D plot of start/end spheres", fontsize=17, pad=20) 
        ax.tick_params(axis='both', which='major', labelsize=13)
        folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/init2D"
        plt.tight_layout()
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(folder_path+"/Start_end_points_in_3D"+"_"+self.date_time+".png", dpi=300)
    
    def plot_spheres(self, ax, fiber_matrix, color):
        ''' plot 3D spheres representing start and end points'''
        color_sphere=0
        u = np.linspace(0, 2 * np.pi, 8)
        v = np.linspace(0, np.pi, 8)
        for sphere_idx, node in enumerate(fiber_matrix):
            sphere_x = node[3] * np.outer(np.cos(u), np.sin(v)) + node[0]
            sphere_y = node[3] * np.outer(np.sin(u), np.sin(v)) + node[1]
            sphere_z = node[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + node[2]
            color_sphere = color
            ax.plot_surface(sphere_x, sphere_y, sphere_z, color=color_sphere, alpha=.2)
