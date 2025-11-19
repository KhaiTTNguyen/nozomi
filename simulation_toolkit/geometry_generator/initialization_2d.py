import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os.path
import numpy as np
from scipy.stats import genextreme
from scipy.optimize import minimize
# import hipa.geometrygen.helper.CollisionDetection2D as CD
# import nozomi.config as config

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
        config.BOX_LENGTH = box_length
        return radii, fiber_id, box_length, circle_centers,\
            mean_d_underlying, sigma_d_underlying
