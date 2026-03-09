import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
import simulation_toolkit.toolkit_params as config_params
class Meshing(object):
    """
    A class for sphere-based meshing of axons, with beading
    """
    def __init__(self, initialization2D, spheres_spacing_ratio, bead_spacing_mean, bead_spacing_stdv, device):
        self.device=device
        self.init2D = initialization2D   
        self.date_time = initialization2D.date_time
        self.box_length = initialization2D.box_length
        self.spheres_spacing_ratio = spheres_spacing_ratio  
        self.bead_spacing_mean, self.bead_spacing_stdv = bead_spacing_mean, bead_spacing_stdv
        self.interpolated_fiber_list = self.interpolate_fibers_with_spheres_start_end()
    
    #========================With beading==========================
    def interpolate_fibers_with_spheres_start_end(self, cylinder=False):
        '''return interpolated_fiber_list torch.Size([N, 6])'''
        interpolated_fiber_list = torch.tensor([], device=self.device) 
        num_spheres = 0
        count = 0        
        for fiber in zip(self.init2D.start_points, self.init2D.end_points):
            fiber = torch.vstack((fiber[0], fiber[1]))
            if len(fiber) > 0:
                count = count+1
                new_fiber = self.interpolate_spheres_for_fiber(fiber, count, cylinder)
                new_fiber_sphere_id = torch.arange(num_spheres, num_spheres + new_fiber.shape[0], device=self.device).unsqueeze(1)
                new_fiber = torch.hstack((new_fiber, new_fiber_sphere_id))
                maxZ = torch.tensor(0)
                maxZ= torch.max(maxZ, torch.max(new_fiber[:,2]))
                if count==1:
                    interpolated_fiber_list=new_fiber
                else:
                    interpolated_fiber_list = torch.vstack((interpolated_fiber_list, new_fiber))
                num_spheres = num_spheres + new_fiber.shape[0]
        return interpolated_fiber_list
    
    def interpolate_spheres_for_fiber(self, fiber, fiber_count, cylinder):
        spacing = fiber[0][3] * self.spheres_spacing_ratio    #start_sphere: x,y,z,r,fid 
        if spacing <= 0:
            print(f"Warning: Invalid spacing {spacing} for fiber {fiber_count}")
            print(f"fiber[0][3] (radius): {fiber[0][3]}")
            print(f"spheres_spacing_ratio: {self.spheres_spacing_ratio}")

        x_coord, y_coord, z_coord, s = self.interpolate_coords_from_spacing_helix(spacing, fiber)
        
        if (cylinder==True):
            new_radius = torch.ones_like(x_coord)*fiber[0][3]
        else:
            # Create radius with beading and spacing here. Then calcualte Coefficient of Variation = (Standard Deviation Radius / Mean Radius).
            new_radius = self.interpolate_radius_with_beading(s, original_radius=fiber[0][3])
        
        new_fid = torch.ones_like(x_coord)*torch.tensor(fiber_count).to(self.device)
        
        new_fiber = torch.stack([x_coord, y_coord, z_coord, new_radius, new_fid]).T
        return new_fiber
    
    def interpolate_radius_with_beading(self, s, original_radius):
        positions_along_axon = s
        beading_spacings = self.get_beading_spacings_along_axon(positions_along_axon)
        beading_positions_along_axon = torch.cumsum(beading_spacings, dim=0)
        beading_positions_along_axon=beading_positions_along_axon[beading_positions_along_axon<positions_along_axon[-1]]
        r0=original_radius
        alpha_mean = config_params.BEAD_ALPHA_MEAN  # Controls mean CV across axons
        alpha_stdv = config_params.BEAD_ALPHA_STDV  # Controls stdv of CV across axons
        alpha_dist = torch.distributions.normal.Normal(loc=alpha_mean, scale=alpha_stdv)
        alpha = alpha_dist.sample((1,)).to(self.device)  # drawn once per axon
        result = torch.full(positions_along_axon.shape,r0.item()).to(self.device)
        for i in range(len(beading_positions_along_axon)):
            z = positions_along_axon
            mean = beading_positions_along_axon[i]
            sigma = torch.tensor(2.3, device=self.device)
            gaussian_peak = torch.exp(-(z - mean)**2 / (2 * sigma**2))
            result = result + alpha * r0 * gaussian_peak
        result = self.process_result_endpoints(result, r0)
        return result
    
    def process_result_endpoints(self, result, target_value):
        x = torch.linspace(0, len(result), len(result),  device=self.device)
        # Calculate the slope of the linear transformation
        slope = (result[-1] - result[0]) / len(result)

        # Create the linear transformation function
        def linear_transformation(x):
            return slope * x #+ result[0]

        # Combine the functions
        def modified_function(x, result):
            return result - linear_transformation(x)

        # Calculate the modified function values
        modified_y = modified_function(x, result)
        modified_y = modified_y - modified_y[0]
        modified_y = modified_y + target_value
        modified_y = torch.clamp(modified_y, min=0.2)
        return modified_y
    
    def get_beading_spacings_along_axon(self, positions_along_axon):
        u = torch.tensor(self.bead_spacing_mean)
        v=torch.tensor(self.bead_spacing_stdv)
        mu = torch.log(u**2 / torch.sqrt(v**2 + u**2))
        sigma = torch.sqrt(torch.log(1 + (v**2 / u**2)))
        lognormal_dist = torch.distributions.log_normal.LogNormal(mu, sigma)
        beading_spacings = lognormal_dist.sample(positions_along_axon.shape).to(self.device)
        return beading_spacings
    
    def interpolate_coords_from_spacing_helix(self, spacing, startend_nodes):
        """
        Interpolate coordinates along a helix curve with dynamic sphere count based on arc length
        """
        start_point = startend_nodes[0, :3]  # [x, y, z]
        end_point = startend_nodes[1, :3]    # [x, y, z]
        
        # Calculate Lz (total z distance)
        Lz = torch.abs(end_point[2] - start_point[2])
        
        # Calculate arc length of the helix
        arc_length = self.calculate_arc_length_numerical(start_point, end_point, Lz)
        
        # Calculate number of spheres based on arc length and spacing
        num_spheres = max(2, int(torch.ceil(arc_length / spacing)))
        
        # Generate parameter values from 0 to 1
        t_values = torch.linspace(0, 1, num_spheres, device=self.device)
        
        # Generate helix coordinates
        xinter, yinter, zinter = self.generate_helix_coordinates(
            start_point, end_point, t_values, Lz
        )
        
        # Create distance array for compatibility with existing code
        s = torch.linspace(0, arc_length.item(), steps=num_spheres, device=self.device)
        
        return xinter, yinter, zinter, s

    def generate_helix_coordinates(self, start_point, end_point, t_values, Lz):
        """
        Generate helix coordinates
        """
        # t_values goes from 0 to 1
        x = (start_point[0] - end_point[0])/2 * torch.cos(t_values * torch.pi) + (start_point[0] + end_point[0])/2
        y = (end_point[1] - start_point[1]) * torch.sin(t_values * torch.pi/2) + start_point[1]
        z = start_point[2] + t_values * Lz

        return x, y, z

    def helix_derivatives(self, t, start, end, Lz):
        """Calculate derivatives of helix parametric equations"""
        # dx/dt
        dx_dt = -(start[0] - end[0])/2 * torch.pi * torch.sin(t * torch.pi)
        
        # dy/dt  
        dy_dt = (end[1] - start[1]) * torch.pi/2 * torch.cos(t * torch.pi/2)
        
        # dz/dt
        dz_dt = Lz
        
        return dx_dt, dy_dt, dz_dt

    def arc_length_integrand(self, t, start, end, Lz):
        """Calculate the integrand for arc length: sqrt((dx/dt)² + (dy/dt)² + (dz/dt)²)"""
        dx_dt, dy_dt, dz_dt = self.helix_derivatives(t, start, end, Lz)
        return torch.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

    def calculate_arc_length_numerical(self, start, end, Lz, num_samples=1000):
        """Calculate total arc length using numerical integration"""
        t_values = torch.linspace(0, 1, num_samples, device=self.device)
        dt = 1.0 / (num_samples - 1)
        
        integrand_values = torch.stack([
            self.arc_length_integrand(t, start, end, Lz) for t in t_values
        ])
        
        # Simple trapezoidal integration
        arc_length = torch.trapz(integrand_values, dx=dt)
        return arc_length
