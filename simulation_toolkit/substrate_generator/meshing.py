import numpy as np
import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os.path
# import hipa.config as config
import torch
import torch.nn.functional as F
import simulation_toolkit.toolkit_params as config_params
class Meshing(object):
    """
    A class for Node Network
    """
    def __init__(self, initialization2D, spheres_spacing_ratio, bead_spacing_mean, bead_spacing_stdv, device):
        self.device=device
        self.init2D = initialization2D   
        self.date_time = initialization2D.date_time
        self.box_length = initialization2D.box_length
        
        self.spheres_spacing_ratio = spheres_spacing_ratio  
        self.bead_spacing_mean, self.bead_spacing_stdv = bead_spacing_mean, bead_spacing_stdv
        # self.interpolated_cylinder_fiber_list = self.interpolate_fibers_with_spheres_start_end(cylinder=True)
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
                # print('At -Lx/2 and Lx/2, maxZ:', maxZ, self.box_length/2)
                if count==1:
                    interpolated_fiber_list=new_fiber
                else:
                    interpolated_fiber_list = torch.vstack((interpolated_fiber_list, new_fiber))
                num_spheres = num_spheres + new_fiber.shape[0]
        return interpolated_fiber_list
    
    def interpolate_spheres_for_fiber(self, fiber, fiber_count, cylinder):
        spacing = fiber[0][3] * self.spheres_spacing_ratio    #start_node: x,y,z,r,fid 
        if spacing <= 0:
            print(f"Warning: Invalid spacing {spacing} for fiber {fiber_count}")
            print(f"fiber[0][3] (radius): {fiber[0][3]}")
            print(f"spheres_spacing_ratio: {self.spheres_spacing_ratio}")

        # x_coord,y_coord,z_coord, s = self.interpolate_coords_from_spacing(spacing, fiber)
        # x_coord,y_coord,z_coord, s = self.interpolate_coords_from_spacing_parabolic(spacing, fiber)
        x_coord, y_coord, z_coord, s = self.interpolate_coords_from_spacing_helix(spacing, fiber)
        
        if (cylinder==True):
            new_radius = torch.ones_like(x_coord)*fiber[0][3]
        else:
            new_radius = self.interpolate_radius_with_beading(s, original_radius=fiber[0][3])
        '''create radius with beading and spacing right here
        beading magnitude ratios of 1.7, 1.5, and 1.2

        Coefficient of Variation = (Standard Deviation / Mean Radius) * 100
        '''
        new_fid = torch.ones_like(x_coord)*torch.tensor(fiber_count).to(self.device)
        
        new_fiber = torch.stack([x_coord, y_coord, z_coord, new_radius, new_fid]).T
        return new_fiber
    
    def interpolate_radius_with_beading(self, s, original_radius):
        positions_along_axon = s
        beading_spacings = self.get_beading_spacings_along_axon(positions_along_axon)
        beading_positions_along_axon = torch.cumsum(beading_spacings, dim=0)
        beading_positions_along_axon=beading_positions_along_axon[beading_positions_along_axon<positions_along_axon[-1]]
        ''' This is for drawing different log-normally distributed radius values
        u = torch.tensor(1.3/2)
        v= torch.tensor(0.3/2)
        mean_r0 = torch.log(u**2 / torch.sqrt(v**2 + u**2))
        std_dev_r0 = torch.sqrt(torch.log(1 + (v**2 / u**2)))
        lognormal_dist = torch.distributions.log_normal.LogNormal(mean_r0, std_dev_r0)
        # Draw samples from the distribution
        r0 = lognormal_dist.sample((1,))
        '''
        r0=original_radius
        mean_r1 = config_params.BEAD_AMPLITUDE_MEAN #1.5  # Adjust this to get the desired mean of the gaussian distribution
        std_r1 = config_params.BEAD_AMPLITUDE_STDV #0.85  # Adjust this to control the spread of the distribution
        gaussian_dist = torch.distributions.normal.Normal(loc=mean_r1, scale=std_r1)   
        result = torch.full(positions_along_axon.shape,r0.item()).to(self.device)
        for i in range(len(beading_positions_along_axon)):
            r1 = gaussian_dist.sample((1,)).to(self.device)
            z = positions_along_axon
            mean = beading_positions_along_axon[i]
            sigma = torch.tensor(2.3, device=self.device)
            gaussian_peak = torch.exp(-(z - mean)**2 / (2 * sigma**2)) / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
            result = result + r1*gaussian_peak
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
        modified_y = torch.clamp(modified_y, min=0.08)
        return modified_y
    
    def get_beading_spacings_along_axon(self, positions_along_axon):
        u = torch.tensor(self.bead_spacing_mean)
        v=torch.tensor(self.bead_spacing_stdv)
        mu = torch.log(u**2 / torch.sqrt(v**2 + u**2))
        sigma = torch.sqrt(torch.log(1 + (v**2 / u**2)))
        lognormal_dist = torch.distributions.log_normal.LogNormal(mu, sigma)
        beading_spacings = lognormal_dist.sample(positions_along_axon.shape).to(self.device)
        # normal_dist = torch.distributions.normal.Normal(u, v)
        # beading_spacings = normal_dist.sample(positions_along_axon.shape)
        return beading_spacings
    
    def interpolate_coords_from_spacing(self, spacing, start_end_nodes):       
        x = start_end_nodes[:,0]
        y = start_end_nodes[:,1]
        z = start_end_nodes[:,2]
        # print(x,y,z)
        # compute the distances, ds, between points
        dx, dy, dz = x[+1:]-x[:-1],  y[+1:]-y[:-1],  z[+1:]-z[:-1]
        ds = torch.tensor((0, torch.sqrt(dx*dx+dy*dy+dz*dz)))

        # segment distance
        s = torch.cumsum(ds, dim=0)
        num_spheres = int(torch.ceil(s[-1]/spacing))
        # print('s',s, 'num_spheres', num_spheres, 'spacing', spacing)
        s = torch.linspace(0,s[-1].item(), steps=num_spheres, device=self.device)
        xinter = F.interpolate(x.unsqueeze(0).unsqueeze(0), (s.size(0),), 
                            mode='linear', align_corners=True).squeeze()
        yinter = F.interpolate(y.unsqueeze(0).unsqueeze(0), (s.size(0),), 
                            mode='linear', align_corners=True).squeeze()
        zinter = F.interpolate(z.unsqueeze(0).unsqueeze(0), (s.size(0),), 
                            mode='linear', align_corners=True).squeeze()
        xinter[0], xinter[-1], yinter[0], yinter[-1], zinter[0], zinter[-1] = x[0], x[-1], y[0], y[-1], z[0], z[-1]

        return xinter,yinter,zinter, s

    def interpolate_coords_from_spacing_parabolic(self, spacing, startend_nodes):
        x = startend_nodes[:,0]
        y = startend_nodes[:,1] 
        z = startend_nodes[:,2]
        
        # compute the distances, ds, between points
        dx, dy, dz = x[+1:]-x[:-1], y[+1:]-y[:-1], z[+1:]-z[:-1]
        ds = torch.tensor((0, torch.sqrt(dx*dx+dy*dy+dz*dz)))
        
        # segment distance
        s = torch.cumsum(ds, dim=0)
        num_spheres = int(torch.ceil(s[-1]/spacing))
        
        # Create parabolic interpolation instead of linear
        t_values = torch.linspace(0, 1, steps=num_spheres, device=self.device)
        
        # Get fiber radius for curve depth calculation
        fiber_radius = startend_nodes[0][3]  # assuming radius is in column 3
        
        # Calculate parabolic curve coordinates
        xinter, yinter, zinter = self.interpolate_parabolic_curve(
            start_point=[x[0], y[0], z[0]], 
            end_point=[x[-1], y[-1], z[-1]], 
            t_values=t_values, 
            fiber_radius=fiber_radius,
            depth_multiplier=config.DEPTH_MULTIPLIER
        )
        
        # Ensure endpoints are exact
        xinter[0], xinter[-1] = x[0], x[-1]
        yinter[0], yinter[-1] = y[0], y[-1] 
        zinter[0], zinter[-1] = z[0], z[-1]
        
        return xinter, yinter, zinter, torch.linspace(0, s[-1].item(), steps=num_spheres, device=self.device)

    def interpolate_parabolic_curve(self, start_point, end_point, t_values, fiber_radius, depth_multiplier=1.5):
        """
        Interpolate points along a parabolic curve between start and end points.
        
        Args:
            start_point: [x, y, z] coordinates of start
            end_point: [x, y, z] coordinates of end  
            t_values: parameter values from 0 to 1
            fiber_radius: radius of the fiber
            depth_multiplier: controls how deep the parabolic curve is (depth = radius * multiplier)
        
        Returns:
            x, y, z coordinates along the parabolic curve
        """
        start = torch.tensor(start_point, device=self.device)
        end = torch.tensor(end_point, device=self.device)
        
        # Calculate curve depth based on fiber radius
        curve_depth = fiber_radius * depth_multiplier
        
        # Find midpoint and direction vector
        midpoint = (start + end) / 2
        direction_vector = end - start
        fiber_length = torch.norm(direction_vector)
        
        # Create perpendicular vector for curve offset
        # Use cross product with a reference z+ unit vector to get perpendicular direction
        reference = torch.tensor([0, 0, 1], device=self.device, dtype=direction_vector.dtype)
    
        perpendicular = torch.cross(direction_vector, reference)
        perpendicular = perpendicular / torch.norm(perpendicular)  # normalize
        
        # Calculate control point (vertex of parabola)
        control_point = midpoint + perpendicular * curve_depth
        
        # Quadratic Bezier curve interpolation
        # P(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        # where P₀ = start, P₁ = control_point, P₂ = end
        
        t = t_values.unsqueeze(1)  # Shape: [num_spheres, 1]
        one_minus_t = 1 - t
        
        # Bezier curve formula
        points = (one_minus_t**2) * start + 2 * one_minus_t * t * control_point + (t**2) * end
        
        return points[:, 0], points[:, 1], points[:, 2]  # x, y, z coordinates
    

    '''BUILDING'''
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
        Generate helix coordinates using the same equations as your original code
        """
        # Convert to the same format as your original numpy code
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

    '''deprecating'''
    # def interpolate_cosine_curve():
    #     '''
    #     Given the coords of
    #     x01,y01,z01,x11,y11,z11
    #     x02,y02,z02,x12,y12,z12
    #     '''
    #     # Parameters for half period (0 to π)
    #     Lz=10

    #     start1 = np.array([-2,-2,-Lz/2])
    #     start2 = np.array([1,1,-Lz/2])
    #     end1 = np.array([1,1,Lz/2])
    #     end2 = np.array([-2,-2,Lz/2])

    #     z = np.linspace(-Lz/2, Lz/2, 200)

    #     mid=(start1+start2)/2
    #     print('mid', mid)

    #     # # Helix 1
    #     x1 = (start1[0]-end1[0])/2 * np.cos((z+Lz/2)/Lz *np.pi) + (start1[0]+end1[0])/2
    #     y1 = (end1[1]-start1[1]) * np.sin((z+Lz/2)/Lz *np.pi/2) + start1[1]
    #     z1 = z  # Goes from 0 to π (bottom to top)

    #     # # Helix 2 - offset by π (180 degrees) to create braiding
    #     x2 = (start2[0]-end2[0])/2 * np.cos((z+Lz/2)/Lz *np.pi) + (start2[0]+end2[0])/2
    #     y2 = (end2[1]-start2[1]) * np.sin((z+Lz/2)/Lz *np.pi/2) + start2[1]
    #     z2 = z  # Same height progression
    #     print('x1 \n', x1)
    #     print('y1 \n', y1)

    #     print('x2 \n', x2)
    #     print('y2 \n', y2)

    #     # Create the plot
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.view_init(elev=90, azim=90)

    #     # Plot both helices
    #     ax.plot(x1, y1, z1, 'g-', linewidth=4, label='Helix 1')
    #     ax.plot(x2, y2, z2, 'y-', linewidth=4, label='Helix 2')
