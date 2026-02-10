import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp1f1
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import simulation_toolkit.toolkit_params as config_params

class WatsonDistribution:
    """
    Watson distribution for modeling orientation dispersion on the unit sphere.
    
    Based on Zhang et al. (2011) "Axon diameter mapping in the presence of 
    orientation dispersion with diffusion MRI"
    """
    
    def __init__(self, mu, kappa):
        """
        Initialize Watson distribution.
        
        Parameters:
        mu : array-like, shape (3,)
            Mean direction (unit vector)
        kappa : float
            Concentration parameter (kappa > 0)
        """
        self.mu = np.array(mu) / np.linalg.norm(mu)  # Ensure unit vector
        self.kappa = float(kappa)
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        
        # Compute normalization constant
        self.norm_const = self._compute_normalization_constant()
    
    def _compute_normalization_constant(self):
        """Compute the normalization constant M^(-1)(1/2, 3/2; kappa)"""
        # Using confluent hypergeometric function
        # M(a,b;z) = hyp1f1(a, b, z)
        return 1.0 / hyp1f1(0.5, 1.5, self.kappa)
    
    def pdf(self, n):
        """
        Compute probability density function.
        
        Parameters:
        n : array-like, shape (..., 3)
            Unit vectors on sphere
            
        Returns:
        pdf_values : array-like
            PDF values
        """
        n = np.atleast_2d(n)
        # Ensure unit vectors
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        
        # Compute (mu · n)^2
        dot_products = np.dot(n, self.mu)
        dot_squared = dot_products**2
        
        # Watson distribution: M^(-1) * exp(kappa * (mu · n)^2)
        pdf_values = self.norm_const * np.exp(self.kappa * dot_squared)
        
        return pdf_values.squeeze()
    
    def sample(self, n_samples=1000):
        """
        Sample from Watson distribution using rejection sampling.
        
        Parameters:
        n_samples : int
            Number of samples to generate
            
        Returns:
        samples : array, shape (n_samples, 3)
            Unit vectors sampled from the distribution
        """
        samples = []
        n_generated = 0
        
        # For rejection sampling, we need an envelope function
        # We use uniform distribution on sphere with appropriate scaling
        max_pdf = self.norm_const * np.exp(self.kappa)  # Maximum possible PDF value
        
        while n_generated < n_samples:
            # Generate candidate samples uniformly on sphere
            n_candidates = min(n_samples - n_generated, n_samples // 10 + 100)
            candidates = self._sample_uniform_sphere(n_candidates)
            
            # Only consider candidates in upper hemisphere (Z > 0)
            upper_hemisphere_mask = candidates[:, 2] > 0
            candidates = candidates[upper_hemisphere_mask]
            
            if len(candidates) == 0:
                continue
            
            # Compute PDF values
            pdf_values = self.pdf(candidates)
            
            # Generate uniform random values for rejection
            u = np.random.uniform(0, max_pdf, len(candidates))
            
            # Accept samples where u < pdf
            accepted = candidates[u < pdf_values]
            
            if len(accepted) > 0:
                samples.append(accepted)
                n_generated += len(accepted)
        
        # Concatenate and return exact number of samples
        all_samples = np.vstack(samples)
        return all_samples
    
    def _sample_uniform_sphere(self, n_samples):
        """Generate uniform samples on unit sphere."""
        # Using normal distribution and normalization
        samples = np.random.randn(n_samples, 3)
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        return samples
    

    def visualize_watson_samples(self, samples, mu, kappa, title_suffix=""):
        """
        Visualize Watson distribution samples.
        
        Parameters:
        samples : array, shape (n_samples, 3)
            Sampled unit vectors
        mu : array, shape (3,)
            Mean direction
        kappa : float
            Concentration parameter
        """
        # fig, ax = plt.subplots(figsize=(12, 6), projection='3d')
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection='3d')
        
        # Create half unit sphere surface (upper hemisphere, z >= 0)
        u = np.linspace(0, 2 * np.pi, 50)  # azimuthal angle
        v = np.linspace(0, np.pi/2, 25)    # polar angle (0 to pi/2 for upper hemisphere)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot the hemisphere surface
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='lightgray')
        
        # 3D scatter plot
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], 
                alpha=0.8, s=20, c='blue')
        
        # Plot mean direction as red arrow
        ax.quiver(0, 0, 0, mu[0], mu[1], mu[2], 
                color='red', arrow_length_ratio=0.1, linewidth=3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Watson Distribution Samples on Upper Half of a Unit Sphere\nκ={kappa}', fontsize=17)
       
        # Set equal aspect ratio
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        plt.tight_layout()
        ODI_folder = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH,"figs", "substrate_stats", "ODI")
        if not os.path.exists(ODI_folder):
                    os.makedirs(ODI_folder)
        plt.savefig(f'{ODI_folder}/watson_samples_kappa_{kappa}.png')

    def demonstrate_watson_distribution(self, kappa_values= [4, 8, 16, 32, 64, 128]):
        """Demonstrate Watson distribution 
        with different concentration parameters.
        Input kappa taken from Table 1 of Zhang (2011): 10.1016/j.neuroimage.2011.01.084
        """
        # Define mean direction (z-axis)
        mu = np.array([0, 0, 1])
        
        for kappa in kappa_values:
            print(f"Generating samples for κ = {kappa}...")
            
            # Create Watson distribution
            watson = WatsonDistribution(mu, kappa)
            
            # Generate samples
            samples = watson.sample(1000)
            
            # Compute some statistics
            dot_products = np.dot(samples, mu)
            angles_deg = np.degrees(np.arccos(np.abs(dot_products)))
            
            # Compare with theoretical values from Table 1
            angles_5deg = np.sum(angles_deg <= 5) / len(angles_deg) * 100
            angles_10deg = np.sum(angles_deg <= 10) / len(angles_deg) * 100
            angles_15deg = np.sum(angles_deg <= 15) / len(angles_deg) * 100
            angles_30deg = np.sum(angles_deg <= 30) / len(angles_deg) * 100
            angles_45deg = np.sum(angles_deg <= 45) / len(angles_deg) * 100
            angles_60deg = np.sum(angles_deg <= 60) / len(angles_deg) * 100
            angles_75deg = np.sum(angles_deg <= 75) / len(angles_deg) * 100
            
            print(f"κ={kappa:3d}:" +
                f"≤5°:{angles_5deg:5.1f}%, " +
                f"≤10°:{angles_10deg:5.1f}%, " +
                f"≤15°:{angles_15deg:5.1f}%, "+
                f"≤30°:{angles_30deg:5.1f}%, "+
                f"≤45°:{angles_45deg:5.1f}%, "+
                f"≤60°:{angles_60deg:5.1f}%, "+
                f"≤75°:{angles_75deg:5.1f}%, ")
            
            # Visualize samples
            self.visualize_watson_samples(samples, mu, kappa, f" (Sample {kappa})")
