# diffusion_simulation.py
import numpy as np
import scipy.io as sio
import os
import json
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiffusionSimulator:
    """
    A class for simulating diffusion MRI data with configurable parameters.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the diffusion simulator with configuration parameters.
        
        Args:
            config_file (str): Path to JSON configuration file
        """
        self.config = self._load_config(config_file)
        self.validate_config()
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "n_directions": 75,
            "b_values": [1000, 3000],
            "matrix_dims": [10, 10, 10],
            "substrate_type": "white_matter",
            "output_dir": "./output",
            "matlab_file": "diffusion_data.mat",
            "seed": 42
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        
        return default_config
    
    def validate_config(self):
        """Validate configuration parameters."""
        required_keys = ["n_directions", "b_values", "matrix_dims"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        if len(self.config["b_values"]) != 2:
            raise ValueError("Exactly 2 b-values must be specified")
        
        if len(self.config["matrix_dims"]) != 3:
            raise ValueError("Matrix dimensions must be 3D")
            
        logger.info("Configuration validated successfully")
    
    def generate_substrate_properties(self, substrate_type: str) -> Dict[str, float]:
        """
        Generate substrate properties based on tissue type.
        
        Args:
            substrate_type (str): Type of substrate ('white_matter', 'gray_matter', 'csf')
            
        Returns:
            dict: Dictionary containing diffusion properties
        """
        properties = {
            "white_matter": {
                "fa": 0.7,  # Fractional anisotropy
                "md": 0.8e-3,  # Mean diffusivity (mm²/s)
                "axial_diffusivity": 1.2e-3,
                "radial_diffusivity": 0.4e-3
            },
            "gray_matter": {
                "fa": 0.2,
                "md": 0.9e-3,
                "axial_diffusivity": 1.0e-3,
                "radial_diffusivity": 0.8e-3
            },
            "csf": {
                "fa": 0.0,
                "md": 3.0e-3,
                "axial_diffusivity": 3.0e-3,
                "radial_diffusivity": 3.0e-3
            }
        }
        
        if substrate_type not in properties:
            logger.warning(f"Unknown substrate type: {substrate_type}. Using white_matter.")
            substrate_type = "white_matter"
        
        return properties[substrate_type]
    
    def generate_gradient_directions(self, n_directions: int) -> np.ndarray:
        """
        Generate uniformly distributed gradient directions on a sphere.
        
        Args:
            n_directions (int): Number of gradient directions
            
        Returns:
            np.ndarray: Array of gradient directions (n_directions x 3)
        """
        np.random.seed(self.config["seed"])
        
        # Generate points on unit sphere using spherical coordinates
        directions = np.zeros((n_directions, 3))
        
        for i in range(n_directions):
            # Use uniform distribution on sphere
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            
            theta = 2 * np.pi * u  # Azimuthal angle
            phi = np.arccos(2 * v - 1)  # Polar angle
            
            directions[i, 0] = np.sin(phi) * np.cos(theta)  # x
            directions[i, 1] = np.sin(phi) * np.sin(theta)  # y
            directions[i, 2] = np.cos(phi)  # z
        
        # Normalize to ensure unit vectors
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        logger.info(f"Generated {n_directions} gradient directions")
        return directions
    
    def compute_diffusion_tensor(self, properties: Dict[str, float]) -> np.ndarray:
        """
        Compute the diffusion tensor from tissue properties.
        
        Args:
            properties (dict): Tissue diffusion properties
            
        Returns:
            np.ndarray: 3x3 diffusion tensor
        """
        # Principal diffusion direction (assume along z-axis)
        principal_direction = np.array([0, 0, 1])
        
        # Create orthogonal basis
        v1 = principal_direction
        v2 = np.array([1, 0, 0])
        v3 = np.cross(v1, v2)
        v2 = np.cross(v3, v1)
        
        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)
        
        # Eigenvalues
        lambda1 = properties["axial_diffusivity"]
        lambda2 = properties["radial_diffusivity"]
        lambda3 = properties["radial_diffusivity"]
        
        # Construct tensor
        V = np.column_stack([v1, v2, v3])
        L = np.diag([lambda1, lambda2, lambda3])
        D = V @ L @ V.T
        
        return D
    
    def simulate_signal(self, directions: np.ndarray, tensor: np.ndarray, 
                       b_value: float) -> np.ndarray:
        """
        Simulate diffusion MRI signal using the diffusion tensor model.
        
        Args:
            directions (np.ndarray): Gradient directions
            tensor (np.ndarray): Diffusion tensor
            b_value (float): b-value in s/mm²
            
        Returns:
            np.ndarray: Simulated signal values
        """
        n_directions = directions.shape[0]
        signals = np.zeros(n_directions)
        
        # Add small amount of noise for realism
        np.random.seed(self.config["seed"])
        
        for i, direction in enumerate(directions):
            # Compute apparent diffusion coefficient (ADC)
            adc = direction.T @ tensor @ direction
            
            # Mono-exponential signal decay
            signal = np.exp(-b_value * adc)
            
            # Add Rician noise (typical for MRI)
            noise_level = 0.02  # 2% noise
            noise = np.random.normal(0, noise_level)
            signal = np.abs(signal + noise)
            
            signals[i] = signal
        
        return signals
    
    def augment_to_matrix(self, signals: np.ndarray, 
                         matrix_dims: Tuple[int, int, int]) -> np.ndarray:
        """
        Augment scalar signals into 3D matrices.
        
        Args:
            signals (np.ndarray): 1D array of signal values
            matrix_dims (tuple): Target matrix dimensions
            
        Returns:
            np.ndarray: 4D array (x, y, z, directions)
        """
        n_directions = len(signals)
        x, y, z = matrix_dims
        
        # Create 4D matrix
        matrix = np.zeros((x, y, z, n_directions))
        
        # Add spatial variation to make it more realistic
        np.random.seed(self.config["seed"])
        
        for i in range(n_directions):
            # Create a base matrix with the signal value
            base_signal = signals[i]
            
            # Add spatial variation (e.g., Gaussian smoothing effect)
            spatial_variation = np.random.normal(1.0, 0.1, (x, y, z))
            spatial_variation = np.clip(spatial_variation, 0.5, 1.5)
            
            matrix[:, :, :, i] = base_signal * spatial_variation
        
        logger.info(f"Augmented signals to {matrix_dims} matrix")
        return matrix
    
    def combine_b_value_data(self, data_b1: np.ndarray, 
                           data_b2: np.ndarray) -> np.ndarray:
        """
        Combine data from two b-values into a single matrix.
        
        Args:
            data_b1 (np.ndarray): Data from first b-value
            data_b2 (np.ndarray): Data from second b-value
            
        Returns:
            np.ndarray: Combined data matrix
        """
        combined_data = np.concatenate([data_b1, data_b2], axis=-1)
        logger.info(f"Combined data shape: {combined_data.shape}")
        return combined_data
    
    def save_for_matlab(self, data: np.ndarray, output_dir: str, 
                       filename: str) -> str:
        """
        Save data in MATLAB format.
        
        Args:
            data (np.ndarray): Data to save
            output_dir (str): Output directory
            filename (str): Output filename
            
        Returns:
            str: Full path to saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        # Prepare data dictionary for MATLAB
        matlab_data = {
            'diffusion_data': data,
            'config': self.config,
            'data_shape': data.shape,
            'b_values': self.config['b_values'],
            'n_directions': self.config['n_directions'],
            'matrix_dims': self.config['matrix_dims']
        }
        
        try:
            sio.savemat(filepath, matlab_data)
            logger.info(f"Data saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            raise
    
    def run_simulation(self) -> str:
        """
        Run the complete diffusion simulation pipeline.
        
        Returns:
            str: Path to saved MATLAB file
        """
        logger.info("Starting diffusion simulation...")
        
        try:
            # Step 1: Generate substrate properties
            substrate_props = self.generate_substrate_properties(
                self.config["substrate_type"]
            )
            logger.info(f"Substrate properties: {substrate_props}")
            
            # Step 2: Generate gradient directions
            directions =self.generate_gradient_directions(self.config["n_directions"])
            
            # Step 3: Compute diffusion tensor
            tensor = self.compute_diffusion_tensor(substrate_props)
            logger.info(f"Diffusion tensor computed")
            
            # Step 4: Simulate signals for both b-values
            b_values = self.config["b_values"]
            
            signals_b1 = self.simulate_signal(directions, tensor, b_values[0])
            signals_b2 = self.simulate_signal(directions, tensor, b_values[1])
            
            logger.info(f"Signals computed for b-values: {b_values}")
            
            # Step 5: Augment signals to 3D matrices
            matrix_dims = tuple(self.config["matrix_dims"])
            
            data_b1 = self.augment_to_matrix(signals_b1, matrix_dims)
            data_b2 = self.augment_to_matrix(signals_b2, matrix_dims)
            
            # Step 6: Combine b-value data
            combined_data = self.combine_b_value_data(data_b1, data_b2)
            
            # Step 7: Save data for MATLAB
            filepath = self.save_for_matlab(
                combined_data, 
                self.config["output_dir"],
                self.config["matlab_file"]
            )
            
            logger.info("Diffusion simulation completed successfully")
            return filepath
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise

def create_config_file(filename: str = "config.json"):
    """Create a sample configuration file."""
    config = {
        "n_directions": 75,
        "b_values": [1000, 3000],
        "matrix_dims": [10, 10, 10],
        "substrate_type": "white_matter",
        "output_dir": "./output",
        "matlab_file": "diffusion_data.mat",
        "seed": 42
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration file created: {filename}")

if __name__ == "__main__":
    # Create sample configuration
    create_config_file()
    
    # Run simulation
    simulator = DiffusionSimulator("config.json")
    output_file = simulator.run_simulation()
    print(f"Simulation complete. Data saved to: {output_file}")


'''
% diffusion_matlab_processor.m
function diffusion_matlab_processor(input_file, output_file)
    % DIFFUSION_MATLAB_PROCESSOR Process diffusion data from Python and save as NIFTI
    %
    % Usage:
    %   diffusion_matlab_processor('diffusion_data.mat', 'diffusion_data.nii')
    %
    % Inputs:
    %   input_file  - Path to .mat file from Python
    %   output_file - Path for output .nii file
    %
    % Requirements:
    %   - Tools for NIfTI toolbox (https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)
    
    try
        % Set default parameters if not provided
        if nargin < 1
            input_file = './output/diffusion_data.mat';
        end
        if nargin < 2
            output_file = './output/diffusion_data.nii';
        end
        
        fprintf('Starting MATLAB processing...\n');
        fprintf('Input file: %s\n', input_file);
        fprintf('Output file: %s\n', output_file);
        
        % Check if input file exists
        if ~exist(input_file, 'file')
            error('Input file does not exist: %s', input_file);
        end
        
        % Load data from Python
        fprintf('Loading data from MATLAB file...\n');
        data = load(input_file);
        
        % Extract diffusion data and metadata
        if ~isfield(data, 'diffusion_data')
            error('diffusion_data field not found in input file');
        end
        
        diffusion_data = data.diffusion_data;
        config = data.config;
        
        fprintf('Data shape: [%s]\n', num2str(size(diffusion_data)));
        fprintf('B-values: [%s]\n', num2str(config.b_values));
        fprintf('Number of directions: %d\n', config.n_directions);
        
        % Validate data dimensions
        expected_dims = [config.matrix_dims, config.n_directions * 2];
        if ~isequal(size(diffusion_data), expected_dims)
            warning('Data dimensions do not match expected size');
        end
        
        % Create NIFTI header
        nii = make_nii(diffusion_data);
        
        % Set voxel dimensions (in mm)
        voxel_size = [2.0, 2.0, 2.0, 1.0]; % 2mm isotropic with TR=1s
        nii.hdr.dime.pixdim(2:5) = voxel_size;
        
        % Set data type
        nii.hdr.dime.datatype = 16; % float32
        nii.hdr.dime.bitpix = 32;
        
        % Set spatial orientation
        nii.hdr.hist.qform_code = 1;
        nii.hdr.hist.sform_code = 1;
        
        % Add description
        description = sprintf('Diffusion MRI simulation: %d dirs, b=[%s]', ...
                            config.n_directions, num2str(config.b_values));
        nii.hdr.hist.descrip = pad_string(description, 80);
        
        % Create output directory if it doesn't exist
        [output_dir, ~, ~] = fileparts(output_file);
        if ~isempty(output_dir) && ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        % Save NIFTI file
        fprintf('Saving NIFTI file...\n');
        save_nii(nii, output_file);
        
        % Verify the saved file
        if exist(output_file, 'file')
            fprintf('Successfully saved: %s\n', output_file);
            
            % Display file info
            file_info = dir(output_file);
            fprintf('File size: %.2f MB\n', file_info.bytes / 1024 / 1024);
            
            % Test loading the saved file
            test_nii = load_nii(output_file);
            fprintf('Verification: Loaded data shape [%s]\n', num2str(size(test_nii.img)));
        else
            error('Failed to create output file');
        end
        
        fprintf('MATLAB processing completed successfully!\n');
        
    catch ME
        fprintf('Error in MATLAB processing: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        rethrow(ME);
    end
end

function padded_str = pad_string(str, target_length)
    % Helper function to pad string to specific length
    if length(str) >= target_length
        padded_str = str(1:target_length);
    else
        padded_str = [str, repmat(' ', 1, target_length - length(str))];
    end
end

function check_nifti_toolbox()
    % Check if NIfTI toolbox is available
    if ~exist('make_nii', 'file')
        error(['Tools for N
'''