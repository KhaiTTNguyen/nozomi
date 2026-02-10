from simulation_toolkit.substrate_generator.initialization_2d import Init2D
from simulation_toolkit.substrate_generator.meshing import Meshing
from simulation_toolkit.substrate_generator.geometric_optimization import GeometricOptimization
import simulation_toolkit.toolkit_params as config_params
import numpy as np
import time
import torch
import os.path
import simulation_toolkit.utils.common_utils as util
from matplotlib.pyplot import cm
from simulation_toolkit.utils import fiber_3D_plot
from simulation_toolkit.utils import along_fiber_plot
from simulation_toolkit.utils import orientation_plot

def substrate_main(params, experiment_folder):
    config_params.ORIENTATION_SHAPE_PARAM = params['orientation_shape_parameter']
    config_params.BOX_LENGTH = params['box_length_init']
    config_params.VOLUME_FRACTION = params['target_volume_fraction']
    config_params.NUM_FIBERS = params['num_fibers']
    config_params.MEAN_DIAMETER = params['mean_diameter']   
    config_params.SIGMA_DIAMETER = params['sigma_diameter']   
    config_params.DISTRIBUTION_SHAPE = params['dist_shape']   
    config_params.SPACE_BUFFER_STARTS_ENDS = params['space_buffer_starts_ends']   
    config_params.SPHERE_SPACING = params['spheres_spacing']   
    config_params.SPACE_BUFFER_REPULSE = params['space_buffer_repulse']  
    config_params.W_OVERLAP = params['w_overlap'] 
    config_params.W_CURVE = params['w_curve']
    config_params.W_LENGTH = params['w_length']
    config_params.BEAD_SPACING_MEAN = params['bead_spacing_mean']   
    config_params.BEAD_SPACING_STDV = params['bead_spacing_stdv']
    config_params.BEAD_AMPLITUDE_MEAN = params['bead_amplitude_mean']
    config_params.BEAD_AMPLITUDE_STDV = params['bead_amplitude_stdv']
    print(f"Building substrate with: \
        diameter_mean={config_params.MEAN_DIAMETER},\
        orientation_shape_parameter={config_params.ORIENTATION_SHAPE_PARAM}, \
        bead_amplitude_mean={config_params.BEAD_AMPLITUDE_MEAN},\
        num_fibers={config_params.NUM_FIBERS}") 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config_params.EXP_DATE_TIME = util.get_date_time()        
    config_params.ODI_INDEX = np.round(2/np.pi * np.arctan(1/config_params.ORIENTATION_SHAPE_PARAM), 4)
    config_params.SUBSTRATE_OUTPUT_FOLDER_PATH = os.path.join(config_params.OUTPUT_FOLDER_PATH, experiment_folder, \
                                                       str(config_params.EXP_DATE_TIME)+\
                                                        '_d'+str(config_params.MEAN_DIAMETER)+\
                                                        '_K'+str(int(config_params.ORIENTATION_SHAPE_PARAM))+\
                                                    '_ODI_'+str(config_params.ODI_INDEX)+\
                                                    '_bead_'+str(config_params.BEAD_AMPLITUDE_MEAN)+'_'+\
                                                        str(config_params.NUM_FIBERS) +'fibers')
    not_converged=True
    while not_converged:
        st = time.time()
        # -------- 2D initialization of axon start/end points --------
        initialization2D = Init2D(  device=device,
                                    date_time=config_params.EXP_DATE_TIME,
                                    orientation_shape_parameter=config_params.ORIENTATION_SHAPE_PARAM, 
                                    target_volume_fraction=config_params.VOLUME_FRACTION, 
                                    num_fibers=config_params.NUM_FIBERS,
                                    dist_shape=config_params.DISTRIBUTION_SHAPE,
                                    mean_diameter=config_params.MEAN_DIAMETER, 
                                    sigma_radii=config_params.SIGMA_DIAMETER,
                                    space_buffer=config_params.SPACE_BUFFER_STARTS_ENDS,
                                    box_length_init=config_params.BOX_LENGTH)

        data_folder = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH,'data')
        if not os.path.exists(data_folder):
                    os.makedirs(data_folder)

        init2d_data_folder = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH,'figs','init2D')
        if not os.path.exists(init2d_data_folder):
                    os.makedirs(init2d_data_folder)

        # save data with Pickle format
        init2d_data_file_name = os.path.join(init2d_data_folder ,'init2d.pkl')
        util.save_data_array_to_pickle(init2d_data_file_name, initialization2D.initial_positions, 
                                initialization2D.box_length.cpu().item()) 

        initialization2D.plot_PBC()
        # -------- Sphere-Based Meshing of Axons --------
        meshing = Meshing(initialization2D, config_params.SPHERE_SPACING, config_params.BEAD_SPACING_MEAN, config_params.BEAD_SPACING_STDV, device)
        # -------- Geometric Optimization --------
        substrate = GeometricOptimization(device, meshing, config_params.SPHERE_SPACING, 
                                            config_params.W_OVERLAP, config_params.W_CURVE, config_params.W_LENGTH, \
                                            config_params.SPACE_BUFFER_REPULSE, initialization2D.mean_d_underlying, initialization2D.sigma_d_underlying,\
                                            config_params.BEAD_SPACING_MEAN, config_params.BEAD_SPACING_STDV)
        not_converged = not substrate.optimized
    if (substrate.optimized == True):

        # get the execution time
        et = time.time()
        elapsed_time = et - st
        
        file_name_root = 'array'+str(config_params.NUM_FIBERS) +'_fibers_boxL_'+str(config_params.BOX_LENGTH.item())+'_'+str(config_params.EXP_DATE_TIME) + \
                        '_avf_'+str(config_params.VOLUME_FRACTION)+\
                        '_d'+str(config_params.MEAN_DIAMETER)+'_sig'+str(config_params.SIGMA_DIAMETER)+\
                            'wo'+str(config_params.W_OVERLAP)+'_wc'+str(config_params.W_CURVE)+'_wl'+str(config_params.W_LENGTH)+'_K'+str(int(config_params.ORIENTATION_SHAPE_PARAM))+\
                            '_ODI_'+str(config_params.ODI_INDEX)+'_'+\
                                str(round(elapsed_time,2))+'_sec'+'.pkl'
        
        data_folder = os.path.join(config_params.SUBSTRATE_OUTPUT_FOLDER_PATH,'data')
        if not os.path.exists(data_folder):
                    os.makedirs(data_folder)

        # # save data with Pickle format
        data_file_name = os.path.join(data_folder, file_name_root)
        util.save_data_array_to_pickle(data_file_name, substrate.optimized_fibers.cpu(), 
                                initialization2D.box_length.cpu().item()) 
        # Plot
        fiber_list = util.map_matrix_to_list_numpy(substrate.optimized_fibers)
        color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list)))
        np.random.shuffle(color)
        '''Choose FOV to plot, can include 3D animation GIF'''
        fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_90') 
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, animation_input=False, optimized=True)
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_0')                    
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_90', animation_input=True)       
        
        along_fiber_plot.plot_along_axon_radius_variation(substrate.optimized_fibers, colors=color)
        along_fiber_plot.plot_diameter_CV_distribution()
        along_fiber_plot.plot_diameter_GEV_distribution(substrate.optimized_fibers)
        orientation_plot.plot_along_axon_OD(substrate.optimized_fibers, optimized=True)
        
        print("----------- Substrate saved in folder:"+ config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+ " -----------")
    else:
        print("Exited unoptimized")
