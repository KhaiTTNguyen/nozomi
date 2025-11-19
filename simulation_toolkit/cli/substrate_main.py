from simulation_toolkit.geometry_generator import initialization_2d
from simulation_toolkit.geometry_generator import meshing
from simulation_toolkit.geometry_generator import optimization
import nozomi.simulation_toolkit.defaults.params as params
import numpy as np
import time

def substrate_main(params, experiment_folder):
    params.ORIENTATION_SHAPE_PARAM = params['orientation_shape_parameter'] # #7.5 ODI 0.08 # and 4.5 ODI 0.13
    box_length_init = params['box_length_init']
    target_volume_fraction = params['target_volume_fraction']
    params.NUM_FIBERS = params['num_fibers']
    params.MEAN_DIAMETER = params['mean_diameter']   
    params.SIGMA_DIAMETER = params['sigma_diameter']   
    params.DISTRIBUTION_SHAPE = params['dist_shape']   
    params.SPACE_BUFFER_STARTS_ENDS = params['space_buffer_starts_ends']   
    params.SPHERE_SPACING = params['spheres_spacing']   
    params.SPACE_BUFFER_REPULSE = params['space_buffer_repulse']  
    params.W_OVERLAP = params['w_overlap'] 
    params.W_CURVE = params['w_curve']
    params.W_LENGTH = params['w_length']
    params.BEAD_SPACING_MEAN = params['bead_spacing_mean']   
    params.BEAD_SPACING_STDV = params['bead_spacing_stdv']
    params.DEPTH_MULTIPLIER = params['depth_multiplier']   
    print(f"Building substrate with: \
          orientation_shape_parameter={params.ORIENTATION_SHAPE_PARAM}, \
          num_fibers={params.NUM_FIBERS}, \
        diameter_mean={params.MEAN_DIAMETER},\
        diameter_stdv={params.SIGMA_DIAMETER}")
    

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    params.EXP_DATE_TIME = util.get_date_time()        
    params.ODI_INDEX = np.round(2/np.pi * np.arctan(1/params.ORIENTATION_SHAPE_PARAM), 4)
    params.SUBSTRATE_OUTPUT_FOLDER_PATH = os.path.join(params.OUTPUT_FOLDER_PATH, experiment_folder, \
                                                       str(params.EXP_DATE_TIME)+\
                                                        '_d'+str(params.MEAN_DIAMETER)+\
                                                        '_K'+str(int(params.ORIENTATION_SHAPE_PARAM))+\
                                                    '_ODI_'+str(params.ODI_INDEX)+'_'+\
                                                        str(params.NUM_FIBERS) +'fibers')
    not_converged=True
    while not_converged:
        st = time.time()
        initialization2D = Init2D(  device=device,
                                    date_time=params.EXP_DATE_TIME,
                                    orientation_shape_parameter=params.ORIENTATION_SHAPE_PARAM, 
                                    target_volume_fraction=target_volume_fraction, 
                                    num_fibers=params.NUM_FIBERS,
                                    dist_shape=params.DISTRIBUTION_SHAPE,
                                    mean_diameter=params.MEAN_DIAMETER, 
                                    sigma_radii=params.SIGMA_DIAMETER,
                                    space_buffer=params.SPACE_BUFFER_STARTS_ENDS,
                                    box_length_init=box_length_init)
        
        # data_folder = os.path.join(config.SUBSTRATE_OUTPUT_FOLDER_PATH,'data')
        # if not os.path.exists(data_folder):
        #             os.makedirs(data_folder)

        init2d_data_folder = os.path.join(params.SUBSTRATE_OUTPUT_FOLDER_PATH,'figs','init2D')
        if not os.path.exists(init2d_data_folder):
                    os.makedirs(init2d_data_folder)

        # # save data with Pickle format
        init2d_data_file_name = os.path.join(init2d_data_folder ,'init2d.pkl')
        util.save_data_array_to_pickle(init2d_data_file_name, initialization2D.initial_positions, 
                                initialization2D.box_length.cpu().item()) 
        

        initialization2D.plot_PBC()
        meshing = Meshing(initialization2D, params.SPHERE_SPACING, params.BEAD_SPACING_MEAN, params.BEAD_SPACING_STDV, device)


        substrate = OverlapRemoval(device, meshing, params.SPHERE_SPACING, 
                                              params.W_OVERLAP, params.W_CURVE, params.W_LENGTH, \
                                                params.SPACE_BUFFER_REPULSE, initialization2D.mean_d_underlying, initialization2D.sigma_d_underlying,\
                                                params.BEAD_SPACING_MEAN, params.BEAD_SPACING_STDV)
        not_converged = not substrate.optimized
    if (substrate.optimized == True):

        # get the execution time
        et = time.time()
        elapsed_time = et - st
        
        file_name_root = 'array'+str(params.NUM_FIBERS) +'_fibers_boxL_'+str(params.BOX_LENGTH.item())+'_'+str(params.EXP_DATE_TIME) + \
                        '_avf_'+str(params.VOLUME_FRACTION)+\
                        '_d'+str(params.MEAN_DIAMETER)+'_sig'+str(params.SIGMA_DIAMETER)+\
                            'wo'+str(params.W_OVERLAP)+'_wc'+str(params.W_CURVE)+'_wl'+str(params.W_LENGTH)+'_K'+str(int(params.ORIENTATION_SHAPE_PARAM))+\
                            '_ODI_'+str(params.ODI_INDEX)+'_'+\
                                str(round(elapsed_time,2))+'_sec'+'.pkl'
        
        data_folder = os.path.join(params.SUBSTRATE_OUTPUT_FOLDER_PATH,'data')
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
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, animation_input=False, optimized=True)
        fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_90') 
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_0')                    
        
        # fiber_3D_plot.plot_fibers(substrate.optimized_fibers, overlap_indices=None, color=color, optimized=True, POV='horizontal_90', animation_input=True)       
        
        '''UNCOMMENT THIS PART AFTER TESTING'''
        # along_fiber_plot.plot_along_axon_radius_variation(substrate.optimized_fibers, colors=color)
        # along_fiber_plot.plot_diameter_CV_distribution()
        # # along_fiber_plot.plot_bead_spacing_distribution()
        # along_fiber_plot.plot_diameter_GEV_distribution(substrate.optimized_fibers)
        # # along_fiber_plot.plot_slices(, substrate.optimized_fibers, N=5, color=color)
        # orientation_plot.plot_along_axon_OD(substrate.optimized_fibers, optimized=True)
        
        print("----------- Substrate saved in folder:"+ params.SUBSTRATE_OUTPUT_FOLDER_PATH+ " -----------")
    else:
        print("Exited unoptimized")
