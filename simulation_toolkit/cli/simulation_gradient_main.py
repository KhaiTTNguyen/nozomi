import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import simulation_toolkit.simulation_engine.geometry as geom
import simulation_toolkit.simulation_engine.diffsim3d as ds3
import simulation_toolkit.simulation_engine.helper.simulation_report as simrep
import simulation_toolkit.simulation_engine.helper.sim_util as sim_util
import simulation_toolkit.utils.common_utils as common_util
import simulation_toolkit.toolkit_params as config_params

import matplotlib.pyplot as pl
import simulation_toolkit.utils.adjust_geometry as ag

from matplotlib.pyplot import cm
import time
import os 

def simulation_gradient_main(params, substrate_file):
    print("Running wide pulse gradient simulation...")
    # # =========== Gather simulation parameters ===========
    # total_sim_time = params['sim_time']
    # time_step = params['time_step']
    # num_spins = int(params['num_spins'])
    # compartment= params['compartment']
    # D0_intra = params['D0_intra']
    # D0_extra = params['D0_extra']
    # nsegx, nsegy, nsegz = params['nseg'], params['nseg'], params['nseg']
    # file_path = substrate_file
    
    # # =========== Prepare output folder ===========
    # target_folder_path = os.path.dirname( os.path.dirname(file_path) )
    # folder_name = os.path.join(target_folder_path, 'sim')
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)
    
    # # =========== Setup simulation ===========
    # optimized_fibers, L = common_util.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
    # config_params.BOX_LENGTH = L
    # fiberlist_xyz_r_fid = common_util.split_matrix_to_list(optimized_fibers)
    # print('box length',config_params.BOX_LENGTH)
    # D = D0_intra # um^2/ms
    # if compartment=='intra':
    #     D = D0_intra # um^2/ms
    # elif compartment=='extra':
    #     D = D0_extra # um^2/ms
    # T2 = 100 # ms
    # rho = 1 # fractional water density
    # sg3 = geom.SimGeometry3D(L,L,L,D,T2,rho)
    # fiber_xyzr_fid_list = ag.createBoundaryZ(fiberlist_xyz_r_fid, L)

    # for fiber in fiber_xyzr_fid_list:
    #     sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
    #     spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
    #     sg3.add_structure(spstruc) # add it to sg3
    
    # nt = int(total_sim_time/time_step) # total number of steps thru time
    # # print('Start setting up structures')
    # sim = ds3.DiffSim3d(sg3,num_spins) 
    # # ========= Pre-compute table that store structures in segments ==========
    # table_st = time.time()
    # print('Num segments', nsegx)
    # sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)
    # if compartment=='intra':
    #     sim.setup(structures=list(np.arange(0, len(fiber_xyzr_fid_list))))  # seed INSIDE structures
    # elif compartment=='extra':
    #     sim.setup(structures=[int(len(fiber_xyzr_fid_list))])  # seed OUTSIDE structures
    # # print('Done setting up structures')
    # table_et = time.time()
    # table_elapsed_time =  np.round(table_et-table_st,2)
    # sim_start_time = time.time()

    # # =========== Pre-allocate GPU result arrays ===========
    # num_steps = int(total_sim_time / time_step) + 1
    # Dx_array, Dy_array, Dz_array = gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32)

    # # =========== Arrays for storing second moments (variance) ===========
    # Kx2_array, Ky2_array, Kz2_array = gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32)
    
    # # =========== Arrays for storing fourth moments ===========
    # Kx4_array, Ky4_array, Kz4_array = gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32)
    
    # # Pre-allocate CPU result arrays
    # Dx_step, Dy_step, Dz_step, diff_time = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
    # # Initial values
    # Dx_step[0], Dy_step[0], Dz_step[0], diff_time[0] = D, D, D, 0

    # print(f'Got to sim {compartment}-axonal simloops')
    # # =========== Simulation loop ===========
    # current_time = 0.0
    # step_idx = 1
    # while current_time < nt*time_step:
    #     # Take time step
    #     sim.step(time_step)
    #     current_time += time_step
        
    #     # Use the GPU kernel to compute displacements
    #     sim.calculate_diffusion_coefficients_and_kurtoses(step_idx, current_time, 
    #                                 Dx_array, Dy_array, Dz_array,
    #                                 Kx2_array, Ky2_array, Kz2_array, 
    #                                 Kx4_array, Ky4_array, Kz4_array)
    #     diff_time[step_idx-1]=current_time
    #     step_idx += 1

    #     # Progress reporting  
    #     if step_idx % 1000 == 0:
    #         progress = (step_idx + 1) / num_steps * 100
    #         print(f"Progress: {progress:.1f}% | Time: {current_time:.3f}s")

    # # After the loop, copy results back to CPU once
    # Dx_step, Dy_step, Dz_step, diff_time = np.array(Dx_array.get())[1:], np.array(Dy_array.get())[1:], np.array(Dz_array.get())[1:], np.array(diff_time)[1:]

    # # Calculate kurtosis using GPU
    # Kx_final_array = Kx4_array / Kx2_array**2 - 3.0
    # Ky_final_array = Ky4_array / Ky2_array**2 - 3.0
    # Kz_final_array = Kz4_array / Kz2_array**2 - 3.0

    # # After the loop, copy results back to CPU once
    # Kx_final = np.array(Kx_final_array.get())[1:]
    # Ky_final = np.array(Ky_final_array.get())[1:]
    # Kz_final = np.array(Kz_final_array.get())[1:]

    # # # get the execution time
    # elapsed_time = np.round(time.time() - sim_start_time,2)

    # #=============== Save coefficient results ===================
    # file_name = os.path.basename(file_path)
    # base_name, extension = os.path.splitext(file_name)
    
    # file_name_new = f'diffcoeff_{compartment}_{str(int(nsegx))}segments_' \
    #     f'{str(num_spins)}spins_{base_name}_TABLEtime{str(table_elapsed_time)}sec_SIMtime{str(round(elapsed_time,2))}sec_timestep{str(time_step)}ms'
    # # Combine new filename with folder path to get the full path
    # data_folder_name = os.path.join(target_folder_path, 'sim', 'ADCdata')
    # if not os.path.exists(data_folder_name):
    #     os.makedirs(data_folder_name)
    # data_file_path = os.path.join(data_folder_name, file_name_new+'.pkl')
    # simrep.save_data_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
    # simrep.plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new )

    # print(f"\nSimulation completed!")
    # # Clean up GPU memory
    # try:
    #     del Dx_array, Dy_array, Dz_array
    #     del Kx2_array, Ky2_array, Kz2_array
    #     del Kx4_array, Ky4_array, Kz4_array
    #     drv.Context.synchronize()
    #     print("GPU memory cleaned up")
    # except Exception as e:
    #     print(f"Warning: Error during cleanup: {e}")

