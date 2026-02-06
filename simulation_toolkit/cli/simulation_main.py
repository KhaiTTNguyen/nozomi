import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import simulation_toolkit.simulation_engine.geometry as geom
import simulation_toolkit.simulation_engine.diffsim3d as ds3
import simulation_toolkit.simulation_engine.helper.simulation_report as simrep
import simulation_toolkit.simulation_engine.helper.sim_util as sim_util
import simulation_toolkit.utils.common_utils as common_util

import matplotlib.pyplot as pl
import simulation_toolkit.utils.adjust_geometry as ag

from matplotlib.pyplot import cm
import time
import os 

def simulation_main(params, substrate_file):
    total_sim_time = params['sim_time']
    time_step = params['time_step']
    num_spins = params['num_spins']
    compartment= params['compartment']     
    file_path = substrate_file
    target_folder_path = os.path.dirname( os.path.dirname(file_path) )
    folder_name = os.path.join(target_folder_path, 'sim')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    optimized_fibers, L = common_util.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
    fiberlist_xyz_r_fid = common_util.split_matrix_to_list_with_box_length(optimized_fibers, L)
    print('box length',L)
    D = 2.25 # um^2/ms
    if compartment=='intra':
        D = 2.25 # um^2/ms
    elif compartment=='extra':
        D = 2.0 # um^2/ms
    T2 = 100 # ms
    rho = 1 # fractional water density
    sg3 = geom.SimGeometry3D(L,L,L,D,T2,rho)
    fiber_xyzr_fid_list = ag.createBoundaryZ(fiberlist_xyz_r_fid, L)

    for fiber in fiber_xyzr_fid_list:
        sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
        spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
        sg3.add_structure(spstruc) # add it to sg3
    dt = time_step # time step in ms
    nt = int(total_sim_time/dt) # total number of steps thru time
    num_spins = int(num_spins)
    print('Start setting up structures')
    sim = ds3.DiffSim3d(sg3,num_spins) 
    nsegx,nsegy,nsegz=20,20,20  # set number of segments
    table_st = time.time()
    sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)
    if compartment=='intra':
        sim.setup(structures=list(np.arange(0, len(fiber_xyzr_fid_list))))  # seed inside structures 0:len(fiberlist)
    elif compartment=='extra':
        sim.setup(structures=[int(len(fiber_xyzr_fid_list))])  # seed OUTSIDE structures len(fiberlist)=index of box
    print('Done setting up structures')
    table_et = time.time()
    table_elapsed_time =  np.round(table_et-table_st,2)
    start_time = time.time()

    #==========================
    # Pre-allocate GPU result arrays
    num_steps = int(total_sim_time / dt) + 1
    Dx_array, Dy_array, Dz_array = gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32)

    # Arrays for storing second moments (variance)
    Kx2_array, Ky2_array, Kz2_array = gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32)
    
    # # Arrays for storing fourth moments
    Kx4_array, Ky4_array, Kz4_array = gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32), gpuarray.zeros(num_steps, dtype=np.float32)
    
    # Pre-allocate CPU result arrays
    Dx_step, Dy_step, Dz_step, diff_time = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
    # Initial values
    Dx_step[0], Dy_step[0], Dz_step[0], diff_time[0] = D, D, D, 0

    print(f'Got to sim {compartment}-axonal simloops')
    current_time = 0.0
    step_idx = 1
    while current_time < nt*dt:
        # Take time step
        sim.step(time_step)
        current_time += time_step
        
        # Use the GPU kernel to compute displacements
        sim.calculate_diffusion_coefficients_and_kurtoses(step_idx, current_time, 
                                    Dx_array, Dy_array, Dz_array,
                                    Kx2_array, Ky2_array, Kz2_array, 
                                    Kx4_array, Ky4_array, Kz4_array)
        diff_time[step_idx-1]=current_time
        step_idx += 1

        # Progress reporting  
        if step_idx % 1000 == 0:
            progress = (step_idx + 1) / num_steps * 100
            print(f"Progress: {progress:.1f}% | Time: {current_time:.3f}s")

    # After the loop, copy results back to CPU once
    Dx_step, Dy_step, Dz_step, diff_time = np.array(Dx_array.get())[1:], np.array(Dy_array.get())[1:], np.array(Dz_array.get())[1:], np.array(diff_time)[1:]

    # Calculate kurtosis using GPU
    Kx_final_array = Kx4_array / Kx2_array**2 - 3.0
    Ky_final_array = Ky4_array / Ky2_array**2 - 3.0
    Kz_final_array = Kz4_array / Kz2_array**2 - 3.0

    # After the loop, copy results back to CPU once
    Kx_final = np.array(Kx_final_array.get())[1:]
    Ky_final = np.array(Ky_final_array.get())[1:]
    Kz_final = np.array(Kz_final_array.get())[1:]

    # # get the execution time
    elapsed_time = np.round(time.time() - start_time,2)

    #===============PUT THESE IN HELPER FILE===================
    #=============== Save coefficient results ===================
    file_name = os.path.basename(file_path)
    base_name, extension = os.path.splitext(file_name)
    date_time = str(common_util.get_date_time())

    file_name_new = f'DiffCoeff_{compartment}_{date_time}_{str(len(fiber_xyzr_fid_list))}_fibers_' \
        f'{str(num_spins)}_spins_{base_name}_TABLEtime{str(table_elapsed_time)}sec_SIMtime{str(round(elapsed_time,2))}_sec_dt{str(dt)}_seg{str(int(nsegx))}'
    # Combine new filename with folder path to get the full path
    data_folder_name = os.path.join(target_folder_path, 'sim', 'ADCdata')
    if not os.path.exists(data_folder_name):
        os.makedirs(data_folder_name)
    data_file_path = os.path.join(data_folder_name, str(int(nsegx))+'SEGMENT_'+file_name_new+'data.pkl')
    simrep.save_data_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
    simrep.plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new )

    # ================== Save kurtosis results ==================
    file_name_new = f'FINAL_kurtosis_{compartment}_{date_time}_{str(len(fiber_xyzr_fid_list))}_fibers_' \
        f'{str(num_spins)}_spins_{base_name}_TABLEtime{str(table_elapsed_time)}sec_SIMtime{str(round(elapsed_time,2))}sec_dt{str(dt)}_SEGMENT{str(int(nsegx))}'
    # Combine new filename with folder path to get the full path
    data_folder_name = os.path.join(target_folder_path, 'sim', 'kurtosis_data')
    if not os.path.exists(data_folder_name):
        os.makedirs(data_folder_name)
    data_file_path = os.path.join(data_folder_name, file_name_new+'data.pkl')    
    simrep.save_data_pickle(data_file_path, np.column_stack((Kx_final, Ky_final, Kz_final, diff_time)))
    simrep.plot_K_vs_time(diff_time, Kx_final, Ky_final, Kz_final, folder_name, file_name_new )    

    print(f"\nSimulation completed!")
    # Clean up GPU memory
    try:
        del Dx_array, Dy_array, Dz_array
        del Kx2_array, Ky2_array, Kz2_array
        del Kx4_array, Ky4_array, Kz4_array
        drv.Context.synchronize()
        print("GPU memory cleaned up")
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

