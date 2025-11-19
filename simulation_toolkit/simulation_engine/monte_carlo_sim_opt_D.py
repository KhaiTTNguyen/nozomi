import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
# import diffsim3d as ds3
import hipa.diffsim.geometry as geom
import hipa.diffsim.diffsim3d_additional as ds3
import hipa.diffsim.helper.simulation_report as simrep
import hipa.diffsim.helper.sim_util as sim_util
import matplotlib.pyplot as pl
import hipa.util.util as util
import hipa.util.adjust_geometry as ag

from matplotlib.pyplot import cm
import time
import os 

# @profile
def monte_carlo_sim(substrate_file, total_sim_time, time_step, num_spins, compartment='intra'):
    file_path = substrate_file
    target_folder_path = os.path.dirname( os.path.dirname(file_path) )
    folder_name = os.path.join(target_folder_path, 'sim')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    optimized_fibers, L = util.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
    fiberlist_xyz_r_fid = util.split_matrix_to_list_with_box_length(optimized_fibers, L)
    print('L box length',L)
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
    # # nt = int(6)
    nt = int(total_sim_time/dt) # total number of steps thru time
    # spins = int(1e3)
    # spins = int(1e3)
    num_spins = int(num_spins)
    print('Start setting up structures')
    sim = ds3.DiffSim3d(sg3,num_spins) 
    nsegx,nsegy,nsegz=20,20,20
    print('nsegx', nsegx)
    table_st = time.time()
    sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)
    if compartment=='intra':
        sim.setup(structures=list(np.arange(0, len(fiber_xyzr_fid_list))))  # seed inside structures 0:len(fiberlist)
    elif compartment=='extra':
        sim.setup(structures=[int(len(fiber_xyzr_fid_list))])  # seed OUTSIDE structures len(fiberlist)=index of box
    print('Done setting up structures')
    table_et = time.time()
    table_elapsed_time =  np.round(table_et-table_st,2)
    st = time.time()

    # after simulating diffusion, are all spins still outside axons
    # if compartment=='intra':
    #     isInside = sim_util.validate_compartment_intra(sim, sg3)
    #     spins_temp = ((sim.spins_d.get().T)[isInside]).T
    #     if not isInside:
    #         print("Error: Not all spins are inside structures")
    #         # return
    # elif compartment=='extra':
    #     isOutside = sim_util.validate_compartment_extra(sim, sg3)
    #     spins_temp = ((sim.spins_d.get().T)[isOutside]).T
    #     if not isOutside:
    #         print("Error: Not all spins are outside structures")
    #         # return
    
    # spins_temp = sim.spins_d.get()
    # fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    # ax.view_init(azim=0, elev=90)
    # ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
    # k=0
    # ax.set_xlim(-L/2, L/2)
    # ax.set_ylim(-L/2, L/2)
    # ax.set_zlim(-L/2, L/2)
    # ax.set_xlabel("x (um)")
    # ax.set_ylabel("y (um)")
    # ax.set_zlabel("z (um)")
    # spin_plot_file_name = 'front_outside.png'
    # spin_plot_file_name = os.path.join(folder_name, spin_plot_file_name)
    # pl.savefig(spin_plot_file_name)

    #==========================

    num_steps = int(total_sim_time / dt) + 1
    Dx_array = gpuarray.zeros(num_steps, dtype=np.float32)
    Dy_array = gpuarray.zeros(num_steps, dtype=np.float32)
    Dz_array = gpuarray.zeros(num_steps, dtype=np.float32)

    # Pre-allocate CPU result arrays
    Dx_step, Dy_step, Dz_step, diff_time = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
    print('diff_time.shape', diff_time.shape)
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
        sim.calculate_displacements(step_idx, current_time, Dx_array, Dy_array, Dz_array)
        diff_time[step_idx-1]=current_time
        step_idx += 1

    # After the loop, copy results back to CPU once
    Dx_step = np.array(Dx_array.get())[1:]
    Dy_step = np.array(Dy_array.get())[1:]
    Dz_step = np.array(Dz_array.get())[1:]
    diff_time = np.array(diff_time)[1:]
    et = time.time()
    # # get the execution time
    elapsed_time = np.round(et - st,2)

    # #===============PUT THESE IN HELPER FILE===================
    file_name = os.path.basename(file_path)
    base_name, extension = os.path.splitext(file_name)
    date_time = str(util.get_date_time())
    file_name_new = f'OPT_{compartment}_{date_time}_{str(len(fiber_xyzr_fid_list))}_fibers_' \
        f'{str(num_spins)}_spins_{base_name}_TABLEtime{str(table_elapsed_time)}sec_SIMtime{str(round(elapsed_time,2))}_sec_dt{str(dt)}_seg{str(int(nsegx))}'
    # print('file_name_new', file_name_new)
    # Combine new filename with folder path to get the full path
    data_folder_name = os.path.join(target_folder_path, 'sim', 'ADCdata')
    if not os.path.exists(data_folder_name):
        os.makedirs(data_folder_name)
    data_file_path = os.path.join(data_folder_name, str(int(nsegx))+'SEGMENT_'+file_name_new+'data.pkl')
    simrep.save_data_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
    simrep.plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new )    

    # after simulating diffusion, are all spins still outside axons
    # if compartment=='intra':
    #     isInside = sim_util.validate_compartment_intra(sim, sg3)
    #     spins_temp = ((sim.spins_d.get().T)[isInside]).T
    #     if not isInside:
    #         print("Error: Not all spins are inside structures")
    #         return
    # elif compartment=='extra':
    #     isOutside = sim_util.validate_compartment_extra(sim, sg3)
    #     spins_temp = ((sim.spins_d.get().T)[isOutside]).T
    #     if not isOutside:
    #         print("Error: Not all spins are outside structures")
    #         return
    # print('elapsed_time', elapsed_time)
    
    # fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    # # ax.view_init(azim=90, elev=0)
    # ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
    # k=0
    # ax.set_xlim(-L/2, L/2)
    # ax.set_ylim(-L/2, L/2)
    # ax.set_zlim(-L/2, L/2)
    # ax.set_xlabel("x (um)")
    # ax.set_ylabel("y (um)")
    # ax.set_zlabel("z (um)")
    # spin_plot_file_name = str(date_time)+'_'+'_outside.png'
    # spin_plot_file_name = os.path.join(folder_name, spin_plot_file_name)
    # pl.savefig(spin_plot_file_name)