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
    # The intra-axonal diffusivity was 2.25 μm2/ms in the simulations, as found out in a previous brain WM study invivo[29]. The extra-axonal diffusivity was 2 μm2/ms [28],
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
    nsegx,nsegy,nsegz=20,20,20
    sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)
    if compartment=='intra':
        sim.setup(structures=list(np.arange(0, len(fiber_xyzr_fid_list))))  # seed inside structures 0:len(fiberlist)
    elif compartment=='extra':
        sim.setup(structures=[int(len(fiber_xyzr_fid_list))])  # seed OUTSIDE structures len(fiberlist)=index of box
    print('Done setting up structures')

    # # Check spin seeding before simulation
    # if compartment=='intra':
    #     isInside = sim_util.validate_compartment_intra(sim, sg3)
    #     # spins_temp = ((sim.spins_d.get().T)[isInside]).T
    #     if not isInside:
    #         print("Error: Not all spins are inside structures")
    #         # return
    # elif compartment=='extra':
    #     isOutside = sim_util.validate_compartment_extra(sim, sg3)
    #     # spins_temp = ((sim.spins_d.get().T)[isOutside]).T
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
    st = time.time()
    num_steps = int(total_sim_time / dt) + 1
    dx_array_gpu = gpuarray.zeros(num_steps, dtype=np.float32)
    dy_array_gpu = gpuarray.zeros(num_steps, dtype=np.float32)
    dz_array_gpu = gpuarray.zeros(num_steps, dtype=np.float32)
    count_array_gpu = gpuarray.zeros(num_steps, dtype=np.int32)
    
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
        # Calculate displacements and counts for central region
        sim.calculate_displacements_central(step_idx, current_time, dx_array_gpu, dy_array_gpu, dz_array_gpu, count_array_gpu)
        # print('step_idx',step_idx-1, 'current_time', current_time)
        diff_time[step_idx-1]=current_time
        step_idx += 1

    # After the loop, copy results back to CPU once
    dx_raw = dx_array_gpu.get()
    dy_raw = dy_array_gpu.get()
    dz_raw = dz_array_gpu.get()
    central_counts = count_array_gpu.get()
    for i in range(1, step_idx):
        if central_counts[i] > 0 and diff_time[i] > 0:
            # D = <r²>/(2t) for each direction
            Dx_step[i] = dx_raw[i] / (2.0 * diff_time[i] * central_counts[i])
            Dy_step[i] = dy_raw[i] / (2.0 * diff_time[i] * central_counts[i])
            Dz_step[i] = dz_raw[i] / (2.0 * diff_time[i] * central_counts[i])
    Dx_step = np.array(Dx_step)
    Dy_step = np.array(Dy_step)
    Dz_step = np.array(Dz_step)
    diff_time = np.array(diff_time)
    et = time.time()
    # # get the execution time
    elapsed_time = np.round(et - st,2)

    # #===============PUT THESE IN HELPER FILE===================
    file_name = os.path.basename(file_path)
    base_name, extension = os.path.splitext(file_name)
    date_time = str(util.get_date_time())
    file_name_new = f'OPT_central_{compartment}_{date_time}_{str(len(fiber_xyzr_fid_list))}_fibers_' \
        f'{str(num_spins)}_spins_{base_name}_SIMtime{str(round(elapsed_time,2))}_sec_dt{str(dt)}_seg{str(int(nsegx))}'
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
    #         # return
    # elif compartment=='extra':
    #     isOutside = sim_util.validate_compartment_extra(sim, sg3)
    #     spins_temp = ((sim.spins_d.get().T)[isOutside]).T
    #     if not isOutside:
    #         print("Error: Not all spins are outside structures")
    #         # return
    
   