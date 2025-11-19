import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
# import diffsim3d as ds3
import hipa.diffsim.geometry as geom
import hipa.diffsim.diffsim3d_additional as ds3
import hipa.diffsim.helper.simulation_report as simrep
import hipa.diffsim.helper.plot_gradient_experiment as plt_ge
import hipa.diffsim.waveforms as waveform
import hipa.diffsim.helper.sim_util as sim_util
import matplotlib.pyplot as pl
import hipa.util.util as util
import hipa.util.adjust_geometry as ag

from matplotlib.pyplot import cm
import time
import os 

# @profile
def monte_carlo_sim(substrate_file, total_sim_time, time_step, num_spins, compartment='intra'):
    big_delta = 70.45
    little_delta=2.62

    file_path = substrate_file
    target_folder_path = os.path.dirname( os.path.dirname(file_path) )
    folder_name = os.path.join(target_folder_path, 'sim')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    optimized_fibers, L = util.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
    fiberlist_xyz_r_fid = util.split_matrix_to_list_with_box_length(optimized_fibers, L)
    print('L box length',L)
    D = 3.0 # um^2/ms
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
    num_spins = int(num_spins)
    print('Start setting up structures')
    sim = ds3.DwiSim3d(sg3,num_spins) 
    nsegx,nsegy,nsegz=20,20,20
    sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)
    if compartment=='intra':
        sim.setup(structures=list(np.arange(0, len(fiber_xyzr_fid_list))))  # seed inside structures 0:len(fiberlist)
    elif compartment=='extra':
        sim.setup(structures=[int(len(fiber_xyzr_fid_list))])  # seed OUTSIDE structures len(fiberlist)=index of box
    print('Done setting up structures')

    # after simulating diffusion, are all spins still outside axons
    if compartment=='intra':
        isInside = sim_util.validate_compartment_intra(sim, sg3)
        spins_temp = ((sim.spins_d.get().T)[isInside]).T
        if not isInside:
            print("Error: Not all spins are inside structures")
            # return
    elif compartment=='extra':
        isOutside = sim_util.validate_compartment_extra(sim, sg3)
        spins_temp = ((sim.spins_d.get().T)[isOutside]).T
        if not isOutside:
            print("Error: Not all spins are outside structures")
            # return

    #==========================
    st = time.time()
    # ===================== Make gwave =====================
    '''......... '''
    gwave = waveform.PGDiffWaveform(big_delta=big_delta, little_delta=little_delta, te=total_sim_time)
    bval = np.linspace(0,3,100) # ms/um^2, diffusion b-value 

    phase_sig = sim.simulate(gwave)

    # calculate gradient amplitudes
    b = gwave.calculate_bvalue_from_wave()
    Gmax = np.sqrt(bval/b) # vector of Gmax's
    print('Gmax', Gmax)
    print('phase_sig', phase_sig.shape)
    print('phase_sig', phase_sig)
    print('phase_sig max', max(phase_sig[0, :]))
    print('phase_sig max', max(phase_sig[1, :]))
    print('phase_sig max', max(phase_sig[2, :]))

    # calculate the dwi signal
    dwsig = np.zeros([3,len(Gmax)])
    for n,gmax in enumerate(Gmax):
        dwsig[:,n] =  np.sum(np.cos(gmax*phase_sig),axis=-1) / num_spins
    
    et = time.time()
    # # get the execution time
    elapsed_time = np.round(et - st,2)

    # #===============PUT THESE IN HELPER FILE===================
    file_name = os.path.basename(file_path)
    base_name, extension = os.path.splitext(file_name)
    date_time = str(util.get_date_time())
    # Combine new filename with folder path to get the full path
    data_folder_name = os.path.join(target_folder_path, 'sim', 'gradient')
    if not os.path.exists(data_folder_name):
        os.makedirs(data_folder_name)
    
    plt_ge.plot_waveform(gwave, data_folder_name, big_delta, little_delta)
    save_waveform_object(gwave, data_folder_name)
    
    # gwave = waveform.CosineOGDiffWaveform(
        #     num_oscillations=5,
        #     frequency=192,  # Hz
        #     gmax=1,        # mT/m
        #     te=0.082,         # 100 ms
        #     time_step=time_step
        # )


    for direction, _ in enumerate(dwsig[:,0]):
        if direction == 0:
            direction_tag='x_'
        elif direction == 1:
            direction_tag='y_'
        else:
            direction_tag='z_'
        data_file_path = os.path.join(data_folder_name, direction_tag+'DWI_vs_b_big_delta_'+str(int(big_delta))+'_little_delta_'+str(int(little_delta))+str(date_time))
        
        if not os.path.exists(data_file_path):
            os.makedirs(data_file_path)
        plt_ge.plot_log_dwsig_vs_bval(bval, np.log(dwsig[direction,:]),data_file_path, compartment)
        
        
    # after simulating diffusion, are all spins still outside axons
    if compartment=='intra':
        isInside = sim_util.validate_compartment_intra(sim, sg3)
        spins_temp = ((sim.spins_d.get().T)[isInside]).T
        if not isInside:
            print("Error: Not all spins are inside structures")
            return
    elif compartment=='extra':
        isOutside = sim_util.validate_compartment_extra(sim, sg3)
        spins_temp = ((sim.spins_d.get().T)[isOutside]).T
        if not isOutside:
            print("Error: Not all spins are outside structures")
            return
    print('elapsed_time', elapsed_time)
    
    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    # ax.view_init(azim=90, elev=0)
    ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
    k=0
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_zlim(-L/2, L/2)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("z (um)")
    spin_plot_file_name = str(date_time)+'_'+'_outside.png'
    spin_plot_file_name = os.path.join(folder_name, spin_plot_file_name)
    pl.savefig(spin_plot_file_name)
