# Legacy test06 residual coverage: sphere-based multi-direction diffusion validation.
# import pycuda.autoinit
# import pycuda.driver as drv
# import pycuda.gpuarray as gpuarray
# import numpy as np
# # import diffsim3d as ds3
# import hipa.diffsim.geometry as geom
# import hipa.diffsim.diffsim3d_additional as ds3
# import hipa.diffsim.waveforms as waveform
# import hipa.diffsim.helper.simulation_report as simrep
# import hipa.diffsim.helper.plot_gradient_experiment as plt_ge
# import hipa.diffsim.helper.sim_util as sim_util
# import matplotlib.pyplot as plt
# import hipa.util.util as util
# import hipa.util.adjust_geometry as ag
# from datetime import datetime
# from matplotlib.pyplot import cm
# import time
# import os 
# import pickle
# import hipa.diffsim.helper.waveform_io as waveform_io

# def testInside2BoundarySpheres():
#         '''
#         diffusion outside of multiple spheres
#         '''
#         compartment = 'intra'
#         time_step=0.0001 # ms
#         # total_sim_time=0.082 # OGSE
#         total_sim_time=17 # PGSE
        
#         big_delta = 12
#         little_delta= 3
#         spins = int(1e5)
#         diffdir_file='<REPO_ROOT>/HIPASimExperiment/biophysical_sim/NODDI/gradient/DWI_10dir.bvec'
#         folder_date_time = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
#         st = time.time()
#         Lx = 20.0 # um
#         Ly = 20.0 # um
#         Lz = 20.0 # um
#         D = 3.0 # um^2/ms
#         T2 = 200 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         sz = np.zeros(1)
#         sy = np.ones(1)*10
#         sx = np.zeros(1)
#         sr = np.ones(1)*5
#         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
 
#         sz2 = np.zeros(1)
#         sy2 = np.ones(1)*(-10)
#         sx2 = np.zeros(1)
#         sr2 = np.ones(1)*5
#         spstruc = geom.Structure3D(sx2,sy2,sz2,sr2,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
 
#         #================================================
#         print('sg3.nstructures',sg3.nstructures)
#         print('sg3.nstructures',sg3.nsphere)
        
#         num_spins=spins
#         sim = ds3.DwiSim3d(sg3,spins)
#         diffdir=waveform_io.load_bvec_file_for_gradient_directions(diffdir_file)
#         sim.set_diffusion_directions(diffdir=diffdir)
#         nsegx,nsegy,nsegz=2,2,2
#         sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)

#         print('Start setting up structures')
#         if compartment=='intra':
#             sim.setup(structures=list(np.arange(0,2))) 
#             # sim.setup(structures=list(np.arange(0, 2)))  # seed inside structures 0:len(fiberlist)

#         # first, check thath all the spins are in fact in spstruc
#         if compartment=='intra':
#             isInside = sim_util.validate_compartment_intra(sim, sg3)
#             spins_temp = ((sim.spins_d.get().T)[isInside]).T
#             if not isInside:
#                 print("Error: Not all spins are inside structures")
#         elif compartment=='extra':
#             isOutside = sim_util.validate_compartment_extra(sim, sg3)
#             spins_temp = ((sim.spins_d.get().T)[isOutside]).T
#             if not isOutside:
#                 print("Error: Not all spins are outside structures")

#         # plot after seed
#         spins_temp = sim.spins_d.get()
#         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#         # ax.view_init(azim=0, elev=90)
#         ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
#         k=0
#         ax.set_xlim(-Lx/2, Lx/2)
#         ax.set_ylim(-Ly/2, Ly/2)
#         ax.set_zlim(-Lz/2, Lz/2)
#         ax.set_xlabel("x (um)")
#         ax.set_ylabel("y (um)")
#         ax.set_zlabel("z (um)")
#         spin_plot_file_name = 'spins_2boundarysphere.png'
#         output_folder = "./hipa/tests/gradient_sim_multi_directions"
#         os.makedirs(os.path.join(output_folder,folder_date_time), exist_ok=True)
#         spin_plot_file_name = os.path.join(output_folder, folder_date_time, spin_plot_file_name)
#         plt.savefig(spin_plot_file_name)
#         # exit()

#         # ===================== Make gwave =====================
#         '''......... '''
#         gwave = waveform.PGDiffWaveform(big_delta=big_delta, little_delta=little_delta, te=total_sim_time)
#         # Create an apodised oscillating gradient with 4 cycles at 50 Hz
#         # Create a trapezoidal-cosine oscillating gradient with:

#         # Create the waveform
#         # gwave = waveform.CosineOGDiffWaveform(
#         #     num_oscillations=5,
#         #     frequency=192,  # Hz
#         #     gmax=1,        # mT/m
#         #     te=0.082,
#         #     time_step=time_step
#         # )

#         # gwave = waveform.ApodisedOGDiffWaveform(freq=50, n_cycles=4, apodisation='gaussian', te=0.1)
#         gradient_file_name = os.path.join(output_folder, folder_date_time, str(gwave.__class__.__name__)+"_")
#         waveform_io.save_waveform_object(gwave, gradient_file_name)
#         gwave2 = waveform_io.load_waveform_object(gradient_file_name+'.pkl')
#         plt_ge.plot_wave(gwave2, os.path.join(output_folder, folder_date_time))

#         bval = np.linspace(0,3,100) # ms/um^2, diffusion b-value 

#         phase_sig = sim.simulate_multi_directions(gwave2)
#         pkl_file_name = os.path.join(gradient_file_name+'multidirr_signal.pkl')
#         with open(pkl_file_name, 'wb') as f:
#             pickle.dump(phase_sig, f, protocol=pickle.HIGHEST_PROTOCOL)
#         print('Done saving signal data!')

#          # save result for later use
#         # calculate gradient amplitudes
#         b = gwave.calculate_bvalue_from_wave()
#         Gmax = np.sqrt(bval/b) # vector of Gmax's
#         # print('Gmax', Gmax)
#         # print('phase_sig', phase_sig.shape)
#         # print('phase_sig', phase_sig)
#         # print('phase_sig max', max(phase_sig[0, :]))
#         # print('phase_sig max', max(phase_sig[1, :]))
#         # print('phase_sig max', max(phase_sig[2, :]))

#         # calculate the dwi signal
#         dwsig = np.zeros([diffdir.shape[1],len(Gmax)])
#         # print(dwsig.shape)
#         for n,gmax in enumerate(Gmax):
#             dwsig[:,n] =  np.sum(np.cos(gmax*phase_sig),axis=-1) / num_spins
        
#         fig, ax = plt.subplots()
#         for dir in range(0,diffdir.shape[1]):
#             # plt_ge.plot_dwsig_vs_bval(bval,dwsig[dir,:],os.path.join(gradient_file_name+'_dir'+str(dir)+'.png'))
#             plt_ge.save_gradient_data(dwsig[dir,:], os.path.join(gradient_file_name+'_dir'+str(dir)+'.pkl'))
#             plt.plot(bval, dwsig[dir,:], label='dir_'+str(dir),markersize=10)
#         plt.grid(True)
#         ax.set_xlabel("b-value (ms/um^2)")
#         ax.set_ylabel("DWI signal")
#         ax.set_title('DWI signal vs. b-value for multiple directions')
#         plt.legend()
#         plt.grid(True, linestyle="--", alpha=0.5)

#         # Tight layout and show/save
#         plt.tight_layout()
#         plt.savefig(os.path.join(gradient_file_name+'multiple directions_dir'+str(diffdir.shape[1])+'.png'))

#         et = time.time()
#         elapsed_time = np.round(et - st,2)

        
#         #-------------------------------------------
#         if compartment=='intra':
#             isInside = sim_util.validate_compartment_intra(sim, sg3)
#             spins_temp = ((sim.spins_d.get().T)[isInside]).T
#             if not isInside:
#                 print("Error: Not all spins are inside structures")
#         elif compartment=='extra':
#             isOutside = sim_util.validate_compartment_extra(sim, sg3)
#             spins_temp = ((sim.spins_d.get().T)[isOutside]).T
#             if not isOutside:
#                 print("Error: Not all spins are outside structures")
#         print('elapsed_time', elapsed_time)

# testInside2BoundarySpheres()