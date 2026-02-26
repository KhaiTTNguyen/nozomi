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
# import matplotlib.pyplot as pl
# import hipa.util.util as util
# import hipa.util.adjust_geometry as ag
# from datetime import datetime
# from matplotlib.pyplot import cm
# import time
# import os 
# import hipa.diffsim.helper.waveform_io as waveform_io

# def testOutsideTwoSetOfMultipleSpheres():
#         '''
#         diffusion outside of multiple spheres
#         '''
#         compartment = 'intra'
#         time_step=0.0001 # ms
#         total_sim_time=51
#         big_delta = 50
#         little_delta=0.0000001
#         spins = int(1e5)
#         diffdir_file='/home/nguyt16@ds.vanderbilt.edu/HIPASimExperiment/biophysical_sim/NODDI/gradient/DWI.bvec'
        
#         st = time.time()
#         Lx = 60.0 # um
#         Ly = 60.0 # um
#         Lz = 60.0 # um
#         D = 3.0 # um^2/ms
#         T2 = 200 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         nsphere = 100
#         sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         sy = np.full(sz.shape, 1)*10
#         sx = np.full(sz.shape, 0)
#         sr = np.full(sz.shape, 5)
#         sz1 = np.concatenate((sz[-30:]-Lz, sz, sz[:30]+Lz ), axis=0)
#         sy1 = np.concatenate((sy[-30:], sy, sy[:30] ), axis=0)
#         sx1 = np.concatenate((sx[-30:], sx, sx[:30] ), axis=0)
#         sr1 = np.concatenate((sr[-30:], sr, sr[:30] ), axis=0)
#         spstruc = geom.Structure3D(sx1,sy1,sz1,sr1,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3

#         nsphere = 100
#         sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         sy = np.full(sz.shape, 1)*(-10)
#         sx = np.full(sz.shape, 0)
#         sr = np.full(sz.shape, 5.)
#         sz2 = np.concatenate((sz[-30:]-Lz, sz, sz[:30]+Lz ), axis=0)
#         sy2 = np.concatenate((sy[-30:], sy, sy[:30] ), axis=0)
#         sx2 = np.concatenate((sx[-30:], sx, sx[:30] ), axis=0)
#         sr2 = np.concatenate((sr[-30:], sr, sr[:30] ), axis=0)
#         spstruc = geom.Structure3D(sx2,sy2,sz2,sr2,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3

#         nsphere = 100
#         sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         sx = np.full(sz.shape, 1)*(-10)
#         sy = np.full(sz.shape, 0)
#         sr = np.full(sz.shape, 5.)
#         sz3 = np.concatenate((sz[-30:]-Lz, sz, sz[:30]+Lz ), axis=0)
#         sy3 = np.concatenate((sy[-30:], sy, sy[:30] ), axis=0)
#         sx3 = np.concatenate((sx[-30:], sx, sx[:30] ), axis=0)
#         sr3 = np.concatenate((sr[-30:], sr, sr[:30] ), axis=0)
#         spstruc = geom.Structure3D(sx3,sy3,sz3,sr3,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
        
#         nsphere = 100
#         sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         sx = np.full(sz.shape, 1)*(10)
#         sy = np.full(sz.shape, 0)
#         sr = np.full(sz.shape, 5.)
#         sz4 = np.concatenate((sz[-30:]-Lz, sz, sz[:30]+Lz ), axis=0)
#         sy4 = np.concatenate((sy[-30:], sy, sy[:30] ), axis=0)
#         sx4 = np.concatenate((sx[-30:], sx, sx[:30] ), axis=0)
#         sr4 = np.concatenate((sr[-30:], sr, sr[:30] ), axis=0)
#         spstruc = geom.Structure3D(sx4,sy4,sz4,sr4,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
        
#         #================================================
#         print('sg3.nstructures',sg3.nstructures)
#         print('sg3.nstructures',sg3.nsphere)
        
#         num_spins=spins
#         sim = ds3.DwiSim3d(sg3,spins)
#         diffdir=waveform_io.load_bvec_file_for_gradient_directions(diffdir_file)
#         sim.set_diffusion_directions(diffdir=diffdir)
#         nsegx,nsegy,nsegz=20,20,20
#         sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)

#         print('Start setting up structures')
#         if compartment=='intra':
#             sim.setup(structures=list(np.arange(0,4))) 
#             # sim.setup(structures=list(np.arange(0, 2)))  # seed inside structures 0:len(fiberlist)
#         elif compartment=='extra':
#             sim.setup(structures=[4])  # seed OUTSIDE structures len(fiberlist)=index of box
#         print('Done setting up structures')
        
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
#         # spins_temp = sim.spins_d.get()
#         # fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#         # ax.view_init(azim=0, elev=90)
#         # ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
#         # k=0
#         # ax.set_xlim(-Lx/2, Lx/2)
#         # ax.set_ylim(-Ly/2, Ly/2)
#         # ax.set_zlim(-Lz/2, Lz/2)
#         # ax.set_xlabel("x (um)")
#         # ax.set_ylabel("y (um)")
#         # ax.set_zlabel("z (um)")
        
#         output_folder = "./hipa/tests/gradients"
#         os.makedirs(output_folder, exist_ok=True)
#         # ===================== Make gwave =====================
#         '''......... '''
#         # gwave = waveform.PGDiffWaveform(big_delta=big_delta, little_delta=little_delta, te=total_sim_time)
#         # Create an apodised oscillating gradient with 4 cycles at 50 Hz
#         # Create a trapezoidal-cosine oscillating gradient with:

#         # Create the waveform
#         gwave = waveform.CosineOGDiffWaveform(
#             num_oscillations=5,
#             frequency=192,  # Hz
#             gmax=1,        # mT/m
#             te=0.082,         # 100 ms
#             time_step=time_step
#         )

#         # gwave = waveform.ApodisedOGDiffWaveform(freq=50, n_cycles=4, apodisation='gaussian', te=0.1)
#         gradient_file_name = os.path.join(output_folder, str(gwave.__class__.__name__)+"_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
#         waveform_io.save_waveform_object(gwave, gradient_file_name)
#         gwave2 = waveform_io.load_waveform_object(gradient_file_name+'.pkl')
#         plt_ge.plot_wave(gwave2, output_folder)
        
#         bval = np.linspace(0,3,100) # ms/um^2, diffusion b-value 

#         phase_sig = sim.simulate_multi_directions(gwave)

#         # calculate gradient amplitudes
#         b = gwave.calculate_bvalue_from_wave()
#         Gmax = np.sqrt(bval/b) # vector of Gmax's
#         print('Gmax', Gmax)
#         print('phase_sig', phase_sig.shape)
#         print('phase_sig', phase_sig)
#         print('phase_sig max', max(phase_sig[0, :]))
#         print('phase_sig max', max(phase_sig[1, :]))
#         print('phase_sig max', max(phase_sig[2, :]))

#         # calculate the dwi signal
#         dwsig = np.zeros([3,len(Gmax)])
#         for n,gmax in enumerate(Gmax):
#             dwsig[:,n] =  np.sum(np.cos(gmax*phase_sig),axis=-1) / num_spins
        
#         plt_ge.plot_dwsig_vs_bval(bval,dwsig[0,:],'./DWI_vs_b/x_dir')
#         plt_ge.plot_dwsig_vs_bval(bval,dwsig[1,:],'./DWI_vs_b/y_dir')
#         plt_ge.plot_dwsig_vs_bval(bval,dwsig[2,:],'./DWI_vs_b/z_dir')
 
#         et = time.time()
#         # # get the execution time
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

# testOutsideTwoSetOfMultipleSpheres()