import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
# import diffsim3d as ds3
import hipa.diffsim.geometry as geom
import hipa.diffsim.diffsim3d_additional as ds3
import hipa.diffsim.waveforms as waveform
import hipa.diffsim.helper.simulation_report as simrep
import hipa.diffsim.helper.plot_gradient_experiment as plt_ge
import hipa.diffsim.helper.sim_util as sim_util
import matplotlib.pyplot as plt
import hipa.util.util as util
import hipa.util.adjust_geometry as ag
from datetime import datetime
from matplotlib.pyplot import cm
from matplotlib.colors import LinearSegmentedColormap

import time
import os 
import pickle
import hipa.diffsim.helper.waveform_io as waveform_io

def testInside1CenterSphere():
        '''
        diffusion outside of multiple spheres
        '''
        compartment = 'intra'
        time_step=0.0001 # ms
        # total_sim_time=0.082 # OGSE
        total_sim_time=17 # PGSE
        
        big_delta = 12
        little_delta= 3
        spins = int(1e5)
        diffdir_file='/home/nguyt16@ds.vanderbilt.edu/HIPASimExperiment/biophysical_sim/NODDI/gradient/DWI_64dir_6shells.bvec'
        bval_file='/home/nguyt16@ds.vanderbilt.edu/HIPASimExperiment/biophysical_sim/NODDI/gradient/DWI_64dir_6shells.bval'
        folder_date_time = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        st = time.time()
        Lx = 20.0 # um
        Ly = 20.0 # um
        Lz = 20.0 # um
        D = 3.0 # um^2/ms
        T2 = 200 # ms
        rho = 1 # fractional water density
        sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

        sz = np.zeros(1)
        sy = np.zeros(1)
        sx = np.zeros(1)
        sr = np.ones(1)*5
        spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
        sg3.add_structure(spstruc) # add it to sg3
 
        #================================================
        print('sg3.nstructures',sg3.nstructures)
        print('sg3.nstructures',sg3.nsphere)
        
        num_spins=spins
        sim = ds3.DwiSim3d(sg3,spins)
        diffdir=waveform_io.load_file(diffdir_file)
        b_values=waveform_io.load_file(bval_file).T
        print('diffdir', diffdir.shape)
        print('b_values', b_values.shape)
        dir_count=np.unique(diffdir[0,:])
        dir_count = dir_count[dir_count != 0].shape[0]
        sim.set_diffusion_directions(diffdir=diffdir)
        nsegx,nsegy,nsegz=2,2,2
        sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)

        print('Start setting up structures')
        if compartment=='intra':
            sim.setup(structures=list(np.arange(0,1))) 
            # sim.setup(structures=list(np.arange(0, 2)))  # seed inside structures 0:len(fiberlist)

        # first, check thath all the spins are in fact in spstruc
        if compartment=='intra':
            isInside = sim_util.validate_compartment_intra(sim, sg3)
            spins_temp = ((sim.spins_d.get().T)[isInside]).T
            if not isInside:
                print("Error: Not all spins are inside structures")
        elif compartment=='extra':
            isOutside = sim_util.validate_compartment_extra(sim, sg3)
            spins_temp = ((sim.spins_d.get().T)[isOutside]).T
            if not isOutside:
                print("Error: Not all spins are outside structures")

        # plot after seed
        spins_temp = sim.spins_d.get()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.view_init(azim=0, elev=90)
        ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
        k=0
        ax.set_xlim(-Lx/2, Lx/2)
        ax.set_ylim(-Ly/2, Ly/2)
        ax.set_zlim(-Lz/2, Lz/2)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_zlabel("z (um)")
        spin_plot_file_name = 'spins_1centersphere.png'
        output_folder = "./hipa/tests/gradient_sim_multi_directions"
        os.makedirs(os.path.join(output_folder,folder_date_time), exist_ok=True)
        spin_plot_file_name = os.path.join(output_folder, folder_date_time, spin_plot_file_name)
        plt.savefig(spin_plot_file_name)
        # exit()

        # ===================== Make gwave =====================
        '''......... '''
        gwave = waveform.PGDiffWaveform(big_delta=big_delta, little_delta=little_delta, te=total_sim_time)
        # Create an apodised oscillating gradient with 4 cycles at 50 Hz
        # Create a trapezoidal-cosine oscillating gradient with:

        # Create the waveform
        # gwave = waveform.CosineOGDiffWaveform(
        #     num_oscillations=5,
        #     frequency=192,  # Hz
        #     gmax=1,        # mT/m
        #     te=0.082,
        #     time_step=time_step
        # )

        # gwave = waveform.CosineOGDiffWaveform(
        #     num_oscillations=5,
        #     frequency=192,  # Hz
        #     gmax=1,        # mT/m
        #     te=82,
        #     time_step=time_step
        # )

        # gwave = waveform.ApodisedOGDiffWaveform(freq=50, n_cycles=4, apodisation='gaussian', te=0.1)
        gradient_file_name = os.path.join(output_folder, folder_date_time, str(gwave.__class__.__name__)+"_")
        waveform_io.save_waveform_object(gwave, gradient_file_name)
        gwave2 = waveform_io.load_waveform_object(gradient_file_name+'.pkl')
        plt_ge.plot_wave(gwave2, os.path.join(output_folder, folder_date_time))

        # exit()
        # bval = np.linspace(0,3,100) # ms/um^2, diffusion b-value 
        unique_bvals = np.unique(b_values)
        bval = unique_bvals
        phase_sig = sim.simulate_multi_directions(gwave2)
        pkl_file_name = os.path.join(gradient_file_name+'multidirr_signal.pkl')
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(phase_sig, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done saving signal data!')

         # save result for later use
        # calculate gradient amplitudes
        b = gwave.calculate_bvalue_from_wave()
        Gmax = np.sqrt(bval/1000/b) # vector of Gmax's
    
        # print('Gmax', Gmax)
        # print('phase_sig', phase_sig.shape)
        # print('phase_sig', phase_sig)
        # print('phase_sig max', max(phase_sig[0, :]))
        # print('phase_sig max', max(phase_sig[1, :]))
        # print('phase_sig max', max(phase_sig[2, :]))

        # calculate the dwi signal
        dwsig = np.zeros([diffdir.shape[1],len(Gmax)])
        print(dwsig.shape)
        
        for n,gmax in enumerate(Gmax):
            dwsig[:,n] =  np.sum(np.cos(gmax*phase_sig),axis=-1) / num_spins

        fig, ax = plt.subplots()

        # First plot b=0 in black
        dir_mask = b_values == 0 # index of directions with this b-value
        b_mask = unique_bvals == 0
        b_signals = dwsig[dir_mask, b_mask]
        # Calculate mean and std across directions
        mean_signal = np.mean(b_signals, axis=0)
        std_signal = np.std(b_signals, axis=0)
        # Plot with error bars
        plt.errorbar(b, mean_signal, yerr=std_signal, 
                    label=f'b={int(b)}', capsize=8, fmt='o', markersize=5,
                    color='black')  # fmt='o' for circle markers

        nonzero_bvals = unique_bvals[unique_bvals > 0]
        # Create a colormap for the b-values
        cm_dict = {'red':  ((0.0, 0.0, 0.0),
                            (1.0, 1.0, 1.0)),
                'green': ((0.0, 0.0, 0.0),
                            (1.0, 0.0, 0.0)),
                'blue':  ((0.0, 1.0, 1.0),
                            (1.0, 0.0, 0.0))}
        blue_red = LinearSegmentedColormap('blue_red', cm_dict)
        

        for idx, b in enumerate(nonzero_bvals):
            # Get signals for all directions with this b-value
            dir_mask = b_values == b # index of directions with this b-value
            b_mask = unique_bvals == b
            b_signals = dwsig[dir_mask, b_mask]
            # Calculate mean and std across directions
            mean_signal = np.mean(b_signals, axis=0)
            std_signal = np.std(b_signals, axis=0)
            # Plot with error bars
            
            plt.errorbar(b, mean_signal, yerr=std_signal, 
                        label=f'b={int(b)}', capsize=8, fmt='o', markersize=5,
                        color=blue_red(b / np.max(nonzero_bvals)))  # fmt='o' for circle markers

        plt.xlabel('b-value (s/mm²)')
        plt.ylabel('DW Signal')
        plt.legend()
        # fig, ax = plt.subplots()
        # max_legend_entries = 5  # Show only first 5 directions in legend
        # bval = bval * 1000  # Convert to s/mm^2 for visualization
        # dir_count=0
        # for dir in range(0, diffdir.shape[1]):
        #     plt_ge.save_gradient_data(dwsig[dir,:], os.path.join(gradient_file_name+'_dir'+str(dir)+'.pkl'))
            
        #     if b_values[dir]==0:
        #         plt.plot(bval, dwsig[dir,:], label=f'b={b_values[dir]}', markersize=10, color='orange')  # First direction in orange
        #     else:    
        #         dir_count += 1
        #         if dir < max_legend_entries:
        #             plt.plot(bval, dwsig[dir,:], label=f'dir_{dir}', markersize=10)
        #         # else:
        #         #     plt.plot(bval, dwsig[dir,:], markersize=10, alpha=0.6)  # Make others more transparent

            # if b_values[dir]==0:
            #     if dir < max_legend_entries:
            #         dir_count += 1
            #         plt.plot(bval, dwsig[dir,:], label=f'dir_{dir}', markersize=10, color='blue')  # First direction in blue
            # else:    
            #     dir_count += 1
            #     if dir < max_legend_entries:
            #         plt.plot(bval, dwsig[dir,:], label=f'dir_{dir}', markersize=10)
            #     # else:
            #     #     plt.plot(bval, dwsig[dir,:], markersize=10, alpha=0.6)  # Make others more transparent

        plt.grid(True)
        ax.set_xlabel("b-value (s/mm²)")
        ax.set_ylabel("DWI signal")
        ax.set_title(f'DWI signal vs. b-value for {int(dir_count)} directions')

        # Add legend for first few directions only
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(gradient_file_name+str(dir_count)+'_directions.png'))
        et = time.time()
        elapsed_time = np.round(et - st,2)

        
        #-------------------------------------------
        if compartment=='intra':
            isInside = sim_util.validate_compartment_intra(sim, sg3)
            spins_temp = ((sim.spins_d.get().T)[isInside]).T
            if not isInside:
                print("Error: Not all spins are inside structures")
        elif compartment=='extra':
            isOutside = sim_util.validate_compartment_extra(sim, sg3)
            spins_temp = ((sim.spins_d.get().T)[isOutside]).T
            if not isOutside:
                print("Error: Not all spins are outside structures")
        print('elapsed_time', elapsed_time)

testInside1CenterSphere()