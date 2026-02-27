import pycuda.autoinit
from datetime import datetime
# from matplotlib.pyplot import cm
# from matplotlib.colors import LinearSegmentedColormap

import os
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from simulation_toolkit.simulation_engine.helper import plot_gradient_experiment as plt_ge
from simulation_toolkit.simulation_engine.helper import waveform_io
from simulation_toolkit.simulation_engine.diffsim3d import DwiSim3d
from simulation_toolkit.simulation_engine.geometry import SimGeometry3D
from simulation_toolkit.simulation_engine.waveforms import *

def test_free_gradient_diffusion_empty_arena():
        '''
        diffusion in empty arena
        '''
        # diffdir_file='SimExperiment/biophysical_sim/NODDI/gradient/DWI_64dir_6shells.bvec'
        # bval_file='SimExperiment/biophysical_sim/NODDI/gradient/DWI_64dir_6shells.bval'
        
        # compartment = 'intra'
        # time_step=0.0001 # ms
        # # total_sim_time=0.082 # OGSE
        total_sim_time=17 # PGSE
        
        big_delta = 12
        little_delta= 3
        output_folder = './tests/validation/wide_pulse/test_figures'
        folder_date_time = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(os.path.join(output_folder, folder_date_time)):
                os.makedirs(os.path.join(output_folder, folder_date_time))
        dt = 0.002 # time step in ms
        nt = int(50000) # total number of steps thru time
        spins = int(1e5)

        st = time.time()
        Lx = 10 # um
        Ly = 10 # um
        Lz = 10 # um
        D = 3.0 # um^2/ms
        T2 = 200 # ms
        rho = 1 # fractional water density
        sg3 = SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

        num_spins=spins
        sim = DwiSim3d(sg3,spins)
        sim.set_segments(nsegx=2,nsegy=2,nsegz=2)

        # ===================== Make gwave =====================
        '''......... '''
        gwave = PGDiffWaveform(big_delta=big_delta, little_delta=little_delta, te=total_sim_time)
        
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

        # # exit()
        bval = np.linspace(0,3,100) # ms/um^2, diffusion b-value 
        # unique_bvals = np.unique(b_values)
        # bval = unique_bvals
        # phase_sig = sim.simulate_multi_directions(gwave2)
        # pkl_file_name = os.path.join(gradient_file_name+'multidirr_signal.pkl')
        # with open(pkl_file_name, 'wb') as f:
        #     pickle.dump(phase_sig, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print('Done saving signal data!')

        #  # save result for later use
        # # calculate gradient amplitudes
        b = gwave.calculate_bvalue_from_wave()
        print('GOTHEREREEEEEEEE', b)
        print('b', b)
        Gmax = np.sqrt(bval/1000/b) # vector of Gmax's
    
        # print('Gmax', Gmax)
        # print('phase_sig', phase_sig.shape)
        # print('phase_sig', phase_sig)
        # print('phase_sig max', max(phase_sig[0, :]))
        # print('phase_sig max', max(phase_sig[1, :]))
        # print('phase_sig max', max(phase_sig[2, :]))

        # # calculate the dwi signal
        # dwsig = np.zeros([diffdir.shape[1],len(Gmax)])
        # print(dwsig.shape)
        
        # for n,gmax in enumerate(Gmax):
        #     dwsig[:,n] =  np.sum(np.cos(gmax*phase_sig),axis=-1) / num_spins

        # fig, ax = plt.subplots()

        # # First plot b=0 in black
        # dir_mask = b_values == 0 # index of directions with this b-value
        # b_mask = unique_bvals == 0
        # b_signals = dwsig[dir_mask, b_mask]
        # # Calculate mean and std across directions
        # mean_signal = np.mean(b_signals, axis=0)
        # std_signal = np.std(b_signals, axis=0)
        # # Plot with error bars
        # plt.errorbar(b, mean_signal, yerr=std_signal, 
        #             label=f'b={int(b)}', capsize=8, fmt='o', markersize=5,
        #             color='black')  # fmt='o' for circle markers

        # nonzero_bvals = unique_bvals[unique_bvals > 0]
        # # Create a colormap for the b-values
        # cm_dict = {'red':  ((0.0, 0.0, 0.0),
        #                     (1.0, 1.0, 1.0)),
        #         'green': ((0.0, 0.0, 0.0),
        #                     (1.0, 0.0, 0.0)),
        #         'blue':  ((0.0, 1.0, 1.0),
        #                     (1.0, 0.0, 0.0))}
        # blue_red = LinearSegmentedColormap('blue_red', cm_dict)
        

        # for idx, b in enumerate(nonzero_bvals):
        #     # Get signals for all directions with this b-value
        #     dir_mask = b_values == b # index of directions with this b-value
        #     b_mask = unique_bvals == b
        #     b_signals = dwsig[dir_mask, b_mask]
        #     # Calculate mean and std across directions
        #     mean_signal = np.mean(b_signals, axis=0)
        #     std_signal = np.std(b_signals, axis=0)
        #     # Plot with error bars
            
        #     plt.errorbar(b, mean_signal, yerr=std_signal, 
        #                 label=f'b={int(b)}', capsize=8, fmt='o', markersize=5,
        #                 color=blue_red(b / np.max(nonzero_bvals)))  # fmt='o' for circle markers

        # plt.xlabel('b-value (s/mm²)')
        # plt.ylabel('DW Signal')
        # plt.legend()
        # # fig, ax = plt.subplots()
        # # max_legend_entries = 5  # Show only first 5 directions in legend
        # # bval = bval * 1000  # Convert to s/mm^2 for visualization
        # # dir_count=0
        # # for dir in range(0, diffdir.shape[1]):
        # #     plt_ge.save_gradient_data(dwsig[dir,:], os.path.join(gradient_file_name+'_dir'+str(dir)+'.pkl'))
            
        # #     if b_values[dir]==0:
        # #         plt.plot(bval, dwsig[dir,:], label=f'b={b_values[dir]}', markersize=10, color='orange')  # First direction in orange
        # #     else:    
        # #         dir_count += 1
        # #         if dir < max_legend_entries:
        # #             plt.plot(bval, dwsig[dir,:], label=f'dir_{dir}', markersize=10)
        # #         # else:
        # #         #     plt.plot(bval, dwsig[dir,:], markersize=10, alpha=0.6)  # Make others more transparent

        #     # if b_values[dir]==0:
        #     #     if dir < max_legend_entries:
        #     #         dir_count += 1
        #     #         plt.plot(bval, dwsig[dir,:], label=f'dir_{dir}', markersize=10, color='blue')  # First direction in blue
        #     # else:    
        #     #     dir_count += 1
        #     #     if dir < max_legend_entries:
        #     #         plt.plot(bval, dwsig[dir,:], label=f'dir_{dir}', markersize=10)
        #     #     # else:
        #     #     #     plt.plot(bval, dwsig[dir,:], markersize=10, alpha=0.6)  # Make others more transparent

        # plt.grid(True)
        # ax.set_xlabel("b-value (s/mm²)")
        # ax.set_ylabel("DWI signal")
        # ax.set_title(f'DWI signal vs. b-value for {int(dir_count)} directions')

        # # Add legend for first few directions only
        # plt.legend()
        # plt.grid(True, linestyle="--", alpha=0.5)

        # plt.tight_layout()
        # plt.savefig(os.path.join(gradient_file_name+str(dir_count)+'_directions.png'))
        # et = time.time()
        # elapsed_time = np.round(et - st,2)

test_free_gradient_diffusion_empty_arena()