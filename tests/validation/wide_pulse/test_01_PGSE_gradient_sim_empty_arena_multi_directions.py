import pycuda.autoinit
import numpy as np
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

import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.pyplot import cm
from matplotlib.colors import LinearSegmentedColormap

import time
import os 
import pickle

def test_free_gradient_diffusion_empty_arena_multi_directions_PGSE():
        '''
        diffusion in empty arena with PGSE waveform
        '''
        time_step=0.001 # ms
        # total_sim_time=78 # PGSE
        total_sim_time=62 # PGSE
        
        big_delta = 50   # ms
        little_delta= 12 # ms
        # teff = 50 - 12/3 = 46ms

        spins = int(1e5)
        diffdir_file='./tests/validation/wide_pulse/DWI_64dir_6shells.bvec'
        bval_file='./tests/validation/wide_pulse/DWI_64dir_6shells.bval'
        folder_date_time = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        st = time.time()
        Lx = 20.0 # um
        Ly = 20.0 # um
        Lz = 20.0 # um
        D = 3.0 # um^2/ms
        T2 = 200 # ms
        rho = 1 # fractional water density
        sg3 = SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

        #================================================        
        num_spins=spins
        sim = DwiSim3d(sg3,spins)
        diffdir=waveform_io.load_file(diffdir_file)
        b_values=waveform_io.load_file(bval_file).T # s/mm^2
        print('diffdir', diffdir.shape)
        print('b_values', b_values.shape)
        dir_count=np.unique(diffdir[0,:])
        dir_count = dir_count[dir_count != 0].shape[0]
        sim.set_diffusion_directions(diffdir=diffdir)
        nsegx,nsegy,nsegz=2,2,2
        sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)

        print('Start setting up structures')
        sim.setup(structures=list(np.arange(0,1))) 
        # sim.setup(structures=list(np.arange(0, 2)))  # seed inside structures 0:len(fiberlist)

        output_folder = './tests/validation/wide_pulse/test_figures'
        os.makedirs(os.path.join(output_folder,folder_date_time), exist_ok=True)
        # ===================== Make gwave =====================
        gwave = PGDiffWaveform(big_delta=big_delta, little_delta=little_delta, te=total_sim_time, time_step=time_step)
      
        gradient_file_name = os.path.join(output_folder, folder_date_time, str(gwave.__class__.__name__)+"_")
        waveform_io.save_waveform_object(gwave, gradient_file_name)
        gwave2 = waveform_io.load_waveform_object(gradient_file_name+'.pkl')
        plt_ge.plot_wave(gwave2, os.path.join(output_folder, folder_date_time))

        unique_bvals = np.unique(b_values)
        phase_accumulated_by_each_spin_in_each_diffdir = sim.simulate_multi_directions(gwave2)
        pkl_file_name = os.path.join(gradient_file_name+'multidirr_signal.pkl')
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(phase_accumulated_by_each_spin_in_each_diffdir, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done saving signal data!')

        # calculate b-value based on the gwave
        '''
        q(t) = \int_0^t gamma*[ G(t') dt'] (cumulative gradient moment)
            b = \int_0^T [ dt * q(t) * q(t) ]        
              = \int_0^T [ dt * ( \int_0^t [gamma * G(t') dt'] )^2 ]
        '''
        b = gwave.calculate_bvalue_from_wave() # ms/um^2 
        # convert b-value: 1 ms/um^2 = 1000 s/mm^2
        # b-value scales with gradient amplitude squared.
        # calculate Gmax to scale to achieve the desired b-values
        Gmax = np.sqrt(unique_bvals/1000/b) # vector of Gmax's
    
        # calculate the dwi signal
        dwsig = np.zeros([diffdir.shape[1],len(Gmax)]) # 390 directions x 7 b-values 
        
        for n,gmax in enumerate(Gmax): # loop throguh each b-value
            #  signal for every direction at the nth unique b-value
            dwsig[:,n] =  np.sum(np.cos(gmax*phase_accumulated_by_each_spin_in_each_diffdir),axis=-1) / num_spins

        fig, ax = plt.subplots()

        # First plot b=0 in black
        dir_mask = b_values == 0 # index of directions with this b-value
        b_mask = unique_bvals == 0
        b_signals = dwsig[dir_mask, b_mask]
        # Calculate mean and std across directions
        mean_signal = np.mean(b_signals, axis=0)
        std_signal = np.std(b_signals, axis=0)
        # Plot with error bars
        plt.errorbar(0, mean_signal, yerr=std_signal, 
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
            dir_mask = b_values == b # index of directions with this b-value [b_values and 
            b_mask = unique_bvals == b
            # dwsig 390 x 7 datapoints -- 390 directions, 7 b-values (including b=0)
            b_signals = dwsig[dir_mask, b_mask]
            # Calculate mean and std across directions
            mean_signal = np.mean(b_signals, axis=0)
            std_signal = np.std(b_signals, axis=0)
            # Plot with error bars
            plt.errorbar(b, mean_signal, yerr=std_signal, 
                        label=f'b={int(b)}', capsize=8, fmt='o', markersize=5,
                        color=blue_red(b / np.max(nonzero_bvals)))  # fmt='o' for circle markers

        # Add theoretical exponential decay curve
        # Signal = exp(-b * D) where D = 3.0 um^2/ms  
        b_theory = np.linspace(0, np.max(unique_bvals), 100)  # Smooth curve in s/mm^2 = 10^3 ms /(10^3)^2 um^2 = 1/1000 ms/um^2
        D_theory = 3.0  # um^2/ms (same as defined in the simulation)
        signal_theory = np.exp(-b_theory * D_theory / 1000)  #  ms/um^2 * um^2/ms 
        plt.plot(b_theory, signal_theory, 'r--', linewidth=2, 
                label=f'Analytical: exp(-bD), D={D_theory} μm²/ms')
        plt.legend()
        plt.grid(True)
        ax.set_xlabel("b-value (s/mm²)")
        ax.set_ylabel("DWI signal")
        ax.set_title(f'DWI signal vs. b-value for {int(dir_count)} directions')

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(gradient_file_name+str(dir_count)+'_directions.png'))
        et = time.time()
        elapsed_time = np.round(et - st,2)

        # plot after simulation
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
        spin_plot_file_name = 'free_diffusion_arena.png'
        spin_plot_file_name = os.path.join(output_folder, folder_date_time, spin_plot_file_name)
        plt.savefig(spin_plot_file_name)
        #-------------------------------------------
        
test_free_gradient_diffusion_empty_arena_multi_directions_PGSE()