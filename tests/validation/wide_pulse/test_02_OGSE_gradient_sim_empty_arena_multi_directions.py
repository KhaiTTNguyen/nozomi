import pycuda.autoinit
import numpy as np
import os
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from simulation_toolkit.simulation_engine.helper import plot_gradient_experiment as plt_ge
from simulation_toolkit.simulation_engine.helper import waveform_io
from simulation_toolkit.simulation_engine.diffsim3d import DwiSim3d
from simulation_toolkit.simulation_engine.geometry import SimGeometry3D
from simulation_toolkit.simulation_engine.waveforms import *

import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

import pickle

def test_free_diffusion_empty_arena_multi_directions_OGSE():
        '''
        diffusion in empty arena with apodized cosine OGSE waveform
        '''
        time_step=0.001 # ms (waveform time step)
        total_sim_time=78 # ms - must be > 2 * T_duration to fit both gradient lobes

        # OGSE parameters
        N_cycles = 1       # number of cosine oscillation cycles
        T_duration = 26.0  # ms, gradient on-time per lobe (t_eff = T / (4*N))
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
        b_values=waveform_io.load_file(bval_file).T
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
        # Apodized cosine OGSE waveform (Does et al. 2003)
        gwave = ApodizedCosineOGSEWaveform(
            N_cycles=N_cycles,
            T_duration=T_duration,
            te=total_sim_time,
            gmax=1.0,
            time_step=time_step
        )
        print(f'OGSE waveform: N={N_cycles}, T={T_duration} ms, t_eff={gwave.get_effective_diffusion_time():.2f} ms')
        info = gwave.get_waveform_info()
        print(f'Cosine frequency: {info["cosine_frequency_Hz"]:.1f} Hz')
        gradient_file_name = os.path.join(output_folder, folder_date_time, str(gwave.__class__.__name__)+"_")
        waveform_io.save_waveform_object(gwave, gradient_file_name)
        gwave2 = waveform_io.load_waveform_object(gradient_file_name+'.pkl')
        plt_ge.plot_wave(gwave2, os.path.join(output_folder, folder_date_time))

        unique_bvals = np.unique(b_values)
        phase_sig = sim.simulate_multi_directions(gwave2)
        pkl_file_name = os.path.join(gradient_file_name+'multidirr_signal.pkl')
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(phase_sig, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Done saving signal data!')

        # save result for later use
        # calculate gradient amplitudes
        b_base = gwave.calculate_bvalue_from_wave()
        Gmax = np.sqrt(unique_bvals/1000/b_base) # vector of unique Gmax's
    
        # calculate the dwi signal
        dwsig = np.zeros([diffdir.shape[1], len(Gmax)])
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
        plt.errorbar(0, mean_signal, yerr=std_signal, 
                    label='b=0', capsize=8, fmt='o', markersize=5,
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
        

        for b_shell in nonzero_bvals:
            # Get signals for all directions with this b-value
            dir_mask = b_values == b_shell # index of directions with this b-value
            b_mask = unique_bvals == b_shell
            b_signals = dwsig[dir_mask, b_mask]
            # Calculate mean and std across directions
            mean_signal = np.mean(b_signals, axis=0)
            std_signal = np.std(b_signals, axis=0)
            # Plot with error bars
            
            plt.errorbar(b_shell, mean_signal, yerr=std_signal, 
                        label=f'b={int(b_shell)}', capsize=8, fmt='o', markersize=5,
                        color=blue_red(b_shell / np.max(nonzero_bvals)))  # fmt='o' for circle markers

        # Add theoretical exponential decay curve
        # For free diffusion, ADC = D regardless of waveform => Signal = exp(-b * D)
        b_theory = np.linspace(0, np.max(unique_bvals), 100)  # Smooth curve
        D_theory = 3.0  # um^2/ms (same as defined in the simulation)
        signal_theory = np.exp(-b_theory * D_theory / 1000)  # Convert to s/mm^2 units
        t_eff = gwave.get_effective_diffusion_time()
        plt.plot(b_theory, signal_theory, 'r--', linewidth=2, 
                label=f'Analytical: exp(-bD), D={D_theory} μm²/ms (t_eff={t_eff:.2f} ms)')

        ax.set_xlabel("b-value (s/mm²)")
        ax.set_ylabel("DWI signal")
        ax.set_title(f'OGSE DWI signal vs. b-value for {int(dir_count)} directions (N={N_cycles}, t_eff={gwave.get_effective_diffusion_time():.2f} ms)')

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(gradient_file_name+str(dir_count)+'_directions.png'))
        et = time.time()
        print(f'Elapsed time: {np.round(et - st, 2)} s')

        # plot after seed
        spins_temp = sim.spins_d.get()
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.view_init(azim=0, elev=90)
        ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
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
        
test_free_diffusion_empty_arena_multi_directions_OGSE()