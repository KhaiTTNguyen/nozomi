# show the gradient waveform
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle

def save_gradient_data(singal_vs_bval, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(singal_vs_bval, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done saving data file')
    return

def plot_wave(gwave, output_folder):

    fig, ax = plt.subplots()
    plt.plot(gwave.t, gwave.wave, linewidth=2)
    plt.grid(True)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Normalized gradient strength")
    ax.set_title('Gradient waveform')
    gradient_file_name = os.path.join(output_folder, str(gwave.__class__.__name__)+"_" + str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))      
    plt.savefig(gradient_file_name)

def plot_waveform(gwave, output_folder, big_delta, little_delta):

    fig, ax = plt.subplots()
    plt.plot(gwave.t, gwave.wave, linewidth=2)
    plt.grid(True)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Normalized gradient strength")
    ax.set_title('Gradient waveform')

    spin_plot_file_name = 'Gradient waveform_big_delta_'+str(int(big_delta))+'_little_delta_'+str(int(little_delta))+'.png'
    spin_plot_file_name = os.path.join(output_folder, spin_plot_file_name)
    plt.savefig(spin_plot_file_name)

def plot_dwsig_vs_bval(bval,dwsig, spin_plot_file_name):
    fig, ax = plt.subplots()
    plt.plot(bval, dwsig, 'b.',markersize=10, linewidth=2)
    plt.grid(True)
    ax.set_xlabel("b-value (ms/um^2)")
    ax.set_ylabel("DWI signal")
    ax.set_title('DWI signal vs. b-value')

    spin_plot_file_name = spin_plot_file_name
    plt.savefig(spin_plot_file_name)

def plot_log_dwsig_vs_bval(bval,dwsig, output_folder, compartment):
    fig, ax = plt.subplots()
    plt.plot(bval, dwsig, 'b.',markersize=10, linewidth=2)
    plt.grid(True)
    ax.set_xlabel("b-value (ms/um^2)")
    ax.set_ylabel("Log DWI signal")
    ax.set_title('Log DWI signal vs. b-value')

    spin_plot_file_name = compartment+'_log_DWI_signal_vs_b-value.png'
    spin_plot_file_name = os.path.join(output_folder, spin_plot_file_name)
    plt.savefig(spin_plot_file_name)