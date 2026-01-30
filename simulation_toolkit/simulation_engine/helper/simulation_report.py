import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name):
    fig, ax = plt.subplots()
    Dxy_step = (Dx_step + Dy_step)/2
    ax.semilogx(diff_time, Dx_step, 'b.',markersize=10)   
    ax.semilogx(diff_time, Dy_step, 'y.',markersize=10)   
    ax.semilogx(diff_time, Dxy_step, 'g.',markersize=10)   
    ax.semilogx(diff_time, Dz_step, 'r.', markersize=10)  
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('D (µm$^2$/ms)') 
    # ax.set_ylim([0, Dx_step[0]*5/4])
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_title(" Diffusion coefficients vs. diffusion time ", fontsize=17)
    ax.grid(True)
    ax.set_xlim([0.0001, max(diff_time)])
    ax.set_ylim([0.0, 3.2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    ax.legend(['Dx','Dy','Dxy','Dz'], fontsize=18)
    file_name = os.path.join(folder_name, 'difftime_'+str(np.round(np.max(diff_time), 2))+'_'+file_name+'.png')
    
    plt.savefig(file_name)

def plot_K_vs_time(diff_time, Kx_step, Ky_step, Kz_step, folder_name, file_name):
    fig, ax = plt.subplots()
    Kxy_step = (Kx_step + Ky_step)/2
    ax.semilogx(diff_time, Kx_step, 'b.',markersize=10)   
    ax.semilogx(diff_time, Ky_step, 'y.',markersize=10)   
    ax.semilogx(diff_time, Kxy_step, 'g.',markersize=10)   
    ax.semilogx(diff_time, Kz_step, 'r.', markersize=10)  
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('K (µm$^2$/ms)') 
    # ax.set_ylim([0, Dx_step[0]*5/4])
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_title(" Diffusion coefficients vs. diffusion time ", fontsize=17)
    ax.grid(True)
    ax.set_xlim([0.0001, max(diff_time)])
    ax.set_ylim([0.0, 2.0])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    ax.legend(['Kx','Ky','Kxy','Kz'], fontsize=18)
    file_name = os.path.join(folder_name, 'difftime_'+str(np.round(np.max(diff_time), 2))+'_'+file_name+'.png')
    
    plt.savefig(file_name)

def save_data_pickle(file_name, data):
    folder_path = os.path.dirname(file_name) 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(file_name, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done saving file')
    return
