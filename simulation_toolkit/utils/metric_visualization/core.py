import os


METRIC_CONFIG = {
    'adc': {
        'ylabel': r'ADC ($\mu m^2/ms$)',
        'output_dir': 'plot_across_{parameter}',
        'filename_prefix': 'ADC_wrt_{parameter}',
        'color_scheme': 'viridis',
        'ylim': (0, 3),
    },
    'kurtosis': {
        'ylabel': r'Kurtosis',
        'output_dir': 'plot_kurtosis_across_{parameter}',
        'filename_prefix': 'kurtosis_wrt_{parameter}',
        'color_scheme': 'plasma',
        'ylim': (0, 5),
    }
}



def plot_metric_vs_parameter(
    data, 
    metric='adc',           # 'adc' or 'kurtosis'
    parameter='diameter',   # 'diameter' or 'watson_k'
    compartment='intra',    # 'intra' or 'extra'
    watson_k=None,
    **kwargs
):
    
    '''
    All common visualization logica are stored here.
    Specific plotting details are handled based on the metric type
    and can be called with a flag defined in the function arguments.

    Input: Folder containing all the experiments
    Output: Diffusion time dependent plots of the metrics in the corresponding folders.

    1. Script will take in the folder /nozomi/experiment/visualization
    2. Loop through all the subfolders in /nozomi/experiment/visualization/data
    3. Each subfolder contains an experiment with multiple repetitions 
    /nozomi/experiment/visualization/data/experiment_d1.68_sig0.45_500axons_reproducibility_OD10/

    4. An example data file is store here: /nozomi/experiment/visualization/data/experiment_d1.68_sig0.45_500axons_reproducibility_OD10/2025-09-12_17-22-10_d1.68_K10_ODI_0.0635_500fibers/sim/ADCdata/20SEGMENT_OPT_extra_2025-09-17_07-15-03_2602_fibers_500000_spins_array500_fibers_boxL_68.0_2025-09-12_17-22-10_avf_0.74_d1.68_sig0.45wo10_wc3_wl3_K10_ODI_0.06359364.8_sec_SIMtime6018.09_sec_dt0.002_seg20data.pkl
    
    5. Gather all the file paths with the same K value (e.g., K=10)
    6. Print the file paths to verify
    
    6. Plot the ADC vs diffusion time for all the diameters in one plot for K=10
    7. Save the plot in the folder /nozomi/experiment/visualization/plots/ADC_wrt_diameter/K10/
    8. Repeat for all K values and for kurtosis metric as well.
    9. Repeat for extra-axonal compartment as well.
    '''
    compartment = str(os.path.basename(folder_path))
    folder_name = os.path.join(folder_path, 'ADC_wrt_diameter')
    fig, ax = plt.subplots(figsize=(5,5))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")  
    
    
    
    
    
    
    """Generic function to plot any metric vs any parameter"""
    config = METRIC_CONFIG[metric]
    
    # Common plotting logic here
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot with metric-specific configuration
    for diameter in data['diameters']:
        ax.plot(data['time'], data[f'{metric}_{diameter}'], 
                label=f'd={diameter}μm')
    
    ax.set_ylabel(config['ylabel'], fontsize=15)
    ax.set_xlabel('Diffusion Time (ms)', fontsize=15)
    # ... more common formatting
    
    return fig, ax