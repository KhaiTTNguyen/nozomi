from util import *
import matplotlib.cm as cm
import os 

def import_array_ADC_full_path(file_name):
    print('file_name', file_name)
    Dx_Dy_Dz_difftime = load_ADC_data_pickle(file_name)
    return Dx_Dy_Dz_difftime

def load_ADC_data_pickle(file_name):
# Open the Pickle file for reading in binary mode ('rb')
    with open(file_name, 'rb') as file:
        # Unpickle the data
        loaded_data = pickle.load(file)
    return loaded_data

def count_num_files_in_folder(folder_path):
    """Counts the number of files in a given directory (excluding subdirectories).
    Args:
        folder_path: The path to the directory.

    Returns:
        The number of files in the directory.
    """
    count = 0
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            count += 1
    return count

def sort_lists_together(list_a, list_b):
  """Sorts list A and reorders list B based on the reordered positions of list A.

  Args:
    list_a: The list to be sorted.
    list_b: The list to be reordered.

  Returns:
    A tuple containing the sorted list A and the reordered list B.
  """
  # Sort list A
  list_a_floats = [float(x) for x in list_a]
  # Create a dictionary mapping elements in list A to their original indices
  index_map = {element: index for index, element in enumerate(list_a_floats)}
  sorted_a_floats = sorted(list_a_floats, reverse=True)
  # Reorder list B based on the sorted order of list A
  reordered_b = [list_b[index_map[element]] for element in sorted_a_floats]

  return sorted_a_floats, reordered_b

def extract_diameter_from_file_path(file_path):
    # Split the path based on underscores (_)
    parts = file_path.split("_")
    # Find the index of the part starting with "avf_"
    avf_index = next((i for i, part in enumerate(parts) if part.startswith("avf")), None)
    # Check if "avf_" was found
    if avf_index is not None:
        pass
    else:
        print("'avf_' section not found in filepath")
    diameter_string=parts[avf_index+2]
    diameter = diameter_string[1:]
    return diameter

def get_path_to_first_file(folder_path):
    """
    Returns the full path to the first file found in a given folder.
    Args:
    folder_path: The path to the folder.
    Returns:
    The full path to the first file, or None if no files are found or the folder 
    does not exist.
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return None

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            return file_path
    return None

def get_dispersion_value_kappa(file_path):
    import re
    match = re.findall(r'K(\d+)', file_path)

    if match:
        return match[0] # return the first occurence of a number after K
    else:
        print("No value found after 'K'")

def plot_Dxy_across_kappa(folder_path, diff_time_limit):
    '''
    input ADCdata folder
    loop through folder, load all data
    plot Dz data for each Kappa value
    '''
    folder_name = os.path.join(folder_path, 'ADC_wrt_kappa')
    fig, ax = plt.subplots()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")

    num_files = count_num_files_in_folder(folder_path)
    print("Number of files:", num_files)

    path_to_first_file = get_path_to_first_file(folder_path)
    diameter = extract_diameter_from_file_path(path_to_first_file)

    colors = {'200': 'crimson', '20': 'violet', '10': 'royalblue', '8': 'darkcyan'}
    kappa_list = []
    i=0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                Dx_Dy_Dz_difftime = pickle.load(f)
                Dx, Dy, Dz, diff_time = Dx_Dy_Dz_difftime[:,0], Dx_Dy_Dz_difftime[:,1], Dx_Dy_Dz_difftime[:,2], Dx_Dy_Dz_difftime[:,3]
                kappa = get_dispersion_value_kappa(file_path)
                kappa_list.append(kappa)
                color_i = colors[kappa]
                Dxy = (Dx + Dy)/2
                ax.semilogx(diff_time, Dxy, color=color_i,  marker='.', label='K='+str(kappa), markersize=10)  
                i+=1
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    unqiue_kappa_list = []
    for handle, label, kappa in zip(handles, labels, kappa_list):
        if label not in unique_labels:
            unique_labels[label] = handle
            unqiue_kappa_list.append(kappa) 
    print('list(unique_labels.keys())', len((list(unique_labels.keys()))))
    print('list(unique_labels.values())', len(list(unique_labels.values())))       
    sorted_kappa_list, reordered_labels = sort_lists_together(unqiue_kappa_list, list(unique_labels.keys()))
    sorted_kappa_list, reordered_handles = sort_lists_together(unqiue_kappa_list, list(unique_labels.values()))
    ax.legend(handles=reordered_handles, labels=reordered_labels)

    plot_file_name = os.path.join(folder_name, f'RD_wrt_kappa_d={diameter}_difftime_limit{diff_time_limit}.png')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel('t (ms)', fontsize=15)
    ax.set_ylabel('RD (µm$^2$/ms)', fontsize=15)
    ax.set_title("Radial diffusivity vs. diffusion time at $d$="+str(diameter)+"µm", fontsize=17)
    ax.grid(True)
    ax.set_xlim([0.0001, diff_time_limit])
    ax.set_ylim([0.0, 3.2])
    plt.savefig(plot_file_name)
    print('Done plotting')

def plot_Dz_across_kappa(folder_path, diff_time_limit):
    '''
    input ADCdata folder
    loop through folder, load all data
    plot Dz data for each Kappa value
    '''
    folder_name = os.path.join(folder_path, 'ADC_wrt_kappa')
    fig, ax = plt.subplots()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")

    num_files = count_num_files_in_folder(folder_path)
    print("Number of files:", num_files)

    path_to_first_file = get_path_to_first_file(folder_path)
    diameter = extract_diameter_from_file_path(path_to_first_file)

    colors = {'200': 'crimson', '20': 'violet', '10': 'royalblue', '8': 'darkcyan'}
    kappa_list = []
    
    # t_min_pos = np.inf
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                Dx_Dy_Dz_difftime = pickle.load(f)
                Dx, Dy, Dz, diff_time = Dx_Dy_Dz_difftime[:,0], Dx_Dy_Dz_difftime[:,1], Dx_Dy_Dz_difftime[:,2], Dx_Dy_Dz_difftime[:,3]
                t_pos = diff_time[diff_time > 0] 
                # t_min_pos = min(t_min_pos, t_pos.min())
                kappa = get_dispersion_value_kappa(file_path)
                kappa_list.append(kappa)
                color_i = colors[kappa]

                mask = diff_time > 0.01
                inv_sqrt_t = 1.0 / np.sqrt(diff_time[mask]) 
                Dz_masked = Dz[mask]/Dz[0] 
                ax.plot(inv_sqrt_t, Dz_masked, color=color_i, marker='.', label='K='+str(kappa), markersize=10)
                
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    unqiue_kappa_list = []
    for handle, label, kappa in zip(handles, labels, kappa_list):
        if label not in unique_labels:
            unique_labels[label] = handle
            unqiue_kappa_list.append(kappa) 
    print('list(unique_labels.keys())', len((list(unique_labels.keys()))))
    print('list(unique_labels.values())', len(list(unique_labels.values())))       
    sorted_kappa_list, reordered_labels = sort_lists_together(unqiue_kappa_list, list(unique_labels.keys()))
    sorted_kappa_list, reordered_handles = sort_lists_together(unqiue_kappa_list, list(unique_labels.values()))
    ax.legend(handles=reordered_handles, labels=reordered_labels, fontsize=18)

    plot_file_name = os.path.join(folder_name, f'AD_wrt_kappa_d={diameter}_difftime_limit{diff_time_limit}.png')
    ax.tick_params(axis='both', which='major', labelsize=13)
    # ax.set_xlabel('t (ms)', fontsize=15)
    ax.set_xlabel(r'$1/\sqrt{t}\;(\mathrm{ms}^{-1/2})$', fontsize=15)
    # ax.set_ylabel('AD (µm$^2$/ms)', fontsize=15)
    ax.set_ylabel(r'$D_{\parallel}/D_0$', fontsize=15)
    # ax.set_title("$d$="+str(diameter)+"µm", fontsize=17)
    ax.set_title(r"$\overline{d}$="+str(diameter)+"µm", fontsize=17)
    ax.grid(True)
    # x_min = 1.0 / np.sqrt(10) # corresponds to the old right boundary in t 
    # x_max = 1.0 / np.sqrt(t_min_pos) # left boundary in t becomes right in 1/√t 
    ax.set_xlim([0.1, 0.5]) # 500ms - 1ms
    # ax.set_xlim([0.0001, diff_time_limit])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    plt.savefig(plot_file_name)
    print('Done plotting')


if __name__ == '__main__':
    AD_diff_time_limit = 100 #ms
    # RD_diff_time_limit = 100 #ms
    
    #d1.68
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d168/extra'
    plot_Dz_across_kappa(folder_path, AD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d168/intra'
    plot_Dz_across_kappa(folder_path, AD_diff_time_limit)        

    # d2.58
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d258/extra'
    plot_Dz_across_kappa(folder_path, AD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d258/intra'
    plot_Dz_across_kappa(folder_path, AD_diff_time_limit)   
    
    # d3.5
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d35/extra'
    plot_Dz_across_kappa(folder_path, AD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d35/intra'
    plot_Dz_across_kappa(folder_path, AD_diff_time_limit)          

    # d4.5
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d45/extra'
    plot_Dz_across_kappa(folder_path, AD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_WatsonK/2025-09-17/d45/intra'
    plot_Dz_across_kappa(folder_path, AD_diff_time_limit)          
    