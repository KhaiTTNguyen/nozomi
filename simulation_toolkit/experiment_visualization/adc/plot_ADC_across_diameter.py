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

def plot_Dxy_across_diameter(folder_path, diff_time_limit):
    '''
    input ADCdata folder
    loop through folder, load all data
    plot data
    '''
    folder_name = os.path.join(folder_path, 'ADC_wrt_diameter')
    fig, ax = plt.subplots(figsize=(5,5))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")  
    
    num_files = count_num_files_in_folder(folder_path)
    print("Number of files:", num_files)

    path_to_first_file = get_path_to_first_file(folder_path)
    kappa = get_dispersion_value_kappa(path_to_first_file)
    
    colors = { '1.68': 'red', '2.58': 'blue', '3.5': 'green', '4.5': 'purple'}
    diameter_list = []
    i=0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                Dx_Dy_Dz_difftime = pickle.load(f)
                Dx, Dy, Dz, diff_time = Dx_Dy_Dz_difftime[:,0], Dx_Dy_Dz_difftime[:,1], Dx_Dy_Dz_difftime[:,2], Dx_Dy_Dz_difftime[:,3]
                Dxy = (Dx + Dy)/2
                diameter = extract_diameter_from_file_path(file_path)
                diameter_list.append(diameter)
                color_i = colors[diameter]
                ax.semilogx(diff_time, Dxy, color=color_i,  marker='.', label=str(diameter)+' µm', markersize=5)  
                i+=1
    handles, labels = ax.get_legend_handles_labels()
    # Adjust plot elements
    # Create a dictionary to store unique labels and their corresponding handles
    # print('list(unique_labels.keys())', type(handles))
    # print('list(unique_labels.values())', type(labels))
    
    unique_labels = {}
    unqiue_diameter_list = []
    for handle, label, diameter in zip(handles, labels, diameter_list):
        if label not in unique_labels:
            unique_labels[label] = handle
            unqiue_diameter_list.append(diameter) 
    print('list(unique_labels.keys())', len((list(unique_labels.keys()))))
    print('list(unique_labels.values())', len(list(unique_labels.values())))
    # Use the unique labels and handles to create the legend
    sorted_diameter_list, reordered_labels = sort_lists_together(unqiue_diameter_list, list(unique_labels.keys()))
    sorted_diameter_list, reordered_handles = sort_lists_together(unqiue_diameter_list, list(unique_labels.values()))
    ax.legend(handles=reordered_handles, labels=reordered_labels, fontsize=18)

    plot_file_name = os.path.join(folder_name, 'RD_wrt_diameter'+'_K='+str(kappa)+'_difftime_limit'+str(diff_time_limit)+'.png')
    ax.tick_params(axis='both', which='major', labelsize=13)
    # ax.set_xlabel('t (ms)', fontsize=15)
    # ax.set_ylabel('RD (µm$^2$/ms)', fontsize=15) 
    ax.set_title(" $\kappa$="+str(kappa), fontsize=17)
    ax.grid(True)
    ax.set_xlim([0.002, diff_time_limit])
    ax.set_ylim([0.0, 3.2])
    plt.savefig(plot_file_name)
    print('Done plotting')

def plot_Dz_across_diameter(folder_path, diff_time_limit):
    '''
    input ADCdata folder
    loop through folder, load all data
    plot data
    '''
    folder_name = os.path.join(folder_path, 'ADC_wrt_diameter')
    fig, ax = plt.subplots()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")  
    
    num_files = count_num_files_in_folder(folder_path)
    print("Number of files:", num_files)

    path_to_first_file = get_path_to_first_file(folder_path)
    kappa = get_dispersion_value_kappa(path_to_first_file)
    
    colors = { '1.68': 'red', '2.58': 'blue', '3.5': 'green', '4.5': 'purple'}
    diameter_list = []
    i=0
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                Dx_Dy_Dz_difftime = pickle.load(f)
                Dx, Dy, Dz, diff_time = Dx_Dy_Dz_difftime[:,0], Dx_Dy_Dz_difftime[:,1], Dx_Dy_Dz_difftime[:,2], Dx_Dy_Dz_difftime[:,3]
                diameter = extract_diameter_from_file_path(file_path)
                diameter_list.append(diameter)
                color_i = colors[diameter]
                ax.semilogx(diff_time, Dz, color=color_i,  marker='.', label=str(diameter)+' µm', markersize=10)  
                i+=1
    handles, labels = ax.get_legend_handles_labels()
    # Adjust plot elements
    # Create a dictionary to store unique labels and their corresponding handles
    # print('list(unique_labels.keys())', type(handles))
    # print('list(unique_labels.values())', type(labels))
    
    unique_labels = {}
    unqiue_diameter_list = []
    for handle, label, diameter in zip(handles, labels, diameter_list):
        if label not in unique_labels:
            unique_labels[label] = handle
            unqiue_diameter_list.append(diameter) 
    print('list(unique_labels.keys())', len((list(unique_labels.keys()))))
    print('list(unique_labels.values())', len(list(unique_labels.values())))
    # Use the unique labels and handles to create the legend
    sorted_diameter_list, reordered_labels = sort_lists_together(unqiue_diameter_list, list(unique_labels.keys()))
    sorted_diameter_list, reordered_handles = sort_lists_together(unqiue_diameter_list, list(unique_labels.values()))
    ax.legend(handles=reordered_handles, labels=reordered_labels, fontsize=18)

    plot_file_name = os.path.join(folder_name, 'AD_wrt_diameter'+'_K='+str(kappa)+'_difftime_limit'+str(diff_time_limit)+'.png')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel('t (ms)', fontsize=15)
    ax.set_ylabel('AD (µm$^2$/ms)', fontsize=15) 
    ax.set_title(" Axial diffusivity vs. diffusion time at $\kappa$="+str(kappa), fontsize=17)
    ax.set_xlim([0.0001, diff_time_limit])
    ax.set_ylim([0.0, 3.2])
    ax.axis('off')
    ax.grid(True)
    plt.savefig(plot_file_name)
    print('Done plotting')


if __name__ == '__main__':
    RD_diff_time_limit = 120 #ms
    AD_diff_time_limit = 120
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-09-17-K10/intra'
    plot_Dxy_across_diameter(folder_path, RD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-09-17-K10/extra'
    plot_Dxy_across_diameter(folder_path, RD_diff_time_limit)          
    
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-09-17-K20/intra'
    plot_Dxy_across_diameter(folder_path, RD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-09-17-K20/extra'
    plot_Dxy_across_diameter(folder_path, RD_diff_time_limit)          
    
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-09-17-K200/intra'
    plot_Dxy_across_diameter(folder_path, RD_diff_time_limit)          
    folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-09-17-K200/extra'
    plot_Dxy_across_diameter(folder_path, RD_diff_time_limit)          
    
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K200/intra'
    # plot_Dxy_across_diameter(folder_path, RD_diff_time_limit)          
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K200/extra'
    # plot_Dxy_across_diameter(folder_path, RD_diff_time_limit)          
    
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K8/intra'
    # plot_Dz_across_diameter(folder_path, AD_diff_time_limit)          
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K8/extra'
    # plot_Dz_across_diameter(folder_path, AD_diff_time_limit)          
    
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K10/intra'
    # plot_Dz_across_diameter(folder_path, AD_diff_time_limit)          
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K10/extra'
    # plot_Dz_across_diameter(folder_path, AD_diff_time_limit)          
    
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K20/intra'
    # plot_Dz_across_diameter(folder_path, AD_diff_time_limit)          
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K20/extra'
    # plot_Dz_across_diameter(folder_path, AD_diff_time_limit)          
    
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K200/intra'
    # plot_Dz_across_diameter(folder_path, AD_diff_time_limit)          
    # folder_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/plot_across_diameter/2025-04-26-K200/extra'
    # plot_Dz_across_diameter(folder_path, AD_diff_time_limit)          
    
