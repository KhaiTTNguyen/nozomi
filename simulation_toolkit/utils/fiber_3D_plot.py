import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os.path
import torch
from mpl_toolkits.mplot3d import Axes3D
import simulation_toolkit.utils.common_utils as util
import simulation_toolkit.toolkit_params as config_params

def plot_box(ax):
    box_vertices, box_x, box_y, box_z = prepare_edges_for_box()
    ax.plot3D(box_x, box_y, box_z, color='b', lw='1.')

def prepare_edges_for_box():
    bx = config_params.BOX_LENGTH.cpu().numpy()
    by = config_params.BOX_LENGTH.cpu().numpy()
    bz = config_params.BOX_LENGTH.cpu().numpy()
    
    box_x = np.array([(0,bx,bx,0 ,0 ,bx,bx,0)]) - bx/2
    box_y = np.array([(0,0 ,by,by,0 ,0 ,by,by)]) - by/2
    box_z = np.array([(0,0 ,0 ,0 ,bz,bz,bz,bz)]) - bz/2
    box_vertices = np.vstack([box_x, box_y, box_z]).T

    edges = [(0,1), (1,2),(2,3),(3,0),(0,4),(4,5),
                (5,6),(6,7), (7,4),(4,7),(7,3),(3,2),(2,6),(6,5),(5,1)]
    # draw lines connecting nodes
    line_x, line_y, line_z = prepare_edges_for_plot(points=box_vertices, edges=edges)
    return box_vertices, line_x, line_y, line_z

def prepare_edges_for_plot( points, edges):
    line_x = np.array([])
    line_y = np.array([])
    line_z = np.array([])
    # draw lines connecting nodes
    for (i,j) in edges:
        line_x = np.append(line_x, [points[i, 0], points[j, 0]])      
        line_y = np.append(line_y, [points[i, 1], points[j, 1]])      
        line_z = np.append(line_z, [points[i, 2], points[j, 2]])
    return line_x, line_y, line_z
    
def plot_fibers( spheres_xyz_r_fid, overlap_indices=None,color=None, animation_input=False, optimized=False, POV='default', num_iter=0):
    fiber_list_xyz_r_fid = util.split_matrix_to_list(spheres_xyz_r_fid.detach().cpu().numpy())
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/visual/" 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if POV=='top_down':
        ax.view_init(azim=-90, elev=90)
    elif POV=='bottom_up':
        ax.view_init(azim=-90, elev=-90)
    elif POV=='horizontal_90':
        ax.view_init(azim=-90, elev=0)
    elif POV=='horizontal_0':
        ax.view_init(azim=0, elev=0)
    
    #-------------------- plot fibers ----------------------
    if color is None:
        from matplotlib.pyplot import cm    
        color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list_xyz_r_fid)))
        np.random.shuffle(color)
    'get index of fiber.original_id in unique_ids --> map to color'
    unique_ids = np.unique([fiber[0][-1] for fiber in fiber_list_xyz_r_fid])
    fiber_idx=0
    for fiber_count, fiber in enumerate(fiber_list_xyz_r_fid):
        color_idx = np.argwhere(np.isin(unique_ids , fiber[0][-1])).ravel()[0]
        fiber_color = color[color_idx]
        plot_spheres_POV(ax, fiber, fiber_idx, fiber_color, overlap_indices, POV)
        fiber_idx=fiber_idx+fiber.shape[0]

    # ------------------- plot box edges ----------------------
    # if POV=='default':
    #     ax.set_axis_off()   
    plot_box(ax) 
    ax.set_xlim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    ax.set_ylim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    ax.set_zlim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    ax.set_xlabel("x (µm)", fontsize=15, labelpad=13)
    ax.set_ylabel("y (µm)", fontsize=15, labelpad=13)
    ax.set_zlabel("z (µm)", fontsize=15, labelpad=13)
    ax.tick_params(axis='both', which='major', labelsize=13)
    if optimized==True:
        plt.title("3D plot of optimized fibers", fontsize=17, pad=20)
        plt.savefig(folder_path+"/array_optimized_"+str(POV)+"_POV_"+str(len(fiber_list_xyz_r_fid))+"_fibers"+str(config_params.EXP_DATE_TIME)+'_avf_'+str(config_params.VOLUME_FRACTION)+".png", 
        dpi=500, edgecolor='b', format='png')
        if animation_input==True:
            def rotate(angle):
                ax.view_init(azim=angle,elev=0)
            print("Making animation")
            rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 5), interval=100)
            rot_animation.save(folder_path+"/GIF_array_optimized"+str(len(fiber_list_xyz_r_fid))+"_fibers_"+str(config_params.EXP_DATE_TIME)+'_avf_'+str(config_params.VOLUME_FRACTION)+".gif", dpi=500, writer='imagemagick')  
    else:   
        plt.title("3D plot of unoptimized fibers", fontsize=17, pad=20)
        plt.savefig(folder_path+"/array_unoptimized_"+str(POV)+"_POV_"+str(len(fiber_list_xyz_r_fid))+"_fibers"+str(config_params.EXP_DATE_TIME)+str(config_params.VOLUME_FRACTION)+"num_iter_"+str(num_iter)+".png", 
            dpi=500, edgecolor='b', format='png')
    plt.close(fig)

def plot_spheres_POV( ax, fiber, fiber_idx, fiber_color, overlap_indices, POV):
    if POV=='top_down':
        plot_spheres(ax, fiber, fiber_idx, fiber_color, overlap_indices=None)
    elif POV=='bottom_up':
        plot_spheres(ax, fiber, fiber_idx, fiber_color, overlap_indices=None)
    elif POV=='horizontal_90':
        plot_spheres(ax, fiber, fiber_idx, fiber_color, overlap_indices=None)
    elif POV=='horizontal_0':
        plot_spheres(ax, fiber, fiber_idx, fiber_color, overlap_indices=None)
    else:
        plot_spheres(ax=ax, fiber_matrix=fiber, fiber_idx=fiber_idx, 
                            color=fiber_color, overlap_indices=overlap_indices)

def plot_spheres( ax, fiber_matrix, fiber_idx, color, overlap_indices=None):
    color_sphere=0
    u = np.linspace(0, 2 * np.pi, 8) # used to be  np.linspace(0, 2 * np.pi, 8)
    v = np.linspace(0, np.pi, 8)
    for sphere_idx, node in enumerate(fiber_matrix):
        sphere_x = node[3] * np.outer(np.cos(u), np.sin(v)) + node[0]
        sphere_y = node[3] * np.outer(np.sin(u), np.sin(v)) + node[1]
        sphere_z = node[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + node[2]
        
        current_idx = (sphere_idx+fiber_idx)
        if np.any(overlap_indices == current_idx): 
            color_sphere = 'red' 
            ax.plot_surface(sphere_x, sphere_y, sphere_z, color=color_sphere, alpha=1.)
            # print('red sphere plotted')
        else: 
            color_sphere = color
            ax.plot_surface(sphere_x, sphere_y, sphere_z, color=color_sphere, alpha=1.)

