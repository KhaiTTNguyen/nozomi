import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os.path
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
    
def _as_numpy(spheres_xyz_r_fid):
    if hasattr(spheres_xyz_r_fid, 'detach'):
        return spheres_xyz_r_fid.detach().cpu().numpy()
    return np.asarray(spheres_xyz_r_fid)


def _plot_component(ax, spheres_xyz_r_fid, color, overlap_indices, POV, alpha=1.0):
    fiber_list_xyz_r_fid = util.split_matrix_to_list(_as_numpy(spheres_xyz_r_fid))
    if len(fiber_list_xyz_r_fid) == 0:
        return fiber_list_xyz_r_fid

    if color is None:
        color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list_xyz_r_fid)))
        np.random.shuffle(color)

    unique_ids = np.unique([fiber[0][util.fiber_id_column(fiber)] for fiber in fiber_list_xyz_r_fid])
    fiber_idx = 0
    for fiber in fiber_list_xyz_r_fid:
        color_idx = np.argwhere(np.isin(unique_ids, fiber[0][util.fiber_id_column(fiber)])).ravel()[0]
        fiber_color = color[color_idx]
        plot_spheres_POV(ax, fiber, fiber_idx, fiber_color, overlap_indices, POV, alpha=alpha)
        fiber_idx = fiber_idx + fiber.shape[0]
    return fiber_list_xyz_r_fid


def _configure_axes(ax, POV):
    if POV == 'horizontal_90':
        ax.view_init(azim=-90, elev=0)
        ax.set_xlabel("x (µm)", fontsize=15, labelpad=13)
        ax.set_yticks([])
        ax.set_zticks([])
    elif POV == 'horizontal_0':
        ax.view_init(azim=0, elev=0)
        ax.set_ylabel("y (µm)", fontsize=15, labelpad=13)
        ax.set_xticks([])
        ax.set_zticks([])
    elif POV == 'topdown':
        ax.view_init(azim=-90, elev=90)
        ax.set_xlabel("x (µm)", fontsize=15, labelpad=13)
        ax.set_ylabel("y (µm)", fontsize=15, labelpad=13)
        ax.set_zticks([])
    else:
        ax.set_xlabel("x (µm)", fontsize=15, labelpad=13)
        ax.set_ylabel("y (µm)", fontsize=15, labelpad=13)
        ax.set_zlabel("z (µm)", fontsize=15, labelpad=13)


def _finish_plot(fig, ax):
    plot_box(ax)
    ax.set_xlim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    ax.set_ylim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    ax.set_zlim(-config_params.BOX_LENGTH/2, config_params.BOX_LENGTH/2)
    ax.tick_params(axis='both', which='major', labelsize=13)


def plot_fibers( spheres_xyz_r_fid, overlap_indices=None,color=None, animation_input=False, optimized=False, POV='default', num_iter=0, component_label=None, alpha=1.0):
    print("Plotting fibers in 3D..." \
    "Can take a few hours for a large number of fibers....")
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/visual/" 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    _configure_axes(ax, POV)
        
    #-------------------- plot fibers ----------------------
    fiber_list_xyz_r_fid = _plot_component(ax, spheres_xyz_r_fid, color, overlap_indices, POV, alpha=alpha)

    # ------------------- plot box edges ----------------------
    _finish_plot(fig, ax)
    label = "" if component_label is None else "_" + component_label
    title_label = "" if component_label is None else " - " + component_label
    if optimized==True:
        plt.title("3D plot of optimized fibers" + title_label, fontsize=17, pad=20)
        plt.savefig(folder_path+"/optimized"+label+"_"+str(POV)+"_POV_"+str(len(fiber_list_xyz_r_fid))+"_fibers_"+str(config_params.EXP_DATE_TIME)+'_VF_'+str(config_params.VOLUME_FRACTION)+".png", 
        dpi=500, edgecolor='b', format='png')
        if animation_input==True:
            def rotate(angle):
                ax.view_init(azim=angle,elev=0)
            print("Making animation")
            rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 5), interval=100)
            rot_animation.save(folder_path+"/GIF_optimized"+label+str(len(fiber_list_xyz_r_fid))+"_fibers_"+str(config_params.EXP_DATE_TIME)+'_VF_'+str(config_params.VOLUME_FRACTION)+".gif", dpi=500, writer='imagemagick')  
    else:   
        plt.title("3D plot of unoptimized fibers" + title_label, fontsize=17, pad=20)
        plt.savefig(folder_path+"/unoptimized"+label+"_"+str(POV)+"_POV_"+str(len(fiber_list_xyz_r_fid))+"_fibers"+str(config_params.EXP_DATE_TIME)+str(config_params.VOLUME_FRACTION)+"num_iter_"+str(num_iter)+".png", 
            dpi=500, edgecolor='b', format='png')
    plt.close(fig)


def plot_myelinated_fibers(outer_fibers, inner_fibers, color=None, POV='topdown'):
    print("Plotting myelinated fibers in 3D...")
    folder_path = config_params.SUBSTRATE_OUTPUT_FOLDER_PATH+"/figs/visual/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    outer_list = util.split_matrix_to_list(_as_numpy(outer_fibers))
    if color is None:
        color = cm.rainbow(np.linspace(0.0, 1.0, len(outer_list)))
        np.random.shuffle(color)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    _configure_axes(ax, POV)
    _plot_component(ax, outer_fibers, color, None, POV, alpha=0.22)
    _plot_component(ax, inner_fibers, color, None, POV, alpha=0.95)
    _finish_plot(fig, ax)
    plt.title("Myelinated substrate: outer and inner membranes", fontsize=17, pad=20)
    plt.savefig(folder_path+"/myelinated_outer_inner_"+str(POV)+"_POV_"+str(len(outer_list))+"_fibers_"+str(config_params.EXP_DATE_TIME)+'_VF_'+str(config_params.VOLUME_FRACTION)+".png",
        dpi=500, edgecolor='b', format='png')
    plt.close(fig)

def plot_spheres_POV( ax, fiber, fiber_idx, fiber_color, overlap_indices, POV, alpha=1.0):
    if POV=='horizontal_90':
        plot_spheres(ax, fiber, fiber_idx, fiber_color, overlap_indices=None, alpha=alpha)
    elif POV=='horizontal_0':
        plot_spheres(ax, fiber, fiber_idx, fiber_color, overlap_indices=None, alpha=alpha)
    elif POV=='topdown':
        plot_spheres(ax, fiber, fiber_idx, fiber_color, overlap_indices=None, alpha=alpha)
    else:
        plot_spheres(ax=ax, fiber_matrix=fiber, fiber_idx=fiber_idx, 
                            color=fiber_color, overlap_indices=overlap_indices, alpha=alpha)

def plot_spheres( ax, fiber_matrix, fiber_idx, color, overlap_indices=None, alpha=1.0):
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
            ax.plot_surface(sphere_x, sphere_y, sphere_z, color=color_sphere, alpha=alpha)
        else: 
            color_sphere = color
            ax.plot_surface(sphere_x, sphere_y, sphere_z, color=color_sphere, alpha=alpha)

