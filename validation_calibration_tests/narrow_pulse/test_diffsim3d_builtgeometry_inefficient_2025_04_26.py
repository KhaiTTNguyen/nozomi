# import pycuda.autoinit
# import pycuda.driver as drv
# import pycuda.gpuarray as gpuarray
# import numpy as np
# import unittest
# import diffsim3d.diffsim3d as ds3
# import diffsim3d.geometry as geom
# import matplotlib.pyplot as pl
# from geometrygen.util import *
# import geometrygen.import_geometry as ig
# import geometrygen.adjust_geometry as ag
# from matplotlib.pyplot import cm
# import time
# import os 

# class TestDiffSim3D(unittest.TestCase):

#     def setUp(self):
#         drv.Device(0).make_context()
    
#     def tearDown(self):
#         drv.Context.pop()
        
#     #-------------------------------------------------------------------------------------------
#     # def testInsideMultiFiberGeometry(self):
#     #     st = time.time()
#     #     #d35-k10
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_19-29-30/data/array32_fibers_boxL_42_2025-04-26_19-29-30_avf_0.57_d3.5_sig0.7w_o10_w_c3_w_l3_K10_ODI_0.0635_bffse0.01_bffsph0.001_1164.01_sec.pkl'
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_19-31-16_array31_fibers_avf_0.6_d3.5_sig0.7w_K10_ODI_0.0635/data/array31_fibers_boxL_42_2025-04-26_19-31-16_avf_0.6_d3.5_sig0.7w_o10_w_c3_w_l3_K10_ODI_0.0635_bffse0.01_bffsph0.001_1297.64_sec.pkl'
# #         #d35-k8
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_22-14-38/data/array30_fibers_boxL_42_2025-04-26_22-14-38_avf_0.58_d3.5_sig0.7w_o10_w_c3_w_l3_K8_ODI_0.0792_bffse0.01_bffsph0.001_1235.42_sec.pkl'
#     #     file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_22-16-23/data/array31_fibers_boxL_42_2025-04-26_22-16-23_avf_0.65_d3.5_sig0.7w_o10_w_c3_w_l3_K8_ODI_0.0792_bffse0.01_bffsph0.001_1195.46_sec.pkl'
#     #     #d258-k10
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_19-33-31_array61_fibers_avf_0.58_d2.58_sig0.516_K10_ODI_0.0635/data/array61_fibers_boxL_42_2025-04-26_19-33-31_avf_0.58_d2.58_sig0.516w_o10_w_c3_w_l3_K10_ODI_0.0635_bffse0.01_bffsph0.001_1593.43_sec.pkl'
#     #     # ============== EXPERIEMNT HERE UPWARDS =========================

#     #     import os
#     #     target_folder_path = os.path.dirname( os.path.dirname(file_path) )
#     #     folder_name = os.path.join(target_folder_path, 'ADC')
#     #     if not os.path.exists(folder_name):
#     #         os.makedirs(folder_name)
#     #         print("Folder created successfully.")
#     #     else:
#     #         print("Folder already exists.")  
#     #     optimized_fibers, L = ig.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
#     #     fiberlist_xyz_r_fid = ag.split_matrix_to_list(optimized_fibers, L)
#     #     print('L box length',L)
#     #     D = 3.0 # um^2/ms
#     #     T2 = 100 # ms
#     #     rho = 1 # fractional water density
#     #     sg3 = geom.SimGeometry3D(L,L,L,D,T2,rho)
#     #     fiber_xyzr_fid_list = ag.createBoundaryZ(fiberlist_xyz_r_fid, L)

#     #     # ag.plot_sphere_fibers(fiber_xyzr_fid_list, L, folder_name)
        
#     #     for fiber in fiber_xyzr_fid_list:
#     #         sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
#     #         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#     #         sg3.add_structure(spstruc) # add it to sg3
#     #         # print('fiber[:,4]', fiber[:,4])
#     #     dt = 0.0001 # time step in ms
#     #     # nt = 20
#     #     nt = int(2e6) # total number of steps thru time
#     #     spins = int(1e5)
#     #     num_spins = spins
#     #     print('Start setting up structures')
#     #     sim = ds3.DiffSim3d(sg3,spins)
#     #     nsegx,nsegy,nsegz=20,20,20
#     #     sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)
#     #     sim.setup(structures=list(np.arange(0, len(fiber_xyzr_fid_list))))  # seed inside structures 0:len(fiberlist)
#     #     print('Done setting up structures')

#     #     # first, check thath all the spins are in fact in the multiple spstruc's
#     #     isInside = False
#     #     for structure in sg3.structures[:-1]:
#     #         isInside = np.logical_or(isInside, structure.isinside(sim.spins))
#     #     self.assertTrue(isInside.all())

#     #     Dx_step = [D]
#     #     Dy_step = [D]
#     #     Dz_step = [D]
#     #     diff_time = [0]
#     #     print('Got to sim intra-axonal simloops')
#     #     current_time = 0.0
#     #     # max_time = 10 #10 #3000  
#     #     while current_time < nt*dt:
#     #     # while current_time < max_time:
#     #         time_step = dt
#     #         current_time+=time_step
#     #         # --------step-----------
#     #         sim.step(time_step)
#     #         Dx_step.append(np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(current_time))
#     #         Dy_step.append(np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(current_time))
#     #         Dz_step.append(np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(current_time))
#     #         diff_time.append(current_time)
#     #     Dx_step = np.array(Dx_step)
#     #     Dy_step = np.array(Dy_step)
#     #     Dz_step = np.array(Dz_step)
#     #     diff_time = np.array(diff_time)
#     #     et = time.time()
#     #     # get the execution time
#     #     elapsed_time = np.round(et - st,2)
        
#     #     #==================================
#     #     file_name = os.path.basename(file_path)
#     #     base_name, extension = os.path.splitext(file_name)
#     #     date_time = str(get_date_time())
#     #     file_name_new = 'intra_'+ date_time + '_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' \
#     #         + str(num_spins)+'_spins_'+ base_name +'_SIMtime'+str(round(elapsed_time,2))+'_sec'+ '_dt' + str(dt)+'_seg'+str(int(nsegx))
#     #     print('file_name_new', file_name_new)
#     #     # Combine new filename with folder path to get the full path
#     #     data_folder_name = os.path.join(target_folder_path, 'ADC', 'ADCdata')
#     #     if not os.path.exists(data_folder_name):
#     #         os.makedirs(data_folder_name)
#     #     data_file_path = os.path.join(data_folder_name, str(int(nsegx))+'SEGMENT_'+file_name_new+'data.pkl')
#     #     save_ADCdata_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
#     #     plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new )    

#     #     # after simulating diffusion, are all spins still inside sg3?
#     #     isInside = False
#     #     for struct in sg3.structures[:-1]:
#     #         isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
#     #     self.assertTrue(isInside.all())
#     #     print('intra elapsed_time', elapsed_time)


#     def testOutsideMultiFiberGeometry(self):
#         #d35-K10
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_19-31-16_array31_fibers_avf_0.6_d3.5_sig0.7w_K10_ODI_0.0635/data/array31_fibers_boxL_42_2025-04-26_19-31-16_avf_0.6_d3.5_sig0.7w_o10_w_c3_w_l3_K10_ODI_0.0635_bffse0.01_bffsph0.001_1297.64_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_19-29-30/data/array32_fibers_boxL_42_2025-04-26_19-29-30_avf_0.57_d3.5_sig0.7w_o10_w_c3_w_l3_K10_ODI_0.0635_bffse0.01_bffsph0.001_1164.01_sec.pkl'
        
#         #d35-k8
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_22-14-38/data/array30_fibers_boxL_42_2025-04-26_22-14-38_avf_0.58_d3.5_sig0.7w_o10_w_c3_w_l3_K8_ODI_0.0792_bffse0.01_bffsph0.001_1235.42_sec.pkl'
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_22-16-23/data/array31_fibers_boxL_42_2025-04-26_22-16-23_avf_0.65_d3.5_sig0.7w_o10_w_c3_w_l3_K8_ODI_0.0792_bffse0.01_bffsph0.001_1195.46_sec.pkl'
#         #d258-k10
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-26_19-33-31_array61_fibers_avf_0.58_d2.58_sig0.516_K10_ODI_0.0635/data/array61_fibers_boxL_42_2025-04-26_19-33-31_avf_0.58_d2.58_sig0.516w_o10_w_c3_w_l3_K10_ODI_0.0635_bffse0.01_bffsph0.001_1593.43_sec.pkl'

#         import os
#         target_folder_path = os.path.dirname( os.path.dirname(file_path) )
#         folder_name = os.path.join(target_folder_path, 'ADC')
#         if not os.path.exists(folder_name):
#             os.makedirs(folder_name)
#             print("Folder created successfully.")
#         else:
#             print("Folder already exists.")  
#         optimized_fibers, L = ig.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
#         fiberlist_xyz_r_fid = ag.split_matrix_to_list(optimized_fibers, L)
#         print('L box length',L)
#         D = 3.0 # um^2/ms
#         T2 = 100 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(L,L,L,D,T2,rho)
#         fiber_xyzr_fid_list = ag.createBoundaryZ(fiberlist_xyz_r_fid, L)

#         for fiber in fiber_xyzr_fid_list:
#             sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
#             spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#             sg3.add_structure(spstruc) # add it to sg3
#         dt = 0.0001 # time step in ms
#         # nt = int(6)
#         nt = int(2e6) # total number of steps thru time
#         spins = int(1e5)
#         num_spins = spins
#         print('Start setting up structures')
#         sim = ds3.DiffSim3d(sg3,spins) 
#         nsegx,nsegy,nsegz=20,20,20
#         sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)
#         sim.setup(structures=[int(len(fiber_xyzr_fid_list))])  # seed OUTSIDE structures len(fiberlist)=index of box
#         print('Done setting up structures')
        
#         # first, check thath all the spins are in fact in the multiple spstruc's
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         self.assertTrue(isOutside.all())

#         st = time.time()
#         Dx_step = [D]
#         Dy_step = [D]
#         Dz_step = [D]
#         diff_time = [0]
#         print('Got to sim extra-axonal simloops')
#         current_time = 0.0
#         # max_time = 10 #10 #3000
#         while current_time < nt*dt:
#         # while current_time < max_time:
#             time_step = dt
#             current_time+=time_step
#             # --------step-----------
#             sim.step(time_step)
#             Dx_step.append(np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(current_time))
#             Dy_step.append(np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(current_time))
#             Dz_step.append(np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(current_time))
#             diff_time.append(current_time)
#         Dx_step = np.array(Dx_step)
#         Dy_step = np.array(Dy_step)
#         Dz_step = np.array(Dz_step)
#         diff_time = np.array(diff_time)
#         et = time.time()
#         # # get the execution time
#         elapsed_time = np.round(et - st,2)

#         #==================================
#         file_name = os.path.basename(file_path)
#         base_name, extension = os.path.splitext(file_name)
#         date_time = str(get_date_time())
#         file_name_new = 'extra_'+ date_time + '_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' \
#             + str(num_spins)+'_spins_'+ base_name +'_SIMtime'+str(round(elapsed_time,2))+'_sec'+ '_dt' + str(dt)+'_seg'+str(int(nsegx))
#         print('file_name_new', file_name_new)
#         # Combine new filename with folder path to get the full path
#         data_folder_name = os.path.join(target_folder_path, 'ADC', 'ADCdata')
#         if not os.path.exists(data_folder_name):
#             os.makedirs(data_folder_name)
#             print("Folder created successfully.")
#         else:
#             print("Folder already exists.")  
#         data_file_path = os.path.join(data_folder_name, str(int(nsegx))+'SEGMENT_'+file_name_new+'data.pkl')
#         save_ADCdata_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
#         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new )    

#         # after simulating diffusion, are all spins still outside axons
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         self.assertTrue(isOutside.all())
#         print('elapsed_time', elapsed_time)
# if __name__ == '__main__':
#     unittest.main()