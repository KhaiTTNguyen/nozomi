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
#     def testInsideMultiFiberGeometry(self):
#         st = time.time()
#         # experiments
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-15_06-53-20/data/array440_fibers_boxL_40_2025-04-15_06-53-20_avf_0.64_d0.82_sig0.164w_o10_w_c3_w_l3_K20_ODI_0.0318_bffse0.24_bffsph0.001_5866.09_sec.pkl'

#         #d082-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_18-14-36_440_fibers_boxL_40_avf_0.64_d0.82_sig0.164w_o10_w_c3_w_l5_K20_ODI_0.0318/data/array440_fibers_boxL_40_2025-03-11_18-14-36_avf_0.63_d0.82_sig0.164w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_11592.71_sec.pkl'
#         #d082-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_18-10-45_500_fibers_boxL_40_avf_0.67_d0.82_sig0.164w_o10_w_c3_w_l5_K200_ODI_0.0032/data/array500_fibers_boxL_40_2025-03-11_18-10-45_avf_0.67_d0.82_sig0.164w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_7121.67_sec.pkl'
        
#         #d1.34-K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_16-06-44_210_fibers_boxL_40_avf_0.65_d1.34_sig0.268_K200_ODI_0.0032/data/array210_fibers_boxL_40_2025-03-04_16-06-44_avf_0.65_d1.34_sig0.268w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_3204.36_sec.pkl'
#         # d134-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_16-09-07_210_fibers_boxL_40_avf_0.68_d1.34_sig0.268_K20_ODI_0.0318/data/array210_fibers_boxL_40_2025-03-04_16-09-07_avf_0.68_d1.34_sig0.268w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_6592.36_sec.pkl'
#         # d134-k75
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_16-12-54_195_fibers_boxL_40_avf_0.65_d1.34_sig0.268_K7_ODI_0.0903/data/array195_fibers_boxL_40_2025-03-04_16-12-54_avf_0.65_d1.34_sig0.268w_o10_w_c3_w_l5_K7_ODI_0.0903_bffse0.24_bffsph0.001_17747.0_sec.pkl'
#         #d134-k45
#         # file_path = ''
        
#         #d2.58-K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-25_03-59-18_array72_fibers_boxL_40_avf_0.79_d2.58_sig0.516_K200_ODI_0.0032/data/array72_fibers_boxL_40_2025-02-25_03-59-18_avf_0.79_d2.58_sig0.516w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_2450.6_sec.pkl'
#         #d2.58-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-25_04-02-02_array72_fibers_boxL_40_avf_0.69_d2.58_sig0.516_K20_ODI_0.0318/data/array72_fibers_boxL_40_2025-02-25_04-02-02_avf_0.69_d2.58_sig0.516w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_1553.98_sec.pkl'
#         #d2.58-k75
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_15-53-55_65_fibers_boxL_40_avf_0.67_d2.58_sig0.516_K7_ODI_0.0903/data/array65_fibers_boxL_40_2025-03-04_15-53-55_avf_0.67_d2.58_sig0.516w_o10_w_c3_w_l5_K7_ODI_0.0903_bffse0.24_bffsph0.001_3114.69_sec.pkl'
#         #d2.58-k45
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_15-55-57_60_fibers_boxL_40_avf_0.66_d2.58_sig0.516_K4_ODI_0.1392/data/array60_fibers_boxL_40_2025-03-04_15-55-57_avf_0.66_d2.58_sig0.516w_o10_w_c3_w_l5_K4_ODI_0.1392_bffse0.24_bffsph0.001_20426.05_sec.pkl'


#         #d45-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_17-07-04_25_fibers_boxL_40_avf_0.73_d4.5_sig0.9w_K200_ODI_0.0032/data/array25_fibers_boxL_40_2025-03-11_17-07-04_avf_0.73_d4.5_sig0.9w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_2220.32_sec.pkl'
#         #d45-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_16-46-55_22_fibers_boxL_40_avf_0.66_d4.5_sig0.9w_o10_w_c3_w_l5_K20_ODI_0.0318/data/array22_fibers_boxL_40_2025-03-11_16-46-55_avf_0.66_d4.5_sig0.9w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_2051.45_sec.pkl'
#         #d45_k75
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_16-56-09_avf_0.64_d4.5_sig0.9_K7/data/array19_fibers_boxL_40_2025-03-11_16-56-09_avf_0.64_d4.5_sig0.9w_o10_w_c3_w_l5_K7_ODI_0.0844_bffse0.24_bffsph0.001_4317.47_sec.pkl'
#         #d45_k45
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_18-37-14_array18_fibers_boxL_40_avf_0.65_d4.5_sig0.9_K4_ODI_0.1392_bffse0.24/data/array18_fibers_boxL_40_2025-03-11_18-37-14_avf_0.65_d4.5_sig0.9w_o10_w_c3_w_l5_K4_ODI_0.1392_bffse0.24_bffsph0.001_7223.2_sec.pkl'
        
#         # ============== EXPERIEMNT HERE UPWARDS =========================

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

#         # ag.plot_sphere_fibers(fiber_xyzr_fid_list, L, folder_name)
        
#         for fiber in fiber_xyzr_fid_list:
#             sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
#             spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#             sg3.add_structure(spstruc) # add it to sg3
#             # print('fiber[:,4]', fiber[:,4])
#         dt = 0.0001 # time step in ms
#         # nt = 20
#         nt = int(1e6) # total number of steps thru time
#         # nt = int(9e6) # total number of steps thru time
#         spins = int(1e5)
#         num_spins = spins
#         print('Start setting up structures')
#         sim = ds3.DiffSim3d(sg3,spins)
#         nsegx,nsegy,nsegz=20,20,20
#         sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)
#         sim.setup(structures=list(np.arange(0, len(fiber_xyzr_fid_list))))  # seed inside structures 0:len(fiberlist)
#         print('Done setting up structures')

#         # first, check thath all the spins are in fact in the multiple spstruc's
#         isInside = False
#         for structure in sg3.structures[:-1]:
#             isInside = np.logical_or(isInside, structure.isinside(sim.spins))
#         self.assertTrue(isInside.all())

#         Dx_step = [D]
#         Dy_step = [D]
#         Dz_step = [D]
#         diff_time = [0]
#         print('Got to sim intra-axonal simloops')
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
#         # get the execution time
#         elapsed_time = np.round(et - st,2)
        
#         #==================================
#         file_name = os.path.basename(file_path)
#         base_name, extension = os.path.splitext(file_name)
#         date_time = str(get_date_time())
#         file_name_new = 'intra_'+ date_time + '_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' \
#             + str(num_spins)+'_spins_'+ base_name +'_SIMtime'+str(round(elapsed_time,2))+'_sec'+ '_dt' + str(dt)+'_seg'+str(int(nsegx))
#         print('file_name_new', file_name_new)
#         # Combine new filename with folder path to get the full path
#         data_folder_name = os.path.join(target_folder_path, 'ADC', 'ADCdata')
#         if not os.path.exists(data_folder_name):
#             os.makedirs(data_folder_name)
#         data_file_path = os.path.join(data_folder_name, str(int(nsegx))+'SEGMENT_'+file_name_new+'data.pkl')
#         # save_ADCdata_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
#         # plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new )    

#         # after simulating diffusion, are all spins still inside sg3?
#         isInside = False
#         for struct in sg3.structures[:-1]:
#             isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
#         self.assertTrue(isInside.all())
#         print('intra elapsed_time', elapsed_time)


#     # def testOutsideMultiFiberGeometry(self):
        
#     #     # test
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-04-15_06-53-20/data/array440_fibers_boxL_40_2025-04-15_06-53-20_avf_0.64_d0.82_sig0.164w_o10_w_c3_w_l3_K20_ODI_0.0318_bffse0.24_bffsph0.001_5866.09_sec.pkl'

#     #     # d082-k200
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_18-10-45_500_fibers_boxL_40_avf_0.67_d0.82_sig0.164w_o10_w_c3_w_l5_K200_ODI_0.0032/data/array500_fibers_boxL_40_2025-03-11_18-10-45_avf_0.67_d0.82_sig0.164w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_7121.67_sec.pkl'
#     #     # d082-k20
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_18-14-36_440_fibers_boxL_40_avf_0.64_d0.82_sig0.164w_o10_w_c3_w_l5_K20_ODI_0.0318/data/array440_fibers_boxL_40_2025-03-11_18-14-36_avf_0.63_d0.82_sig0.164w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_11592.71_sec.pkl'

#     #     # d1.34-K200
#     #     file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_16-06-44_210_fibers_boxL_40_avf_0.65_d1.34_sig0.268_K200_ODI_0.0032/data/array210_fibers_boxL_40_2025-03-04_16-06-44_avf_0.65_d1.34_sig0.268w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_3204.36_sec.pkl'
#     #     # d134-k20
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_16-09-07_210_fibers_boxL_40_avf_0.68_d1.34_sig0.268_K20_ODI_0.0318/data/array210_fibers_boxL_40_2025-03-04_16-09-07_avf_0.68_d1.34_sig0.268w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_6592.36_sec.pkl'
#     #     # d134-k75
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_16-12-54_195_fibers_boxL_40_avf_0.65_d1.34_sig0.268_K7_ODI_0.0903/data/array195_fibers_boxL_40_2025-03-04_16-12-54_avf_0.65_d1.34_sig0.268w_o10_w_c3_w_l5_K7_ODI_0.0903_bffse0.24_bffsph0.001_17747.0_sec.pkl'

#     #     # d2.58-K200
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-25_03-59-18_array72_fibers_boxL_40_avf_0.79_d2.58_sig0.516_K200_ODI_0.0032/data/array72_fibers_boxL_40_2025-02-25_03-59-18_avf_0.79_d2.58_sig0.516w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_2450.6_sec.pkl'
#     #     # d2.58-k20
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-25_04-02-02_array72_fibers_boxL_40_avf_0.69_d2.58_sig0.516_K20_ODI_0.0318/data/array72_fibers_boxL_40_2025-02-25_04-02-02_avf_0.69_d2.58_sig0.516w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_1553.98_sec.pkl'
#     #     # d2.58-k75
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_15-53-55_65_fibers_boxL_40_avf_0.67_d2.58_sig0.516_K7_ODI_0.0903/data/array65_fibers_boxL_40_2025-03-04_15-53-55_avf_0.67_d2.58_sig0.516w_o10_w_c3_w_l5_K7_ODI_0.0903_bffse0.24_bffsph0.001_3114.69_sec.pkl'
#     #     # d2.58-k45
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-04_15-55-57_60_fibers_boxL_40_avf_0.66_d2.58_sig0.516_K4_ODI_0.1392/data/array60_fibers_boxL_40_2025-03-04_15-55-57_avf_0.66_d2.58_sig0.516w_o10_w_c3_w_l5_K4_ODI_0.1392_bffse0.24_bffsph0.001_20426.05_sec.pkl'


#     #     # d45-k200 RUN
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_17-07-04_25_fibers_boxL_40_avf_0.73_d4.5_sig0.9w_K200_ODI_0.0032/data/array25_fibers_boxL_40_2025-03-11_17-07-04_avf_0.73_d4.5_sig0.9w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_2220.32_sec.pkl'
#     #     # d45-k20 RUN
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_16-46-55_22_fibers_boxL_40_avf_0.66_d4.5_sig0.9w_o10_w_c3_w_l5_K20_ODI_0.0318/data/array22_fibers_boxL_40_2025-03-11_16-46-55_avf_0.66_d4.5_sig0.9w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_2051.45_sec.pkl'
#     #     # d45_k75
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_16-56-09_avf_0.64_d4.5_sig0.9_K7/data/array19_fibers_boxL_40_2025-03-11_16-56-09_avf_0.64_d4.5_sig0.9w_o10_w_c3_w_l5_K7_ODI_0.0844_bffse0.24_bffsph0.001_4317.47_sec.pkl'
#     #     # d45_k45
#     #     # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-03-11_18-37-14_array18_fibers_boxL_40_avf_0.65_d4.5_sig0.9_K4_ODI_0.1392_bffse0.24/data/array18_fibers_boxL_40_2025-03-11_18-37-14_avf_0.65_d4.5_sig0.9w_o10_w_c3_w_l5_K4_ODI_0.1392_bffse0.24_bffsph0.001_7223.2_sec.pkl'
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

#     #     for fiber in fiber_xyzr_fid_list:
#     #         sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
#     #         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#     #         sg3.add_structure(spstruc) # add it to sg3
#     #     dt = 0.0001 # time step in ms
#     #     # nt = 20
#     #     # nt = int(6)
#     #     nt = int(1e6) # total number of steps thru time
#     #     # spins = int(1000)
        
#     #     spins = int(1e5)
#     #     num_spins = spins
#     #     print('Start setting up structures')
#     #     sim = ds3.DiffSim3d(sg3,spins) 
#     #     nsegx,nsegy,nsegz=20,20,20
#     #     sim.set_segments(nsegx=nsegx,nsegy=nsegy,nsegz=nsegz)
#     #     sim.setup(structures=[int(len(fiber_xyzr_fid_list))])  # seed OUTSIDE structures len(fiberlist)=index of box
#     #     print('Done setting up structures')
        
#     #     # first, check thath all the spins are in fact in the multiple spstruc's
#     #     isOutside = False
#     #     for struct in sg3.structures[:-1]:
#     #         isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#     #     isOutside = np.invert(isOutside)
#     #     self.assertTrue(isOutside.all())

#     #     st = time.time()
#     #     Dx_step = [D]
#     #     Dy_step = [D]
#     #     Dz_step = [D]
#     #     diff_time = [0]
#     #     print('Got to sim extra-axonal simloops')
#     #     current_time = 0.0
#     #     # max_time = 10 #10 #3000
#     #     while current_time < nt*dt:
#     #     # while current_time < max_time:
#     #         time_step = dt
#     #         current_time+=time_step
#     #         # --------step-----------
#     #         sim.step(time_step)
#     #         # Dx_step.append(np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(current_time))
#     #         # Dy_step.append(np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(current_time))
#     #         # Dz_step.append(np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(current_time))
#     #         diff_time.append(current_time)
#     #     Dx_step = np.array(Dx_step)
#     #     Dy_step = np.array(Dy_step)
#     #     Dz_step = np.array(Dz_step)
#     #     diff_time = np.array(diff_time)
#     #     et = time.time()
#     #     # # get the execution time
#     #     elapsed_time = np.round(et - st,2)

#     #     #==================================
#     #     file_name = os.path.basename(file_path)
#     #     base_name, extension = os.path.splitext(file_name)
#     #     date_time = str(get_date_time())
#     #     file_name_new = 'extra_'+ date_time + '_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' \
#     #         + str(num_spins)+'_spins_'+ base_name +'_SIMtime'+str(round(elapsed_time,2))+'_sec'+ '_dt' + str(dt)+'_seg'+str(int(nsegx))
#     #     print('file_name_new', file_name_new)
#     #     # Combine new filename with folder path to get the full path
#     #     data_folder_name = os.path.join(target_folder_path, 'ADC', 'ADCdata')
#     #     if not os.path.exists(data_folder_name):
#     #         os.makedirs(data_folder_name)
#     #         print("Folder created successfully.")
#     #     else:
#     #         print("Folder already exists.")  
#     #     data_file_path = os.path.join(data_folder_name, str(int(nsegx))+'SEGMENT_'+file_name_new+'data.pkl')
#     #     # save_ADCdata_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
#     #     # plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new )    

#     #     # after simulating diffusion, are all spins still outside axons
#     #     isOutside = False
#     #     for struct in sg3.structures[:-1]:
#     #         isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#     #     isOutside = np.invert(isOutside)
#     #     self.assertTrue(isOutside.all())
#     #     print('elapsed_time', elapsed_time)
# if __name__ == '__main__':
#     unittest.main()