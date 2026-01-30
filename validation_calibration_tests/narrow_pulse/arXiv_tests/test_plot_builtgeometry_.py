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
        

#     def testOutsideMultiFiberGeometry(self):
#         '''
#         # find /path/to/folder -type f -newermt "2024-05-06"
#         find ./ -depth -name "*2024-06-27_17-44-24*"
#         '''
#         st = time.time()
#         file_path = './geometrygen/animation/2024-07-07_03-06-59/data/array100_fibers_2024-07-07_03-06-59_avf_0.68w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_3618.52_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-05_19-24-55/data/array20_fibers_2024-07-05_19-24-55_avf_0.52w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_418.1_sec.pkl'
#         file_path = './geometrygen/animation/2024-07-08_17-56-42_AVF_0.65_500fibers_K200/data/array250_fibers_2024-07-08_17-56-42_avf_1.57w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_14522.92_sec.pkl'
#         file_path = './geometrygen/animation/2024-07-10_09-21-59/data/array250_fibers_2024-07-10_09-21-59_avf_8.48w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_129321.43_sec.pkl'
#         file_path = './geometrygen/animation/2024-07-08_17-56-42_AVF_0.65_500fibers_K200/data/Farray250_fibers_2024-07-08_17-56-42_avf_1.57w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_14522.92_sec.pkl'
#         file_path = './geometrygen/animation/2024-07-08_17-55-14_AVF_0.66_500fibers_K200/data/array250_fibers_2024-07-08_17-55-14_avf_1.31w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_11987.36_sec.pkl'
#         file_path = './geometrygen/animation/2024-07-08_17-52-37_AVF0.65_500fibers_K200/data/array250_fibers_2024-07-08_17-52-37_avf_1.6w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_15947.77_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers_usethis/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-08_18-30-33/data/array250_fibers_2024-07-08_18-30-33_avf_5.38w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_101024.14_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-11_15-37-56/data/array200_fibers_2024-07-11_15-37-56_avf_1.11w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_5843.84_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-11_15-37-56/data/array200_fibers_2024-07-11_15-37-56_avf_1.11w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_5843.84_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers_usethis/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
        
#         file_path = './geometrygen/animation/2024-07-08_17-34-27_AVF0.67_500fibers_K20/data/array250_fibers_2024-07-08_17-34-27_avf_1.53w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_13168.29_sec.pkl'
#         #230 fibers K200
#         # file_path = './geometrygen/animation/2024-07-31_08-14-47/data/array230_fibers_2024-07-31_08-14-47_avf_0.0_d5.0_sig2.5w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_11497.37_sec.pkl'
#         # # 250 fibers K200
#         # file_path = './geometrygen/animation/2024-07-31_07-03-42/data/array250_fibers_2024-07-31_07-03-42_avf_0.0_d5.0_sig2.5w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_19811.39_sec.pkl'
#         # # 300 fibers K200
#         # file_path = './geometrygen/animation/2024-07-31_06-01-04_300fibersK200/data/array300_fibers_2024-07-31_06-01-04_avf_0.0_d5.0_sig2.5w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_16806.44_sec.pkl'
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_17-34-37_AVF0.65_250fibers_K20/data/array250_fibers_2024-07-08_17-34-37_avf_1.45w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_14463.59_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_17-52-37_AVF0.65_500fibers_K200/data/array250_fibers_2024-07-08_17-52-37_avf_1.6w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_15947.77_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers_usethis/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_17-34-27_AVF0.67_500fibers_K20/data/array250_fibers_2024-07-08_17-34-27_avf_1.53w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_13168.29_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_17-55-14_AVF_0.66_500fibers_K200/data/array250_fibers_2024-07-08_17-55-14_avf_1.31w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_11987.36_sec.pkl'
#         #exp1
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78_K20/data/array250_fibers_2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_20090.62_sec.pkl'
#         #exp2
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-08-30_avf_0.64_d2.1_sig0.78_K200/data/array250_fibers_2024-09-06_20-08-30_avf_0.64_d2.1_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_14568.68_sec.pkl'
#         # #exp3
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-20-17_avf_0.65_d3.14_sig0.78_K200/data/array300_fibers_2024-09-06_20-20-17_avf_0.65_d3.14_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_17259.06_sec.pkl'
#         # #exp4
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-36-58_avf_0.63_d5.68_sig0.78_K200/data/array300_fibers_2024-09-06_20-36-58_avf_0.63_d5.68_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_102966.59_sec.pkl'
#         # ---20 fiberes 
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-22_19-46-58_converged_20fib/data/array20_fibers_2024-10-22_19-46-58_avf_0.72_d1.38_sig0.3w_o1000_w_c1_w_l100_K200_bffse0.01_bffsph0.01_28008.44_sec.pkl'        
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-07_03-07-44_small40fibers-try-sim/data/array20_fibers_2024-07-07_03-07-44_avf_0.55w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_582.43_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-24_10-15-11/data/array2_fibers_2024-10-24_10-15-11_avf_0.37_d1.38_sig0.3w_o1000_w_c1_w_l50_K200_bffse0.01_bffsph0.01_143.17_sec.pkl'
#         #-------- ODI 0.13 spread much
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-11-03_14-09-19_array250_fibers_boxL_45.0_2024-11-03_14-09-19_avf_0.63_d1.4_sig0.4w_o10_w_c3_w_l3_K4_ODI_0.1392089745461279_bffse0.25_bffsph0.001_3047.77_sec/data/array250_fibers_boxL_45.0_2024-11-03_14-09-19_avf_0.63_d1.4_sig0.4w_o10_w_c3_w_l3_K4_ODI_0.1392089745461279_bffse0.25_bffsph0.001_3047.77_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-11-03_14-09-19_array250_fibers_boxL_45.0_2024-11-03_14-09-19_avf_0.63_d1.4_sig0.4w_o10_w_c3_w_l3_K4_ODI_0.1392089745461279_bffse0.25_bffsph0.001_3047.77_sec/data/array250_fibers_boxL_45.0_2024-11-03_14-09-19_avf_0.63_d1.4_sig0.4w_o10_w_c3_w_l3_K4_ODI_0.1392089745461279_bffse0.25_bffsph0.001_3047.77_sec.pkl'
#         # ------ ODI 0.08
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-11-03_14-09-12_array250_fibers_boxL_47.0_2024-11-03_14-09-12_avf_0.57_d1.4_sig0.4w_o10_w_c3_w_l3_K7_ODI_0.08438492631768273_bffse0.25_bffsph0.001_10733.25_sec/data/array250_fibers_boxL_47.0_2024-11-03_14-09-12_avf_0.57_d1.4_sig0.4w_o10_w_c3_w_l3_K7_ODI_0.08438492631768273_bffse0.25_bffsph0.001_10733.25_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-11-03_14-09-08_array250_fibers_boxL_46.0_2024-11-03_14-09-08_avf_0.6_d1.4_sig0.4w_o10_w_c3_w_l3_K7_ODI_0.08438492631768273_bffse0.25_bffsph0.001_2651.48_sec/data/array250_fibers_boxL_46.0_2024-11-03_14-09-08_avf_0.6_d1.4_sig0.4w_o10_w_c3_w_l3_K7_ODI_0.08438492631768273_bffse0.25_bffsph0.001_2651.48_sec.pkl'
        
#         # straight
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-08-30_avf_0.64_d2.1_sig0.78_K200/data/array250_fibers_2024-09-06_20-08-30_avf_0.64_d2.1_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_14568.68_sec.pkl'
#         # intermediate spread
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78_K20/data/array250_fibers_2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_20090.62_sec.pkl'
#         # extreme spread
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-11-03_14-09-08_array250_fibers_boxL_46.0_2024-11-03_14-09-08_avf_0.6_d1.4_sig0.4w_o10_w_c3_w_l3_K7_ODI_0.08438492631768273_bffse0.25_bffsph0.001_2651.48_sec/data/array250_fibers_boxL_46.0_2024-11-03_14-09-08_avf_0.6_d1.4_sig0.4w_o10_w_c3_w_l3_K7_ODI_0.08438492631768273_bffse0.25_bffsph0.001_2651.48_sec.pkl'
        
#     #     import os
#     #     # Extract the target folder path (without the filename)
#     #     target_folder_path = os.path.dirname( os.path.dirname(file_path) )
#     #     folder_name = os.path.join(target_folder_path, 'ADC')
#     #     if not os.path.exists(folder_name):
#     #         os.makedirs(folder_name)
#     #         print("Folder created successfully.")
#     #     else:
#     #         print("Folder already exists.")  
#     #     optimized_fibers, L = ig.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
#     #     ###### only exp------------------------
#     #     # optimized_fibers[:, 0:4], L = optimized_fibers[:, 0:4], L

#     #     print('optimized_fibers', optimized_fibers.shape)
#     #     fiberlist_xyz_r_fid = ag.split_matrix_to_list(optimized_fibers, L)
#     #     print('L box length',L)
#     #     D = 3.0 # um^2/ms
#     #     T2 = 100 # ms
#     #     rho = 1 # fractional water density
#     #     sg3 = geom.SimGeometry3D(L,L,L,D,T2,rho)
#     #     fiber_xyzr_fid_list = ag.createBoundaryZ(fiberlist_xyz_r_fid, L)
#     #     print('len(fiber_xyzr_fid_list)',len(fiber_xyzr_fid_list))
        
#     # # #     #==================================
#     #     file_name = os.path.basename(file_path)
#     #     base_name, extension = os.path.splitext(file_name)
#     #     print('base_name', base_name)
#     #     date_time = str(get_date_time())
#     #     file_name_new = 'extra_'+ date_time + '_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_'+ base_name 
#     #     # Combine new filename with folder path to get the full path

#     #     fiber_list = ag.map_matrix_to_list_numpy(optimized_fibers)
#     #     color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list)))
#     #     np.random.shuffle(color)
#     #     # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, animation_input=True, optimized=True, folder_name=folder_name) 
#     #     # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_90', folder_name=folder_name) 
#     #     # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_0', folder_name=folder_name)                    
#     #     # ag.plot_radius_distribution()
#     #     # ag.plot_diameter_GEV_distribution(optimized_fibers, L, folder_name=target_folder_path)
#     #     # ag.plot_slices(L, folder_name, N=5, color=color, spheres_xyz_r_fid=optimized_fibers, folder_name=folder_name)
#     #     ag.plot_along_axon_OD(fiberlist_xyz_r_fid, L, optimized=True, folder_name=folder_name)


#         # Replace 'your_root_folder' with the actual path to your root folder
#         def compute_OD_for_all(file_path):
#             # Implement your function A logic here
#             print(f"Running OD comp for file: {file_path}")
#             # Extract the target folder path (without the filename)
#             # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/skip_2024-09-08_13-09-14/data/array250_fibers_2024-09-08_13-09-14_avf_0.69_d3.14_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_70257.75_sec.pkl'
#             target_folder_path = os.path.dirname( os.path.dirname(file_path) )
#             folder_name = os.path.join(target_folder_path, 'ADC')
#             if not os.path.exists(folder_name):
#                 os.makedirs(folder_name)
#                 print("Folder created successfully.")
#             else:
#                 print("Folder already exists.")  
#             optimized_fibers, L = ig.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
#             print('optimized_fibers', optimized_fibers.shape)
#             fiberlist_xyz_r_fid = ag.split_matrix_to_list(optimized_fibers, L)
#             print('L box length',L)
#             # Combine new filename with folder path to get the full path
#             ag.plot_along_axon_OD(fiberlist_xyz_r_fid, L, optimized=True, folder_name=folder_name)

#         def process_subfolders(root_folder):
#             for subfolder_name in os.listdir(root_folder):
#                 print('Current subfolder_name', subfolder_name)
#                 subfolder_path = os.path.join(root_folder, subfolder_name)
#                 # Skip subfolders that match a certain criteria
#                 if subfolder_name.startswith("skip_"):  # Example: skip folders starting with "skip_"
#                     print('skipped ',subfolder_path )
#                     continue
#                 if os.path.isdir(subfolder_path):
#                     data_folder_path = os.path.join(subfolder_path, 'data')
#                     if os.path.isdir(data_folder_path):
#                         for filename in os.listdir(data_folder_path):
#                             file_path = os.path.join(data_folder_path, filename)
#                             if os.path.isfile(file_path):
#                                 compute_OD_for_all(file_path)
        
#         root_folder = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation'
#         process_subfolders(root_folder) 

# if __name__ == '__main__':
#     unittest.main()