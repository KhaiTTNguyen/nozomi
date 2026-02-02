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
#         # small num axons
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-17_17-45-45/data/array3_fibers_boxL_6.0_2025-01-17_17-45-45_avf_0.26_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_141.72_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-08_13-16-09/data/array20_fibers_boxL_14.0_2025-01-08_13-16-09_avf_0.44_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_338.09_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-08_12-10-57/data/array200_fibers_boxL_48.0_2025-01-08_12-10-57_avf_0.49_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_2562.17_sec.pkl'
        
#         # more straight
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_17-36-52/data/array200_fibers_boxL_44.0_2025-01-22_17-36-52_avf_0.59_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_2991.21_sec.pkl'
#         # more spread
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_17-33-31/data/array200_fibers_boxL_43.0_2025-01-22_17-33-31_avf_0.64_d1.4_sig0.4w_o10_w_c3_w_l3_K18_ODI_0.0353_bffse0.24_bffsph0.001_2864.4_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_10-21-07_AD_shoot/data/array1_fibers_boxL_40_2025-01-22_10-21-07_avf_0.0_d1.4_sig0.4w_o10_w_c3_w_l3_K18_ODI_0.0353_bffse0.24_bffsph0.001_33.31_sec.pkl'
#         # more spread cylinder
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_11-28-53/data/array1_fibers_boxL_40_2025-01-22_11-28-53_avf_0.0_d1.4_sig0.4w_o10_w_c3_w_l3_K18_ODI_0.0353_bffse0.24_bffsph0.001_41.43_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_13-10-23/data/array20_fibers_boxL_40_2025-01-22_13-10-23_avf_0.06_d1.4_sig0.4w_o10_w_c3_w_l3_K18_ODI_0.0353_bffse0.24_bffsph0.001_172.75_sec.pkl'
        
#         # more straight
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_17-36-52/data/array200_fibers_boxL_44.0_2025-01-22_17-36-52_avf_0.59_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_2991.21_sec.pkl'
        
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_10-19-33/data/array1_fibers_boxL_40_2025-01-22_10-19-33_avf_0.0_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_24.34_sec.pkl'
#         # more straight cylinder
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_11-30-12/data/array1_fibers_boxL_40_2025-01-22_11-30-12_avf_0.0_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_18.0_sec.pkl'
        
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_10-21-07_AD_shoot/data/array1_fibers_boxL_40_2025-01-22_10-21-07_avf_0.0_d1.4_sig0.4w_o10_w_c3_w_l3_K18_ODI_0.0353_bffse0.24_bffsph0.001_33.31_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_15-53-56/data/array20_fibers_boxL_20.0_2025-01-22_15-53-56_avf_0.22_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_132.91_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-07_16-45-47/data/array50_fibers_boxL_24.0_2025-01-07_16-45-47_avf_0.41_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_474.62_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-07_16-45-50/data/array50_fibers_boxL_24.0_2025-01-07_16-45-50_avf_0.37_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_403.22_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_13-23-34/data/array20_fibers_boxL_40_2025-01-22_13-23-34_avf_0.06_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_209.72_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_17-20-06/data/array5_fibers_boxL_10.0_2025-01-22_17-20-06_avf_0.22_d1.4_sig0.4w_o10_w_c5_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_149.54_sec.pkl'
        
#         # experiments
#         #d2-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_19-31-42_avf_0.69_d2.1_sig0.78_K20/data/array250_fibers_2024-09-08_19-31-42_avf_0.69_d2.1_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_15086.51_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78_K20/data/array250_fibers_2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_20090.62_sec.pkl'
#         #d2-k80
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-09_23-59-34_250_fibers_avf_0.66_d2.1_sig0.78w_o10_w_c5_w_l5_K80/data/array250_fibers_2024-10-09_23-59-34_avf_0.66_d2.1_sig0.78w_o10_w_c5_w_l5_K80_bffse0_bffsph0.001_74132.1_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-09_23-52-51_250_fibers_2024-10-09_23-52-51_avf_0.65_d2.1_sig0.78w_o10_w_c5_w_l5_K80/data/array250_fibers_2024-10-09_23-52-51_avf_0.65_d2.1_sig0.78w_o10_w_c5_w_l5_K80_bffse0_bffsph0.001_59711.43_sec.pkl'
#         #d2-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-08-41_avf_0.64_d2.1_sig0.78_K200/data/array250_fibers_2024-09-06_20-08-41_avf_0.64_d2.1_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_17943.35_sec.pkl'
#         ### file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-31_06-01-04_300fibersK200/data/array300_fibers_2024-07-31_06-01-04_avf_0.0_d5.0_sig2.5w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_16806.44_sec.pkl'
#         ### file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_18-30-33/data/array250_fibers_2024-07-08_18-30-33_avf_5.38w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_101024.14_sec.pkl'
#         #d3-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_12-50-29_avf_0.7_d3.14_sig0.78_K20/data/array250_fibers_2024-09-08_12-50-29_avf_0.7_d3.14_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_39188.4_sec.pkl'
#         #d3-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-16-15_avf_0.66_d3.14_sig0.78_K200/data/array250_fibers_2024-09-06_20-16-15_avf_0.66_d3.14_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_15506.54_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-20-17_avf_0.65_d3.14_sig0.78_K200/data/array300_fibers_2024-09-06_20-20-17_avf_0.65_d3.14_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_17259.06_sec.pkl'
#         #d5-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_14-06-48_avf_0.7_d5.68_sig0.78_K20/data/array250_fibers_2024-09-08_14-06-48_avf_0.7_d5.68_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_134780.23_sec.pkl'
#         #d5-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-36-42_avf_0.65_d5.68_sig0.78w_o10_w_c5_w_l5_K200/data/array300_fibers_2024-09-06_20-36-42_avf_0.65_d5.68_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_72934.88_sec.pkl'

#         # d8-K20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_22-47-25_avf_0.69_d8.48_sig0.78_K20/data/array250_fibers_2024-09-08_22-47-25_avf_0.69_d8.48_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_247597.86_sec.pkl'
#         # d8-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-41-46_300_fibers_2024-09-06_20-41-46_avf_0.65_d8.48_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_226712.87_sec/data/array300_fibers_2024-09-06_20-41-46_avf_0.65_d8.48_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_226712.87_sec.pkl'
        
#         #d2-K200
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-29_23-41-58_70_fibers_boxL_40_avf_0.68_d2.58_sig0.58_K200_ODI_0.0032/data/array70_fibers_boxL_40_2025-01-29_23-41-58_avf_0.68_d2.58_sig0.58w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_4318.15_sec.pkl'
#         # d2-20
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-30_10-28-51_70_fibers_boxL_40_avf_0.68_d2.58_sig0.58_K20_ODI_0.0318_8329.32_sec/data/array70_fibers_boxL_40_2025-01-30_10-28-51_avf_0.68_d2.58_sig0.58w_o10_w_c3_w_l3_K20_ODI_0.0318_bffse0.24_bffsph0.001_8329.32_sec.pkl'
#         # d1.34-K200
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-30_13-34-33_200_fibers_boxL_40_avf_0.7_d1.34_sig0.4_K200_ODI_0.0032/data/array200_fibers_boxL_40_2025-01-30_13-34-33_avf_0.7_d1.34_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_8469.23_sec.pkl'
#         # d1.34-K20
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-30_13-35-09_200_fibers_boxL_40_avf_0.67_d1.34_sig0.4_K20_ODI_0.0318_10559.23sec/data/array200_fibers_boxL_40_2025-01-30_13-35-09_avf_0.67_d1.34_sig0.4w_o10_w_c3_w_l3_K20_ODI_0.0318_bffse0.24_bffsph0.001_10559.23_sec.pkl'
#         # d1.34 - k7
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-05_01-33-00_210_fibers_boxL_40_avf_0.68_d1.34_sig0.2w_o10_w_c3_w_l5_K7_ODI_0.0903/data/array210_fibers_boxL_40_2025-02-05_01-33-00_avf_0.68_d1.34_sig0.2w_o10_w_c3_w_l5_K7_ODI_0.0903_bffse0.24_bffsph0.001_28053.79_sec.pkl'
        
#         #d0.82-sig0.4-K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-04_12-52-19/data/array370_fibers_boxL_40_2025-02-04_12-52-19_avf_0.68_d0.82_sig0.4w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_8412.87_sec.pkl'
#         # file_pathOLD = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-31_10-06-21_280_fibers_boxL_40_avf_0.5_d0.82_sig0.4_K200_ODI_0.0032/data/array280_fibers_boxL_40_2025-01-31_10-06-21_avf_0.5_d0.82_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_9825.99_sec.pkl'
                
#         # d0.82-sig0.4-K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-31_10-06-21_280_fibers_boxL_40_avf_0.5_d0.82_sig0.4_K200_ODI_0.0032/data/array280_fibers_boxL_40_2025-01-31_10-06-21_avf_0.5_d0.82_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_9825.99_sec.pkl'
#         # d0.82-sig0.4-K20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-31_10-06-09/data/array280_fibers_boxL_40_2025-01-31_10-06-09_avf_0.54_d0.82_sig0.4w_o10_w_c3_w_l3_K20_ODI_0.0318_bffse0.24_bffsph0.001_11025.93_sec.pkl'
#         # sample w luck d0.82-sig0.2-K20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-04_17-44-35/data/array460_fibers_boxL_40_2025-02-04_17-44-35_avf_0.7_d0.82_sig0.2w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_21574.65_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-04_17-43-14/data/array460_fibers_boxL_40_2025-02-04_17-43-14_avf_0.7_d0.82_sig0.2w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_22359.07_sec.pkl'

#         #d0.67- 0.3sig - K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-31_10-19-26/data/array500_fibers_boxL_40_2025-01-31_10-19-26_avf_0.66_d0.67_sig0.3w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_9913.47_sec.pkl'
        
        
#         #d0.74_sig0.35 - K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-31_19-35-53/data/array430_fibers_boxL_40_2025-01-31_19-35-53_avf_0.64_d0.74_sig0.35w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_8606.42_sec.pkl'
#         #d0.74_sig0.1 - K20        
#         # file_pathOLD = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-02_13-49-49/data/array610_fibers_boxL_40_2025-02-02_13-49-49_avf_0.7_d0.74_sig0.1w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_18621.19_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-03_10-18-24/data/array570_fibers_boxL_40_2025-02-03_10-18-24_avf_0.65_d0.74_sig0.1w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_12947.38_sec.pkl'
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
#         dt = 0.0002 # time step in ms
#         # nt = 20
#         nt = int(2e6) # total number of steps thru time
#         # nt = int(9e6) # total number of steps thru time
#         spins = int(1e4)
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
#         max_time = 10 #10 #3000  
#         # while current_time < nt*dt:
#         while current_time < max_time:
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
#         save_ADCdata_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
#         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new )    

#         # after simulating diffusion, are all spins still inside sg3?
#         isInside = False
#         for struct in sg3.structures[:-1]:
#             isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
#         self.assertTrue(isInside.all())

#     def testOutsideMultiFiberGeometry(self):
#         st = time.time()
#         # small num axons
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-17_17-45-45/data/array3_fibers_boxL_6.0_2025-01-17_17-45-45_avf_0.26_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_141.72_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-08_13-16-09/data/array20_fibers_boxL_14.0_2025-01-08_13-16-09_avf_0.44_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_338.09_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-08_12-10-57/data/array200_fibers_boxL_48.0_2025-01-08_12-10-57_avf_0.49_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_2562.17_sec.pkl'
#          # more straight
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_17-36-52/data/array200_fibers_boxL_44.0_2025-01-22_17-36-52_avf_0.59_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_2991.21_sec.pkl'
#         # more spread
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-22_17-33-31/data/array200_fibers_boxL_43.0_2025-01-22_17-33-31_avf_0.64_d1.4_sig0.4w_o10_w_c3_w_l3_K18_ODI_0.0353_bffse0.24_bffsph0.001_2864.4_sec.pkl'
        
#         # experiments
#         # d2-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_19-31-42_avf_0.69_d2.1_sig0.78_K20/data/array250_fibers_2024-09-08_19-31-42_avf_0.69_d2.1_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_15086.51_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78_K20/data/array250_fibers_2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_20090.62_sec.pkl'
#         # d2-k80
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-09_23-59-34_250_fibers_avf_0.66_d2.1_sig0.78w_o10_w_c5_w_l5_K80/data/array250_fibers_2024-10-09_23-59-34_avf_0.66_d2.1_sig0.78w_o10_w_c5_w_l5_K80_bffse0_bffsph0.001_74132.1_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-09_23-52-51_250_fibers_2024-10-09_23-52-51_avf_0.65_d2.1_sig0.78w_o10_w_c5_w_l5_K80/data/array250_fibers_2024-10-09_23-52-51_avf_0.65_d2.1_sig0.78w_o10_w_c5_w_l5_K80_bffse0_bffsph0.001_59711.43_sec.pkl'
#         # d2-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-08-41_avf_0.64_d2.1_sig0.78_K200/data/array250_fibers_2024-09-06_20-08-41_avf_0.64_d2.1_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_17943.35_sec.pkl'
#         # ## file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-31_06-01-04_300fibersK200/data/array300_fibers_2024-07-31_06-01-04_avf_0.0_d5.0_sig2.5w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_16806.44_sec.pkl'
#         # ## file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_18-30-33/data/array250_fibers_2024-07-08_18-30-33_avf_5.38w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_101024.14_sec.pkl'
#         # d3-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_12-50-29_avf_0.7_d3.14_sig0.78_K20/data/array250_fibers_2024-09-08_12-50-29_avf_0.7_d3.14_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_39188.4_sec.pkl'
#         # d3-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-16-15_avf_0.66_d3.14_sig0.78_K200/data/array250_fibers_2024-09-06_20-16-15_avf_0.66_d3.14_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_15506.54_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-20-17_avf_0.65_d3.14_sig0.78_K200/data/array300_fibers_2024-09-06_20-20-17_avf_0.65_d3.14_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_17259.06_sec.pkl'
#         # d5-k20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_14-06-48_avf_0.7_d5.68_sig0.78_K20/data/array250_fibers_2024-09-08_14-06-48_avf_0.7_d5.68_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_134780.23_sec.pkl'
#         # d5-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-36-42_avf_0.65_d5.68_sig0.78w_o10_w_c5_w_l5_K200/data/array300_fibers_2024-09-06_20-36-42_avf_0.65_d5.68_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_72934.88_sec.pkl'

#         # d8-K20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_22-47-25_avf_0.69_d8.48_sig0.78_K20/data/array250_fibers_2024-09-08_22-47-25_avf_0.69_d8.48_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_247597.86_sec.pkl'
#         # d8-k200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-41-46_300_fibers_2024-09-06_20-41-46_avf_0.65_d8.48_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_226712.87_sec/data/array300_fibers_2024-09-06_20-41-46_avf_0.65_d8.48_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_226712.87_sec.pkl'
        
#         #d2-K200
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-29_23-41-58_70_fibers_boxL_40_avf_0.68_d2.58_sig0.58_K200_ODI_0.0032/data/array70_fibers_boxL_40_2025-01-29_23-41-58_avf_0.68_d2.58_sig0.58w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_4318.15_sec.pkl'
#         # d2-20
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-30_10-28-51_70_fibers_boxL_40_avf_0.68_d2.58_sig0.58_K20_ODI_0.0318_8329.32_sec/data/array70_fibers_boxL_40_2025-01-30_10-28-51_avf_0.68_d2.58_sig0.58w_o10_w_c3_w_l3_K20_ODI_0.0318_bffse0.24_bffsph0.001_8329.32_sec.pkl'
#         # d1.34-K200
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-30_13-34-33_200_fibers_boxL_40_avf_0.7_d1.34_sig0.4_K200_ODI_0.0032/data/array200_fibers_boxL_40_2025-01-30_13-34-33_avf_0.7_d1.34_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_8469.23_sec.pkl'
#         # d1.34-K20
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-30_13-35-09_200_fibers_boxL_40_avf_0.67_d1.34_sig0.4_K20_ODI_0.0318_10559.23sec/data/array200_fibers_boxL_40_2025-01-30_13-35-09_avf_0.67_d1.34_sig0.4w_o10_w_c3_w_l3_K20_ODI_0.0318_bffse0.24_bffsph0.001_10559.23_sec.pkl' 
#         # d1.34 - k7
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-05_01-33-00_210_fibers_boxL_40_avf_0.68_d1.34_sig0.2w_o10_w_c3_w_l5_K7_ODI_0.0903/data/array210_fibers_boxL_40_2025-02-05_01-33-00_avf_0.68_d1.34_sig0.2w_o10_w_c3_w_l5_K7_ODI_0.0903_bffse0.24_bffsph0.001_28053.79_sec.pkl'
        
#         # d0.82-sig0.4-K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-04_12-52-19/data/array370_fibers_boxL_40_2025-02-04_12-52-19_avf_0.68_d0.82_sig0.4w_o10_w_c3_w_l5_K200_ODI_0.0032_bffse0.24_bffsph0.001_8412.87_sec.pkl'
#         # file_pathOLD = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-31_10-06-21_280_fibers_boxL_40_avf_0.5_d0.82_sig0.4_K200_ODI_0.0032/data/array280_fibers_boxL_40_2025-01-31_10-06-21_avf_0.5_d0.82_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_9825.99_sec.pkl'
#         # d0.82-sig0.4-K20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-31_10-06-09/data/array280_fibers_boxL_40_2025-01-31_10-06-09_avf_0.54_d0.82_sig0.4w_o10_w_c3_w_l3_K20_ODI_0.0318_bffse0.24_bffsph0.001_11025.93_sec.pkl'
#         # sample w luck d0.82-sig0.2-K20
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-04_17-44-35/data/array460_fibers_boxL_40_2025-02-04_17-44-35_avf_0.7_d0.82_sig0.2w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_21574.65_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-04_17-43-14/data/array460_fibers_boxL_40_2025-02-04_17-43-14_avf_0.7_d0.82_sig0.2w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_22359.07_sec.pkl'
        
#         #d0.67- 0.3sig - K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-31_10-19-26/data/array500_fibers_boxL_40_2025-01-31_10-19-26_avf_0.66_d0.67_sig0.3w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_9913.47_sec.pkl'
        
#         #d0.74_sig0.35 - K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-31_19-35-53/data/array430_fibers_boxL_40_2025-01-31_19-35-53_avf_0.64_d0.74_sig0.35w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_8606.42_sec.pkl'
#         #d0.74_sig0.1 - K20        
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-02_13-49-49/data/array610_fibers_boxL_40_2025-02-02_13-49-49_avf_0.7_d0.74_sig0.1w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_18621.19_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-02-03_10-18-24/data/array570_fibers_boxL_40_2025-02-03_10-18-24_avf_0.65_d0.74_sig0.1w_o10_w_c3_w_l5_K20_ODI_0.0318_bffse0.24_bffsph0.001_12947.38_sec.pkl'
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

#         for fiber in fiber_xyzr_fid_list:
#             sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
#             spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#             sg3.add_structure(spstruc) # add it to sg3
#         dt = 0.0002 # time step in ms
#         # nt = 20
#         nt = int(2e6) # total number of steps thru time
#         # nt = int(9e6) # total number of steps thru time
#         #higher number of spins easier to come out..!
#         spins = int(1e4)
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

#         Dx_step = [D]
#         Dy_step = [D]
#         Dz_step = [D]
#         diff_time = [0]
#         print('Got to sim extra-axonal simloops')
#         current_time = 0.0
#         max_time = 10 #10 #3000
#         # while current_time < nt*dt:
#         while current_time < max_time:
#             # if current_time%50==0:
#             #     print('current_time', current_time)
#             #     isOutside = False
#             #     for struct in sg3.structures[:-1]:
#             #         isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#             #     isOutside = np.invert(isOutside)
#             #     if ~isOutside.all():
#             #         print('stopped at ', current_time)
#             #         spins_temp = sim.spins_d.get()
#             #         spins_temp = ((spins_temp.T)[np.invert(isOutside)]).T
#             #         ig.plot_fibers(spins_temp, fiber_xyzr_fid_list, L, current_time, folder_name, file_path)
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

# if __name__ == '__main__':
#     unittest.main()