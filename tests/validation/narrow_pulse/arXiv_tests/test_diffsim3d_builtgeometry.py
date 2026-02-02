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
#         '''
#         # find /path/to/folder -type f -newermt "2024-05-06"
#         find ./ -depth -name "*2024-06-27_17-44-24*"
#         '''
#         st = time.time()
#         # file_name = '5_500000_2024-01-23_11-03-02_avf_0.008333333333333333_K20_space_buffer_end1e-05_space_buffer_sphere1e-05_2229_sec.pkl'
#         # file_name = '6_500000_2023-11-28_16-06-53_avf_0.0632_K200_1462_sec.pkl'
#         # file_name = '15_30000_2024-03-25_16-35-32_avf_0.0626_K20_space_buffer_startend1_space_buffer_sphere0.5_1050_sec.pkl'
#         # file_name = '43_500000_2024-01-19_16-24-40_avf_0.155_K200_space_buffer_end1e-05_space_buffer_sphere1e-05_100230_sec.pkl'
#         # file_name = 'SAP_10_5000_2024-05-06_18-12-17_avf_0.0658_K200_space_buffer_startend1e-05_space_buffer_sphere1e-05_52_sec.pkl'
#         # file_name = 'SAP_5_5000_2024-05-07_10-29-04_avf_0.0_K200_space_buffer_startend1e-05_space_buffer_sphere1e-05_24_sec.pkl' DELETE this
#         # file_name =         file_name = 'SAP_80_100000_2024-05-08_04-29-32_avf_0.3867_K200_space_buffer_startend1e-06_space_buffer_sphere0.0001_151856_sec.pkl ERROR
#         # file_name = 'SAP_5_5000_2024-05-07_19-58-08_avf_0.0342_K200_space_buffer_startend1e-05_space_buffer_sphere1e-05_7_sec.pkl'
#         # file_name = 'SAP_5_5000_2024-05-07_20-07-05_avf_0.0366_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_9_sec.pkl'
#         # file_name = 'SAP_5_5000_2024-05-07_20-15-32_avf_0.0524_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_18_sec.pkl'
#         # file_name = 'SAP_5_5000_2024-05-07_20-15-27_avf_0.0344_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_7_sec.pkl'
#         # file_name = 'SAP_5_5000_2024-05-07_20-36-49_avf_0.0334_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_27_sec.pkl'
#         # file_name = 'SAP_5_5000_2024-05-07_20-36-52_avf_0.0436_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_27_sec.pkl'
#         # file_name = 'SAP_60_100000_2024-05-08_02-42-22_avf_0.234_K50_space_buffer_startend1e-05_space_buffer_sphere0.0001_4571_sec.pkl' # checking why
#         # file_name = 'SAP_60_100000_2024-05-08_02-39-41_avf_0.2948_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_5657_sec.pkl' # CANNOT LOAD
#         # file_name = ' SAP_60_100000_2024-05-08_02-40-39_avf_0.215_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_4016_sec.pkl' # cannot lood
#         # file_name = 'SAP_60_100000_2024-05-08_02-42-54_avf_0.3938_K10_space_buffer_startend1e-05_space_buffer_sphere0.0001_156365_sec.pkl'        
#         # file_name = 'SAP_10_10000_2024-05-10_17-22-47_avf_0.01824_K15_space_buffer_startend1e-06_space_buffer_sphere0.0001_48_sec.pkl' worked
#         # file_name = 'SAP_10_10000_2024-05-10_17-25-02_avf_0.02164_K15_space_buffer_startend1e-06_space_buffer_sphere0.0001_59_sec.pkl' worked
#         # file_name = 'SAP_array10_5000_2024-05-11_18-53-54_avf_0.0742_K200_space_buffer_startend1e-06_space_buffer_sphere0.001_90_sec.pkl'
#         # file_name = 'SAP_array10_5000_2024-05-11_19-01-09_avf_0.04322_K20_space_buffer_startend1e-06_space_buffer_sphere0.001_92_sec.pkl'
#         # file_path = './geometrygen/animation/2024-06-27_17-44-24/data/array2_fibers_2024-06-27_17-44-24_avf_0.0077_w_overlap10_w_curve5_w_length5_K-200_buffstartend0_buffsphere0.001_703_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-03_14-14-05_200fibers_avf0.7/data/array200_fibers_2024-07-03_14-14-05_avf_0.53w_o10_w_c5_w_l5_K-200_buffstartend0_buffsphere0.001_6216.1_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-06_01-07-11/data/array20_fibers_2024-07-06_01-07-11_avf_0.57w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_560.95_sec.pkl'
#         # 600fibers file_path = './geometrygen/animation/2024-07-06_03-54-08_AVF0.91_600fibers/data/array300_fibers_2024-07-06_03-54-08_avf_0.91w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_15666.68_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-06_21-02-00/data/array2_fibers_2024-07-06_21-02-00_avf_0.01w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_44.63_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-07_03-07-44/data/array20_fibers_2024-07-07_03-07-44_avf_0.55w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_582.43_sec.pkl'
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
#         file_path = './geometrygen/animation/2024-07-11_15-37-56/data/array200_fibers_2024-07-11_15-37-56_avf_1.11w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_5843.84_sec.pkl'
#         file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers_usethis/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
        
#         # file_path = './geometrygen/animation/2024-07-08_17-34-27_AVF0.67_500fibers_K20/data/array250_fibers_2024-07-08_17-34-27_avf_1.53w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_13168.29_sec.pkl'
#         # file_path = './geometrygen/animation/2024-07-08_17-34-37_AVF0.65_250fibers_K20/data/array250_fibers_2024-07-08_17-34-37_avf_1.45w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_14463.59_sec.pkl'
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-06_21-44-31_40fibers_Avf0.1_K200/data/array20_fibers_2024-07-06_21-44-31_avf_0.12w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_816.17_sec.pkl'
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-10_18-32-13/data/array2_fibers_2024-10-10_18-32-13_avf_0.01_d2.1_sig0.78w_o10_w_c5_w_l5_K80_bffse0_bffsph0.001_498.18_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-08_23-37-30/data/array5_fibers_2024-10-08_23-37-30_avf_0.12_d2.1_sig0.78w_o10_w_c3_w_l5_K5_bffse0_bffsph0.001_85.28_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-03_10-22-13_AVF0.5_10fibers/data/array10_fibers_2024-07-03_10-22-13_avf_0.5w_o10_w_c5_w_l5_K-200_buffstartend0_buffsphere0.001_50.31_sec.pkl'
        
#         # experiments
#         #d2-k20
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_19-31-42_avf_0.69_d2.1_sig0.78_K20/data/array250_fibers_2024-09-08_19-31-42_avf_0.69_d2.1_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_15086.51_sec.pkl'
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
#         # ============== EXPERIEMNT HERE UPWARDS =========================
        
#         # 50fibers K80
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-20_16-38-23_50fib_test/data/array50_fibers_2024-10-20_16-38-23_avf_0.52_d2.1_sig0.78w_o10_w_c5_w_l5_K20_bffse0.5_bffsph0.5_624.62_sec.pkl'
#         # 50fibers K200
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-21_13-48-42/data/array50_fibers_2024-10-21_13-48-42_avf_0.48_d2.1_sig0.78w_o10_w_c5_w_l5_K200_bffse0.5_bffsph0.5_104.52_sec.pkl'
#         # 50fibers big geometry
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-21_14-25-54/data/array50_fibers_2024-10-21_14-25-54_avf_0.61_d5.68_sig0.78w_o10_w_c5_w_l5_K200_bffse0.5_bffsph0.5_812.34_sec.pkl'
        
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-21_14-26-34/data/array50_fibers_2024-10-21_14-26-34_avf_0.65_d5.68_sig0.78w_o10_w_c5_w_l5_K20_bffse0.5_bffsph0.5_1983.98_sec.pkl'
#         # 20 fibers sample varying time steps
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-07_03-07-44_small40fibers-try-sim/data/array20_fibers_2024-07-07_03-07-44_avf_0.55w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_582.43_sec.pkl'
#         import os
#         target_folder_path = os.path.dirname( os.path.dirname(file_path) )
#         folder_name = os.path.join(target_folder_path, 'ADC')
#         if not os.path.exists(folder_name):
#             os.makedirs(folder_name)
#             print("Folder created successfully.")
#         else:
#             print("Folder already exists.")  
#         optimized_fibers, L = ig.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
#         ###### only exp------------------------
#         # optimized_fibers[:, 0:4], L = optimized_fibers[:, 0:4]*2.0, L*2.0
#         print('optimized_fibers', optimized_fibers.shape)
#         print('L', L)
#         fiberlist_xyz_r_fid = ag.split_matrix_to_list(optimized_fibers, L)
#         print('L box length',L)
#         D = 3.0 # um^2/ms
#         T2 = 100 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(L,L,L,D,T2,rho)
#         fiber_xyzr_fid_list = ag.createBoundaryZ(fiberlist_xyz_r_fid, L)
#         print('len(fiber_xyzr_fid_list)',len(fiber_xyzr_fid_list))
        
#         for fiber in fiber_xyzr_fid_list:
#             sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
#             spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#             sg3.add_structure(spstruc) # add it to sg3
#         dt = 0.1 # time step in ms
#         nt = 300 # total number of steps thru time

#         #higher number of spins easier to come out..!
#         spins = int(5e3)
#         num_spins = spins
#         print('Start setting up structures')
#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=list(np.arange(0, len(fiber_xyzr_fid_list))))  # seed inside structures 0:len(fiberlist)
#         print('Done setting up structures')

#         # first, check thath all the spins are in fact in the multiple spstruc's
#         isInside = False
#         for structure in sg3.structures[:-1]:
#             isInside = np.logical_or(isInside, structure.isinside(sim.spins))
#         self.assertTrue(isInside.all())
#         # WORKING
#         # Dx_step = np.zeros(nt+1)
#         # Dy_step = np.zeros(nt+1)
#         # Dz_step = np.zeros(nt+1)
#         # diff_time = np.zeros(nt+1)
#         # Dx_step[0], Dy_step[0], Dz_step[0] = D, D, D
#         # print('Got to sim loops')
#         # for n in range(1, nt+1):
#         #     print(n)
#         #     isInside = False
#         #     for struct in sg3.structures[:-1]:
#         #         isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
#         #     # self.assertTrue(isInside.all())

#         #     if ~isInside.all():
#         #         print('stopped at', n)
#         #         spins_temp = sim.spins_d.get()
#         #         spins_temp = ((spins_temp.T)[np.invert(isInside)]).T
#         #         ig.plot_fibers(spins_temp, fiber_xyzr_fid_list, L, n, folder_name, file_path)
            
#         #     # --------step-----------
#         #     sim.step(dt)
#         #     Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
#         #     Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
#         #     Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
#         #     diff_time[n] = n*dt
        
#         Dx_step = [D]
#         Dy_step = [D]
#         Dz_step = [D]
#         diff_time = [0]
#         print('Got to sim loops')
#         current_time = 0
#         while current_time < nt*dt:
#             print('current_time', current_time)
#             isInside = False
#             for struct in sg3.structures[:-1]:
#                 isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
#             self.assertTrue(isInside.all())
#             # if ~isInside.all():
#             #     print('stopped at', current_time)
#             #     spins_temp = sim.spins_d.get()
#             #     spins_temp = ((spins_temp.T)[np.invert(isInside)]).T
#             #     ig.plot_fibers(spins_temp, fiber_xyzr_fid_list, L, current_time, folder_name, file_path)
            
#             time_step = ag.adjust_time_step(current_time=current_time, max_dt=dt)
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
#         elapsed_time = et - st
#         # save_data_pickle(file_name, np.array([Dx_step, Dy_step, Dz_step, diff_time]))
#         import os
#         file_name = os.path.basename(file_path)
#         base_name, extension = os.path.splitext(file_name)
#         print('base_name', base_name)
#         date_time = str(get_date_time())
#         pickle_file_name = file_name + '_'+str(date_time)
#         file_name_new = 'intra_'+ date_time + '_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' \
#             + str(num_spins)+'_spins_'+ base_name +'_SIMtime'+str(round(elapsed_time,2))+'_sec'+ '_' + str(dt)+'_maxstep_'
#         # Combine new filename with folder path to get the full path
#         data_folder_name = os.path.join(target_folder_path, 'ADC', 'ADCdata')
#         if not os.path.exists(data_folder_name):
#             os.makedirs(data_folder_name)
#             print("Folder created successfully.")
#         else:
#             print("Folder already exists.")  
#         data_file_path = os.path.join(data_folder_name, 'NoSEGMENT_'+file_name_new+'data.pkl')
#         save_ADCdata_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
#         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new )    

#         # after simulating diffusion, are all spins still inside sg3?
#         isInside = False
#         for struct in sg3.structures[:-1]:
#             isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
#         self.assertTrue(isInside.all())

#         fiber_list = ag.map_matrix_to_list_numpy(optimized_fibers)
#         color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list)))
#         np.random.shuffle(color)
#         # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, animation_input=True, optimized=True, folder_name=folder_name) 
#         # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_90', folder_name=folder_name) 
#         # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_0', folder_name=folder_name)                    
#         # ag.plot_radius_distribution()
#         # ag.plot_diameter_distribution()
#         # ag.plot_slices(L, folder_name, N=5, color=color, spheres_xyz_r_fid=optimized_fibers, folder_name=folder_name)
        
#         # ag.plot_along_axon_OD(fiberlist_xyz_r_fid, L, optimized=True, folder_name=folder_name)
#         ####################################################################################
#         print('optimized_fibers', type(optimized_fibers))
#         fiber_list = ag.map_matrix_to_list_numpy(optimized_fibers)
#         color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list)))
#         np.random.shuffle(color)
#         # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, animation_input=True, optimized=True) 
#         # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_90') 
#         # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_0')                    
#         # ag.plot_radius_distribution()
#         # ag.plot_diameter_distribution()
#         # ag.plot_slices(L, N=5, color=color, spheres_xyz_r_fid=optimized_fibers)


#         # signal from T2 decay
#         # self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)
#         # ===========================================
#         # save data with Pickle format
#         save_data_pickle(pickle_file_name, np.array([Dx_step, Dy_step, Dz_step, diff_time]))


# #     def testOutsideMultiFiberGeometry(self):
# #         '''
# #         # find /path/to/folder -type f -newermt "2024-05-06"
# #         find ./ -depth -name "*2024-06-27_17-44-24*"
# #         '''
# #         st = time.time()
# #         file_path = './geometrygen/animation/2024-07-07_03-06-59/data/array100_fibers_2024-07-07_03-06-59_avf_0.68w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_3618.52_sec.pkl'
# #         # file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
# #         # file_path = './geometrygen/animation/2024-07-05_19-24-55/data/array20_fibers_2024-07-05_19-24-55_avf_0.52w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_418.1_sec.pkl'
# #         file_path = './geometrygen/animation/2024-07-08_17-56-42_AVF_0.65_500fibers_K200/data/array250_fibers_2024-07-08_17-56-42_avf_1.57w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_14522.92_sec.pkl'
# #         file_path = './geometrygen/animation/2024-07-10_09-21-59/data/array250_fibers_2024-07-10_09-21-59_avf_8.48w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_129321.43_sec.pkl'
# #         file_path = './geometrygen/animation/2024-07-08_17-56-42_AVF_0.65_500fibers_K200/data/Farray250_fibers_2024-07-08_17-56-42_avf_1.57w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_14522.92_sec.pkl'
# #         file_path = './geometrygen/animation/2024-07-08_17-55-14_AVF_0.66_500fibers_K200/data/array250_fibers_2024-07-08_17-55-14_avf_1.31w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_11987.36_sec.pkl'
# #         file_path = './geometrygen/animation/2024-07-08_17-52-37_AVF0.65_500fibers_K200/data/array250_fibers_2024-07-08_17-52-37_avf_1.6w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_15947.77_sec.pkl'
# #         # file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers_usethis/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
# #         # file_path = './geometrygen/animation/2024-07-08_18-30-33/data/array250_fibers_2024-07-08_18-30-33_avf_5.38w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_101024.14_sec.pkl'
# #         # file_path = './geometrygen/animation/2024-07-11_15-37-56/data/array200_fibers_2024-07-11_15-37-56_avf_1.11w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_5843.84_sec.pkl'
# #         # file_path = './geometrygen/animation/2024-07-11_15-37-56/data/array200_fibers_2024-07-11_15-37-56_avf_1.11w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_5843.84_sec.pkl'
# #         # file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers_usethis/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
        
# #         file_path = './geometrygen/animation/2024-07-08_17-34-27_AVF0.67_500fibers_K20/data/array250_fibers_2024-07-08_17-34-27_avf_1.53w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_13168.29_sec.pkl'
# #         #230 fibers K200
# #         # file_path = './geometrygen/animation/2024-07-31_08-14-47/data/array230_fibers_2024-07-31_08-14-47_avf_0.0_d5.0_sig2.5w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_11497.37_sec.pkl'
# #         # # 250 fibers K200
# #         # file_path = './geometrygen/animation/2024-07-31_07-03-42/data/array250_fibers_2024-07-31_07-03-42_avf_0.0_d5.0_sig2.5w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_19811.39_sec.pkl'
# #         # # 300 fibers K200
# #         # file_path = './geometrygen/animation/2024-07-31_06-01-04_300fibersK200/data/array300_fibers_2024-07-31_06-01-04_avf_0.0_d5.0_sig2.5w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_16806.44_sec.pkl'
# #         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_17-34-37_AVF0.65_250fibers_K20/data/array250_fibers_2024-07-08_17-34-37_avf_1.45w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_14463.59_sec.pkl'
# #         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_17-52-37_AVF0.65_500fibers_K200/data/array250_fibers_2024-07-08_17-52-37_avf_1.6w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_15947.77_sec.pkl'
# #         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers_usethis/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
# #         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_17-34-27_AVF0.67_500fibers_K20/data/array250_fibers_2024-07-08_17-34-27_avf_1.53w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_13168.29_sec.pkl'
# #         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-08_17-55-14_AVF_0.66_500fibers_K200/data/array250_fibers_2024-07-08_17-55-14_avf_1.31w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_11987.36_sec.pkl'
# #         #exp1
# #         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78_K20/data/array250_fibers_2024-09-08_12-44-54_avf_0.68_d2.1_sig0.78w_o10_w_c5_w_l5_K20_bffse0_bffsph0.001_20090.62_sec.pkl'
# #         #exp2
# #         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-08-30_avf_0.64_d2.1_sig0.78_K200/data/array250_fibers_2024-09-06_20-08-30_avf_0.64_d2.1_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_14568.68_sec.pkl'
# #         # #exp3
# #         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-20-17_avf_0.65_d3.14_sig0.78_K200/data/array300_fibers_2024-09-06_20-20-17_avf_0.65_d3.14_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_17259.06_sec.pkl'
# #         # #exp4
# #         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-09-06_20-36-58_avf_0.63_d5.68_sig0.78_K200/data/array300_fibers_2024-09-06_20-36-58_avf_0.63_d5.68_sig0.78w_o10_w_c5_w_l5_K200_bffse0_bffsph0.001_102966.59_sec.pkl'
# #         # ---20 fiberes 
# #         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-22_19-46-58_converged_20fib/data/array20_fibers_2024-10-22_19-46-58_avf_0.72_d1.38_sig0.3w_o1000_w_c1_w_l100_K200_bffse0.01_bffsph0.01_28008.44_sec.pkl'        
# #         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-07-07_03-07-44_small40fibers-try-sim/data/array20_fibers_2024-07-07_03-07-44_avf_0.55w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_582.43_sec.pkl'
# #         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2024-10-24_10-15-11/data/array2_fibers_2024-10-24_10-15-11_avf_0.37_d1.38_sig0.3w_o1000_w_c1_w_l50_K200_bffse0.01_bffsph0.01_143.17_sec.pkl'
# #         import os
# #         # Extract the target folder path (without the filename)
# #         target_folder_path = os.path.dirname( os.path.dirname(file_path) )
# #         folder_name = os.path.join(target_folder_path, 'ADC')
# #         if not os.path.exists(folder_name):
# #             os.makedirs(folder_name)
# #             print("Folder created successfully.")
# #         else:
# #             print("Folder already exists.")  
# #         optimized_fibers, L = ig.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
# #         ###### only exp------------------------
# #         # optimized_fibers[:, 0:4], L = optimized_fibers[:, 0:4], L

# #         print('optimized_fibers', optimized_fibers.shape)
# #         fiberlist_xyz_r_fid = ag.split_matrix_to_list(optimized_fibers, L)
# #         print('L box length',L)
# #         D = 3.0 # um^2/ms
# #         T2 = 100 # ms
# #         rho = 1 # fractional water density
# #         sg3 = geom.SimGeometry3D(L,L,L,D,T2,rho)
# #         fiber_xyzr_fid_list = ag.createBoundaryZ(fiberlist_xyz_r_fid, L)
# #         print('len(fiber_xyzr_fid_list)',len(fiber_xyzr_fid_list))
        
# #         for fiber in fiber_xyzr_fid_list:
# #             sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
# #             spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
# #             sg3.add_structure(spstruc) # add it to sg3
# #         dt = 0.005 # time step in ms
# #         nt = 10000 # total number of steps thru time

# #         # #higher number of spins easier to come out..!
# #         spins = int(5e3)
# #         # #  spins=int(1e5)
# #         num_spins = spins
# #         print('Start setting up structures')
# #         sim = ds3.DiffSim3d(sg3,spins)
# #         # sim.setup(structures=list(np.arange(0, int(len(fiber_xyzr_fid_list)/20))))  # seed INSIDE structures 0:len(fiberlist)
# #         # sim.setup(structures=list(np.arange(0, int(len(fiber_xyzr_fid_list)/2))))  # seed INSIDE structures 0:len(fiberlist)
# #         # print('sg3.nstructures',sg3.nstructures)
        
# #         sim.setup(structures=[int(len(fiber_xyzr_fid_list))])  # seed OUTSIDE structures len(fiberlist)=index of box
# #         print('Done setting up structures')
        
# #         # first, check thath all the spins are in fact in the multiple spstruc's
# #         isOutside = False
# #         for struct in sg3.structures[:-1]:
# #             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
# #         isOutside = np.invert(isOutside)
# #         print('num outside', np.sum(isOutside))
# #         self.assertTrue(isOutside.all())
# #         if ~isOutside.all():
# #             exit()
        
# #         Dx_step = np.zeros(nt+1)
# #         Dy_step = np.zeros(nt+1)
# #         Dz_step = np.zeros(nt+1)
# #         diff_time = np.zeros(nt+1)
# #         Dx_step[0], Dy_step[0], Dz_step[0] = D, D, D
# #         print('Got to extra-axonal sim loops')
# #         for n in range(1, nt+1):
# #             print(n)
# #             isOutside = False
# #             for struct in sg3.structures[:-1]:
# #                 isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
# #             isOutside = np.invert(isOutside)

# #             if ~isOutside.all() or n==1 or n==9999 or n==19999:
# #                 print('stopped at ', n)
# #                 spins_temp = sim.spins_d.get()
# #                 spins_temp = ((spins_temp.T)[np.invert(isOutside)]).T
# #                 ig.plot_fibers(spins_temp, fiber_xyzr_fid_list, L, n, folder_name, file_path)
            
# #             # --------step-----------
# #             sim.step(dt)
# #             Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
# #             Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
# #             Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
# #             diff_time[n] = n*dt

# #     #     #=======================================
# #     #     # msdx = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins
# #     #     # msdy = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins
# #     #     # msdz = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins
# #     #     #print(msdx/2/dt/nt,msdy/2/dt/nt,msdz/2/dt/nt)

# #     #     # Diffusion in the x direction should be close to D
# #     #     # self.assertGreater(msdx/2/dt/nt,D-0.2)
# #     #     # self.assertLess(msdx/2/dt/nt,D+0.2)
# #     #     # Diffusion in the y&z direction shoudl be close to 0 due to restriction
# #     #     # self.assertLess(msdy/2/dt/nt,0.01)
# #     #     # self.assertLess(msdz/2/dt/nt,0.01)

# #         et = time.time()
# #         # # get the execution time
# #         elapsed_time = et - st

# #     # #     #==================================
# #         file_name = os.path.basename(file_path)
# #         base_name, extension = os.path.splitext(file_name)
# #         print('base_name', base_name)
# #         date_time = str(get_date_time())
# #         file_name_new = 'extra_'+ date_time + '_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' \
# #             + str(num_spins)+'_spins_'+ base_name +'_'+str(round(elapsed_time,2))+'_sec'+'_' + str(dt)+'_step_'

# #         # Combine new filename with folder path to get the full path
# #         data_file_path = os.path.join(target_folder_path, 'ADC', 'ADCdata', 'NoSEGMENT_'+file_name_new+'data.pkl')
        
# #         save_ADCdata_pickle(data_file_path, np.column_stack((Dx_step, Dy_step, Dz_step, diff_time)))
# #         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, folder_name, file_name_new)    

# #         # after simulating diffusion, are all spins still outside axons
# #         isOutside = False
# #         for struct in sg3.structures[:-1]:
# #             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
# #         isOutside = np.invert(isOutside)
# #         print('num outside', np.sum(isOutside))
# #         self.assertTrue(isOutside.all())

# #         # fiber_list = ag.map_matrix_to_list_numpy(optimized_fibers)
# #         # color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list)))
# #         # np.random.shuffle(color)
# #         # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, animation_input=True, optimized=True, folder_name=folder_name) 
# #         # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_90', folder_name=folder_name) 
# #         # ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_0', folder_name=folder_name)                    
# #         # ag.plot_radius_distribution()
# #         # ag.plot_diameter_distribution()
# #         # ag.plot_slices(L, folder_name, N=5, color=color, spheres_xyz_r_fid=optimized_fibers, folder_name=folder_name)
        
# #         ag.plot_along_axon_OD(fiberlist_xyz_r_fid, L, optimized=True, folder_name=folder_name)

# if __name__ == '__main__':
#     unittest.main()