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
# from pycuda.reduction import ReductionKernel
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
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-08_12-10-57/data/array200_fibers_boxL_48.0_2025-01-08_12-10-57_avf_0.49_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_2562.17_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-08_13-16-09/data/array20_fibers_boxL_14.0_2025-01-08_13-16-09_avf_0.44_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_338.09_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-07_15-59-44/data/array3_fibers_boxL_31.0_2025-01-07_15-59-44_avf_0.01_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_64.49_sec.pkl'
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-17_17-45-45/data/array3_fibers_boxL_6.0_2025-01-17_17-45-45_avf_0.26_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_141.72_sec.pkl'
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
#         dt = 0.001 # time step in ms
#         nt = int(2000000) # total number of steps thru time

#         #higher number of spins easier to come out..!
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

    
#         ### ------------------------------------------------------
#         # Assuming nt, dt, spins, block_size, grid_size, and sim are defined
#         Dx_step = gpuarray.zeros(nt + 1, np.float32)
#         Dy_step = gpuarray.zeros(nt + 1, np.float32)
#         Dz_step = gpuarray.zeros(nt + 1, np.float32)
#         diff_time = gpuarray.zeros(nt + 1, np.float32)

#         Dx_step[0] = D
#         Dy_step[0] = D
#         Dz_step[0] = D

#         # Reduction kernels for summing on the GPU
#         sum_reduction = ReductionKernel(
#             np.float32,
#             neutral="0",
#             reduce_expr="a + b",
#             map_expr="x[i]",
#             arguments="float *x"
#         )

#         current_time = 0.0
#         idx = 0
#         print("Starting simulation loops...")
#         while current_time < nt * dt:
#             idx += 1
#             current_time += dt

#             # Simulate step in GPU
#             sim.step(dt)

#             # Compute squared displacements directly on GPU
#             dx_squared = gpuarray.empty(sim.spins_d.shape[1], np.float32)
#             dy_squared = gpuarray.empty(sim.spins_d.shape[1], np.float32)
#             dz_squared = gpuarray.empty(sim.spins_d.shape[1], np.float32)

#             sim.compute_squared_displacements(
#                 sim.spins_d, sim.spins0_d, dx_squared, dy_squared, dz_squared)

#             # Perform the summation on the GPU
#             Dx_step[idx] = sum_reduction(dx_squared) / (spins * 2 * current_time)
#             Dy_step[idx] = sum_reduction(dy_squared) / (spins * 2 * current_time)
#             Dz_step[idx] = sum_reduction(dz_squared) / (spins * 2 * current_time)

#             # Store the current time
#             diff_time[idx] = current_time

#         # Retrieve results back to CPU after the loop
#         Dx_step_cpu = Dx_step.get()
#         Dy_step_cpu = Dy_step.get()
#         Dz_step_cpu = Dz_step.get()
#         diff_time_cpu = diff_time.get()

#         print("Simulation completed!")
#         Dx_step = np.array(Dx_step_cpu)
#         Dy_step = np.array(Dy_step_cpu)
#         Dz_step = np.array(Dz_step_cpu)
#         diff_time = np.array(diff_time_cpu)
#         et = time.time()
#         # get the execution time
#         elapsed_time = et - st
        
#         #==================================
#         file_name = os.path.basename(file_path)
#         base_name, extension = os.path.splitext(file_name)
#         print('base_name', base_name)
#         date_time = str(get_date_time())
#         file_name_new = 'GPU_intra_'+ date_time + '_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' \
#             + str(num_spins)+'_spins_'+ base_name +'_SIMtime'+str(round(elapsed_time,2))+'_sec'+ '_' + str(dt)+'_maxstep_'
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

#         # after simulating diffusion, are all spins still inside sg3?
#         isInside = False
#         for struct in sg3.structures[:-1]:
#             isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
#         self.assertTrue(isInside.all())

#     def testOutsideMultiFiberGeometry(self):
#         st = time.time()
#         # small num axons
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-08_12-10-57/data/array200_fibers_boxL_48.0_2025-01-08_12-10-57_avf_0.49_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_2562.17_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-08_13-16-09/data/array20_fibers_boxL_14.0_2025-01-08_13-16-09_avf_0.44_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_338.09_sec.pkl'
#         # file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-07_15-59-44/data/array3_fibers_boxL_31.0_2025-01-07_15-59-44_avf_0.01_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_64.49_sec.pkl'
#         file_path = '/home/nguyt16@ds.vanderbilt.edu/GPU_McDiffusionSim3D/geometrygen/animation/2025-01-17_17-45-45/data/array3_fibers_boxL_6.0_2025-01-17_17-45-45_avf_0.26_d1.4_sig0.4w_o10_w_c3_w_l3_K200_ODI_0.0032_bffse0.24_bffsph0.001_141.72_sec.pkl'
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
#         dt = 0.001 # time step in ms
#         nt = int(2000000) # total number of steps thru time
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

#         ### ------------------------------------------------------
#         # Assuming nt, dt, spins, block_size, grid_size, and sim are defined
#         Dx_step = gpuarray.zeros(nt + 1, np.float32)
#         Dy_step = gpuarray.zeros(nt + 1, np.float32)
#         Dz_step = gpuarray.zeros(nt + 1, np.float32)
#         diff_time = gpuarray.zeros(nt + 1, np.float32)

#         Dx_step[0] = D
#         Dy_step[0] = D
#         Dz_step[0] = D

#         # Reduction kernels for summing on the GPU
#         sum_reduction = ReductionKernel(
#             np.float32,
#             neutral="0",
#             reduce_expr="a + b",
#             map_expr="x[i]",
#             arguments="float *x"
#         )

#         current_time = 0.0
#         idx = 0
#         print("Starting simulation loops...")
#         while current_time < nt * dt:
#             idx += 1
#             current_time += dt

#             # Simulate step in GPU
#             sim.step(dt)

#             # Compute squared displacements directly on GPU
#             dx_squared = gpuarray.empty(sim.spins_d.shape[1], np.float32)
#             dy_squared = gpuarray.empty(sim.spins_d.shape[1], np.float32)
#             dz_squared = gpuarray.empty(sim.spins_d.shape[1], np.float32)

#             sim.compute_squared_displacements(
#                 sim.spins_d, sim.spins0_d, dx_squared, dy_squared, dz_squared)

#             # Perform the summation on the GPU
#             Dx_step[idx] = sum_reduction(dx_squared) / (spins * 2 * current_time)
#             Dy_step[idx] = sum_reduction(dy_squared) / (spins * 2 * current_time)
#             Dz_step[idx] = sum_reduction(dz_squared) / (spins * 2 * current_time)

#             # Store the current time
#             diff_time[idx] = current_time

#         # Retrieve results back to CPU after the loop
#         Dx_step_cpu = Dx_step.get()
#         Dy_step_cpu = Dy_step.get()
#         Dz_step_cpu = Dz_step.get()
#         diff_time_cpu = diff_time.get()

#         print("Simulation completed!")
#         Dx_step = np.array(Dx_step_cpu)
#         Dy_step = np.array(Dy_step_cpu)
#         Dz_step = np.array(Dz_step_cpu)
#         diff_time = np.array(diff_time_cpu)
#         et = time.time()
#         # get the execution time
#         elapsed_time = et - st
        
#         #==================================
#         file_name = os.path.basename(file_path)
#         base_name, extension = os.path.splitext(file_name)
#         print('base_name', base_name)
#         date_time = str(get_date_time())
#         file_name_new = 'GPU_extra_'+ date_time + '_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' \
#             + str(num_spins)+'_spins_'+ base_name +'_SIMtime'+str(round(elapsed_time,2))+'_sec'+ '_' + str(dt)+'_maxstep_'
        
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
#         # print('num outside', np.sum(isOutside))
#         self.assertTrue(isOutside.all())

# if __name__ == '__main__':
#     unittest.main()