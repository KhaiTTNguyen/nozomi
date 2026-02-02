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

# class TestDiffSim3D(unittest.TestCase):

#     def setUp(self):
#         drv.Device(0).make_context()
    
#     def tearDown(self):
#         drv.Context.pop()

    # def testInsideTwoSetOfMultipleSpheres(self):
    #     '''
    #     diffusion inside of multiple spheres
    #     '''
    #     Lx = 10.0 # um
    #     Ly = 10.0 # um
    #     Lz = 10.0 # um
    #     D = 3.0 # um^2/ms
    #     T2 = 200 # ms
    #     rho = 1 # fractional water density
    #     sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

    #     nsphere = 200 # fail at 100 high num of spheres & high curvature fails
    #     sz = np.linspace(-Lz/2, Lz/2, num=nsphere)
    #     sz = np.concatenate((sz[-30:-1]-Lz, sz, sz[1:30]+Lz ), axis=0)
    #     sy = np.sin(2*np.pi*sz/Ly)
 
    #     # sz = np.arange(-nsphere/2-11,nsphere/2+11)*Lx/nsphere   # needs to include +-Lx/2
    #     # sy = np.sin(2*np.pi*np.arange(-11,nsphere+11)/nsphere)*1.5 
    #     sx = np.full(sz.shape, 0)
        
    #     sx = np.concatenate((np.full(sz.shape, 0)+Lx/2,np.full(sz.shape, 0)-Lx/2))
    #     print('sz', sz)
    #     print('sy', sy)
    #     sy = np.concatenate((sy, sy))
    #     sz = np.concatenate((sz,sz))
    #     sr = np.full(sz.shape, 1)

    #     # exit()
    #     spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
    #     sg3.add_structure(spstruc) # add it to sg3
    #     print('sg3.nstructures',sg3.nstructures)
    #     # 100000
    #     dt = 0.01 # time step in ms
    #     nt = 100 # total number of steps thru time
    #     # nt = 10000 # total number of steps thru time
    #     spins = int(1e5)
    #     num_spins=spins
    #     sim = ds3.DiffSim3d(sg3,spins)
    #     sim.setup(structures=[0,])

    #     # first, check thath all the spins are in fact in spstruc
    #     isInside = False
    #     for struct in sg3.structures[:-1]:
    #         print(type(struct))
    #         isInside = np.logical_or(isInside, struct.isinside(sim.spins)) # str.isinside(sim.spins) return vector length num_spins & OR with isInside
    #     self.assertTrue(isInside.all())

    #     # ---------------- original --------------
    #     # for n in range(nt):
    #     #     sim.step(dt)
        
    #     Dx_step = np.zeros(nt+1)
    #     Dy_step = np.zeros(nt+1)
    #     Dz_step = np.zeros(nt+1)
    #     diff_time = np.zeros(nt+1)
    #     Dx_step[0], Dy_step[0], Dz_step[0] = D, D, D
    #     # file_name = '1_5000_2023-11-28_12-42-07_avf_0.058_K200_1_sec.pkl' #12min
    #     # optimization_problem = ig.import_geometry(file_name)
        
    #     for n in range(1, nt+1):
    #         print(n)
    #         # ---------- step -----------
    #         isInside = False
    #         for struct in sg3.structures[:-1]:
    #             isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
    #         # self.assertTrue(isInside.all())

    #         if n==9 or ~isInside.all(): #and n%10==0:
    #             print(n)
    #             fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    #             # ax.view_init(azim=45, elev=45)
    #             spins_temp = sim.spins_d.get()
    #             print('spins', spins_temp)
    #             ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=0.1, marker='.')
                
    #             'get index of fiber.original_id in unique_ids --> map to color'
    #             # fiber_idx=0
    #             # optimization_problem.plot_spheres(ax=ax, node_list=fiber, fiber_idx=fiber_idx, 
    #             #                 color='red')
    #             # print('fiber_'+str(fiber_count)+' plotted.')
    #             # fiber_idx=fiber_idx+len(fiber)

    #             ax.set_xlim(-Lx/2, Lx/2)
    #             ax.set_ylim(-Ly/2, Ly/2)
    #             ax.set_zlim(-Lz/2, Lz/2)
    #             ax.set_xlabel("x (um)")
    #             ax.set_ylabel("y (um)")
    #             ax.set_zlabel("z (um)")
    #             spin_plot_file_name = 'outside_at'+str(n)+'.png'
    #             spin_plot_file_name = os.path.join('.','diffsim3d','animation','debug_spin', spin_plot_file_name)
    #             pl.savefig(spin_plot_file_name)
    #             self.assertTrue(isInside.all())

  
    #         # ---------- step -----------
    #         sim.step(dt)
    #         Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
    #         Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
    #         Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
    #         diff_time[n] = n*dt

    #     plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, str(nsphere) +'_vecticle_z_'+ str(dt) +'_step_')

    #     #-------------------------------------------
    #     # all spins are still inside the arena
    #     self.assertTrue(spstruc.isinside(sim.spins_d.get()).all())

    #     # print(sim.spins_d.get()-sim.spins0_d.get())
    #     msdx = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins
    #     msdy = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins
    #     msdz = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins
    #     #print(msdx/2/dt/nt,msdy/2/dt/nt,msdz/2/dt/nt)

    #     # Diffusion in the z direction should be close to D
    #     # self.assertGreater(msdz/2/dt/nt,D-0.2)
    #     # self.assertLess(msdz/2/dt/nt,D+0.2)
    #     # Diffusion in the x&y direction shoudl be close to 0 due to restriction
    #     # self.assertLess(msdy/2/dt/nt,0.02)
    #     # self.assertLess(msdx/2/dt/nt,0.02)

    #     # after simulating diffusion, are all spins still inside spstruc?
    #     self.assertEqual(np.sum(~spstruc.isinside(sim.spins_d.get())),0)

    #     # signal from T2 decay
    #     self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)
    #     #-------------------------------------------

    #     # after simulating diffusion, are all spins still inside sg3?
    #     isInside = False
    #     for struct in sg3.structures[:-1]:
    #         isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get())) # str.isinside(sim.spins) return vector length num_spins & OR with isInside
    #     self.assertTrue(isInside.all())

    # def testInside1FiberGeometry(self):
    #     '''
    #     diffusion inside fibers
    #     '''
    #     st = time.time()
    #     file_name = '1_5000_2023-11-28_12-42-07_avf_0.058_K200_1_sec.pkl' #12min
    #     # file_name = '1_5000_2023-11-28_12-42-07_avf_0.058_K200_1_sec.pkl'
    #     # file_name = '43_500000_2023-11-01_11-26-53_avf_0.626.pkl'
    #     # file_name='35_500000_2023-10-26_13-48-40_avf_0.519386_5hr.pkl'
    #     # file_name='10_50000_2023-10-28_22-35-24_avf_0.106696.pkl'
    #     # file_name='5_1000_2023-11-08_00-23-06_avf_0.402_92_sec.pkl'
    #     # file_name='4_1000_2023-11-08_00-32-22_avf_0.376_118_sec.pkl'
    #     fibers_xyzr_fid, Lx, Ly, Lz = ig.import_array_geometry_full_path(file_name)
    #     xa, ya, za, ra, f_id = unpack_matrix_to_vectors(fibers_xyzr_fid)
    #     print('Lx',Lx, 'Ly', Ly, 'Lz', Lz)
    #     D = 3.0 # um^2/ms
    #     T2 = 100 # ms
    #     rho = 1 # fractional water density
    #     sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)
        
    #     fiber_xyzr_fid_list = ag.createBoundaryZ(fibers_xyzr_fid, Lz)
    #     # fiber_xyzr_fid_list = ag.shiftFOVtoPlusMinusHalfLx(fiber_xyzr_fid_list, Lx, Ly, Lz)
    #     fiber_xyzr_fid_list = ag.shiftToEdgeAndCreateAnotherAtReciprocal(fiber_xyzr_fid_list, Lx, Ly, Lz)
    #     print('len(fiber_xyzr_fid_list)',len(fiber_xyzr_fid_list))
    #     for fiber in fiber_xyzr_fid_list:
    #         sx, sy, sz, sr = fiber.x, fiber.y, fiber.z, fiber.radius
    #         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
    #         sg3.add_structure(spstruc) # add it to sg3

    #     dt = 0.01 # time step in ms
    #     nt = 100 # total number of steps thru time
    #     # dt = 0.01 # time step in ms
    #     # nt = 10000 # total number of steps thru time
    #     spins = int(1e4)
    #     num_spins = spins
    #     print('Start setting up structures')
    #     sim = ds3.DiffSim3d(sg3,spins)
    #     sim.setup(structures=[0,1])
    #     print('Done setting up structures')
        
    #     # first, check thath all the spins are in fact in the multiple spstruc's
    #     isInside = False
    #     for struct in sg3.structures[:-1]:
    #         isInside = np.logical_or(isInside, struct.isinside(sim.spins)) # str.isinside(sim.spins) return vector length num_spins & OR with isInside
    #     self.assertTrue(isInside.all())

    #     Dx_step = np.zeros(nt+1)
    #     Dy_step = np.zeros(nt+1)
    #     Dz_step = np.zeros(nt+1)
    #     diff_time = np.zeros(nt+1)
    #     Dx_step[0], Dy_step[0], Dz_step[0] = D, D, D
    #     print('Got to sim loops')
    #     for n in range(1, nt+1):
    #         print(n)
    #         isInside = False
    #         for struct in sg3.structures[:-1]:
    #             isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
    #         # self.assertTrue(isInside.all())

    #         if n==200 or ~isInside.all():
    #             fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    #             ax.view_init(azim=0, elev=0)
    #             spins_temp = sim.spins_d.get()
    #             print('spins', spins_temp)
    #             ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=0.5, marker='.')
                
    #             from matplotlib.pyplot import cm    
    #             color = cm.rainbow(np.linspace(0.0, 1.0, len(optimization_problem.optimized_fibers)))
    #             np.random.shuffle(color)
    #             'get index of fiber.original_id in unique_ids --> map to color'
    #             unique_ids = np.unique([fiber.original_fiber[0] for fiber in fiber_xyzr_fid_list])
    #             fiber_idx=0
    #             for fiber_count, fiber in enumerate(fiber_xyzr_fid_list):
    #                 color_idx = np.argwhere(np.isin(unique_ids , fiber.original_fiber[0])).ravel()[0]
    #                 fiber_color = color[color_idx]
    #                 optimization_problem.plot_spheres(ax=ax, node_list=fiber, fiber_idx=fiber_idx, 
    #                                 color=fiber_color)
    #                 print('fiber_'+str(fiber_count)+' plotted.')
    #                 fiber_idx=fiber_idx+len(fiber)

    #             ax.set_xlim(-1/2*Lx, 1/2*Lx)
    #             ax.set_ylim(-1/2*Ly, 1/2*Ly)
    #             ax.set_zlim(-1/2*Lz, 1/2*Lz)
    #             ax.set_xlabel("x (um)")
    #             ax.set_ylabel("y (um)")
    #             ax.set_zlabel("z (um)")
    #             spin_plot_file_name = 'outside_at'+str(n)+'.png'
    #             spin_plot_file_name = os.path.join('.','diffsim3d','animation','debug_spin', spin_plot_file_name)
    #             pl.savefig(spin_plot_file_name)
                
    #         # ---------- step -----------
    #         sim.step(dt)
    #         Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
    #         Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
    #         Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
    #         diff_time[n] = n*dt

    #     # msdx = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins
    #     # msdy = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins
    #     # msdz = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins
        
    #     et = time.time()
    #     # get the execution time
    #     elapsed_time = et - st
    #     file_name = 'diffsim3d_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' + str(num_spins)+'_spins_'+ str(optimization_problem.date_time) + \
    #                     '_avf_'+str(optimization_problem.volumne_fraction)+'_'+str(elapsed_time)+'_sec'
    #     # file_name = os.path.join('.','diffsim3d','data',file_name)

    #     plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, file_name + '_' + str(dt)+'_step_')

    #     #print(msdx/2/dt/nt,msdy/2/dt/nt,msdz/2/dt/nt)

    #     # Diffusion in the x direction should be close to D
    #     # self.assertGreater(msdx/2/dt/nt,D-0.2)
    #     # self.assertLess(msdx/2/dt/nt,D+0.2)
    #     # Diffusion in the y&z direction shoudl be close to 0 due to restriction
    #     # self.assertLess(msdy/2/dt/nt,0.01)
    #     # self.assertLess(msdz/2/dt/nt,0.01)

    #     # after simulating diffusion, are all spins still inside sg3?
    #     isInside = False
    #     for struct in sg3.structures[:-1]:
    #         isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
    #     self.assertTrue(isInside.all())
    # -------------------------------------------------------------------------------------------
    # def testInsideMultiFiberGeometry(self):
    #     '''
    #     # find /path/to/folder -type f -newermt "2024-05-06"
    #     find ./ -depth -name "*2024-06-27_17-44-24*"
    #     '''
    #     st = time.time()
    #     # file_name = '5_500000_2024-01-23_11-03-02_avf_0.008333333333333333_K20_space_buffer_end1e-05_space_buffer_sphere1e-05_2229_sec.pkl'
    #     # file_name = '6_500000_2023-11-28_16-06-53_avf_0.0632_K200_1462_sec.pkl'
    #     # file_name = '15_30000_2024-03-25_16-35-32_avf_0.0626_K20_space_buffer_startend1_space_buffer_sphere0.5_1050_sec.pkl'
    #     # file_name = '43_500000_2024-01-19_16-24-40_avf_0.155_K200_space_buffer_end1e-05_space_buffer_sphere1e-05_100230_sec.pkl'
    #     # file_name = 'SAP_10_5000_2024-05-06_18-12-17_avf_0.0658_K200_space_buffer_startend1e-05_space_buffer_sphere1e-05_52_sec.pkl'
    #     # file_name = 'SAP_5_5000_2024-05-07_10-29-04_avf_0.0_K200_space_buffer_startend1e-05_space_buffer_sphere1e-05_24_sec.pkl' DELETE this
    #     # file_name =         file_name = 'SAP_80_100000_2024-05-08_04-29-32_avf_0.3867_K200_space_buffer_startend1e-06_space_buffer_sphere0.0001_151856_sec.pkl ERROR
    #     # file_name = 'SAP_5_5000_2024-05-07_19-58-08_avf_0.0342_K200_space_buffer_startend1e-05_space_buffer_sphere1e-05_7_sec.pkl'
    #     # file_name = 'SAP_5_5000_2024-05-07_20-07-05_avf_0.0366_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_9_sec.pkl'
    #     # file_name = 'SAP_5_5000_2024-05-07_20-15-32_avf_0.0524_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_18_sec.pkl'
    #     # file_name = 'SAP_5_5000_2024-05-07_20-15-27_avf_0.0344_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_7_sec.pkl'
    #     # file_name = 'SAP_5_5000_2024-05-07_20-36-49_avf_0.0334_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_27_sec.pkl'
    #     # file_name = 'SAP_5_5000_2024-05-07_20-36-52_avf_0.0436_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_27_sec.pkl'
    #     # file_name = 'SAP_60_100000_2024-05-08_02-42-22_avf_0.234_K50_space_buffer_startend1e-05_space_buffer_sphere0.0001_4571_sec.pkl' # checking why
    #     # file_name = 'SAP_60_100000_2024-05-08_02-39-41_avf_0.2948_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_5657_sec.pkl' # CANNOT LOAD
    #     # file_name = ' SAP_60_100000_2024-05-08_02-40-39_avf_0.215_K200_space_buffer_startend1e-05_space_buffer_sphere0.0001_4016_sec.pkl' # cannot lood
    #     # file_name = 'SAP_60_100000_2024-05-08_02-42-54_avf_0.3938_K10_space_buffer_startend1e-05_space_buffer_sphere0.0001_156365_sec.pkl'        
    #     # file_name = 'SAP_10_10000_2024-05-10_17-22-47_avf_0.01824_K15_space_buffer_startend1e-06_space_buffer_sphere0.0001_48_sec.pkl' worked
    #     # file_name = 'SAP_10_10000_2024-05-10_17-25-02_avf_0.02164_K15_space_buffer_startend1e-06_space_buffer_sphere0.0001_59_sec.pkl' worked
    #     # file_name = 'SAP_array10_5000_2024-05-11_18-53-54_avf_0.0742_K200_space_buffer_startend1e-06_space_buffer_sphere0.001_90_sec.pkl'
    #     # file_name = 'SAP_array10_5000_2024-05-11_19-01-09_avf_0.04322_K20_space_buffer_startend1e-06_space_buffer_sphere0.001_92_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-06-27_17-44-24/data/array2_fibers_2024-06-27_17-44-24_avf_0.0077_w_overlap10_w_curve5_w_length5_K-200_buffstartend0_buffsphere0.001_703_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-03_14-14-05_200fibers_avf0.7/data/array200_fibers_2024-07-03_14-14-05_avf_0.53w_o10_w_c5_w_l5_K-200_buffstartend0_buffsphere0.001_6216.1_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-06_01-07-11/data/array20_fibers_2024-07-06_01-07-11_avf_0.57w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_560.95_sec.pkl'
    #     # 600fibers file_path = './geometrygen/animation/2024-07-06_03-54-08_AVF0.91_600fibers/data/array300_fibers_2024-07-06_03-54-08_avf_0.91w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_15666.68_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-06_21-02-00/data/array2_fibers_2024-07-06_21-02-00_avf_0.01w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_44.63_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-07_03-07-44/data/array20_fibers_2024-07-07_03-07-44_avf_0.55w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_582.43_sec.pkl'
    #     file_path = './geometrygen/animation/2024-07-07_03-06-59/data/array100_fibers_2024-07-07_03-06-59_avf_0.68w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_3618.52_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-05_19-24-55/data/array20_fibers_2024-07-05_19-24-55_avf_0.52w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_418.1_sec.pkl'
    #     file_path = './geometrygen/animation/2024-07-08_17-56-42_AVF_0.65_500fibers_K200/data/array250_fibers_2024-07-08_17-56-42_avf_1.57w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_14522.92_sec.pkl'
    #     file_path = './geometrygen/animation/2024-07-10_09-21-59/data/array250_fibers_2024-07-10_09-21-59_avf_8.48w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_129321.43_sec.pkl'
    #     file_path = './geometrygen/animation/2024-07-08_17-56-42_AVF_0.65_500fibers_K200/data/Farray250_fibers_2024-07-08_17-56-42_avf_1.57w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_14522.92_sec.pkl'
    #     file_path = './geometrygen/animation/2024-07-08_17-55-14_AVF_0.66_500fibers_K200/data/array250_fibers_2024-07-08_17-55-14_avf_1.31w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_11987.36_sec.pkl'
    #     file_path = './geometrygen/animation/2024-07-08_17-52-37_AVF0.65_500fibers_K200/data/array250_fibers_2024-07-08_17-52-37_avf_1.6w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_15947.77_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers_usethis/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-08_18-30-33/data/array250_fibers_2024-07-08_18-30-33_avf_5.38w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_101024.14_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-11_15-37-56/data/array200_fibers_2024-07-11_15-37-56_avf_1.11w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_5843.84_sec.pkl'
    #     file_path = './geometrygen/animation/2024-07-11_15-37-56/data/array200_fibers_2024-07-11_15-37-56_avf_1.11w_o10_w_c5_w_l5_K200_buffstartend0_buffsphere0.001_5843.84_sec.pkl'
    #     file_path = './geometrygen/animation/2024-07-09_02-02-19_K100_avf0.65_250fibers_usethis/data/array250_fibers_2024-07-09_02-02-19_avf_4.37w_o10_w_c5_w_l5_K100_buffstartend0_buffsphere0.001_51405.92_sec.pkl'
        
    #     # file_path = './geometrygen/animation/2024-07-08_17-34-27_AVF0.67_500fibers_K20/data/array250_fibers_2024-07-08_17-34-27_avf_1.53w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_13168.29_sec.pkl'
    #     # file_path = './geometrygen/animation/2024-07-08_17-34-37_AVF0.65_250fibers_K20/data/array250_fibers_2024-07-08_17-34-37_avf_1.45w_o10_w_c5_w_l5_K20_buffstartend0_buffsphere0.001_14463.59_sec.pkl'
        
        
    #     # file_name = 'SAP_10_5000_2024-05-10_12-19-04_avf_0.03864_K10_space_buffer_startend1e-06_space_buffer_sphere0.0001_100_sec.pkl'
    #     # file_name = 'SAP_50_80000_2024-04-17_18-33-32_avf_0.2248_K20_space_buffer_startend0.001_space_buffer_sphere0.0_29614_sec.pkl'
    #     # file_name = 'SAP_8_5000_2024-04-02_11-16-29_avf_0.0892_K200_space_buffer_startend0.001_space_buffer_sphere2.5_25_sec.pkl'
    #     # file_name='35_500000_2023-10-26_13-48-40_avf_0.519386_5hr.pkl'
    #     # file_name='10_50000_2023-10-28_22-35-24_avf_0.106696.pkl'
    #     # optimization_problem = ig.import_geometry(file_name)
    #     # Lx = optimization_problem.box_length # um
    #     # Ly = optimization_problem.box_length # um
    #     # Lz = optimization_problem.box_length # um

    #     optimized_fibers, L = ig.import_array_geometry_full_path(file_path) # optimized_fibers = xyz_r_fid
    #     ###### only exp------------------------
    #     # optimized_fibers[:, 0:4], L = optimized_fibers[:, 0:4]*2.0, L*2.0

    #     print('optimized_fibers', optimized_fibers.shape)
    #     print('L', L)
    #     fiberlist_xyz_r_fid = ag.split_matrix_to_list(optimized_fibers, L)
    #     print('L box length',L)
    #     D = 3.0 # um^2/ms
    #     T2 = 100 # ms
    #     rho = 1 # fractional water density
    #     sg3 = geom.SimGeometry3D(L,L,L,D,T2,rho)
    #     # fiber_xyzr_fid_list = ag.createBoundaryZ(fiberlist_xyz_r_fid, L)
    #     # print('len(fiber_xyzr_fid_list)',len(fiber_xyzr_fid_list))
        
    #     # for fiber in fiber_xyzr_fid_list:
    #     #     sx, sy, sz, sr = fiber[:,0], fiber[:,1], fiber[:,2], fiber[:,3]
    #     #     spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
    #     #     sg3.add_structure(spstruc) # add it to sg3
    #     # dt = 0.1 # time step in ms
    #     # nt = 200 # total number of steps thru time

    #     # #higher number of spins easier to come out..!
    #     # spins = int(5e3)
    #     # num_spins = spins
    #     # print('Start setting up structures')
    #     # sim = ds3.DiffSim3d(sg3,spins)
    #     # sim.setup(structures=list(np.arange(0, len(fiber_xyzr_fid_list))))  # seed inside structures 0:len(fiberlist)
    #     # print('Done setting up structures')

    #     # # first, check thath all the spins are in fact in the multiple spstruc's
    #     # isInside = False
    #     # for structure in sg3.structures[:-1]:
    #     #     isInside = np.logical_or(isInside, structure.isinside(sim.spins))
    #     # self.assertTrue(isInside.all())
        
    #     # Dx_step = np.zeros(nt+1)
    #     # Dy_step = np.zeros(nt+1)
    #     # Dz_step = np.zeros(nt+1)
    #     # diff_time = np.zeros(nt+1)
    #     # Dx_step[0], Dy_step[0], Dz_step[0] = D, D, D
    #     # print('Got to sim loops')
    #     # for n in range(1, nt+1):
    #     #     print(n)
    #     #     isInside = False
    #     #     for struct in sg3.structures[:-1]:
    #     #         isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
    #     #     # self.assertTrue(isInside.all())

    #     #     if ~isInside.all():
    #     #         print(n)
    #     #         spins_temp = sim.spins_d.get()
    #     #         ig.plot_fibers(spins_temp, fiber_xyzr_fid_list, L, n, file_path)
            
    #     #     # --------step-----------
    #     #     sim.step(dt)
    #     #     Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
    #     #     Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
    #     #     Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
    #     #     diff_time[n] = n*dt
                
    #     # msdx = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins
    #     # msdy = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins
    #     # msdz = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins
    #     # #print(msdx/2/dt/nt,msdy/2/dt/nt,msdz/2/dt/nt)

    #     # Diffusion in the x direction should be close to D
    #     # self.assertGreater(msdx/2/dt/nt,D-0.2)
    #     # self.assertLess(msdx/2/dt/nt,D+0.2)
    #     # Diffusion in the y&z direction shoudl be close to 0 due to restriction
    #     # self.assertLess(msdy/2/dt/nt,0.01)
    #     # self.assertLess(msdz/2/dt/nt,0.01)

    #     # et = time.time()
    #     # # get the execution time
    #     # elapsed_time = et - st
    #     # # save_data_pickle(file_name, np.array([Dx_step, Dy_step, Dz_step, diff_time]))
    #     # import os
    #     # file_name = os.path.basename(file_path)
    #     # base_name, extension = os.path.splitext(file_name)
    #     # print(base_name)
    #     # file_name_new = 'diffsim3d_'+ str(len(fiber_xyzr_fid_list)) + '_fibers_' + str(num_spins)+'_spins_'+ base_name +'_'+str(round(elapsed_time,2))+'_sec'
    #     # plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, file_name_new + '_' + str(dt)+'_step_')    

    #     # # after simulating diffusion, are all spins still inside sg3?
    #     # isInside = False
    #     # for struct in sg3.structures[:-1]:
    #     #     isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get()))
    #     # self.assertTrue(isInside.all())

    #     print('optimized_fibers', type(optimized_fibers))
    #     fiber_list = ag.map_matrix_to_list_numpy(optimized_fibers)
    #     color = cm.rainbow(np.linspace(0.0, 1.0, len(fiber_list)))
    #     np.random.shuffle(color)
    #     ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, animation_input=True, optimized=True) 
    #     ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_90') 
    #     ag.plot_opt_fibers(optimized_fibers, L, overlap_indices=None, color=color, optimized=True, POV='horizontal_0')                    
    #     # ag.plot_radius_distribution()
    #     # ag.plot_diameter_distribution()
    #     ag.plot_slices(L, N=5, color=color, spheres_xyz_r_fid=optimized_fibers)


    #     signal from T2 decay
    #     self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)
    #     ####################################################################################
    #     save data with Pickle format
       
    #     save_data_pickle(file_name, np.array([Dx_step, Dy_step, Dz_step, diff_time]))

    # def testInsideMultiFiberGeometry(self):
    #     '''
    #     diffusion inside fibers
    #     '''
    #     st = time.time()
    #     # file_name = '20_500000_2023-11-08_00-47-52_avf_0.246_15182_sec.pkl' #12min
    #     file_name = '6_500000_2023-11-28_16-06-53_avf_0.0632_K200_1462_sec.pkl'
    #     # file_name = '43_500000_2023-11-01_11-26-53_avf_0.626.pkl'
    #     # file_name='35_500000_2023-10-26_13-48-40_avf_0.519386_5hr.pkl'
    #     # file_name='10_50000_2023-10-28_22-35-24_avf_0.106696.pkl'
    #     # file_name='5_1000_2023-11-08_00-23-06_avf_0.402_92_sec.pkl'
    #     # file_name='4_1000_2023-11-08_00-32-22_avf_0.376_118_sec.pkl'
    #     optimization_problem = ig.import_geometry(file_name)
    #     Lx = optimization_problem.box_length # um
    #     Ly = optimization_problem.box_length # um
    #     Lz = optimization_problem.box_length # um
    #     print('Lx',Lx)
    #     print('Ly',Ly)
    #     print('Lz',Lz)
    #     D = 3.0 # um^2/ms
    #     T2 = 100 # ms
    #     rho = 1 # fractional water density
    #     sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

    #     # make a single sphere and put it in the middle of the geometry
    #     sx = np.array([0,])
    #     sy = np.array([0,])
    #     sz = np.array([0,])
    #     sr = np.array([3,])
    #     spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
    #     sg3.add_structure(spstruc) # add it to sg3

    #     dt = 0.001 # time step in ms
    #     nt = 100000 # total number of steps thru time

    #     spins = 10000

    #     sim = ds3.DiffSim3d(sg3,spins)
    #     sim.setup(structures=[0])

    #     # first, check thath all the spins are in fact in spstruc
    #     self.assertTrue(spstruc.isinside(sim.spins).all())

    #     for n in range(nt):
    #         sim.step(dt)

    #     # after simulating diffusion, are all spins still inside spstruc?
    #     self.assertTrue(spstruc.isinside(sim.spins_d.get()).all())

    #     # signal from T2 decay
    #     self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)

# if __name__ == '__main__':
#     unittest.main()

