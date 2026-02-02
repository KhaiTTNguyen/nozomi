# import pycuda.autoinit
# import pycuda.driver as drv
# import pycuda.gpuarray as gpuarray
# import numpy as np
# import unittest
# import diffsim3d.diffsim3d as ds3
# import diffsim3d.geometry as geom
# import matplotlib.pyplot as pl
# from geometrygen.util import *
# import os
# import time

# class TestDiffSim3D(unittest.TestCase):

#     def setUp(self):
#         drv.Device(0).make_context()
    
#     def tearDown(self):
#         drv.Context.pop()
        
#     def testOutsideMultipleSpheres(self):
#         '''
#         diffusion inside of a multiple spheres
#         '''
#         st = time.time()
#         date_time=get_date_time()
#         Lx = 10.0 # um
#         Ly = 10.0 # um
#         Lz = 10.0 # um
#         D = 3.0 # um^2/ms
#         T2 = 200 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)
        
#         undu_amp = 3
#         undu_freq = 1
#         nsphere = 200
#         sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         sy = np.cos(undu_freq*np.pi*sz/Ly) * undu_amp
#         sx = np.full(sz.shape, 0)
#         sr = np.full(sz.shape, 0.5)
#         sy[-1]=sy[0]
#         sx[-1]=sx[0]
#         sz2 = np.concatenate((sz[-150:-1] - Lz, sz, sz[1:150] + Lz), axis=0)
#         sy2 = np.concatenate((sy[-150:-1], sy, sy[1:150]), axis=0)
#         sx2 = np.concatenate((sx[-150:-1], sx, sx[1:150]), axis=0)
#         sr2 = np.concatenate((sr[-150:-1], sr, sr[1:150]), axis=0)

#         spstruc = geom.Structure3D(sx2,sy2,sz2,sr2,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
#         fib1 = np.column_stack((sx2,sy2,sz2,sr2))
#         fib_list = [fib1]
#         color = ['blue']

#         dt = 0.002 # time step in ms
#         nt = int(5e6) # total number of steps thru time
#         spins = int(1e4)

#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.set_segments(nsegx=20,nsegy=20,nsegz=20)
#         sim.setup(structures=[1])
#         # CREATE DIRECTORY
#         spin_folder_name = os.path.join('.','diffsim3d','animation','debug_spin', 'ADC', 'extra_axonal_JAN15', str(date_time)+'_R'+str(sr[-1])+'_undu_freq'+str(undu_freq)+'_undu_amp'+str(undu_amp)+'_dt_'+str(dt)+'_Lz_'+str(Lz))
#         if not os.path.exists(spin_folder_name):
#             os.makedirs(spin_folder_name)
#             print("Folder created successfully.")
#         else:
#             print("Folder already exists.")

#         # check if all spins are outside axons
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         self.assertTrue(isOutside.all())

#         # sim
#         Dx_step = np.zeros(nt+1)
#         Dy_step = np.zeros(nt+1)
#         Dz_step = np.zeros(nt+1)
#         diff_time = np.zeros(nt+1)
#         Dx_step[0], Dy_step[0], Dz_step[0] = D, D, D
#         for n in range(1, nt+1):
#             sim.step(dt)
#             Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
#             Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
#             Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
#             diff_time[n] = n*dt
#         #-------------------------------------------
#         # check if spins are still outside 
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         self.assertTrue(isOutside.all())
#         et = time.time()
#         elapsed_time = np.round(et - st,2)
        
#         # ------------------- plotting -------------------   
#         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, spin_folder_name , str(date_time)+'_R'+str(sr[-1])+'_undu_freq'+str(undu_freq)+'_undu_amp'+str(undu_amp)+'_dt_'+str(dt)+'_Lz_'+str(Lz)+'_'+str(elapsed_time)+'sec')
        
#         fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#         # ax.view_init(azim=0, elev=0)
#         k=0
#         for fib in fib_list:
#             plot_spheres(ax=ax, node_list=fib, color=color[k])
#             k+=1
#         ax.set_xlim(-Lx/2, Lx/2)
#         ax.set_ylim(-Ly/2, Ly/2)
#         ax.set_zlim(-Lz/2, Lz/2)
#         ax.set_xlabel("x (um)")
#         ax.set_ylabel("y (um)")
#         ax.set_zlabel("z (um)")
#         spin_plot_file_name = str(date_time)+'_'+'SINGLE_fiber.png'
#         spin_plot_file_name = os.path.join(spin_folder_name, spin_plot_file_name)
#         pl.savefig(spin_plot_file_name)
        
#     def testOutsideTwoSetOfMultipleSpheresSINE(self):
#         '''
#         diffusion inside of multiple spheres
#         '''
#         st = time.time()
#         date_time=get_date_time()
#         Lx = 10.0 # um
#         Ly = 10.0 # um
#         Lz = 10.0 # um
#         D = 3.0 # um^2/ms
#         T2 = 200 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         undu_amp = 3
#         undu_freq = 1
#         nsphere = 200
#         sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         sy = np.cos(undu_freq*np.pi*sz/Ly) * undu_amp
#         sx = np.linspace(-Lx/2, Lx/2, nsphere)/5
#         sr = np.full(sz.shape, 0.5)
#         nsphere = 200 
#         sz2 = np.linspace(-Lx/2, Lx/2, nsphere)
#         sy2 = -np.cos(undu_freq*np.pi*sz/Ly) * undu_amp #-np.cos(2*np.pi*sz2/Ly/2)*2 
#         sx2 = -np.linspace(-Lx/2, Lx/2, nsphere)/5
#         sr2 = np.full(sz2.shape, 0.5)
#         sy[-1]=sy2[0]
#         sx[-1]=sx2[0]
#         sz[-1]=Lz/2
#         sy2[-1]=sy[0]
#         sx2[-1]=sx[0]
#         sz2[-1]=Lz/2
#         swpz, swpy,swpx, swpr = sz2,sy2,sx2,sr2
#         sz22 = np.concatenate((sz[-150:]-Lz, sz2, sz[:150]+Lz ), axis=0)
#         sy22 = np.concatenate((sy[-150:], sy2, sy[:150] ), axis=0)
#         sx22 = np.concatenate((sx[-150:], sx2, sx[:150] ), axis=0)
#         sr22 = np.concatenate((sr[-150:], sr2, sr[:150] ), axis=0)
        
#         sy11 = np.concatenate((swpy[-150:], sy, swpy[:150] ), axis=0)
#         sx11 = np.concatenate((swpx[-150:], sx, swpx[:150] ), axis=0)
#         sz11 = np.concatenate((swpz[-150:]-Lz, sz, swpz[:150]+Lz ), axis=0)
#         sr11 = np.concatenate((swpr[-150:], sr, swpr[:150] ), axis=0)

#         spstruc = geom.Structure3D(sx11,sy11,sz11,sr11,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
#         fib1 = np.column_stack((sx11,sy11,sz11,sr11))
#         # fib1 = np.column_stack((sx,sy,sz,sr))
#         spstruc = geom.Structure3D(sx22,sy22,sz22,sr22,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
#         fib2 = np.column_stack((sx22,sy22,sz22,sr22))
#         # fib2 = np.column_stack((sx2,sy2,sz2,sr2))
        
#         fib_list = [fib1, fib2]
#         color = ['red', 'blue']
        
#         dt = 0.002 # time step in ms
#         nt = int(5e6) # total number of steps thru time
#         spins = int(1e4)
        
#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.set_segments(nsegx=20,nsegy=20,nsegz=20)
#         sim.setup(structures=[2])

#         # CREATE DIRECTORY
#         spin_folder_name = os.path.join('.','diffsim3d','animation','debug_spin', 'ADC', 'extra_axonal_JAN15', str(date_time)+'_R'+str(sr[-1])+'_undu_freq'+str(undu_freq)+'_undu_amp'+str(undu_amp)+'_dt_'+str(dt)+'_Lz_'+str(Lz))
#         if not os.path.exists(spin_folder_name):
#             os.makedirs(spin_folder_name)
#             print("Folder created successfully.")
#         else:
#             print("Folder already exists.")

#         # first, check thath all the spins are in fact all outside axons
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         self.assertTrue(isOutside.all())

#         # sim
#         Dx_step = np.zeros(nt+1)
#         Dy_step = np.zeros(nt+1)
#         Dz_step = np.zeros(nt+1)
#         diff_time = np.zeros(nt+1)
#         Dx_step[0], Dy_step[0], Dz_step[0] = D, D, D
#         for n in range(1, nt+1):
#             sim.step(dt)
#             Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
#             Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
#             Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
#             diff_time[n] = n*dt

#         # check if spins still outside axons
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         self.assertTrue(isOutside.all())
#         et = time.time()
#         elapsed_time = np.round(et - st,2)
#         #--------------------- plotting ----------------------
#         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, spin_folder_name , str(date_time)+'_R'+str(sr[-1])+'_undu_freq'+str(undu_freq)+'_undu_amp'+str(undu_amp)+'_dt_'+str(dt)+'_Lz_'+str(Lz)+'_'+str(elapsed_time)+'sec')

#         fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#         # ax.view_init(azim=0, elev=0)
#         k=0
#         for fib in fib_list:
#             plot_spheres(ax=ax, node_list=fib, color=color[k])
#             k+=1
#         ax.set_xlim(-Lx/2, Lx/2)
#         ax.set_ylim(-Ly/2, Ly/2)
#         ax.set_zlim(-Lz/2, Lz/2)
#         ax.set_xlabel("x (um)")
#         ax.set_ylabel("y (um)")
#         ax.set_zlabel("z (um)")
#         spin_plot_file_name = str(date_time)+'_'+'SINE_fibers.png'
#         spin_plot_file_name = os.path.join(spin_folder_name, spin_plot_file_name)
#         pl.savefig(spin_plot_file_name)

# if __name__ == '__main__':
#     unittest.main()