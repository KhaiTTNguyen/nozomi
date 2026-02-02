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

#     def testInsideTwoSetOfMultipleSpheres_LargeRadius(self):
#         '''
#         diffusion inside of multiple spheres
#         '''
#         Lx = 40 # um
#         Ly = 40 # um
#         Lz = 40 # um
#         D = 3.0 # um^2/ms
#         T2 = 200 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         nsphere = 40
#         sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         # sy = np.sin(2*np.pi*sz/Ly*3)*3 +10
#         sy = np.sin(2*np.pi*sz/Ly*3)*3
#         sx = np.full(sz.shape, 0)
#         sr = np.full(sz.shape, 5.)
#         sy[-1]=sy[0]
#         sx[-1]=sx[0]
#         # swpz, swpy,swpx, swpr = sz,sy,sx,sr
#         szz = np.concatenate((sz[-10:-1] - Lz, sz, sz[1:10] + Lz), axis=0)
#         syz = np.concatenate((sy[-10:-1], sy, sy[1:10]), axis=0)
#         sxz = np.concatenate((sx[-10:-1], sx, sx[1:10]), axis=0)
#         srz = np.concatenate((sr[-10:-1], sr, sr[1:10]), axis=0)
#         spstruc = geom.Structure3D(sxz,syz,szz,srz,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
#         fib1 = np.column_stack((sxz,syz,szz,srz))

#         # nsphere = 40
#         # sz11 = np.linspace(-Lx/2, Lx/2, nsphere)
#         # # sy = np.sin(2*np.pi*sz/Ly*3)*3 +10
#         # sy11 = np.full(sz11.shape, 20.5)
#         # sx11 = np.sin(2*np.pi*sz11/Ly*3)*3 -20.5
#         # sr11 = np.full(sz11.shape, 5.)
#         # swpz, swpy,swpx, swpr = sz11,sy11,sx11,sr11
#         # sy11 = np.concatenate((swpy[-90:], sy11, swpy[:90] ), axis=0)
#         # sx11 = np.concatenate((swpx[-90:], sx11, swpx[:90] ), axis=0)
#         # sz11 = np.concatenate((swpz[-90:]-Lz, sz11, swpz[:90]+Lz ), axis=0)
#         # sr11 = np.concatenate((swpr[-90:], sr11, swpr[:90] ), axis=0)
#         # spstruc = geom.Structure3D(sx11,sy11,sz11,sr11,D,T2,rho)
#         # sg3.add_structure(spstruc) # add it to sg3
#         # fib11 = np.column_stack((sx11,sy11,sz11,sr11))

#         # nsphere = 40
#         # sz111 = np.linspace(-Lx/2, Lx/2, nsphere)
#         # # sy = np.sin(2*np.pi*sz/Ly*3)*3 +10
#         # sy111 = np.full(sz111.shape, 20.5)
#         # sx111 = np.sin(2*np.pi*sz111/Ly*3)*3 + 20.5
#         # sr111 = np.full(sz111.shape, 5.)
#         # swpz, swpy,swpx, swpr = sz111,sy111,sx111,sr111
#         # sy111 = np.concatenate((swpy[-90:], sy111, swpy[:90] ), axis=0)
#         # sx111 = np.concatenate((swpx[-90:], sx111, swpx[:90] ), axis=0)
#         # sz111 = np.concatenate((swpz[-90:]-Lz, sz111, swpz[:90]+Lz ), axis=0)
#         # sr111 = np.concatenate((swpr[-90:], sr111, swpr[:90] ), axis=0)
#         # spstruc = geom.Structure3D(sx111,sy111,sz111,sr111,D,T2,rho)
#         # sg3.add_structure(spstruc) # add it to sg3
#         # fib111 = np.column_stack((sx111,sy111,sz111,sr111))

#         # nsphere = 40 
#         # sz2 = np.linspace(-Lx/2, Lx/2, nsphere)
#         # sy2 = np.full(sz2.shape, 0)
#         # sx2 = np.sin(2*np.pi*sz2/Ly*3)*3
#         # sr2 = np.full(sz2.shape, 5.)
#         # swpz, swpy,swpx, swpr = sz2,sy2,sx2,sr2
#         # sz2 = np.concatenate((swpz[-90:]-Lz, sz2, swpz[:90]+Lz ), axis=0)
#         # sy2 = np.concatenate((swpy[-90:], sy2, swpy[:90] ), axis=0)
#         # sx2 = np.concatenate((swpx[-90:], sx2, swpx[:90] ), axis=0)
#         # sr2 = np.concatenate((swpr[-90:], sr2, swpr[:90] ), axis=0)
#         # spstruc = geom.Structure3D(sx2,sy2,sz2,sr2,D,T2,rho)
#         # sg3.add_structure(spstruc) # add it to sg3
#         # fib2 = np.column_stack((sx2,sy2,sz2,sr2))

#         # nsphere = 40 
#         # sz3 = np.linspace(-Lx/2, Lx/2, nsphere)
#         # sx3 = np.full(sz3.shape, 20.5)
#         # sy3 = np.sin(2*np.pi*sz3/Ly*3)*3
#         # sr3 = np.full(sz3.shape, 5.)
#         # swpz, swpy,swpx, swpr = sz3,sy3,sx3,sr3
#         # sz3 = np.concatenate((swpz[-90:]-Lz, sz3, swpz[:90]+Lz ), axis=0)
#         # sy3 = np.concatenate((swpy[-90:], sy3, swpy[:90] ), axis=0)
#         # sx3 = np.concatenate((swpx[-90:], sx3, swpx[:90] ), axis=0)
#         # sr3 = np.concatenate((swpr[-90:], sr3, swpr[:90] ), axis=0)
#         # spstruc = geom.Structure3D(sx3,sy3,sz3,sr3,D,T2,rho)
#         # sg3.add_structure(spstruc) # add it to sg3
#         # fib3 = np.column_stack((sx3,sy3,sz3,sr3))

#         # nsphere = 40 
#         # sz33 = np.linspace(-Lx/2, Lx/2, nsphere)
#         # sx33 = np.full(sz33.shape, 20.5)
#         # sy33 = np.sin(2*np.pi*sz33/Ly*3)*3 -20.5
#         # sr33 = np.full(sz33.shape, 5.)
#         # swpz, swpy,swpx, swpr = sz33,sy33,sx33,sr33
#         # sz33 = np.concatenate((swpz[-90:]-Lz, sz33, swpz[:90]+Lz ), axis=0)
#         # sy33 = np.concatenate((swpy[-90:], sy33, swpy[:90] ), axis=0)
#         # sx33 = np.concatenate((swpx[-90:], sx33, swpx[:90] ), axis=0)
#         # sr33 = np.concatenate((swpr[-90:], sr33, swpr[:90] ), axis=0)
#         # spstruc = geom.Structure3D(sx33,sy33,sz33,sr33,D,T2,rho)
#         # sg3.add_structure(spstruc) # add it to sg3
#         # fib33 = np.column_stack((sx33,sy33,sz33,sr33))

#         # nsphere = 40 
#         # sz333 = np.linspace(-Lx/2, Lx/2, nsphere)
#         # sx333 = np.full(sz333.shape, -20.5)
#         # sy333 = np.sin(2*np.pi*sz333/Ly*3)*3 -20.5
#         # sr333 = np.full(sz333.shape, 5.)
#         # swpz, swpy,swpx, swpr = sz333,sy333,sx333,sr333
#         # sz333 = np.concatenate((swpz[-90:]-Lz, sz333, swpz[:90]+Lz ), axis=0)
#         # sy333 = np.concatenate((swpy[-90:], sy333, swpy[:90] ), axis=0)
#         # sx333 = np.concatenate((swpx[-90:], sx333, swpx[:90] ), axis=0)
#         # sr333 = np.concatenate((swpr[-90:], sr333, swpr[:90] ), axis=0)
#         # spstruc = geom.Structure3D(sx333,sy333,sz333,sr333,D,T2,rho)
#         # sg3.add_structure(spstruc) # add it to sg3
#         # fib333 = np.column_stack((sx333,sy333,sz333,sr333))

#         # nsphere = 40 
#         # sz4 = np.linspace(-Lx/2, Lx/2, nsphere)
#         # sx4 = np.sin(2*np.pi*sz4/Ly*3)*3-20.5
#         # sy4 = np.full(sz4.shape, 0)
#         # sr4 = np.full(sz4.shape, 5.)
#         # swpz, swpy,swpx, swpr = sz4,sy4,sx4,sr4
#         # sz4 = np.concatenate((swpz[-90:]-Lz, sz4, swpz[:90]+Lz ), axis=0)
#         # sy4 = np.concatenate((swpy[-90:], sy4, swpy[:90] ), axis=0)
#         # sx4 = np.concatenate((swpx[-90:], sx4, swpx[:90] ), axis=0)
#         # sr4 = np.concatenate((swpr[-90:], sr4, swpr[:90] ), axis=0)
#         # spstruc = geom.Structure3D(sx4,sy4,sz4,sr4,D,T2,rho)
#         # sg3.add_structure(spstruc) # add it to sg4
#         # fib4 = np.column_stack((sx4,sy4,sz4,sr4))

#         # nsphere = 40 
#         # sz5 = np.linspace(-Lx/2, Lx/2, nsphere)
#         # sy5 = np.sin(2*np.pi*sz5/Ly*3)*3 -20.5
#         # sx5 = np.full(sz5.shape, 0)
#         # sr5 = np.full(sz5.shape, 5.)
#         # swpz, swpy,swpx, swpr = sz5,sy5,sx5,sr5
#         # sz5 = np.concatenate((swpz[-90:]-Lz, sz5, swpz[:90]+Lz ), axis=0)
#         # sy5 = np.concatenate((swpy[-90:], sy5, swpy[:90] ), axis=0)
#         # sx5 = np.concatenate((swpx[-90:], sx5, swpx[:90] ), axis=0)
#         # sr5 = np.concatenate((swpr[-90:], sr5, swpr[:90] ), axis=0)
#         # spstruc = geom.Structure3D(sx5,sy5,sz5,sr5,D,T2,rho)
#         # sg3.add_structure(spstruc) # add it to sg5
#         # fib5 = np.column_stack((sx5,sy5,sz5,sr5))

#         fib_list = [fib1] #, fib11, fib111, fib2, fib3, fib33, fib333, fib4, fib5]
#         color = ['red'] #, 'pink', 'blue', 'purple', 'orange', 'green', 'brown', 'olive', 'cyan']
#         dt = 0.005 # time step in ms
#         nt = 10000 # total number of steps thru time
#         # nt = 100 # total number of steps thru time
#         #spins = int(1e5)
#         spins = int(5e3)
#         num_spins=spins
#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=[0])
        
#         # sim.setup(structures=[0,1,2,3,4])
        
#         # first, check thath all the spins are in fact in spstruc
#         isInside = False
#         for struct in sg3.structures[:-1]:
#             isInside = np.logical_or(isInside, struct.isinside(sim.spins)) # str.isinside(sim.spins) return vector length num_spins & OR with isInside
#         self.assertTrue(isInside.all())
        
#         Dx_step = np.zeros(nt+1)
#         Dy_step = np.zeros(nt+1)
#         Dz_step = np.zeros(nt+1)
#         diff_time = np.zeros(nt+1)
#         Dx_step[0], Dy_step[0], Dz_step[0] = D, D, D
#         date_time=get_date_time()
#         spin_folder_name = os.path.join('.','diffsim3d','animation','debug_spin', 'ADC', 'intra_axonal', str(date_time))
#         if not os.path.exists(spin_folder_name):
#             os.makedirs(spin_folder_name)
#             print("Folder created successfully.")
#         else:
#             print("Folder already exists.")   
#         for n in range(1, nt+1):
#             print(n)
#             isInside = False
#             for struct in sg3.structures[:-1]:
#                 isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get())) # str.isinside(sim.spins) return vector length num_spins & OR with isInside
#             # self.assertTrue(isInside.all())
#             if n==1 or n==999 or n==9999 or n==19999 or n==49999 or n==99999 or ~isInside.all():
#                 print('intra loop stopped at ', n)
#                 fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#                 ax.view_init(azim=0, elev=90)
#                 spins_temp = ((sim.spins_d.get().T)[np.invert(isInside)]).T
#                 ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
#                 k=0
#                 for fib in fib_list:
#                     plot_spheres(ax=ax, node_list=fib, color=color[k])
#                     k+=1
#                 ax.set_xlim(-Lx/2, Lx/2)
#                 ax.set_ylim(-Ly/2, Ly/2)
#                 ax.set_zlim(-Lz/2, Lz/2)
#                 ax.set_xlabel("x (um)")
#                 ax.set_ylabel("y (um)")
#                 ax.set_zlabel("z (um)")
#                 spin_plot_file_name = str(date_time)+'_'+'outside_at'+str(n)+'.png'
#                 spin_plot_file_name = os.path.join(spin_folder_name, spin_plot_file_name)
#                 pl.savefig(spin_plot_file_name)
#                 pl.close(fig)
#             # ---------- step -----------
#             sim.step(dt)
#             Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
#             Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
#             Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
#             diff_time[n] = n*dt
#         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, spin_folder_name , get_date_time()+'_dt_'+str(dt))

#         #-------------------------------------------
#         # after simulating diffusion, are all spins still inside sg3?
#         isInside = False
#         for struct in sg3.structures[:-1]:
#             print('isInside pre', isInside)
#             isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get())) # str.isinside(sim.spins) return vector length num_spins & OR with isInside
#             print('isInside post', isInside)
#         self.assertTrue(isInside.all())

#     def testOutsideTwoSetOfMultipleSpheres(self):
#         '''
#         diffusion inside of multiple spheres
#         '''
#         Lx = 40 # um
#         Ly = 40 # um
#         Lz = 40 # um
#         D = 3.0 # um^2/ms
#         T2 = 200 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         nsphere = 100
#         undu = 4
#         # sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         # # sy = np.sin(2*np.pi*sz/Ly*3)*3 +10
#         # sy = np.sin(2*np.pi*sz/Ly*3)*3
#         sz = np.arange(-nsphere / 2, nsphere / 2 + 1) * Lx / nsphere
#         sy = np.sin(20 * np.pi * np.arange(0, nsphere+1) / nsphere) * undu
#         sx = np.full(sz.shape, 0)
#         sr = np.full(sz.shape, 5.)
#         sy[-1]=sy[0]
#         sx[-1]=sx[0]
#         # swpz, swpy,swpx, swpr = sz,sy,sx,sr
#         szz = np.concatenate((sz[-10:-1] - Lz, sz, sz[1:10] + Lz), axis=0)
#         syz = np.concatenate((sy[-10:-1], sy, sy[1:10]), axis=0)
#         sxz = np.concatenate((sx[-10:-1], sx, sx[1:10]), axis=0)
#         srz = np.concatenate((sr[-10:-1], sr, sr[1:10]), axis=0)
#         spstruc = geom.Structure3D(sxz,syz,szz,srz,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
#         fib1 = np.column_stack((sxz,syz,szz,srz))

#         fib_list = [fib1] #, fib11, fib111, fib2, fib3, fib33, fib333, fib4, fib5]
#         color = ['red'] #, 'pink', 'blue', 'purple', 'orange', 'green', 'brown', 'olive', 'cyan']
#         dt = 0.005 # time step in ms
#         nt = 10000 # total number of steps thru time
#         #spins = int(1e5)
#         spins = int(5e3)
#         num_spins=spins
#         sim = ds3.DiffSim3d(sg3,spins)
#         # sim.setup(structures=[9])
#         sim.setup(structures=[1])
        
#         # first, check thath all the spins are in fact in spstruc
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         print('num outside', np.sum(isOutside))
#         self.assertTrue(isOutside.all())

#         Dx_step = np.zeros(nt+1)
#         Dy_step = np.zeros(nt+1)
#         Dz_step = np.zeros(nt+1)
#         diff_time = np.zeros(nt+1)
#         Dx_step[0], Dy_step[0], Dz_step[0] = D, D, D
#         date_time=get_date_time()
#         spin_folder_name = os.path.join('.','diffsim3d','animation','debug_spin', 'ADC', 'extra_axonal', str(date_time))
#         if not os.path.exists(spin_folder_name):
#             os.makedirs(spin_folder_name)
#             print("Folder created successfully.")
#         else:
#             print("Folder already exists.")   
#         spin_position_file_name = os.path.join(spin_folder_name, str(date_time)+'_'+'positions'+'.txt')
#         f = open(spin_position_file_name, "a")
#         for n in range(1, nt+1):
#             # print(n)
#             # isOutside = False
#             # for struct in sg3.structures[:-1]:
#             #     isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#             # isOutside = np.invert(isOutside)
            
#             # if ~isOutside.all() or n==1 or n==99 or n==999 or n==4999 or n==19999 or n==49999 or n==99999:
#             #     print('stopped at', n)
#             #     fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#             #     ax.view_init(azim=0, elev=00)
#             #     k=0
#             #     for fib in fib_list:
#             #         plot_spheres(ax=ax, node_list=fib, color=color[k])
#             #         k+=1
#             #     spins_temp = sim.spins_d.get()
#             #     spins_temp = ((spins_temp.T)[np.invert(isOutside)]).T
#             #     f.write(str(spins_temp.T))
#             #     f.write("\n")
#             #     ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
#             #     ax.set_xlim(-Lx/2, Lx/2)
#             #     ax.set_ylim(-Ly/2, Ly/2)
#             #     ax.set_zlim(-Lz/2, Lz/2)
#             #     ax.set_xlabel("x (um)")
#             #     ax.set_ylabel("y (um)")
#             #     ax.set_zlabel("z (um)")
#             #     spin_plot_file_name = str(date_time)+'_'+'INside_at'+str(n)+'.png'
#             #     spin_plot_file_name = os.path.join(spin_folder_name, spin_plot_file_name)
#             #     pl.savefig(spin_plot_file_name)
#             #     pl.close(fig)
#             #     # self.assertTrue(isOutside.all())
#             #     if n>=2000 and ~isOutside.all() : 
#             #         f.close()
#             #         exit()

#             # ---------- step -----------
#             sim.step(dt)
#             Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
#             Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
#             Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
#             diff_time[n] = n*dt
#         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, spin_folder_name , get_date_time()+'_dt_'+str(dt))
#         f.close()
#         #-------------------------------------------
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         print('num outside', np.sum(isOutside))
#         self.assertTrue(isOutside.all())
#         #============================END EXTRA AXONAL====================================
# if __name__ == '__main__':
#     unittest.main()