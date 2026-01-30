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
#         Lx = 60.0 # um
#         Ly = 60.0 # um
#         Lz = 60.0 # um
#         D = 3.0 # um^2/ms
#         T2 = 200 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         nsphere = 200
#         sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         sy = np.cos(2*np.pi*sz/Ly/2)*10
#         sx = np.sin(2*np.pi*sz/Ly/2)*10
#         sr = np.full(sz.shape, 2.)
        
#         nsphere = 200 
#         sz2 = np.linspace(-Lx/2, Lx/2, nsphere)
#         sy2 = -np.cos(2*np.pi*sz2/Ly/2)*10
#         sx2 = -np.sin(2*np.pi*sz/Ly/2)*10
#         sr2 = np.full(sz2.shape, 2.)

#         sy[-1]=sy2[0]
#         sy[0]=sy2[-1]
#         swpz, swpy,swpx, swpr = sz2,sy2,sx2,sr2
#         sz2 = np.concatenate((sz[-90:]-Lz, sz2, sz[:90]+Lz ), axis=0)
#         sy2 = np.concatenate((sy[-90:], sy2, sy[:90] ), axis=0)
#         sx2 = np.concatenate((sx[-90:], sx2, sx[:90] ), axis=0)
#         sr2 = np.concatenate((sr[-90:], sr2, sr[:90] ), axis=0)
        
#         sy = np.concatenate((swpy[-90:], sy, swpy[:90] ), axis=0)
#         sx = np.concatenate((swpx[-90:], sx, swpx[:90] ), axis=0)
#         sz = np.concatenate((swpz[-90:]-Lz, sz, swpz[:90]+Lz ), axis=0)
#         sr = np.concatenate((swpr[-90:], sr, swpr[:90] ), axis=0)

#         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
#         fib1 = np.column_stack((sx,sy,sz,sr))
#         spstruc = geom.Structure3D(sx2,sy2,sz2,sr2,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
#         fib2 = np.column_stack((sx2,sy2,sz2,sr2))
#         fib_list = [fib1, fib2]
#         color = ['red', 'blue']
#         dt = 0.01 # time step in ms
#         nt = 3000 # total number of steps thru time
#         # nt = 100 # total number of steps thru time
#         spins = int(5e3)
#         num_spins=spins
#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=[0,1])
        
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
#                 # ax.view_init(azim=0, elev=0)
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
#                 spin_plot_file_name = str(date_time)+'_'+'SINE_outside_at'+str(n)+'.png'
#                 spin_plot_file_name = os.path.join(spin_folder_name, spin_plot_file_name)
#                 pl.savefig(spin_plot_file_name)
#             # ---------- step -----------
#             sim.step(dt)
#             Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
#             Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
#             Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
#             diff_time[n] = n*dt
#         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, spin_folder_name , get_date_time())

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
#         Lx = 60.0 # um
#         Ly = 60.0 # um
#         Lz = 60.0 # um
#         D = 3.0 # um^2/ms
#         T2 = 200 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         nsphere = 200
#         sz = np.linspace(-Lx/2, Lx/2, nsphere)
#         sy = np.cos(2*np.pi*sz/Ly/2)*10
#         sx = np.linspace(-Lz/3, Lz/3, nsphere)
#         sr = np.full(sz.shape, 2.)
        
#         nsphere = 200 
#         sz2 = np.linspace(-Lx/2, Lx/2, nsphere)
#         sy2 = -np.cos(2*np.pi*sz2/Ly/2)*10
#         sx2 = np.linspace(Lz/3, -Lz/3, nsphere)
#         sr2 = np.full(sz2.shape, 2.)

#         swpz, swpy,swpx, swpr = sz2,sy2,sx2,sr2
#         sz2 = np.concatenate((sz[-90:]-Lz, sz2, sz[:90]+Lz ), axis=0)
#         sy2 = np.concatenate((sy[-90:], sy2, sy[:90] ), axis=0)
#         sx2 = np.concatenate((sx[-90:], sx2, sx[:90] ), axis=0)
#         sr2 = np.concatenate((sr[-90:], sr2, sr[:90] ), axis=0)
        
#         sy = np.concatenate((swpy[-90:], sy, swpy[:90] ), axis=0)
#         sx = np.concatenate((swpx[-90:], sx, swpx[:90] ), axis=0)
#         sz = np.concatenate((swpz[-90:]-Lz, sz, swpz[:90]+Lz ), axis=0)
#         sr = np.concatenate((swpr[-90:], sr, swpr[:90] ), axis=0)

#         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
#         fib1 = np.column_stack((sx,sy,sz,sr))
#         spstruc = geom.Structure3D(sx2,sy2,sz2,sr2,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3
#         fib2 = np.column_stack((sx2,sy2,sz2,sr2))
        
#         #================================================
#         dt = 0.01 # time step in ms
#         nt = 5000 # total number of steps thru time
#         # nt = 100 # total number of steps thru time
#         spins = int(5e3)
#         num_spins=spins
#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=[2])
#         # first, check thath all the spins are in fact in spstruc
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         print('num outside', np.sum(isOutside))
#         self.assertTrue(isOutside.all())

#         fib_list = [fib1, fib2]
#         color=['red', 'blue']
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
#             print(n)
#             isOutside = False
#             for struct in sg3.structures[:-1]:
#                 isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#             isOutside = np.invert(isOutside)
            
#             if ~isOutside.all() or n==1 or n==99 or n==999 or n==9999 or n==19999 or n==49999 or n==99999:
#                 print('stopped at', n)
#                 fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#                 # ax.view_init(azim=90, elev=0)
#                 k=0
#                 for fib in fib_list:
#                     plot_spheres(ax=ax, node_list=fib, color=color[k])
#                     k+=1
#                 spins_temp = sim.spins_d.get()
#                 spins_temp = ((spins_temp.T)[np.invert(isOutside)]).T
#                 f.write(str(spins_temp.T))
#                 f.write("\n")
#                 ax.scatter(spins_temp[0,:].T,spins_temp[1,:].T,spins_temp[2,:].T,alpha=1., marker='o')
#                 ax.set_xlim(-Lx/2, Lx/2)
#                 ax.set_ylim(-Ly/2, Ly/2)
#                 ax.set_zlim(-Lz/2, Lz/2)
#                 ax.set_xlabel("x (um)")
#                 ax.set_ylabel("y (um)")
#                 ax.set_zlabel("z (um)")
#                 spin_plot_file_name = str(date_time)+'_'+'SINE_inside_at'+str(n)+'.png'
#                 spin_plot_file_name = os.path.join(spin_folder_name, spin_plot_file_name)
#                 pl.savefig(spin_plot_file_name)
#                 plt.close(fig)
#                 # self.assertTrue(isOutside.all())
#                 if n>=2000 and ~isOutside.all() : 
#                     f.close()
#                     exit()

#             # ---------- step -----------
#             sim.step(dt)
#             Dx_step[n] = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins/2/(n*dt)
#             Dy_step[n] = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins/2/(n*dt)
#             Dz_step[n] = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins/2/(n*dt)
#             diff_time[n] = n*dt
#         plot_ADC_vs_time(diff_time, Dx_step, Dy_step, Dz_step, spin_folder_name , get_date_time())
#         f.close()
#         #-------------------------------------------
#         isOutside = False
#         for struct in sg3.structures[:-1]:
#             isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
#         isOutside = np.invert(isOutside)
#         print('num outside', np.sum(isOutside))
#         self.assertTrue(isOutside.all())
    
# if __name__ == '__main__':
#     unittest.main()