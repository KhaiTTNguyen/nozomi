# import pycuda.autoinit
# import pycuda.driver as drv
# import pycuda.gpuarray as gpuarray
# import numpy as np
# import unittest
# import diffsim3d_arXiv.diffsim3d as ds3
# import diffsim3d_arXiv.geometry as geom
# import matplotlib.pyplot as pl
# from geometrygen.util import *
# import os
# class TestDiffSim3D(unittest.TestCase):

#     def setUp(self):
#         drv.Device(0).make_context()
    
#     def tearDown(self):
#         drv.Context.pop()

#     def testEmptyGeometry(self):
#         '''
#         Simulate Gaussian diffusion in an empty simulation arena 
#         at the end of the simulation:
#         * all spins should still be in the geometry bounded by 0 and Lx,Ly,Lz
#         * the estimated diffusion coefficient should be approximately equal to D
#         '''
#         Lx = 10 # um
#         Ly = 10 # um
#         Lz = 10 # um
#         D = 3.0 # um^2/ms
#         T2 = 100 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         dt = 0.01 # time step in ms
#         nt = int(10000) # total number of steps thru time

#         spins = int(5000)

#         sim = ds3.DiffSim3d(sg3,spins)

#         for n in range(nt):
#             sim.step(dt)

#         # for all spins, -Lx/2 <= x <= Lx/2
#         self.assertTrue((-Lx/2 <= sim.spins_d[0,:].get()).all())
#         self.assertTrue((sim.spins_d[0,:].get() <= Lx/2).all())
#         #  -Ly/2 <= y <= Ly/2
#         self.assertTrue((-Ly/2 <= sim.spins_d[1,:].get()).all())
#         self.assertTrue((sim.spins_d[1,:].get() <= Ly/2).all())
#         #  -Lz/2 <= z <= Lz/2
#         self.assertTrue((-Lz/2 <= sim.spins_d[2,:].get()).all())
#         self.assertTrue((sim.spins_d[2,:].get() <= Lz/2).all())

#         # calculate the mean squared displacment and require it be close to 
#         # 2*D*dt*nt
#         msd = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get()/spins)
#         self.assertAlmostEqual(np.abs(msd/2/dt/nt),D,1)

#         # signal from T2 decay
#         self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),3)

#     def testInsideSingleSphere(self):
#         '''
#         diffusion inside of a single sphere
#         '''
#         Lx = 10 # um
#         Ly = 10 # um
#         Lz = 10 # um
#         D = 3.0 # um^2/ms
#         T2 = 100 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         # make a single sphere and put it in the middle of the geometry
#         sx = np.array([0,])
#         sy = np.array([0,])
#         sz = np.array([0,])
#         sr = np.array([3,])
#         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3

#         dt = 0.001 # time step in ms - STABLE TIME STEP
#         nt = int(1000) # total number of steps thru time

#         spins = int(1000)

#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=[0])

#         # first, check thath all the spins are in fact in spstruc
#         self.assertTrue(spstruc.isinside(sim.spins).all())

#         for n in range(nt):
#             sim.step(dt)

#         # not_finished = sim.out_d[0,:].get()>-10
#         # fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#         # spins = sim.spins_d.get()
#         # ax.scatter(spins[0,not_finished].T,spins[1,not_finished].T,
#         #   spins[2,not_finished].T)
#         # pl.savefig('scatter.png')

#         # fig, ax = pl.subplots()
#         # ax.plot(sim.out_d[0,:].get().T)
#         # #ax.plot(np.isnan(sim.out_d[0,:].get()).any())
#         # pl.savefig('output.png')

#         # spins = sim.spins_d.get()
#         # fig, ax = pl.subplots()
#         # dist2boundary = np.sqrt((spins[0,:]-spstruc.x)**2 + 
#         #             (spins[1,:]-spstruc.y)**2 + 
#         #             (spins[2,:]-spstruc.z)**2) - spstruc.r
#         # ax.hist(dist2boundary.T,bins=100)
#         # #ax.set_ylim(-0.01,0.01)
#         # pl.savefig('distance.png')

#         # after simulating diffusion, are all spins still inside spstruc?
#         self.assertTrue(spstruc.isinside(sim.spins_d.get()).all())

#         # signal from T2 decay
#         self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)

#     def testOutsideSingleSphere(self):
#         '''
#         diffusion inside of a single sphere
#         '''
#         Lx = 10 # um
#         Ly = 10 # um
#         Lz = 10 # um
#         D = 3.0 # um^2/ms
#         T2 = 100 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         # make a single sphere and put it in the middle of the geometry
#         sx = np.array([0,])
#         sy = np.array([0,])
#         sz = np.array([0,])
#         sr = np.array([3,])
#         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3

#         dt = 0.01 # time step in ms
#         nt = 1000 # total number of steps thru time

#         spins = 1000

#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=[1,])

#         # fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#         # spins = sim.spins
#         # ax.scatter(spins[0,:].T,spins[1,:].T,spins[2,:].T)
#         # pl.savefig('scatter_pre.png')

#         # first, check thath all the spins are in fact in spstruc
#         self.assertTrue((~spstruc.isinside(sim.spins)).all())

#         for n in range(nt):
#             sim.step(dt)

#         # not_finished = sim.out_d[0,:].get()>-10
#         # fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#         # spins = sim.spins_d.get()
#         # ax.scatter(spins[0,not_finished].T,spins[1,not_finished].T,
#         #   spins[2,not_finished].T)
#         # pl.savefig('scatter.png')

#         # fig, ax = pl.subplots()
#         # ax.plot(sim.out_d[0,:].get().T)
#         # #ax.plot(np.isnan(sim.out_d[0,:].get()).any())
#         # pl.savefig('output.png')

#         # spins = sim.spins_d.get()
#         # fig, ax = pl.subplots()
#         # dist2boundary = ((spins[0,:]-spstruc.x)**2 + 
#         #             (spins[1,:]-spstruc.y)**2 + 
#         #             (spins[2,:]-spstruc.z)**2 - 
#         #             spstruc.r**2)
#         # ax.plot(dist2boundary.T)
#         # ax.set_ylim(-0.01,0.01)
#         # pl.savefig('distance.png')

#         # after simulating diffusion, are all spins still inside spstruc?
#         self.assertTrue((~spstruc.isinside(sim.spins_d.get())).all())

#         # signal from T2 decay
#         self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)

#     def testInsideTwoSpheres(self):
#         '''
#         diffusion inside two spheres
#         '''
#         Lx = 10 # um
#         Ly = 10 # um
#         Lz = 10 # um
#         D = 3.0 # um^2/ms
#         T2 = 100 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         # make a single sphere and put it in the middle of the geometry
#         sx = np.array([1.1,-1.1])
#         sy = np.array([0,0])
#         sz = np.array([0,0])
#         sr = np.array([2,2])
#         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3

#         dt = 0.001 # time step in ms
#         nt = 1000 # total number of steps thru time
#         '''The number of spins simulated was set to 200,000, providing a lower bound on the SNR of the simulated signal (Harkins and Does, 2016)'''
#         spins = 1000

#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=[0,])

#         for n in range(nt):
#             sim.step(dt)

#         self.assertTrue((spstruc.isinside(sim.spins_d.get())).all())

#         # signal from T2 decay
#         self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)

#     def testOutsideTwoSpheres(self):
#         '''
#         diffusion inside two spheres
#         '''
#         Lx = 10 # um
#         Ly = 10 # um
#         Lz = 10 # um
#         D = 3.0 # um^2/ms
#         T2 = 100 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         # make a single sphere and put it in the middle of the geometry
#         sx = np.array([1.1,-1.1])
#         sy = np.array([0,0])
#         sz = np.array([0,0])
#         sr = np.array([2,2])
#         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3

#         dt = 0.001 # time step in ms
#         nt = 1000 # total number of steps thru time
#         '''The number of spins simulated was set to 200,000, providing a lower bound on the SNR of the simulated signal (Harkins and Does, 2016)'''
#         spins = int(5e3)
        

#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=[1,])

#         # first, check thath all the spins are in fact in spstruc
#         self.assertTrue((~spstruc.isinside(sim.spins)).all())

#         for n in range(nt):
#             sim.step(dt)

#         # not_finished = sim.out_d[0,:].get()>-10
#         # fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#         # spins = sim.spins_d.get()
#         # ax.scatter(spins[0,not_finished].T,spins[1,not_finished].T,
#         #   spins[2,not_finished].T,marker='.')
#         # pl.savefig('scatter2.png')

#         # fig, ax = pl.subplots()
#         # ax.plot(sim.out_d[0,:].get().T)
#         # #ax.plot(np.isnan(sim.out_d[0,:].get()).any())
#         # pl.savefig('output2.png')

#         # spins = sim.spins_d.get()
#         # fig, ax = pl.subplots()
#         # dist2boundary = ((spins[0,:]-spstruc.x)**2 + 
#         #             (spins[1,:]-spstruc.y)**2 + 
#         #             (spins[2,:]-spstruc.z)**2 - 
#         #             spstruc.r**2)
#         # ax.plot(dist2boundary.T)
#         # ax.set_ylim(-0.01,0.01)
#         # pl.savefig('distance.png')

#         # after simulating diffusion, are all spins still inside spstruc?
#         self.assertTrue((~spstruc.isinside(sim.spins_d.get())).all())

#         # signal from T2 decay
#         self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)

# if __name__ == '__main__':
#     unittest.main()