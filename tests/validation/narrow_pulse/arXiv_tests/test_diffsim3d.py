# import pycuda.autoinit
# import pycuda.driver as drv
# import pycuda.gpuarray as gpuarray
# import numpy as np
# import unittest
# import diffsim3d.diffsim3d as ds3
# import diffsim3d.geometry as geom
# import matplotlib.pyplot as pl

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

#         spins = int(100000)

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

#         dt = 0.01 # time step in ms
#         nt = 10000 # total number of steps thru time

#         spins = 100000

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
#         nt = 10000 # total number of steps thru time

#         spins = 100000

#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=[1,])

#         # fig = pl.figure()
#         # ax = fig.add_subplot(111, projection='3d')
#         # spins = sim.spins
#         # ax.scatter(spins[0,:].T,spins[1,:].T,spins[2,:].T)
#         # pl.savefig('scatter_pre.png')

#         # first, check thath all the spins are in fact in spstruc
#         self.assertTrue((~spstruc.isinside(sim.spins)).all())

#         for n in range(nt):
#             sim.step(dt)
#             # mask = spstruc.isinside(sim.spins_d.get())
#             # if mask.any():
#             #     print(n)
#             #     print(np.sum(mask))
#             #     print(sim.sig_d.get()[0,mask])
#             #     slkjdfd

#         # fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#         # spins = sim.spins_d.get()
#         # not_inside = spstruc.isinside(spins)
#         # ax.scatter(spins[0,not_inside].T,spins[1,not_inside].T,
#         #   spins[2,not_inside].T)
#         # ax.set_xlim(-5,5)
#         # ax.set_ylim(-5,5)
#         # ax.set_zlim(-5,5)
#         # pl.savefig('scatter.png')

#         # print(sim.geom.nsphere)
#         # print(sim.spheres_d.get())

#         # fig, ax = pl.subplots()
#         # #ax.plot(sim.sig_d.get().T)
#         # #ax.set_ylim(-0.1,1)
#         # ax.plot(spins[0,not_inside].T,spins[2,not_inside].T,'o')
#         # ax.set_xlim(-5,5)
#         # ax.set_ylim(-5,5)
#         # pl.savefig('output.png')

#         # sig = sim.sig_d.get()
#         # print(sig.shape,not_inside.shape)
#         # print('sig',sig[0,not_inside])

#         # spins = sim.spins_d.get()
#         # fig, ax = pl.subplots()
#         # dist2boundary = ((spins[0,:]-spstruc.x)**2 + 
#         #             (spins[1,:]-spstruc.y)**2 + 
#         #             (spins[2,:]-spstruc.z)**2 - 
#         #             spstruc.r**2)
#         # ax.plot(dist2boundary.T)
#         # ax.set_ylim(-0.1,0.1)
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

#         dt = 0.01 # time step in ms
#         nt = 1000 # total number of steps thru time

#         spins = 100000

#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.setup(structures=[0,])

#         for n in range(nt):
#             sim.step(dt)

#         # fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#         # spins = sim.spins_d.get()
#         # not_inside = spstruc.isinside(spins)
#         # ax.scatter(spins[0,not_inside].T,spins[1,not_inside].T,
#         #   spins[2,not_inside].T)
#         # ax.set_xlim(-5,5)
#         # ax.set_ylim(-5,5)
#         # ax.set_zlim(-5,5)
#         # pl.savefig('scatter.png')

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

#         dt = 0.01 # time step in ms
#         nt = 1000 # total number of steps thru time

#         spins = 100000

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

#     def testInsideMultipleSpheres(self):
#         '''
#         diffusion inside of a multiple spheres
#         '''
#         Lx = 10 # um
#         Ly = 10 # um
#         Lz = 10 # um
#         D = 3.0 # um^2/ms
#         T2 = 100 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         # make a series of spheres along the x direction, similar to an axon
#         nsphere = 20
#         noverlap = 2
#         sz = np.arange(-nsphere/2-noverlap,nsphere/2+1+noverlap)*Lz/nsphere
#         sy = np.cos(2*np.pi*sz/Lz)*0.25
#         sx = np.full(sz.shape, 0)
#         sr = np.full(sz.shape, 1)

#         # print(sz)
#         # print(sy)
#         # print(sx)

#         # fig, ax = pl.subplots()
#         # ax.plot(sz.T,sy.T,marker='.',linestyle='None')
#         # ax.set_xlim(-Lz/2,Lz/2)
#         # ax.set_ylim(-Ly/2,Ly/2)
#         # ax.grid(True)
#         # pl.savefig('wavetN.png')

#         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3

#         dt = 0.01 # time step in ms
#         nt = 1000 # total number of steps thru time

#         spins = 100000

#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.set_segments(nsegx=2,nsegy=2,nsegz=2)
#         sim.setup(structures=[0])

#         # print(sim.pars['nspheres_per_seg'])
#         # print(sim.segments.shape)
#         # print(sim.segments.reshape((2,sim.pars['nspheres_per_seg'])))
#         # print(np.sort(sim.segments))
#         # print(sz)

#         # first, check thath all the spins are in fact in spstruc
#         self.assertTrue(spstruc.isinside(sim.spins).all())

#         spin_prev = 0
#         for n in range(nt):
#             sim.step(dt)
#             # isInside = spstruc.isinside(sim.spins_d.get())
#             # if ~(isInside.all()):
#             #     print(np.sum(~isInside))
#             #     print(n)
#             #     fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
#             #     spinsg = sim.spins_d.get()
#             #     # ax.scatter(spinsg[0,isInside].T,spinsg[1,isInside].T,
#             #     #   spinsg[2,isInside].T,marker='.')
#             #     ax.scatter(spinsg[0,~isInside].T,
#             #                spinsg[1,~isInside].T,
#             #                spinsg[2,~isInside].T,marker='.')
#             #     ax.set_xlim(-Lx/2,Lx/2)
#             #     ax.set_ylim(-Ly/2,Ly/2)
#             #     ax.set_zlim(-Lz/2,Lz/2)
#             #     pl.savefig('scatterM.png')

#             #     fig, ax = pl.subplots()
#             #     ax.plot(spinsg[1,~isInside].T,spinsg[2,~isInside].T,marker='.',linestyle='None')
#             #     ax.set_xlim(-5,5)
#             #     ax.set_ylim(-5,5)
#             #     ax.grid(True)
#             #     pl.savefig('outputN.png')

#             #     self.assertTrue(spstruc.isinside(sim.spins_d.get()).all())

#         # all spins are still inside the arena
#         self.assertTrue(spstruc.isinside(sim.spins_d.get()).all())

#         # print(sim.spins_d.get()-sim.spins0_d.get())
#         msdx = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins
#         msdy = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins
#         msdz = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins

#         # Diffusion in the z direction should be close to D
#         self.assertGreater(msdz/2/dt/nt,D-0.2)
#         self.assertLess(msdz/2/dt/nt,D+0.2)
#         # Diffusion in the x&y direction shoudl be close to 0 due to restriction
#         self.assertLess(msdy/2/dt/nt,0.04)
#         self.assertLess(msdx/2/dt/nt,0.04)

#         # after simulating diffusion, are all spins still inside spstruc?
#         self.assertEqual(np.sum(~spstruc.isinside(sim.spins_d.get())),0)

#         # signal from T2 decay
#         self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)

#     def testOutsideMultipleSpheres(self):
#         '''
#         diffusion inside of a multiple spheres
#         '''
#         Lx = 10 # um
#         Ly = 10 # um
#         Lz = 10 # um
#         D = 3.0 # um^2/ms
#         T2 = 100 # ms
#         rho = 1 # fractional water density
#         sg3 = geom.SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

#         # make a series of spheres along the x direction, similar to an axon
#         nsphere = 20
#         noverlap = 2
#         sz = np.arange(-nsphere/2-noverlap,nsphere/2+1+noverlap)*Lz/nsphere
#         sy = np.cos(2*np.pi*sz/Lz)*0.25
#         sx = np.full(sz.shape, 0)
#         sr = np.full(sz.shape, 1)
#         # sz = np.arange(-nsphere/2-2,nsphere/2+3)*Lx/nsphere
#         # sy = np.sin(2*np.pi*np.arange(-2,nsphere+3)/nsphere)*0.25
#         # sx = np.full(sz.shape, 0)
#         # sr = np.full(sz.shape, 2)

#         spstruc = geom.Structure3D(sx,sy,sz,sr,D,T2,rho)
#         sg3.add_structure(spstruc) # add it to sg3

#         dt = 0.01 # time step in ms
#         nt = 1000 # total number of steps thru time

#         spins = 100000

#         sim = ds3.DiffSim3d(sg3,spins)
#         sim.set_segments(nsegx=2,nsegy=2,nsegz=2)
#         sim.setup(structures=[1])

#         # first, check thath all the spins are in fact in spstruc
#         self.assertTrue((~spstruc.isinside(sim.spins)).all())

#         spin_prev = 0
#         for n in range(nt):
#             sim.step(dt)

#         # all spins are still inside the arena
#         self.assertTrue((~spstruc.isinside(sim.spins)).all())

#         # print(sim.spins_d.get()-sim.spins0_d.get())
#         msdx = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins
#         msdy = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins
#         msdz = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins

#         # Diffusion in the z direction should be close to D
#         self.assertGreater(msdz/2/dt/nt,D-0.2)
#         self.assertLess(msdz/2/dt/nt,D+0.2)
#         # Diffusion in the x*y direction shoudl be close to 3.0, but lower
#         self.assertGreater(msdy/2/dt/nt,D-0.5)
#         self.assertGreater(msdz/2/dt/nt,D-0.5)

#         # signal from T2 decay
#         self.assertAlmostEqual(np.mean(sim.sig_d.get()),np.exp(-nt*dt/T2),2)


# if __name__ == '__main__':
#     unittest.main()
