import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from simulation_toolkit.simulation_engine.diffsim3d import DiffSim3d
from simulation_toolkit.simulation_engine.geometry import SimGeometry3D, Structure3D
import simulation_toolkit.defaults.params as config_params
import os
import matplotlib.pyplot as pl

def test_inside_single_sphere():
    '''
    diffusion inside of a single sphere
    '''
    Lx = 10 # um
    Ly = 10 # um
    Lz = 10 # um
    D = 3.0 # um^2/ms
    T2 = 100 # ms
    rho = 1 # fractional water density
    sg3 = SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

    # make a single sphere and put it in the middle of the geometry
    sx = np.array([0,])
    sy = np.array([0,])
    sz = np.array([0,])
    sr = np.array([1,])
    spstruc = Structure3D(sx,sy,sz,sr,D,T2,rho)
    sg3.add_structure(spstruc) # add it to sg3

    dt = 0.002 # time step in ms
    nt = int(10000) # total number of steps thru time
    spins = int(100000)
 
    sim = DiffSim3d(sg3,spins)
    sim.set_segments()
    sim.setup(structures=[0])

    # first, check that all the spins are in fact in spstruc
    assert (spstruc.isinside(sim.spins).all())

    config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH = "./tests/validation/narrow_pulse/test_figures/" 
    if not os.path.exists(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH):
        os.makedirs(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH)
    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    spins_np = sim.spins_d.get()
    ax.scatter(spins_np[0,:].T,spins_np[1,:].T,spins_np[2,:].T)
    ax.set_xlim(-Lx/2,Lx/2)
    ax.set_ylim(-Ly/2,Ly/2)
    ax.set_zlim(-Lz/2,Lz/2)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_zlabel('z (μm)')
    pl.savefig(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH + "test_inside_single_sphere_pre_sim_spins.png")

    # ======= Simulate diffusion =======
    for n in range(nt):
        sim.step(dt)

    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    spins_np = sim.spins_d.get()
    ax.scatter(spins_np[0,:].T,spins_np[1,:].T,spins_np[2,:].T)
    ax.set_xlim(-Lx/2,Lx/2)
    ax.set_ylim(-Ly/2,Ly/2)
    ax.set_zlim(-Lz/2,Lz/2)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_zlabel('z (μm)')
    pl.savefig(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH + "test_inside_single_sphere_post_sim_spins.png")

    # after simulating diffusion, are all spins still inside spstruc?
    assert (spstruc.isinside(sim.spins_d.get()).all())


def test_outside_single_sphere():
    '''
    diffusion outside of a single sphere
    '''
    Lx = 10 # um
    Ly = 10 # um
    Lz = 10 # um
    D = 3.0 # um^2/ms
    T2 = 100 # ms
    rho = 1 # fractional water density
    sg3 = SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

    # make a single sphere and put it in the middle of the geometry
    sx = np.array([0,])
    sy = np.array([0,])
    sz = np.array([0,])
    sr = np.array([1,])
    spstruc = Structure3D(sx,sy,sz,sr,D,T2,rho)
    sg3.add_structure(spstruc) # add it to sg3

    dt = 0.002 # time step in ms
    nt = int(10000) # total number of steps thru time
    spins = int(100000)

    sim = DiffSim3d(sg3,spins)
    sim.set_segments()  
    sim.setup(structures=[1,])  # sim_arena=last item in structures list --> seed in arena a.k.a. outside single sphere

    # first, check thath all the spins are not in spstruc
    isOutside = False
    for struct in sg3.structures[:-1]:
        isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d)) # str.isInside(sim.spins) return vector length num_spins & OR with isOutside
    isOutside = np.invert(isOutside)
    assert isOutside.all()

    config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH = "./tests/validation/narrow_pulse/test_figures/" 
    if not os.path.exists(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH):
        os.makedirs(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH)
    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    spins_np = sim.spins_d.get()
    ax.scatter(spins_np[0,:].T,spins_np[1,:].T,spins_np[2,:].T)
    ax.set_xlim(-Lx/2,Lx/2)
    ax.set_ylim(-Ly/2,Ly/2)
    ax.set_zlim(-Lz/2,Lz/2)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_zlabel('z (μm)')
    pl.savefig(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH + "test_outside_single_sphere_pre_sim_spins.png")

    # ======= Simulate diffusion =======
    for n in range(nt):
        sim.step(dt)

    # after simulating diffusion, are all spins still not inside spstruc?
    isOutside = False
    for struct in sg3.structures[:-1]:
        isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins_d.get()) return vector length num_spins & OR with isOutside
    isOutside = np.invert(isOutside)
    assert isOutside.all()

    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    spins_np = sim.spins_d.get()
    ax.scatter(spins_np[0,:].T,spins_np[1,:].T,spins_np[2,:].T)
    ax.set_xlim(-Lx/2,Lx/2)
    ax.set_ylim(-Ly/2,Ly/2)
    ax.set_zlim(-Lz/2,Lz/2)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_zlabel('z (μm)')
    pl.savefig(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH + "test_outside_single_sphere_post_sim_spins.png")

    # signal from T2 decay
    assert abs(np.mean(sim.sig_d.get()) - np.exp(-nt*dt/T2)) < 1e-2

def test_inside_multiple_spheres():
    '''
    diffusion inside of multiple spheres
    '''
    Lx = 10.0 # um
    Ly = 10.0 # um
    Lz = 10.0 # um
    D = 3.0 # um^2/ms
    T2 = 100 # ms
    rho = 1 # fractional water density
    sg3 = SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

    nsphere = 20
    sz = np.arange(-nsphere/2-2,nsphere/2+3)*Lx/nsphere
    sy = np.sin(2*np.pi*np.arange(-2,nsphere+3)/nsphere)*0.25
    sx = np.full(sz.shape, 0)
    sr = np.full(sz.shape, 1)
    spstruc = Structure3D(sx,sy,sz,sr,D,T2,rho)
    sg3.add_structure(spstruc) # add it to sg3

    dt = 0.002 # time step in ms
    nt = int(10000) # total number of steps thru time
    spins = int(100000)

    sim = DiffSim3d(sg3,spins)
    sim.set_segments()
    sim.setup(structures=[0,])

    # first, check thath all the spins are in fact in spstruc
    isInside = False
    for struct in sg3.structures[:-1]:
        isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get())) # str.isinside(sim.spins) return vector length num_spins & OR with isInside
    assert isInside.all()

    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    spins_np = sim.spins_d.get()
    ax.scatter(spins_np[0,:].T,spins_np[1,:].T,spins_np[2,:].T)
    ax.set_xlim(-Lx/2,Lx/2)
    ax.set_ylim(-Ly/2,Ly/2)
    ax.set_zlim(-Lz/2,Lz/2)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_zlabel('z (μm)')
    pl.savefig(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH + "test_inside_multiple_spheres_pre_sim_spins.png")

    for n in range(nt):
        sim.step(dt)
    
    #-------------------------------------------
    # all spins are still inside the arena
    isInside = False
    for struct in sg3.structures[:-1]:
        isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get())) # str.isinside(sim.spins) return vector length num_spins & OR with isInside
    assert isInside.all()

    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    spins_np = sim.spins_d.get()
    ax.scatter(spins_np[0,:].T,spins_np[1,:].T,spins_np[2,:].T)
    ax.set_xlim(-Lx/2,Lx/2)
    ax.set_ylim(-Ly/2,Ly/2)
    ax.set_zlim(-Lz/2,Lz/2)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_zlabel('z (μm)')
    pl.savefig(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH + "test_inside_multiple_spheres_post_sim_spins.png")

    msdx = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins
    msdy = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins
    msdz = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins
    #print(msdx/2/dt/nt,msdy/2/dt/nt,msdz/2/dt/nt)

    # Diffusion in the z direction should be close to D
    assert msdz/2/dt/nt > D-0.2
    assert msdz/2/dt/nt < D+0.2
    # # Diffusion in the x&y direction shoudl be close to 0 due to restriction
    assert msdy/2/dt/nt < 0.2
    assert msdx/2/dt/nt < 0.2

    # signal from T2 decay
    assert abs(np.mean(sim.sig_d.get()) - np.exp(-nt*dt/T2)) < 1e-2
    #-------------------------------------------

    # after simulating diffusion, are all spins still inside sg3?
    isInside = False
    for struct in sg3.structures[:-1]:
        isInside = np.logical_or(isInside, struct.isinside(sim.spins_d.get())) # str.isinside(sim.spins) return vector length num_spins & OR with isInside
    assert isInside.all()


def test_outside_multiple_spheres():
    '''
    diffusion outside of a multiple spheres
    '''
    Lx = 10 # um
    Ly = 10 # um
    Lz = 10 # um
    D = 3.0 # um^2/ms
    T2 = 100 # ms
    rho = 1 # fractional water density
    sg3 = SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

    # make a series of spheres along the x direction, similar to an axon
    nsphere = 20
    sz = np.arange(-nsphere/2-2,nsphere/2+3)*Lx/nsphere
    sy = np.sin(2*np.pi*np.arange(-2,nsphere+3)/nsphere)*0.25
    sx = np.full(sz.shape, 0)
    sr = np.full(sz.shape, 1)

    spstruc = Structure3D(sx,sy,sz,sr,D,T2,rho)
    sg3.add_structure(spstruc) # add it to sg3

    dt = 0.002 # time step in ms
    nt = int(10000) # total number of steps thru time
    spins = int(100000)

    sim = DiffSim3d(sg3,spins)
    sim.set_segments()
    sim.setup(structures=[1])

    # first, check thath all the spins are in fact in spstruc
    isOutside = False
    for struct in sg3.structures[:-1]:
        isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins_d.get()) return vector length num_spins & OR with isOutside
    isOutside = np.invert(isOutside)
    assert isOutside.all()
    
    config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH = "./tests/validation/narrow_pulse/test_figures/" 
    if not os.path.exists(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH):
        os.makedirs(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH)
    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    spins_np = sim.spins_d.get()
    ax.scatter(spins_np[0,:].T,spins_np[1,:].T,spins_np[2,:].T)
    ax.set_xlim(-Lx/2,Lx/2)
    ax.set_ylim(-Ly/2,Ly/2)
    ax.set_zlim(-Lz/2,Lz/2)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_zlabel('z (μm)')
    pl.savefig(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH + "test_outside_multiple_spheres_pre_sim_spins.png")

    # ======= Simulate diffusion =======
    for n in range(nt):
        sim.step(dt)

    # all spins are still inside the arena
    isOutside = False
    for struct in sg3.structures[:-1]:
        isOutside = np.logical_or(isOutside, struct.isinside(sim.spins_d.get())) # str.isInside(sim.spins_d.get()) return vector length num_spins & OR with isOutside
    isOutside = np.invert(isOutside)
    assert isOutside.all()

    fig, ax = pl.subplots(subplot_kw={"projection": "3d"})
    spins_np = sim.spins_d.get()
    ax.scatter(spins_np[0,:].T,spins_np[1,:].T,spins_np[2,:].T)
    ax.set_xlim(-Lx/2,Lx/2)
    ax.set_ylim(-Ly/2,Ly/2)
    ax.set_zlim(-Lz/2,Lz/2)
    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_zlabel('z (μm)')
    pl.savefig(config_params.SINGLE_SPHERE_VALIDATION_TEST_FOLDER_PATH + "test_outside_multiple_spheres_post_sim_spins.png")

    msdx = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get())/spins
    msdy = np.sum(((sim.spins_d[1,:]-sim.spins0_d[1,:])**2).get())/spins
    msdz = np.sum(((sim.spins_d[2,:]-sim.spins0_d[2,:])**2).get())/spins

    # Diffusion in the z direction should be close to D
    assert msdz/2/dt/nt > D-0.2
    assert msdz/2/dt/nt < D+0.2
    # # Diffusion in the x*y direction shoudl be close to 3.0, but lower
    assert msdy/2/dt/nt > D-0.5
    assert msdx/2/dt/nt > D-0.5

    # signal from T2 decay
    assert abs(np.mean(sim.sig_d.get()) - np.exp(-nt*dt/T2)) < 1e-2

