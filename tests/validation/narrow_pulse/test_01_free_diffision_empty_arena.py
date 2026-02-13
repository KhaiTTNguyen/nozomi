import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from simulation_toolkit.simulation_engine.diffsim3d import DiffSim3d
from simulation_toolkit.simulation_engine.geometry import SimGeometry3D

def test_free_diffision_empty_arena():
    '''
    Simulate Gaussian diffusion in an empty simulation arena 
    at the end of the simulation:
    * all spins should still be in the geometry bounded by 0 and Lx,Ly,Lz
    * the estimated diffusion coefficient should be approximately equal to D
    '''
    Lx = 10 # um
    Ly = 10 # um
    Lz = 10 # um
    D = 3.0 # um^2/ms
    T2 = 100 # ms
    rho = 1 # fractional water density
    sg3 = SimGeometry3D(Lx,Ly,Lz,D,T2,rho)

    dt = 0.002 # time step in ms
    nt = int(50000) # total number of steps thru time

    spins = int(100000)

    sim = DiffSim3d(sg3,spins)
    sim.set_segments()  # Initialize segments before stepping

    for n in range(nt):
        sim.step(dt)

    # for all spins, -Lx/2 <= x <= Lx/2
    assert (-Lx/2 <= sim.spins_d[0,:].get()).all()
    assert ((sim.spins_d[0,:].get() <= Lx/2).all())
    #  -Ly/2 <= y <= Ly/2
    assert (-Ly/2 <= sim.spins_d[1,:].get()).all()
    assert ((sim.spins_d[1,:].get() <= Ly/2).all())
    #  -Lz/2 <= z <= Lz/2
    assert (-Lz/2 <= sim.spins_d[2,:].get()).all()
    assert ((sim.spins_d[2,:].get() <= Lz/2).all())

    # calculate the mean squared displacment and require it be close to 
    # 2*D*dt*nt
    msd = np.sum(((sim.spins_d[0,:]-sim.spins0_d[0,:])**2).get()/spins)
    assert np.isclose(np.abs(msd/2/dt/nt),D,rtol=1e-1)