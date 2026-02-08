import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from scipy import integrate
from pycuda.compiler import SourceModule
kernel_file = './simulation_toolkit/simulation_engine/sim3d_kernel.cu'

class DiffSim3d:
    '''
    Simulates diffusion within a 3D geometry of structures
    '''
    def __init__(self,geom,nspins,diffdir=None):

        kernel = open(kernel_file,mode='r')
        self.code = kernel.read()
        kernel.close()

        self.pars = dict()

        # set the number of spins
        self.set_spins(nspins)

        # set the simulation geometry
        self.set_geometry(geom)

        # set the number and set of diffusion directions
        self.set_diffusion_directions(diffdir)

        # next: how many blocks and grids to use?
        self.set_block_grid(nspins)

        self.issetup = False

    # @profile
    def setup(self,structures = None,initstates_int = None):

        # use the provided seeder class to find the inital seed of spins, 
        # then pass them to the GPU
        self.mod = SourceModule(self.code % self.pars,no_extern_c=True)

        self.gpu_seed_kernel = self.mod.get_function("seedSpinsKernel")
        self.spins = self.geom.gpu_seed(self.gpu_seed_kernel, self.nspins, structIdxs=structures).astype(np.float32)        
        self.spins_d = gpuarray.to_gpu(self.spins)
        self.spins0_d = self.spins_d.copy()
        self.phase_d = gpuarray.zeros(self.ndiffdir * self.nspins, dtype=np.float32)
        
        self.randomWalk3d = self.mod.get_function("randomWalk3d")
        self.randomWalk3d_phase = self.mod.get_function("randomWalk3d_phase")
        self.randomWalk3d_phase_multi_dirr = self.mod.get_function("randomWalk3d_phase_multidirr")
        self.initstates = self.mod.get_function("initstates")
        self.compute_diffusion_coefficients_and_kurtosis = self.mod.get_function("computeDiffusionCoefficientsAndKurtosis")

        # pass the list of spheres in the geom to the gpu
        self.spheres_d = gpuarray.to_gpu(self.geom.spheres.astype(np.float32))

        if initstates_int is None:
            # use a random integer to initialize the random number generator
            initstates_int = np.random.randint(np.iinfo(np.int32).max,dtype=np.int32)
        self.initstates(initstates_int,block=(self.nblock,1,1), grid=(self.ngrid,1))

        # initialize signal level to M0=1
        self.sig_d = gpuarray.ones([1,self.nspins],dtype=np.float32)
        self.issetup = True

    def set_diffusion_directions(self,diffdir=None):
        if diffdir is None:
            # set x,y,z as default directions
            diffdir = np.array([[1,0,0],[0,1,0],[0,0,1]])
        
        if np.isscalar(diffdir):
            raise TypeError("diffdir cannot be a scalar (so far)")
        
        # check dimensions
        if (diffdir.shape[0] != 3) or (diffdir.ndim != 2):
            raise TypeError("diffdir must be a numpy array of size 3xN")
        
        self.ndiffdir = diffdir.shape[1]
        self.pars['ndiffdir'] = self.ndiffdir
        self.diffdir = gpuarray.to_gpu(diffdir.astype(np.float32))
        self.issetup = False

    def set_spins(self,nspins):
        self.nspins = nspins
        self.pars['nspins'] = self.nspins
        self.issetup = False
    
    def set_geometry(self,geom):
        self.geom = geom
        self.pars['nspheres'] = geom.nsphere
        self.pars['nstructures'] = geom.nstructures
        self.pars['Lx'] = geom.Lx
        self.pars['Ly'] = geom.Ly
        self.pars['Lz'] = geom.Lz

        D = []
        for struct in geom.structures:
            D.append(struct.D)
        self.pars['D'] = ','.join(map(str,D))

        T2 = []
        for struct in geom.structures:
            T2.append(struct.T2)
        self.pars['T2'] = ','.join(map(str,T2))
        self.issetup = False

    # @profile
    def set_segments(self,nsegx=1,nsegy=1,nsegz=1):
        '''return segments = Mx1 vector storing sph ids for each segment. 
        M = k*max_sphere_per_seg, where 'k' is the number of segments. '''
        jump_tol = 1 # um, tolerance to include a sphere inside a boundary...?
        self.nsegx = nsegx
        self.nsegy = nsegy
        self.nsegz = nsegz

        self.pars['dLx'] = self.pars['Lx']/self.nsegx
        self.pars['dLy'] = self.pars['Ly']/self.nsegy
        self.pars['dLz'] = self.pars['Lz']/self.nsegz
        self.pars['nLx'] = self.nsegx
        self.pars['nLy'] = self.nsegy
        self.pars['nLz'] = self.nsegz

        # what is the maximum number of spheres in each segment?
        max_sph_per_seg = 0
        for n in np.arange(nsegx):
            for m in np.arange(nsegy):
                for p in np.arange(nsegz):
                    # how far is each sphere from the 6 boundaries of this segment?
                    inx_min = n*self.pars['dLx']-self.geom.spheres[3,:] - self.geom.spheres[0,:] - self.pars['Lx']/2
                    inx_max = self.geom.spheres[0,:] - ((n+1)*self.pars['dLx']+self.geom.spheres[3,:] - self.pars['Lx']/2)
                    iny_min = m*self.pars['dLy']-self.geom.spheres[3,:] - self.geom.spheres[1,:] - self.pars['Ly']/2
                    iny_max = self.geom.spheres[1,:] - ((m+1)*self.pars['dLy']+self.geom.spheres[3,:] - self.pars['Ly']/2)
                    inz_min = p*self.pars['dLz']-self.geom.spheres[3,:] - self.geom.spheres[2,:] - self.pars['Lz']/2
                    inz_max = self.geom.spheres[2,:] - ((p+1)*self.pars['dLz']+self.geom.spheres[3,:] - self.pars['Lz']/2)

                    in_segment = np.maximum(inx_min,inx_max)
                    in_segment = np.maximum(in_segment,iny_min)
                    in_segment = np.maximum(in_segment,iny_max)
                    in_segment = np.maximum(in_segment,inz_min)
                    in_segment = np.maximum(in_segment,inz_max)
                    # negative value of in_segment = inside/overlapping the segment from this boundary
                    sph_this_seg = np.count_nonzero(in_segment < jump_tol)
                    max_sph_per_seg = max(max_sph_per_seg,sph_this_seg)
        
        self.spheres_per_segment = max_sph_per_seg

        # make a list of spheres in each segment
        self.segments = np.zeros([nsegx*nsegy*nsegz*self.spheres_per_segment,],dtype=np.intc)
        for n in np.arange(nsegx):
            for m in np.arange(nsegy):
                for p in np.arange(nsegz):
                    # how far is each sphere from the 6 boundaries of this segment?
                    inx_min = n*self.pars['dLx']-self.geom.spheres[3,:] - self.geom.spheres[0,:] - self.pars['Lx']/2
                    inx_max = self.geom.spheres[0,:] - ((n+1)*self.pars['dLx']+self.geom.spheres[3,:] - self.pars['Lx']/2)
                    iny_min = m*self.pars['dLy']-self.geom.spheres[3,:] - self.geom.spheres[1,:] - self.pars['Ly']/2
                    iny_max = self.geom.spheres[1,:] - ((m+1)*self.pars['dLy']+self.geom.spheres[3,:] - self.pars['Ly']/2)
                    inz_min = p*self.pars['dLz']-self.geom.spheres[3,:] - self.geom.spheres[2,:] - self.pars['Lz']/2
                    inz_max = self.geom.spheres[2,:] - ((p+1)*self.pars['dLz']+self.geom.spheres[3,:] - self.pars['Lz']/2)

                    in_segment = np.maximum(inx_min,inx_max)
                    in_segment = np.maximum(in_segment,iny_min)
                    in_segment = np.maximum(in_segment,iny_max)
                    in_segment = np.maximum(in_segment,inz_min)
                    in_segment = np.maximum(in_segment,inz_max)
                    # negative value of in_segment = inside/overlapping the segment from this boundary

                    seg_idx = self.spheres_per_segment*(p*nsegx*nsegy + m*nsegx + n)
                    self.segments[seg_idx:seg_idx+self.spheres_per_segment] = np.sort(np.argsort(in_segment)[:self.spheres_per_segment])
                    #self.segments[seg_idx:seg_idx+self.spheres_per_segment] = np.argsort(in_segment)[:self.spheres_per_segment]

        self.segments_d = gpuarray.to_gpu(self.segments)

        self.pars['nspheres_per_seg'] = self.spheres_per_segment

        self.issetup = False

    def set_block_grid(self,nspins):
        self.nblock = int(1000)
        self.ngrid = int(round(nspins/self.nblock))
    
    def calculate_diffusion_coefficients_and_kurtoses(self, step_idx, current_time, 
                                dx_array, dy_array, dz_array,
                                Kx2_array, Ky2_array, Kz2_array,
                                Kx4_array, Ky4_array, Kz4_array):
        """Calculate diffusion coefficients 
        and kurtoses using GPU kernel"""
        block_size = 256
        grid_size = int(np.ceil(self.nspins / block_size))
        
        self.compute_diffusion_coefficients_and_kurtosis(
            self.spins_d, self.spins0_d,
            dx_array, dy_array, dz_array,
            Kx2_array, Ky2_array, Kz2_array,  # Second moment arrays
            Kx4_array, Ky4_array, Kz4_array,  # Fourth moment arrays
            np.int32(step_idx),  np.int32(self.nspins),
            np.float32(current_time),
            block=(block_size, 1, 1), grid=(grid_size, 1)
        )

    def step(self,dt):
        if not self.issetup:
            self.setup()
    
        self.randomWalk3d(np.float32(dt),self.spheres_d,self.segments_d,
                          self.spins_d,self.spins0_d,self.sig_d,
                          block=(self.nblock,1,1), grid=(self.ngrid,1))
    
    def dwi_step(self, gwave_dt, G, G_area_at_each_time_step):
        if not self.issetup:
            self.setup()
    
        self.randomWalk3d_phase(np.float32(gwave_dt),self.spheres_d,self.segments_d,
                          self.spins_d,self.spins0_d,
                          self.phase_d, 
                          np.float32(G), np.float32(G_area_at_each_time_step),  
                          block=(self.nblock,1,1), grid=(self.ngrid,1))
    
    def dwi_multidirections_step(self, gwave_dt, G, G_area_at_each_time_step):
        if not self.issetup:
            self.setup()

        self.randomWalk3d_phase_multi_dirr(np.float32(gwave_dt),self.spheres_d,self.segments_d,
                          self.spins_d,self.spins0_d,
                          self.phase_d, 
                          np.float32(G), np.float32(G_area_at_each_time_step),
                          self.diffdir, np.int32(self.ndiffdir), 
                          block=(self.nblock,1,1), grid=(self.ngrid,1))
    
class DwiSim3d(DiffSim3d):

    def simulate(self,gwave,structures=None,initstates=None):
        '''
        Simulates DWI signal in a simulation gometry
        '''
        G_area_at_each_time_step = integrate.cumulative_trapezoid(gwave.wave, dx=1.0, initial=0)*gwave.dt

        for n,_ in enumerate(gwave.wave):
            self.dwi_step(gwave.dt, 
                          gwave.wave[n]*gwave.dt*267.5/10000, 
                          G_area_at_each_time_step[n]*267.7/10000)
        # Get the result back from GPU
        phase_accumulated = self.phase_d.get()
        
        # Reshape to match expected format (3, nspin)
        return phase_accumulated.reshape(self.ndiffdir, self.nspins)
    
    def simulate_multi_directions(self,gwave,structures=None,initstates=None):
        '''
        Simulates DWI signal in a simulation gometry
        '''
        G_area_at_each_time_step = integrate.cumulative_trapezoid(gwave.wave, dx=1.0, initial=0)*gwave.dt

        for n,_ in enumerate(gwave.wave):
            self.dwi_multidirections_step(gwave.dt, 
                          gwave.wave[n]*gwave.dt*267.5/10000, 
                          G_area_at_each_time_step[n]*267.7/10000)

        # Get the result back from GPU
        phase_accumulated = self.phase_d.get()
        
        # Reshape to match expected format (3, nspin)
        return phase_accumulated.reshape(self.ndiffdir, self.nspins)
