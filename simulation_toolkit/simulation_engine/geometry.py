import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.curandom as curandom
from pycuda.compiler import SourceModule
import numpy as np
import random

class Structure3D:
    '''
    Defines a 3D structure as a list of spheres, and provides methods to check 
    if a list of spins is inside the structure
    '''
    def __init__(self,x,y,z,r,D,T2,rho):
        # here, x,y&z should be row vectors. ensure that is the scase
        self.nspheres = x.size
        self.x = np.reshape(x,[self.nspheres,1])
        self.y = np.reshape(y,[self.nspheres,1])
        self.z = np.reshape(z,[self.nspheres,1])
        self.r = np.reshape(r,[self.nspheres,1])

        self.D = D
        self.T2 = T2
        self.rho = rho
        self.tol = 1e-4

    def isinside(self,rnd):
        '''
        check to see is rnd is inside any sphere in this structure
        '''
        inside_any = ((rnd[0,:]-self.x)**2 + 
                    (rnd[1,:]-self.y)**2 + 
                    (rnd[2,:]-self.z)**2 -
                    self.r**2 < 0)
        
        return inside_any.any(axis=0)

class SimGeometry3D:
    def __init__(self,Lx,Ly,Lz,D,T2,rho):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.nsphere = 0
        self.spheres = np.zeros([5,self.nsphere]) 

        # number of structures inside this geometry
        self.nstructures = 0
        # the last item in this list is always this simulation geometry
        self.structures = [self]

        self.D = D
        self.T2 = T2
        self.rho = rho

    def add_structure(self,struct):
        self.structures[-1] = struct
        self.structures.append(self)

        newsphere = np.zeros([5,struct.nspheres])
        newsphere[0,:] = struct.x.T
        newsphere[1,:] = struct.y.T
        newsphere[2,:] = struct.z.T
        newsphere[3,:] = struct.r.T
        newsphere[4,:] = self.nstructures
        self.spheres = np.concatenate([self.spheres,newsphere],axis=1)

        self.nstructures += 1
        self.nsphere += struct.nspheres

    def isinside(self,rnd):
        return ( (rnd[0,:] > -self.Lx/2) & (rnd[1,:] > -self.Ly/2) & (rnd[2,:] > -self.Lz/2) & 
            (rnd[0,:] < self.Lx/2) & (rnd[1,:] < self.Ly/2) & (rnd[2,:] < self.Lz/2) )

    def gpu_seed(self, gpu_seed_kernel, nspin, structIdxs=None):
        """
        Seed 'nspin' spins into this arena using complete GPU acceleration.
        
        Parameters:
        gpu_seed_kernel : The compiled CUDA kernel function for seeding spins.
        nspin : Number of spins to seed
        structIdxs : Indices of structures to seed spins into
            
        Returns:
        numpy.ndarray
            Array of shape (3, nspin) containing the seeded positions
        """

        # If no structure index is given, seed into all structures
        if structIdxs is None:
            structIdxs = np.arange(self.nstructures + 1, dtype=np.int32)  # All structures + external box
        
        # Convert to numpy array if not already
        structIdxs = np.array(structIdxs, dtype=np.int32)
        
        # --- Prepare sphere data ---
        # Collect all spheres from all structures into one array
        all_spheres = []
        sphere_counts = []
        
        # Process each structure (except the box) to get sphere data
        for struct_idx in range(self.nstructures):
            structure = self.structures[struct_idx]
            if structure.nspheres > 0:
                # Create array of [x, y, z, r] for each sphere in this structure
                spheres = np.vstack([
                    structure.x.flatten(),
                    structure.y.flatten(),
                    structure.z.flatten(),
                    structure.r.flatten()
                ]).T.astype(np.float32)
                
                all_spheres.append(spheres)
                sphere_counts.append(structure.nspheres)
            else:
                sphere_counts.append(0)
        
        # Add a count for the "outside" structure (the box)
        sphere_counts.append(0)
        
        # Combine all sphere data
        if all_spheres:
            all_spheres = np.vstack(all_spheres).astype(np.float32)
        else:
            all_spheres = np.zeros((0, 4), dtype=np.float32)
        
        # --- Create GPU arrays ---
        # Output array for positions (format: x1...xN, y1...yN, z1...zN)
        positions_d = gpuarray.zeros(3 * nspin, dtype=np.float32)
        
        # Structure indices array
        structIdxs_d = gpuarray.to_gpu(structIdxs)
        
        # Sphere data arrays
        spheres_d = gpuarray.to_gpu(all_spheres.flatten())
        sphere_counts_d = gpuarray.to_gpu(np.array(sphere_counts, dtype=np.int32))

        # --- Setup execution parameters ---
        block_size = 256
        grid_size = (nspin + block_size - 1) // block_size
        
        # Create seeding status array (1 = needs seeding, 0 = seeded)
        seeding_status_d = gpuarray.ones(nspin, dtype=np.int32)
        
        # --- Execute seeding kernel until all spins are seeded ---
        iterations = 0
        
        while gpuarray.sum(seeding_status_d).get() > 0:
            # Generate a new random seed for each iteration
            rand_seed = random.randint(1, 2**31 - 1)
            
            # Launch the kernel
            gpu_seed_kernel(
                np.int32(nspin),
                np.int32(self.nstructures + 1),  # +1 for the box
                spheres_d,
                sphere_counts_d,
                positions_d,
                np.int32(len(structIdxs)),
                structIdxs_d,
                np.float32(self.Lx),
                np.float32(self.Ly),
                np.float32(self.Lz),
                np.uint32(rand_seed),
                seeding_status_d,
                block=(block_size, 1, 1),
                grid=(grid_size, 1)
            )
            
            iterations += 1
           
        # Get the result back from GPU
        positions = positions_d.get()
        
        # Reshape to match expected format (3, nspin)
        return positions.reshape(3, nspin)
