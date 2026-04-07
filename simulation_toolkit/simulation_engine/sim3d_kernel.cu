#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define sign(a) ((a) > 0 ? +1 : ((a) < 0 ? -1 : 0))

#define MIN(a, b) (((a) > (b))? (b): (a))
#define MAX(a, b) (((a) < (b))? (b): (a))

// specify constants that are known at compile time
const int nspins = %(nspins)s; // number of spins in the simulation
const int nspheres = %(nspheres)s; // number of spheres in the simulation
const int nspheres_per_seg = %(nspheres_per_seg)s; // num of spheres per segment
const int nstructures = %(nstructures)s; // number of structures

// length of the simulation arena
const float Lx = %(Lx)s;
const float Ly = %(Ly)s;
const float Lz = %(Lz)s;

// length of the simulation arena segments
const float dLx = %(dLx)s;
const float dLy = %(dLy)s;
const float dLz = %(dLz)s;

const float nLx = %(nLx)s;
const float nLy = %(nLy)s;
const float nLz = %(nLz)s;

__device__ float D[] = {%(D)s};
__device__ float T2[] = {%(T2)s};

// error tolerance for floating point precision
const float tol = 1e-4;
const float someLargeNumber = 1e6;

// sphere properties:
//  cx,cy,cz = center of the sphere
//  r = radius of the sphere
//  sidx = structure number
//
// spheres are indexed as [6,nspheres], where cx=0, cy=1, cz=2, r=3, sidx=4

// spin properties: 
//  x,y,z = location of the spin
//  sidx = structure this spin is in
// 
// spins are indexed as [4,nspins], where x=0, y=1, z=2, sidx=3
//
//  jx,jy,jz = direction of the jump, but this not part of the structure

// structure properties:
// the index of a structure signifies importance: lower IDs take precedence
// structure id == nspheres indicates the rectangular arena of the simulation

// segments: all spins are inside a segment. Segments split the simulation arena  
// into to limit the number of spheres that need to be checked. 
// therefore segmentList order is given by [Nsegments,spheres_per_segment]
// and the index to the segment is idxSeg = idxLz*nLx*nLy + idxLy*nLx + idxLx

__device__ curandState states[nspins];

extern "C"
{
    __global__ void initstates(int seed)
    {
        const int spinIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (spinIdx < nspins) 
            curand_init(seed, spinIdx, 0, &states[spinIdx]);
    }

    __device__ float isInsideSphere(float dx,float dy,float dz,float r)
    {
        return dx*dx + dy*dy + dz*dz - r*r;
    }
    
    __device__ float findStepSizeToSphereInsideTol(float dx,float dy,float dz,float r,
                                          float jx,float jy,float jz)
    {

        // if the spin is outside, we can ignore this boundary
        if ((dx*dx + dy*dy + dz*dz - r*r)>0)
            return someLargeNumber;

        // from here on, all spins are inside
        float a = jx*jx + jy*jy + jz*jz;
        float b = 2*(jx*dx + jy*dy + jz*dz);
        float c = dx*dx + dy*dy + dz*dz - (r-tol)*(r-tol);

        float sqterm = b*b-4*a*c;

        // if the sqrt term is negative, there is no collision.
        // That can only occur if (1) the spin is within the radius tolerance
        // and (2) the directon (dx,dy,dz) points tangentially to the sphere.
        // Return 0 and let the routine calculate a new random direction
        if (sqterm < 0) 
            return 0;

        // find both solutions of the quadratic formula. 
        // Choose the stable quadratic formula based on the sign of b
        sqterm = sqrt(sqterm);
        float q = -0.5*(b + sqterm);
        if (b<0) q = -0.5*(b - sqterm);
        float fpos = q/a;
        float fneg = c/q;

        // here there are three options:
        // * 1 postive & 1 negative solution. Jump to the positive
        // * 2 positive solutions. This only occurs if (1) the spin is within 
        //      the radius tolerance and (2) the directon (dx,dy,dz) points 
        //      INTO the sphere. Proceed to the largest positive solution
        // * 2 negative solutions. This only occurs if (1) the spin is within 
        //      the radius tolerance and (2) the directon (dx,dy,dz) points 
        //      AWAY from the sphere. Return 0 to calculate a new random direction
        return MAX(MAX(fpos,fneg),0);
    }

    __device__ float findStepSizeToSphereOutsideTol(float dx,float dy,float dz,float r,
                                          float jx,float jy,float jz)
    {
        float a = jx*jx + jy*jy + jz*jz;
        float b = 2*(jx*dx + jy*dy + jz*dz);
        float c = dx*dx + dy*dy + dz*dz - (r+tol)*(r+tol);

        float sqterm = b*b-4*a*c;

        // if the sqrt term is negative, there is no collision.
        // Proceed as far as allowed
        if (sqterm < 0) 
            return someLargeNumber;

        // find both solutions of the quadratic formula. 
        // Choose the stable quadratic formula based on the sign of b
        // https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        sqterm = sqrt(sqterm);
        float q = -0.5*(b + sqterm);
        if (b<0) q = -0.5*(b - sqterm);
        float fpos = q/a;
        float fneg = c/q;

        if ((dx*dx + dy*dy + dz*dz - r*r)<0)
        {
            // the spin is INSIDE the sphere. Return the only positive solution
            return MAX(fpos,fneg);
        }

        // here, the spin is OUTSIDE the sphere, and there are three options:
        // * 2 positive solutions. Return the smallest positive solution
        // * 2 negative solutions. Return someLargeNumber
        // * 1 postive & 1 negative solution. This only occurs if the spin is 
        //      within the radius tolerance. 
        //      if b>0, the jump points TOWARDS the sphere. return 0
        //      if b<0, the jump points AWAY FROM the sphere. return someLargeNumber
        if (fpos > 0 && fneg > 0) // Both solutions are positive
            return MIN(fpos, fneg);
        else if ((fpos < 0 && fneg < 0) || (b>0))
            return someLargeNumber; 
        // else if (b<0) 
        return 0;
    }

    __global__ void computeDiffusionCoefficientsAndKurtosis(float *spins, float *spins0, 
    /**
    * Kernel to compute second and fourth moments for kurtosis calculation
    */
        float *dx_result, float *dy_result, float *dz_result, 
        float *kx2_result, float *ky2_result, float *kz2_result, 
        float *kx4_result, float *ky4_result, float *kz4_result, 
        int step_idx, int n_spins, float current_time) {
        // Calculate block-level sums using shared memory
        // __shared__ float dx_sum_block[256]; // Assuming block size of 256
        // __shared__ float dy_sum_block[256];
        // __shared__ float dz_sum_block[256];
        __shared__ float dx2_sum_block[256]; // Second moments
        __shared__ float dy2_sum_block[256];
        __shared__ float dz2_sum_block[256];
        __shared__ float dx4_sum_block[256]; // Fourth moments
        __shared__ float dy4_sum_block[256];
        __shared__ float dz4_sum_block[256];
        
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        
        float dx_squared = 0.0f;
        float dy_squared = 0.0f;
        float dz_squared = 0.0f;
        float dx_fourth = 0.0f;
        float dy_fourth = 0.0f;
        float dz_fourth = 0.0f;
        
        if (idx < n_spins) {
            // Compute squared displacements for this spin
            dx_squared = (spins[idx] - spins0[idx]) * (spins[idx] - spins0[idx]);
            dy_squared = (spins[n_spins + idx] - spins0[n_spins + idx]) * (spins[n_spins + idx] - spins0[n_spins + idx]);
            dz_squared = (spins[2*n_spins + idx] - spins0[2*n_spins + idx]) * (spins[2*n_spins + idx] - spins0[2*n_spins + idx]);
        
            dx_fourth = dx_squared * dx_squared;
            dy_fourth = dy_squared * dy_squared;
            dz_fourth = dz_squared * dz_squared;
        }
        
        // Store in shared memory
        dx2_sum_block[tid] = dx_squared;
        dy2_sum_block[tid] = dy_squared;
        dz2_sum_block[tid] = dz_squared;
        dx4_sum_block[tid] = dx_fourth;
        dy4_sum_block[tid] = dy_fourth;
        dz4_sum_block[tid] = dz_fourth;

        __syncthreads();
        
        // Perform parallel reduction to sum values
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                dx2_sum_block[tid] += dx2_sum_block[tid + s];
                dy2_sum_block[tid] += dy2_sum_block[tid + s];
                dz2_sum_block[tid] += dz2_sum_block[tid + s];
                dx4_sum_block[tid] += dx4_sum_block[tid + s];
                dy4_sum_block[tid] += dy4_sum_block[tid + s];
                dz4_sum_block[tid] += dz4_sum_block[tid + s];
            }
            __syncthreads();
        }
        
        // Write block results to global memory
        if (tid == 0) {
            atomicAdd(&dx_result[step_idx], dx2_sum_block[0] / (2.0f * current_time * n_spins));
            atomicAdd(&dy_result[step_idx], dy2_sum_block[0] / (2.0f * current_time * n_spins));
            atomicAdd(&dz_result[step_idx], dz2_sum_block[0] / (2.0f * current_time * n_spins));

            atomicAdd(&kx2_result[step_idx], dx2_sum_block[0]/n_spins);
            atomicAdd(&ky2_result[step_idx], dy2_sum_block[0]/n_spins);
            atomicAdd(&kz2_result[step_idx], dz2_sum_block[0]/n_spins);
            atomicAdd(&kx4_result[step_idx], dx4_sum_block[0]/n_spins);
            atomicAdd(&ky4_result[step_idx], dy4_sum_block[0]/n_spins);
            atomicAdd(&kz4_result[step_idx], dz4_sum_block[0]/n_spins);
        }
    }
    
    __global__ void seedSpinsKernel(
        int nspin,                // Number of spins to generate
        int nstructures,          // Number of structures (including the box)
        float* spheres,           // Array of sphere data [x1,y1,z1,r1,x2,y2,z2,r2,...]
        int* sphere_counts,       // Number of spheres in each structure
        float* positions,         // Output: generated positions [x1...xN, y1...yN, z1...zN]
        int n_struct_idx,         // Number of structure indices to seed in
        int* struct_idxs,         // Array of structure indices to seed in
        float Lx,                 // Domain size in x direction
        float Ly,                 // Domain size in y direction
        float Lz,                 // Domain size in z direction
        unsigned int seed,        // Random seed
        int* seeding_status       // Status (1 = needs seeding, 0 = seeded)
    ) {
        // Each thread handles one spin
        int spinIdx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (spinIdx >= nspin || seeding_status[spinIdx] == 0)
            return;
            
        // Initialize random state for this thread
        curandState localState;
        curand_init(seed, spinIdx, 0, &localState);
    
        // Generate a random position in the box
        float pos_x = (curand_uniform(&localState) - 0.5f) * Lx;
        float pos_y = (curand_uniform(&localState) - 0.5f) * Ly;
        float pos_z = (curand_uniform(&localState) - 0.5f) * Lz;
        
        bool excluded_by_earlier_struct = false;
        bool is_seeded = false;
        
        // Keep track of sphere index as we iterate through structures
        int sphere_idx = 0;
        
        // Check each structure
        for (int struct_idx = 0; struct_idx < nstructures; struct_idx++) {
            int sphere_count = sphere_counts[struct_idx];
            bool in_this_struct = false;
            
            if (struct_idx == nstructures - 1) {
                // This is the box - check if point is in the box
                in_this_struct = (
                    pos_x > -Lx/2 && pos_x < Lx/2 && 
                    pos_y > -Ly/2 && pos_y < Ly/2 && 
                    pos_z > -Lz/2 && pos_z < Lz/2
                );
            } else if (sphere_count > 0) {
                // Check if the point is inside any sphere of this structure
                for (int s = 0; s < sphere_count; s++) {
                    float sphere_x = spheres[sphere_idx + 0];
                    float sphere_y = spheres[sphere_idx + 1];
                    float sphere_z = spheres[sphere_idx + 2];
                    float sphere_r = spheres[sphere_idx + 3];
                    
                    float dx = pos_x - sphere_x;
                    float dy = pos_y - sphere_y;
                    float dz = pos_z - sphere_z;
                    
                    if (dx*dx + dy*dy + dz*dz < sphere_r*sphere_r) {
                        in_this_struct = true;
                        break;  // Found inside this structure
                    }
                    
                    sphere_idx += 4;  // Move to next sphere (x,y,z,r)
                }
            } else {
                sphere_idx += sphere_count * 4;  // Skip empty structures
            }
            
            // Check if this is a target structure
            bool is_target_struct = false;
            for (int i = 0; i < n_struct_idx; i++) {
                if (struct_idxs[i] == struct_idx) {
                    is_target_struct = true;
                    break;
                }
            }
            
            if (is_target_struct) {
                // We want points in this structure
                if (in_this_struct && !excluded_by_earlier_struct) {
                    is_seeded = true;
                }
            } else {
                // We don't want points in this structure
                excluded_by_earlier_struct = excluded_by_earlier_struct || in_this_struct;
            }
        }
        
        // If we found a valid position, store it
        if (is_seeded) {
            positions[spinIdx] = pos_x;
            positions[spinIdx + nspin] = pos_y;
            positions[spinIdx + 2*nspin] = pos_z;
            seeding_status[spinIdx] = 0;  // Mark as seeded
        }
    }

    
    __global__ void randomWalk3d(float dt,float *spheres,int *segmentList,
                                 float *spins,float *spins0,float* sig)
    {   
        // index to the diffusing spin
        const int spinIdx = blockIdx.x*blockDim.x + threadIdx.x;

        // shortcuts for spin properties
        float* x = &spins[0];
        float* y = &spins[nspins];
        float* z = &spins[2*nspins];
        //float* sidx = &spins[3*nspins];

        // shortcuts for sphere properties
        float* cx = &spheres[0];
        float* cy = &spheres[nspheres];
        float* cz = &spheres[2*nspheres];
        float* r = &spheres[3*nspheres];
        float* sphSID = &spheres[4*nspheres];

        float ds;
        float jx;
        float jy;
        float jz;
        float jump_current;
        float jump_remaining = 1.0;

        // find the index to the current segment
        int idxSeg = MIN(nLz-1,MAX(0,floor((z[spinIdx]+Lz/2)/dLz)))*nLx*nLy + 
                     MIN(nLy-1,MAX(0,floor((y[spinIdx]+Ly/2)/dLy)))*nLx +
                     MIN(nLx-1,MAX(0,floor((x[spinIdx]+Lx/2)/dLx)));

        // loop for multiple interactions
        int niter = 0;
        while ((jump_remaining > tol) && (niter<100))
        {
            niter++;
            jump_current = jump_remaining;

            // first, figure out what structure the spin is in
            int spinSID = nstructures;
            for (int n = 0; n<nspheres_per_seg; n++)
            {
                int current_sphere_idx = segmentList[idxSeg*nspheres_per_seg + n];

                if (sphSID[current_sphere_idx] < spinSID)
                {
                    // isInside is negative if the spin is inside the current sphere
                    float isInside = isInsideSphere(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx]);

                    if (isInside < 0)
                    { // this new sphere takes precedence
                        spinSID = sphSID[current_sphere_idx];
                    }
                }
            }

            if (niter == 1)
            {
                ds = sqrt(2*D[spinSID]*dt);
                jx = ds*curand_normal(&states[spinIdx]);
                jy = ds*curand_normal(&states[spinIdx]);
                jz = ds*curand_normal(&states[spinIdx]);
            }

            float isInside;
            float isInsideAny = 0;
            float furthest_fstep_in = 0;
            float closest_fstep_out = someLargeNumber;

            // loop thru all the spheres in the segment to check for interactions
            for (int n=0; n<nspheres_per_seg; n++)
            {
                int current_sphere_idx = segmentList[idxSeg*nspheres_per_seg + n];

                if (sphSID[current_sphere_idx] > spinSID)
                    continue;
                    
                // isInside is negative if the spin is inside the current sphere
                isInside = isInsideSphere(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx]);

                if (isInside < 0) // the spin is inside this sphere
                {
                    isInsideAny = 1;
                    float fstep_in = findStepSizeToSphereInsideTol(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx],
                        jx,jy,jz);

                    if (furthest_fstep_in < fstep_in)
                        furthest_fstep_in = fstep_in;
                }
                else if (sphSID[current_sphere_idx] < spinSID) 
                {
                    // the spin is outside this sphere
                    // stay outside of this sphere
                    float fstep_out = findStepSizeToSphereOutsideTol(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx],
                        jx,jy,jz);

                    if (fstep_out < closest_fstep_out)
                        closest_fstep_out = fstep_out;
                }
            }

            jump_current = MIN(jump_current,isInsideAny? furthest_fstep_in : someLargeNumber);
            jump_current = MIN(jump_current,closest_fstep_out);

            x[spinIdx] += jump_current*jx;
            y[spinIdx] += jump_current*jy;
            z[spinIdx] += jump_current*jz;

            if (jump_current < jump_remaining)
            {
                // the direction is incoherent
                // variance is linear with diffusion time, <x^2> = 2Dt
                // therefore, st dev goes with the sqrt of diffusion time sqrt(t)
                // scale jump_remaining appropriately
                jump_remaining = sqrt(MAX(0,jump_remaining*jump_remaining - 
                                            jump_current*jump_current));

                sig[spinIdx] *= exp(-jump_current*jump_current*dt/T2[spinSID]);

                // scatter at the boundary
                jx = ds*curand_normal(&states[spinIdx]);
                jy = ds*curand_normal(&states[spinIdx]);
                jz = ds*curand_normal(&states[spinIdx]);
            }
            else 
            {
                // not scattering, so the direction is coherent; 
                // continue as normal
                jump_remaining -= jump_current;
                sig[spinIdx] *= exp(-jump_current*dt/T2[spinSID]);
            }
        }

        //enforce periodic boundary conditions in the arena
        if (x[spinIdx] < -Lx/2)
        {
            x[spinIdx] += Lx;
            spins0[spinIdx] += Lx;
        }
        else if (x[spinIdx] > Lx/2)
        {
            x[spinIdx] -= Lx;
            spins0[spinIdx] -= Lx;
        }

        if (y[spinIdx] < -Ly/2)
        {
            y[spinIdx] += Ly;
            spins0[nspins+spinIdx] += Ly;
        }
        else if (y[spinIdx] > Ly/2)
        {
            y[spinIdx] -= Ly;
            spins0[nspins+spinIdx] -= Ly;
        }

        if (z[spinIdx] < -Lz/2)
        {
            z[spinIdx] += Lz;
            spins0[2*nspins+spinIdx] += Lz;
        }
        else if (z[spinIdx] > Lz/2)
        {
            z[spinIdx] -= Lz;
            spins0[2*nspins+spinIdx] -= Lz;
        }
    }

    __global__ void randomWalk3d_phase(float dt,float *spheres,int *segmentList,
                                 float *spins,float *spins0, float* phase_accumulated, 
                                 float G, float G_area_at_this_time_step)
    {   
        // index to the diffusing spin
        const int spinIdx = blockIdx.x*blockDim.x + threadIdx.x;

        // shortcuts for spin properties
        float* x = &spins[0];
        float* y = &spins[nspins];
        float* z = &spins[2*nspins];
        //float* sidx = &spins[3*nspins];

        // shortcuts for sphere properties
        float* cx = &spheres[0];
        float* cy = &spheres[nspheres];
        float* cz = &spheres[2*nspheres];
        float* r = &spheres[3*nspheres];
        float* sphSID = &spheres[4*nspheres];

        float ds;
        float jx;
        float jy;
        float jz;
        float jump_current;
        float jump_remaining = 1.0;

        // find the index to the current segment
        int idxSeg = MIN(nLz-1,MAX(0,floor((z[spinIdx]+Lz/2)/dLz)))*nLx*nLy + 
                     MIN(nLy-1,MAX(0,floor((y[spinIdx]+Ly/2)/dLy)))*nLx +
                     MIN(nLx-1,MAX(0,floor((x[spinIdx]+Lx/2)/dLx)));

        // loop for multiple interactions
        int niter = 0;
        while ((jump_remaining > tol) && (niter<100))
        {
            niter++;
            jump_current = jump_remaining;

            // first, figure out what structure the spin is in
            int spinSID = nstructures;
            for (int n = 0; n<nspheres_per_seg; n++)
            {
                int current_sphere_idx = segmentList[idxSeg*nspheres_per_seg + n];

                if (sphSID[current_sphere_idx] < spinSID)
                {
                    // isInside is negative if the spin is inside the current sphere
                    float isInside = isInsideSphere(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx]);

                    if (isInside < 0)
                    { // this new sphere takes precedence
                        spinSID = sphSID[current_sphere_idx];
                    }
                }
            }

            if (niter == 1)
            {
                ds = sqrt(2*D[spinSID]*dt);
                jx = ds*curand_normal(&states[spinIdx]);
                jy = ds*curand_normal(&states[spinIdx]);
                jz = ds*curand_normal(&states[spinIdx]);
            }

            float isInside;
            float isInsideAny = 0;
            float furthest_fstep_in = 0;
            float closest_fstep_out = someLargeNumber;

            // loop thru all the spheres in the segment to check for interactions
            for (int n=0; n<nspheres_per_seg; n++)
            {
                int current_sphere_idx = segmentList[idxSeg*nspheres_per_seg + n];

                if (sphSID[current_sphere_idx] > spinSID)
                    continue;
                    
                // isInside is negative if the spin is inside the current sphere
                isInside = isInsideSphere(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx]);

                if (isInside < 0) // the spin is inside this sphere
                {
                    isInsideAny = 1;
                    float fstep_in = findStepSizeToSphereInsideTol(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx],
                        jx,jy,jz);

                    if (furthest_fstep_in < fstep_in)
                        furthest_fstep_in = fstep_in;
                }
                else if (sphSID[current_sphere_idx] < spinSID) 
                {
                    // the spin is outside this sphere
                    // stay outside of this sphere
                    float fstep_out = findStepSizeToSphereOutsideTol(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx],
                        jx,jy,jz);

                    if (fstep_out < closest_fstep_out)
                        closest_fstep_out = fstep_out;
                }
            }

            jump_current = MIN(jump_current,isInsideAny? furthest_fstep_in : someLargeNumber);
            jump_current = MIN(jump_current,closest_fstep_out);

            x[spinIdx] += jump_current*jx;
            y[spinIdx] += jump_current*jy;
            z[spinIdx] += jump_current*jz;

            if (jump_current < jump_remaining)
            {
                // the direction is incoherent
                // variance is linear with diffusion time, <x^2> = 2Dt
                // therefore, st dev goes with the sqrt of diffusion time sqrt(t)
                // scale jump_remaining appropriately
                jump_remaining = sqrt(MAX(0,jump_remaining*jump_remaining - 
                                            jump_current*jump_current));

                // sig[spinIdx] *= exp(-jump_current*jump_current*dt/T2[spinSID]);

                // scatter at the boundary
                jx = ds*curand_normal(&states[spinIdx]);
                jy = ds*curand_normal(&states[spinIdx]);
                jz = ds*curand_normal(&states[spinIdx]);
            }
            else 
            {
                // not scattering, so the direction is coherent; 
                // continue as normal
                jump_remaining -= jump_current;
                // sig[spinIdx] *= exp(-jump_current*dt/T2[spinSID]);
            }
        }
        
        phase_accumulated[spinIdx] += x[spinIdx]*G;
        phase_accumulated[spinIdx + nspins] += y[spinIdx]*G;
        phase_accumulated[spinIdx + 2*nspins] += z[spinIdx]*G;

        //enforce periodic boundary conditions in the arena
        if (x[spinIdx] < -Lx/2)
        {
            x[spinIdx] += Lx;
            spins0[spinIdx] += Lx;
            phase_accumulated[spinIdx] += Lx*G_area_at_this_time_step;
        }
        else if (x[spinIdx] > Lx/2)
        {
            x[spinIdx] -= Lx;
            spins0[spinIdx] -= Lx;
            phase_accumulated[spinIdx] -= Lx*G_area_at_this_time_step;
        }

        if (y[spinIdx] < -Ly/2)
        {
            y[spinIdx] += Ly;
            spins0[nspins+spinIdx] += Ly;
            phase_accumulated[spinIdx + nspins] += Ly*G_area_at_this_time_step;
        }
        else if (y[spinIdx] > Ly/2)
        {
            y[spinIdx] -= Ly;
            spins0[nspins+spinIdx] -= Ly;
            phase_accumulated[spinIdx + nspins] -= Ly*G_area_at_this_time_step;
        }

        if (z[spinIdx] < -Lz/2)
        {
            z[spinIdx] += Lz;
            spins0[2*nspins+spinIdx] += Lz;
            phase_accumulated[spinIdx + 2*nspins] += Lz*G_area_at_this_time_step;
        }
        else if (z[spinIdx] > Lz/2)
        {
            z[spinIdx] -= Lz;
            spins0[2*nspins+spinIdx] -= Lz;
            phase_accumulated[spinIdx + 2*nspins] -= Lz*G_area_at_this_time_step;
        }
    }

    __global__ void randomWalk3d_phase_multidirr(float dt,float *spheres,int *segmentList,
                                 float *spins,float *spins0, float* phase_accumulated, 
                                 float G, float G_area_at_this_time_step, 
                                 float* diffdir, int ndiffdir)
    {   
        // index to the diffusing spin
        const int spinIdx = blockIdx.x*blockDim.x + threadIdx.x;

        // shortcuts for spin properties
        float* x = &spins[0];
        float* y = &spins[nspins];
        float* z = &spins[2*nspins];
        //float* sidx = &spins[3*nspins];

        // shortcuts for sphere properties
        float* cx = &spheres[0];
        float* cy = &spheres[nspheres];
        float* cz = &spheres[2*nspheres];
        float* r = &spheres[3*nspheres];
        float* sphSID = &spheres[4*nspheres];
        
        // convenient accessors (3 x ndiffdir, row-major), d defines the d-th direction
        // diffdir is shape 3xN
        #define DIRX(d) diffdir[(0)*ndiffdir + (d)]
        #define DIRY(d) diffdir[(1)*ndiffdir + (d)]
        #define DIRZ(d) diffdir[(2)*ndiffdir + (d)]

        float ds;
        float jx;
        float jy;
        float jz;
        float jump_current;
        float jump_remaining = 1.0;

        // find the index to the current segment
        int idxSeg = MIN(nLz-1,MAX(0,floor((z[spinIdx]+Lz/2)/dLz)))*nLx*nLy + 
                     MIN(nLy-1,MAX(0,floor((y[spinIdx]+Ly/2)/dLy)))*nLx +
                     MIN(nLx-1,MAX(0,floor((x[spinIdx]+Lx/2)/dLx)));

        // loop for multiple interactions
        int niter = 0;
        while ((jump_remaining > tol) && (niter<100))
        {
            niter++;
            jump_current = jump_remaining;

            // first, figure out what structure the spin is in
            int spinSID = nstructures;
            for (int n = 0; n<nspheres_per_seg; n++)
            {
                int current_sphere_idx = segmentList[idxSeg*nspheres_per_seg + n];

                if (sphSID[current_sphere_idx] < spinSID)
                {
                    // isInside is negative if the spin is inside the current sphere
                    float isInside = isInsideSphere(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx]);

                    if (isInside < 0)
                    { // this new sphere takes precedence
                        spinSID = sphSID[current_sphere_idx];
                    }
                }
            }

            if (niter == 1)
            {
                ds = sqrt(2*D[spinSID]*dt);
                jx = ds*curand_normal(&states[spinIdx]);
                jy = ds*curand_normal(&states[spinIdx]);
                jz = ds*curand_normal(&states[spinIdx]);
            }

            float isInside;
            float isInsideAny = 0;
            float furthest_fstep_in = 0;
            float closest_fstep_out = someLargeNumber;

            // loop thru all the spheres in the segment to check for interactions
            for (int n=0; n<nspheres_per_seg; n++)
            {
                int current_sphere_idx = segmentList[idxSeg*nspheres_per_seg + n];

                if (sphSID[current_sphere_idx] > spinSID)
                    continue;
                    
                // isInside is negative if the spin is inside the current sphere
                isInside = isInsideSphere(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx]);

                if (isInside < 0) // the spin is inside this sphere
                {
                    isInsideAny = 1;
                    float fstep_in = findStepSizeToSphereInsideTol(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx],
                        jx,jy,jz);

                    if (furthest_fstep_in < fstep_in)
                        furthest_fstep_in = fstep_in;
                }
                else if (sphSID[current_sphere_idx] < spinSID) 
                {
                    // the spin is outside this sphere
                    // stay outside of this sphere
                    float fstep_out = findStepSizeToSphereOutsideTol(
                        x[spinIdx]-cx[current_sphere_idx],
                        y[spinIdx]-cy[current_sphere_idx],
                        z[spinIdx]-cz[current_sphere_idx],
                        r[current_sphere_idx],
                        jx,jy,jz);

                    if (fstep_out < closest_fstep_out)
                        closest_fstep_out = fstep_out;
                }
            }

            jump_current = MIN(jump_current,isInsideAny? furthest_fstep_in : someLargeNumber);
            jump_current = MIN(jump_current,closest_fstep_out);

            x[spinIdx] += jump_current*jx;
            y[spinIdx] += jump_current*jy;
            z[spinIdx] += jump_current*jz;

            if (jump_current < jump_remaining)
            {
                // the direction is incoherent
                // variance is linear with diffusion time, <x^2> = 2Dt
                // therefore, st dev goes with the sqrt of diffusion time sqrt(t)
                // scale jump_remaining appropriately
                jump_remaining = sqrt(MAX(0,jump_remaining*jump_remaining - 
                                            jump_current*jump_current));

                // sig[spinIdx] *= exp(-jump_current*jump_current*dt/T2[spinSID]);

                // scatter at the boundary
                jx = ds*curand_normal(&states[spinIdx]);
                jy = ds*curand_normal(&states[spinIdx]);
                jz = ds*curand_normal(&states[spinIdx]);
            }
            else 
            {
                // not scattering, so the direction is coherent; 
                // continue as normal
                jump_remaining -= jump_current;
                // sig[spinIdx] *= exp(-jump_current*dt/T2[spinSID]);
            }
        }
        
        // phase_accumulated[spinIdx] += x[spinIdx]*G;
        // phase_accumulated[spinIdx + nspins] += y[spinIdx]*G;
        // phase_accumulated[spinIdx + 2*nspins] += z[spinIdx]*G;
        // Project r = (x,y,z) onto each diffusion direction and accumulate
        const float px = x[spinIdx];
        const float py = y[spinIdx];
        const float pz = z[spinIdx];
        for (int d = 0; d < ndiffdir; ++d) {
            // dot product between spin displacement & unit vector defining the gradient direction
            const float proj = px*DIRX(d) + py*DIRY(d) + pz*DIRZ(d); 
            phase_accumulated[spinIdx + d*nspins] += proj * G;
        }
        
        //enforce periodic boundary conditions in the arena
        
        // G(t) here is scaled by gamma
        // phase = ∫ G(t) * (wrapped_position(t)) * dt
        // so when a spin crosses a periodic boundary, 
        // we need to add/subtract the contribution of the area swept by the spin as 
        // it crosses the boundary to the accumulated phase. 

        // If a spin crosses periodic boundaries, stored position jumps by ±L (wrapped), 
        // but the physical path is continuous (unwrapped).
        // So wrapped position differs from unwrapped by an integer box shift:
            
        // wrapped_position = unwrapped_position + L
        
        // Plugging this into phase gives a missing term:

        // phase = ∫ G(t) * (unwrapped_position(t) + L) * dt
        //       = ∫ G(t) * unwrapped_position(t) * dt + L * ∫ G(t) * dt
        //       = (original phase calculation) + L * G_area_at_this_time_step

        // where G_area_at_this_time_step is the integral of G(t) over the time step during which the crossing occurs.
        
        // X periodic boundaries
        if (x[spinIdx] < -Lx/2)
        {
            x[spinIdx] += Lx;
            spins0[spinIdx] += Lx;
            for (int d = 0; d < ndiffdir; ++d) {
                phase_accumulated[spinIdx + d*nspins] += Lx * DIRX(d) * G_area_at_this_time_step;
            }
        }
        else if (x[spinIdx] > Lx/2)
        {
            x[spinIdx] -= Lx;
            spins0[spinIdx] -= Lx;
            for (int d = 0; d < ndiffdir; ++d) {
                phase_accumulated[spinIdx + d*nspins] -= Lx * DIRX(d) * G_area_at_this_time_step;
            }
        }

        // Y periodic boundaries
        if (y[spinIdx] < -Ly/2)
        {
            y[spinIdx] += Ly;
            spins0[nspins+spinIdx] += Ly;
            for (int d = 0; d < ndiffdir; ++d) {
                phase_accumulated[spinIdx + d*nspins] += Ly * DIRY(d) * G_area_at_this_time_step;
            }
        }
        else if (y[spinIdx] > Ly/2)
        {
            y[spinIdx] -= Ly;
            spins0[nspins+spinIdx] -= Ly;
            for (int d = 0; d < ndiffdir; ++d) {
                phase_accumulated[spinIdx + d*nspins] -= Ly * DIRY(d) * G_area_at_this_time_step;
            }
        }

        // Z periodic boundaries
        if (z[spinIdx] < -Lz/2)
        {
            z[spinIdx] += Lz;
            spins0[2*nspins+spinIdx] += Lz;
            for (int d = 0; d < ndiffdir; ++d) {
                phase_accumulated[spinIdx + d*nspins] += Lz * DIRZ(d) * G_area_at_this_time_step;
            }
        }
        else if (z[spinIdx] > Lz/2)
        {
            z[spinIdx] -= Lz;
            spins0[2*nspins+spinIdx] -= Lz;
            for (int d = 0; d < ndiffdir; ++d) {
                phase_accumulated[spinIdx + d*nspins] -= Lz * DIRZ(d) * G_area_at_this_time_step;
            }
        }
    }
}
