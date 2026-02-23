# 1) Molecule-Membrane Interaction Simulation

Molecule-Membrane Interaction search is done by computing collision distances between the molecule jump trajectory and spheres, and identifying the sphere that the molecule interacts with. In a segment, a molecule’s 3D position relative to a sphere, defined as $\textbf{p}_\textbf{jump}$, is parameterized as

$\textbf{p}_\textbf{jump}=\textbf{p}_\textbf{0}+f\cdot \textbf{j},$

where $\textbf{p}_\textbf{0}=(\delta x,\delta y,\delta z)$ is the initial distance vector between the molecule and the center of the sphere in $x, y, z$. $\textbf{j}=(j_x,j_y,j_z)$ is the vector of the jump, and $f∈[0,1]$ represents the fraction of the jump to be used. To find where the molecule intersects the sphere, an analytic ray-sphere intersection equation was used

$(δx+f\cdot j_x )^2+(δy+f\cdot j_y )^2+(δz+f\cdot j_z )^2=(R±tol)^2.$

Expanding and rearranging, we arrive at

$A\cdot f^2+2B \cdot f+C=0,$

where

$(A=j_x^2+j_y^2+j_z^2,)$

$(B=δx\cdot j_x+δy\cdot j_y+δz\cdot j_z,)$

$(C=(δx)^2+(δy)^2+(δz)^2-(R±tol)^2.)$

The jump fraction $f$ is solved using Muller's method to account for numerical precision and avoid catastrophic cancellation<sup>1</sup>. A tolerance around the radius $R$ is added to prevent leakage of molecules between compartments caused by floating-point precision. Two distinct tolerance values are used: $(R-tol)$ for spins inside spheres, and $(R+tol)$ for those outside.

Note: Alternative to our multi-step collision and membrane scattering approach, we also acknowledge that molecule-membrane interaction has been modeled by rejection sampling<sup>2,3,4,5</sup>, where a step encountering a membrane is canceled, and the molecule stays still for the step. However, bias in $D_\parallel$ was shown when steps toward the membrane are rejected<sup>6</sup>. Our collision-scattering method preserves Brownian statistics by truncating steps at boundaries, and generating new random directions with variance-corrected remaining jump distance.

# 2) Numerically Stable Quadratic Equation Solving
Collision detection was obtained from solving the quadratic equation for sphere-ray intersections. 

### Problem: 
Collision detection requires solving: 
```bash
ax² + bx + c = 0 where:
a = jx² + jy² + jz² (squared jump magnitude)
b = 2(jx·dx + jy·dy + jz·dz) (dot product terms)
c = dx² + dy² + dz² - r² (distance to sphere surface)

### Naive Implementation Issues
// UNSTABLE - suffers from catastrophic cancellation
float discriminant = sqrt(b*b - 4*a*c);
float x1 = (-b + discriminant) / (2*a);
float x2 = (-b - discriminant) / (2*a);
```
But, standard formula `(-b ± √(b²-4ac))/2a` suffers from catastrophic cancellation when `b²≫4ac`

### Approach: 
Muller's method was used to avoid catastrophic cancellation:
* The stable version always adds terms of the same sign in the numerator
* Uses Vieta's formulas (x₁x₂ = c/a) to get the second root

```bash
float sqterm = sqrt(b*b - 4*a*c);
float q = -0.5 * (b + copysign(sqterm, b));
float x1 = q / a;
float x2 = c / q;
```
# 2) Tolerance Handling Around Sphere Boundaries
### Problem:
If molecule's step ends exactly on boundary, adding a tolerance prevents particles from getting "stuck" exactly on boundaries and to prevent leakage of molecules between compartments caused by floating-point precision. 

### Approach: 
Create a "transition zone" of width `2 × tol` around each sphere
* Inner tolerance: `r-tol`
* Outer tolerance: `r+tol`

#### Tolerance Selection Rationale
A 32-bit single-precision float has ~7 decimal digits.
Our tolerance `tol = 1e-4` provides safety margin while avoiding excessive buffer zones.

This tolerance buffer around each sphere was implemented as:
```bash
const float tol = 1e-4;  // Global tolerance

// Inside sphere: collision with (r - tol)
float c = dx*dx + dy*dy + dz*dz - (r-tol)*(r-tol);

// Outside sphere: collision with (r + tol) 
float c = dx*dx + dy*dy + dz*dz - (r+tol)*(r+tol);
```

## References
1.	Accuracy and Stability of Numerical Algorithms | SIAM Publications Library. Other Titles in Applied Mathematics. Accessed October 8, 2025. https://epubs.siam.org/doi/book/10.1137/1.9780898718027
2. Ford JC, Hackney DB. Numerical model for calculation of apparent diffusion coefficients (ADC) in permeable cylinders—comparison with measured ADC in spinal cord white matter. Magnetic Resonance in Medicine. 1997;37(3):387-394. doi:10.1002/mrm.1910370315
3. Nguyen KV, Hernández-Garzón E, Valette J. Efficient GPU-based Monte-Carlo simulation of diffusion in real astrocytes reconstructed from confocal microscopy. Journal of Magnetic Resonance. 2018;296:188-199. doi:10.1016/j.jmr.2018.09.013
4. Palombo M, Ligneul C, Hernandez-Garzon E, Valette J. Can we detect the effect of spines and leaflets on the diffusion of brain intracellular metabolites? NeuroImage. 2018;182:283-293. doi:10.1016/j.neuroimage.2017.05.003
5. Waudby CA, Christodoulou J. GPU accelerated Monte Carlo simulation of pulsed-field gradient NMR experiments. Journal of Magnetic Resonance. 2011;211(1):67-73. doi:10.1016/j.jmr.2011.04.004
6. Lee HH, Fieremans E, Novikov DS. Realistic Microstructure Simulator (RMS): Monte Carlo simulations of diffusion in three-dimensional cell segmentations of microscopy images. J Neurosci Methods. 2021;350:109018. doi:10.1016/j.jneumeth.2020.109018