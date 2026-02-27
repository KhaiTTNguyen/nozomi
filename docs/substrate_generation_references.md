# Substrate Generation Details
## 1) Beading design

Bead amplitudes $R_\mathrm{amp}$ are sampled from a Normal $\mathcal{N}\left(\mu_{R_\mathrm{amp}},\sigma_{R_\mathrm{amp}}^2\right)$. Bead spacings along the length of the axon are drawn from a $Lognormal\ \left(\mu_\mathrm{spacing},\ \sigma_{\mathrm{spacing}}^2\right)$, and are used to determine bead locations along the axon, $l_\mathrm{bead}$. The radii of spheres in a beaded axon, $R_{\text{bead axon}}$, are computed by modifying the pre-bead radii of spheres $R_\mathrm{axon}$ at bead locations $l_\mathrm{bead}$, and smoothing of the axon radii local to the bead region using spheres’ positions along the axon $l$ and bead spatial spread $\sigma_\mathrm{bead}$, defined as

$R_{\text{bead axon}}=\ R_\mathrm{axon}+R_\mathrm{amp}\cdot\frac{1}{\sigma_\mathrm{bead}\sqrt{2π}}\cdot\exp\left(-\frac{1}{2}\cdot \frac{(l - l_\mathrm{bead})^2}{\sigma_\mathrm{bead}^2} \right).$


## 2) Substrate Optimization Design
### a) 2D Initialization
Overlaps disks in 2D initialization are removed by minimizing the following cost function

$\text{argmin}_{x_i,y_i}\sum_{k=1}^{N_\mathrm{overlaps}}\alpha_k^2$

$\alpha_k=\mathrm{min}\left(0, R_i  + R_j - \sqrt{(x_i-x_j )^2+(y_i-y_j )^2}  +l_\mathrm{min} \right);i,j∈N,i≠j $

where $N_\mathrm{overlaps}$ is the number of overlaps; $\alpha_k$ is the overlap distance between any two disks $i,j$; $x_i,y_i,R_i$ are the position in $x,y$ and the radius of each disk; $l_\mathrm{min}$ is the minimum distance between any two disks.

### b) 3D Geometric Optimization
After meshing, overlaps between spheres of different axons are removed with an L-BFGS procedure that balances overlap removal, curvature regularization, and length preservation. The total cost function is

${\mathrm{argmin}}_{x_i,y_i,z_i}[w_{\mathrm{overlap}}*\mathrm{F}_{\mathrm{overlap}} + w_\mathrm{curve}*\mathrm{F}_\mathrm{curve}+w_\mathrm{length}*\mathrm{F}_\mathrm{length}]$

where $w_\mathrm{overlap}, w_\mathrm{curve}, w_\mathrm{length}$ are the weights of each component cost terms. The overlap component cost term is defined as

$\text{F}_\mathrm{overlap}=\frac{R_i R_j}{\sqrt{N_\mathrm{spheres}}}
\sum_{k=1}^{N_\mathrm{overlaps}}\left(\frac{\mathrm{Overlap}(S_i^a,S_j^b )}{R_i+R_j}\right)^2$

$\text{Overlap}(S_i^a,S_j^b)=\mathrm{max}\left(R_i+R_j-|S_i^a-S_j^b|+l_\mathrm{min},0 \right)$

where $S_i^a,S_j^b$ are the 3D position vectors for the $i,j$ overlapping spheres of different fibers $\textbf{\textit{a}}$ and $\textbf{\textit{b}}$, $S_i^a=(x_i^a,y_i^a,z_i^a)$. The curvature component cost term is defined as

$\text{F_\mathrm{curve}}=\sum_{a=1}^{N_\mathrm{fibers}}[\sum_{i=1}^{m_a-2}f_\mathrm{curve}] $

$f_\mathrm{curve}=\frac{1}{3\overline{R^a}} \left(1-\frac{v_i^a \cdot v_{i+1}^a}{|v_i^a||v_{i+1}^a|}\right)^2;v_i^a=S_{i+1}^a-S_i^a;i=1,2,3,..,m_a $

where $m_a$ is the number of spheres in fiber $\textbf{\textit{a}}$; $\overline{R^a}$ is the mean radius of fiber $\textbf{\textit{a}}$. The length component cost term is defined as

$\text{F}_\mathrm{length}=\sum_{a=1}^{N_\mathrm{fibers}}\left[\sum_{i=1}^{m_{a-1}}f_\mathrm{length} \right]$



$f_\mathrm{length}=\frac{1}{\overline{R^a}}\left(|v_i^a|-L_a \right)^2;L_a=\frac{|S_1^a-S_{m_a}^a|}{m_a}$

where $L_a$ defines the average sphere spacing between the endpoints of fiber $\textbf{\textit{a}}$.
