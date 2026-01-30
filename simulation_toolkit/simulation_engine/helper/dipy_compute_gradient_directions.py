"""
This is possible using the ``disperse_charges`` which is an implementation of
electrostatic repulsion :footcite:t:`Jones1999` .
"""
import numpy as np
import os
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, Sphere, disperse_charges

OUTPUT_FOLDER_PATH="../../../../HIPASimExperiment/biophysical_sim/NODDI/"

b_value=[1000, 3000]
num_directions = 75

rng = np.random.default_rng()
n_pts = num_directions
theta = np.pi * rng.random(n_pts)
phi = 2 * np.pi * rng.random(n_pts)
hsph_initial = HemiSphere(theta=theta, phi=phi)

###############################################################################
# Next, we call ``disperse_charges`` which will iteratively move the points so
# that the electrostatic potential energy is minimized.

hsph_updated, potential = disperse_charges(hsph_initial, 5000)

vertices = hsph_updated.vertices
values = np.ones(vertices.shape[0])

###############################################################################
bvecs = np.vstack((vertices, vertices))
bvals = np.hstack((b_value[0] * values, b_value[1] * values))

###############################################################################
# We can also add some b0s. Let's add one at the beginning and one at the end.

bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
bvals = np.insert(bvals, (0, bvals.shape[0]), 0)

# save data
transposed = bvecs.T

# Extract x, y, z components
x_values = transposed[0]
y_values = transposed[1]
z_values = transposed[2]

# Write to bvec file format
if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)
with open(OUTPUT_FOLDER_PATH+'/DWI.bvec', 'w') as f:
    # Write x-values on first. second, third line
    f.write(' '.join(map(str, x_values)) + '\n')
    f.write(' '.join(map(str, y_values)) + '\n')
    f.write(' '.join(map(str, z_values)) + '\n')

with open(OUTPUT_FOLDER_PATH+'/DWI.bval', 'w') as f:
    f.write(' '.join(map(str, bvals)) + '\n')
