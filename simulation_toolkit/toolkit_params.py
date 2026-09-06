OUTPUT_FOLDER_PATH="./experiment/result/"
EXPERIMENT_FOLDER=""
SUBSTRATE_OUTPUT_FOLDER_PATH=""
VALIDATION_TEST_FOLDER_PATH=""
NUM_MOL_TIMESTEP_CALIBRATION_FOLDER_PATH=""
NUM_SEGMENT_CALIBRATION_FOLDER_PATH=""


NUM_NODES_FOR_IVF_CALCULATION = int(5e3)

BOX_LENGTH = 0
# In-plane box size is BOX_LENGTH (Lx=Ly). BOX_LENGTH_Z is the (thin) z-height Lz.
# 0 => auto-compute from the diffusion-length rule Lz = ceil(2*sqrt(2*D0*t)).
BOX_LENGTH_Z = 0
# Reject near-in-plane fibers (|dz| below this) when sampling Watson directions.
# cos(85 deg) ~= 0.08716; bounds helix arc-length blow-up at low orientation K.
ORIENTATION_DZ_FLOOR = 0.0871557

EXP_DATE_TIME = 0
W_OVERLAP = 0
W_CURVE = 0
W_LENGTH = 0
VOLUME_FRACTION = 0
ORIENTATION_SHAPE_PARAM = 0
ODI_INDEX = 0
NUM_FIBERS = 0
MEAN_DIAMETER = 0
SIGMA_DIAMETER = 0
DISTRIBUTION_SHAPE=0
SPHERE_SPACING = 0
BEAD_SPACING_MEAN = 0
BEAD_SPACING_STDV = 0
BEAD_ALPHA_MEAN = 0
BEAD_ALPHA_STDV = 0

SPACE_BUFFER_REPULSE = 0
SPACE_BUFFER_STARTS_ENDS = 0
CV_OUTER_MEAN = 0
CV_OUTER_STDV = 0
CV_RADII = 0
GEV_DIAMETER_MEAN = 0
GEV_DIAMETER_STDV = 0

# Shared axis limits for the CV and diameter-distribution histograms so every
# substrate is plotted on identical axes. None => fall back to per-plot auto/
# default ranges. Populated (as (min, max) tuples) from a data-driven limits
# JSON computed by cli.reprocess_substrate_stats over the whole dataset.
CV_DIAMETER_XLIM = None
CV_DIAMETER_YLIM = None
DIAMETER_DIST_XLIM = None
DIAMETER_DIST_YLIM = None
# Path to a data-driven shared-axis-limits JSON (written by
# cli.reprocess_substrate_stats). When set and present, substrate_main loads it
# so newly generated substrates share the batch axes. None => per-plot defaults.
SUBSTRATE_STATS_AXIS_LIMITS_FILE = None

SIM_CONFIG_FILE = './experiment/setup/simulation/default-sim.json'