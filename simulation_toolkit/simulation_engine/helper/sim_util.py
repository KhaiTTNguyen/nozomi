import numpy as np

def validate_compartment_intra(sim, sg3):
    isInside = np.zeros(sim.nspins, dtype=bool)
    for structure in sg3.structures[:-1]:
        isInside = np.logical_or(isInside, structure.isinside(sim.spins_d.get()))
    return isInside.all()

def validate_compartment_extra(sim, sg3):
    isOutside = np.zeros(sim.nspins, dtype=bool)
    for structure in sg3.structures[:-1]:
        isOutside = np.logical_or(isOutside, structure.isinside(sim.spins_d.get()))
    isOutside = np.invert(isOutside)
    return isOutside.all()