import numpy as np
from simulation_function import makeSimulation

# make simulaiton class and save data directory
simul = makeSimulation("simulation result free water")

# Set transducer spec
simul.source_freq = 25e4
simul.ROC = 99
simul.width = 95
simul.focal_length = 85

# Set sonication condition
simul.end_time = 100e-6
simul.points_per_wavelength = np.pi*2
simul.CFL = 0.1

# Run simulation
simul.recording = False
simul.free_water_run_simulation()