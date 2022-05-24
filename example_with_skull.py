import numpy as np
from simulation_function import makeSimulation

# make simulaiton class and save data directory
simul = makeSimulation("simulation result")

# Set transducer spec
simul.source_freq = 25e4
simul.ROC = 99
simul.width = 95
simul.focal_length = 85

# Set sonication condition
simul.end_time = 100e-6
simul.points_per_wavelength = np.pi*2
simul.CFL = 0.1

# set target and transducer placement
target = [-15.538, 7.803, 22.017] # thalamus
tran_pos = [-15.538-85, 7.803, 22.017]

# Pre-process (Crop and resampling)
simul.preprocessing_rotate('Test Data\\Skull.nii', tran_pos, target)  #(skull image path, target location)

# Make transducer
simul.make_transducer(tran_pos)

# Run simulation
simul.recording = False
simul.run_simulation()