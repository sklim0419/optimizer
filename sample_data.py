import numpy as np
import slotine_Li_regressor as reg

def generate_sample_data(true_pi, num_samples=1000):
    samples = []
    for _ in range(num_samples):
        accel = np.random.normal(-1, 1, 6)
        vel = np.random.normal(-1, 1, 6)
        Y = reg.compute_regressor(accel, vel)
        force = Y @ true_pi
        samples.append({'Y': Y, 'force': force})
    return samples