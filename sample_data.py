import numpy as np
import slotine_Li_regressor as reg

def generate_sample_data(num_samples=1000):
    true_pi = np.array([
        1.0,    # mass
        0.1, 0.1, 0.1,      # first moments (hx, hy, hz)
        0.2, 0.2, 0.2,      # diagonal inertia (Ixx, Iyy, Izz)
        0.05, 0.05, 0.05    # off-diagonal inertia (Ixy, Ixz, Iyz)
    ])

    samples = []
    for _ in range(num_samples):
        accel = np.random.normal(0, 1, 6)
        vel = np.random.normal(0, 1, 6)
        Y = reg.compute_regressor(accel, vel)
        force = Y @ true_pi
        samples.append({'Y': Y, 'force': force})
    return samples, true_pi