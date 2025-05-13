import numpy as np
import log_cholesky as lch
import slotine_Li_regressor as Y
import gradient_descent as gd
import sample_data as data

if __name__ == "__main__":
    samples, true_pi = data.generate_sample_data()
    est_pi = gd.estimate_inertial_parameters(samples, learning_rate=1e-4, max_iter=1000,numerical=False)
    print("\n[True Inertial Parameters]\n", true_pi)
    print("\n[Estimated Inertial Parameters]\n", est_pi)
    print("\n[Relative Error (%)]\n", 100 * np.abs((est_pi - true_pi) / true_pi))
