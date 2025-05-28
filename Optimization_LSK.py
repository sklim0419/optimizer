import numpy as np
import sample_data as data
import MSE
import exponential_map
import log_cholesky

if __name__== "__main__":
    true_pi = np.array([
        1.0,    # mass
        0.1, 0.1, 0.1,      # first moments (hx, hy, hz)
        0.2, 0.2, 0.2,      # diagonal inertia (Ixx, Iyy, Izz)
        0.05, 0.05, 0.05    # off-diagonal inertia (Ixy, Ixz, Iyz)
    ])

    # Optimization Option
    learning_rate = 1e-2
    max_iteration = 10000
    gradient_method = True # choose True: "numerical" / False: "analytical"

    samples = data.generate_sample_data(true_pi)
    """ MSE """
    #est_pi= MSE.mse_estimate_inertial_parameters(samples, learning_rate=learning_rate, max_iter=max_iteration, true_pi= true_pi)
    
    """ exponential map """
    est_pi = exponential_map.estimate_inertial_parameters(samples, learning_rate=learning_rate, max_iter=max_iteration,true_pi=true_pi)
    
    """ log_cholesky """
    #est_pi = log_cholesky.log_estimate_inertial_parameters(samples, learning_rate=learning_rate, max_iter=max_iteration, gradient_method=gradient_method, true_pi=true_pi)




