import numpy as np
import sample_data as data
#import MSE
#import log_cholesky
import user_pymanopt

if __name__== "__main__":
    true_pi = np.array([
        1.0,    # mass
        0.1, 0.1, 0.1,      # first moments (hx, hy, hz)
        0.2, 0.2, 0.2,      # diagonal inertia (Ixx, Iyy, Izz)
        0.1, 0.1, 0.1    # off-diagonal inertia (Ixy, Ixz, Iyz)
    ])

    # Optimization Option
    learning_rate = 1e-4
    max_iteration = 1000
    gradient_method = True # choose True: "numerical" / False: "analytical"

    samples = data.generate_sample_data(true_pi)
    """ MSE """
    #est_pi= MSE.mse_estimate_inertial_parameters(samples, learning_rate=learning_rate, max_iter=max_iteration, true_pi= true_pi)
    
    """ Riemannian Gradient Descent"""    
    """ pymanopt """
    est_pi = user_pymanopt.estimate_inertial_parameters(samples, true_pi=true_pi, max_iterations=max_iteration)          
    """ log_cholesky """
    #est_pi = log_cholesky.log_estimate_inertial_parameters(samples, learning_rate=learning_rate, max_iter=max_iteration, gradient_method=gradient_method, true_pi=true_pi)




