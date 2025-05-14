import numpy as np
import sample_data as data
import MSE
# import exponential
import log_cholesky

if __name__== "__main__":
    true_pi = np.array([
        1.0,    # mass
        0.1, 0.1, 0.1,      # first moments (hx, hy, hz)
        0.2, 0.2, 0.2,      # diagonal inertia (Ixx, Iyy, Izz)
        0.05, 0.05, 0.05    # off-diagonal inertia (Ixy, Ixz, Iyz)
    ])

    # Tuning
    learning_rate = 1e-2
    max_iteration = 10000

    samples = data.generate_sample_data(true_pi)
  
    print("Choose optimizer")
    print("1.MSE 2.esponential 3.log_cholesky")
    optimizer=int(input())
    gradient_method = None
    match optimizer:
        # MSE
        case 1:
            print("Select log_cholesky parameterization")
            print("1. Yes 2. No")
            log_cholesky_parameterization=int(input())
            match log_cholesky_parameterization:
                case 1:
                    log_cholesky_parameterization = True
                    print("Choose gradient method")
                    print("1. numerical 2. analytical")
                    gradient_method = int(input())
                    match gradient_method:
                        case 1:
                            gradient_method = "numerical_grad"
                        case 2:
                            gradient_method = "analytical_grad"
                
                case 2:
                    log_cholesky_parameterization = False

            est_pi= MSE.mse_estimate_inertial_parameters(samples, learning_rate=learning_rate, max_iter=max_iteration,log_cholesky_param=log_cholesky_parameterization, gradient_method=gradient_method, true_pi= true_pi)

        # exponential map
        # case 2:
            # est_pi = exponential.estimate_inertial_parameters(samples, learning_rate=1e-4, max_iter=10000,numerical_grad=True, true_pi=true_pi)


        # log_cholesky
        case 3:
            print("Choose gradient method")
            print("1. numerical 2. analytical")
            gradient_method = int(input())
            match gradient_method:
                case 1:
                    gradient_method = "numerical_grad"
                case 2:
                    gradient_method = "analytical_grad"
            
            est_pi = log_cholesky.log_estimate_inertial_parameters(samples, learning_rate=learning_rate, max_iter=max_iteration, gradient_method=gradient_method, true_pi=true_pi)




