import numpy as np
import log_cholesky
import log_cholesky_parameterization

def mse_estimate_inertial_parameters(samples, learning_rate, max_iter,true_pi=None):
    N = len(samples)
    
    pi = np.array([
    0.5,     # mass
    0.0, 0.0, 0.0,      # h_x, h_y, h_z
    0.1, 0.1, 0.1,      # I_xx, I_yy, I_zz
    0.0, 0.0, 0.0       # I_xy, I_xz, I_yz
    ])  

    N = len(samples)

    for it in range(max_iter):
        cost = 0.0
        grad = np.zeros_like(pi)

        for s in samples:
            Y, tau = s['Y'], s['force']
            err = Y @ pi - tau
            grad += Y.T @ err
            cost += 0.5 * np.sum(err**2)

        grad /= N
        cost /= N

        if it % 100 == 0:
            print(f"Iter {it+1} | MSE Cost: {cost:.6f}")

        pi -= learning_rate * grad
        est_pi = pi

        if it % 100==0:
            print("\n[Iteration ",it,"]")
            print("[True Inertial Parameters]\n", true_pi)
            print("[Estimated Inertial Parameters]\n", est_pi)
            print("[Relative Error (%)]\n", 100 * np.abs((est_pi - true_pi) / true_pi))
        if it == max_iter:
            print("\n[MSE Final]")
            print("[True Inertial Parameters]\n", true_pi)
            print("[Estimated Inertial Parameters]\n", est_pi)
            print("[Relative Error (%)]\n", 100 * np.abs((est_pi - true_pi) / true_pi))

    return est_pi