import numpy as np
from scipy.linalg import sqrtm, inv, expm

def estimate_inertial_parameters(samples, learning_rate, max_iter, true_pi=None):
    # step 1
    def compute_loss(J, samples):
        pi = pseudo2pi(J)
        cost = 0.0
        for s in samples:
            Y, tau = s['Y'], s['force']
            err = Y @ pi - tau
            cost += 0.5 * np.sum(err**2)
        return cost / len(samples)

    # step 2
    def numerical_gradient(J, samples, epsilon=1e-6):
        grad = np.zeros_like(J)
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                E = np.zeros_like(J)
                E[i, j] = 1.0
                loss1 = compute_loss(J + epsilon * E, samples)
                loss2 = compute_loss(J - epsilon * E, samples)
                grad[i, j] = (loss1 - loss2) / (2 * epsilon)
        return grad

    # step 3
    def riemannian_gradient(J, grad_e):
        return J @ grad_e @ J

    #step 4
    def exponential_map(J, V):
        J_sqrt = sqrtm(J)
        J_inv_sqrt = inv(J_sqrt)
        inner = J_inv_sqrt @ V @ J_inv_sqrt
        return J_sqrt @ expm(inner) @ J_sqrt

    # J -> pi
    def pseudo2pi(J):
        m = J[3, 3]
        h = J[0:3, 3]
        I_c = J[0:3, 0:3]
        c = h / m
        I = I_c - m * np.outer(c, c)

        pi = np.zeros(10)
        pi[0] = m
        pi[1:4] = h
        pi[4] = I[0, 0]
        pi[5] = I[1, 1]
        pi[6] = I[2, 2]
        pi[7] = I[0, 1]
        pi[8] = I[0, 2]
        pi[9] = I[1, 2]
        return pi

    J = np.eye(4)
    J[3, 3] = 1.0 

    for it in range(max_iter):
        grad_e = numerical_gradient(J, samples)
        grad_r = riemannian_gradient(J, grad_e)
        V = -learning_rate * grad_r
        J_new = exponential_map(J, V)

        if np.linalg.norm(J_new - J, ord='fro') < 1e-8:
            print(f"[Converged at iteration {it}]")
            break

        J = J_new
        est_pi = pseudo2pi(J)
        cost = compute_loss(J, samples)

        if it % 100==0:
            print("\n[Iteration ",it,"]")
            print("[True Inertial Parameters]\n", true_pi)
            print("[Estimated Inertial Parameters]\n", est_pi)
            print("[Relative Error (%)]\n", 100 * np.abs((est_pi - true_pi) / true_pi))
        if it+1 == max_iter:
            print("\n[exponential map Final]")
            print("[True Inertial Parameters]\n", true_pi)
            print("[Estimated Inertial Parameters]\n", est_pi)
            print("[Relative Error (%)]\n", 100 * np.abs((est_pi - true_pi) / true_pi))

    return est_pi
