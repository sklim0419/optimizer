import numpy as np
import log_cholesky_parameterization


def numerical_gradient(theta, samples, epsilon=1e-6):
    grad = np.zeros_like(theta)
    base_U = log_cholesky_parameterization.logchol2chol(theta)
    base_J = log_cholesky_parameterization.compute_J(base_U)
    base_pi = log_cholesky_parameterization.pseudo2pi(base_J)
    base_cost = sum(0.5 * np.sum((s['Y'] @ base_pi - s['force'])**2) for s in samples)

    for i in range(len(theta)):
        theta_eps = np.copy(theta)
        theta_eps[i] += epsilon

        U_eps = log_cholesky_parameterization.logchol2chol(theta_eps)
        J_eps = log_cholesky_parameterization.compute_J(U_eps)
        pi_eps = log_cholesky_parameterization.pseudo2pi(J_eps)
        cost_eps = sum(0.5 * np.sum((s['Y'] @ pi_eps - s['force'])**2) for s in samples)

        grad[i] = (cost_eps - base_cost) / epsilon
        
    return grad

def compute_jacobian_logcholesky(theta):
    alpha, d1, d2, d3, s12, s13, s23, t1, t2, t3 = theta
    e_alpha = np.exp(alpha)
    e_2alpha = e_alpha ** 2
    e_d1 = np.exp(d1)
    e_d2 = np.exp(d2)
    e_d3 = np.exp(d3)
    m = e_2alpha

    U_bar = np.array([
        [e_d1, s12,  s13,  t1],
        [0.0,  e_d2, s23,  t2],
        [0.0,  0.0,  e_d3, t3],
        [0.0,  0.0,  0.0,  1.0]
    ])

    Sigma_bar = U_bar[:3, :] @ U_bar[:3, :].T

    J = np.zeros((10, 10))

    J[0, 0] = 2 * m

    for i, t in enumerate([t1, t2, t3]):
        J[1 + i, 0] = 2 * m * t
        J[1 + i, 7 + i] = m

    for i in range(3):
        J[4 + i, 0] = 2 * e_2alpha * (np.trace(Sigma_bar) - Sigma_bar[i, i])

    J[7, 0] = 2 * e_2alpha * Sigma_bar[0, 1]  # 부호
    J[8, 0] = 2 * e_2alpha * Sigma_bar[0, 2]
    J[9, 0] = 2 * e_2alpha * Sigma_bar[1, 2]

    def partial_Ubar(j_idx):
        dU = np.zeros_like(U_bar)
        if j_idx == 1:
            dU[0, 0] = e_d1
        elif j_idx == 2:
            dU[1, 1] = e_d2
        elif j_idx == 3:
            dU[2, 2] = e_d3
        elif j_idx == 4:
            dU[0, 1] = 1.0
        elif j_idx == 5:
            dU[0, 2] = 1.0
        elif j_idx == 6:
            dU[1, 2] = 1.0
        elif j_idx == 7:
            dU[0, 3] = 1.0
        elif j_idx == 8:
            dU[1, 3] = 1.0
        elif j_idx == 9:
            dU[2, 3] = 1.0
        return dU

    for j in range(1, 10):
        dU = partial_Ubar(j)
        dSigma_bar = dU[:3, :] @ U_bar[:3, :].T + U_bar[:3, :] @ dU[:3, :].T
        dSigma = e_2alpha * dSigma_bar
        tr_dSigma = np.trace(dSigma)

        for i in range(3):
            J[4 + i, j] = m * (tr_dSigma - dSigma[i, i])

        J[7, j] = m * dSigma[0, 1]  #부호
        J[8, j] = m * dSigma[0, 2]
        J[9, j] = m * dSigma[1, 2]

    return J

def log_estimate_inertial_parameters(samples, learning_rate, max_iter, gradient_method=None, true_pi=None):
    theta = np.array([
            0,                   # alpha = log(sqrt(1))
            0.1, 0.1, -0.8,      # d_i
            0.5, 0.2, 0.05,      # s_ij
            0.1, 0.1, 0.1          # t_i
        ])

    for it in range(max_iter):
        U = log_cholesky_parameterization.logchol2chol(theta)
        J = log_cholesky_parameterization.compute_J(U)
        pi = log_cholesky_parameterization.pseudo2pi(J)

        grad_pi = np.zeros(10)
        cost = 0
        for s in samples:
            Y, tau = s['Y'], s['force']
            e = Y @ pi - tau
            grad_pi += Y.T @ e
            cost += 0.5 * np.sum(e**2)

        if gradient_method == "numerical_grad":
            grad_theta = numerical_gradient(theta, samples)
        elif gradient_method == "analytical_grad":
            J_pi = compute_jacobian_logcholesky(theta)
            grad_theta = J_pi.T @ grad_pi

        grad_theta /= len(samples)
        theta -= learning_rate * grad_theta

        if it % 100==0:
            U = log_cholesky_parameterization.logchol2chol(theta)
            J = log_cholesky_parameterization.compute_J(U)
            est_pi =log_cholesky_parameterization.pseudo2pi(J)
            print("\n[Iteration ",it,"]")
            print("[True Inertial Parameters]\n", true_pi)
            print("[Estimated Inertial Parameters]\n", est_pi)
            print("[Relative Error (%)]\n", 100 * np.abs((est_pi - true_pi) / true_pi))

    U = log_cholesky_parameterization.logchol2chol(theta)
    J = log_cholesky_parameterization.compute_J(U)
    est_pi =log_cholesky_parameterization.pseudo2pi(J)
    print("\n[Final]")
    print("[True Inertial Parameters]\n", true_pi)
    print("[Estimated Inertial Parameters]\n", est_pi)
    print("[Relative Error (%)]\n", 100 * np.abs((est_pi - true_pi) / true_pi))

    return est_pi