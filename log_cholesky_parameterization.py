import numpy as np

def compute_J(U):
    return U @ U.T

def logchol2chol(theta):
    alpha, d1, d2, d3, s12, s13, s23, t1, t2, t3 = theta
    e_alpha = np.exp(alpha)
    U = np.zeros((4, 4))
    U[0, 0] = e_alpha * np.exp(d1)
    U[0, 1] = e_alpha * s12
    U[0, 2] = e_alpha * s13
    U[0, 3] = e_alpha * t1
    U[1, 1] = e_alpha * np.exp(d2)
    U[1, 2] = e_alpha * s23
    U[1, 3] = e_alpha * t2
    U[2, 2] = e_alpha * np.exp(d3)
    U[2, 3] = e_alpha * t3
    U[3, 3] = e_alpha
    return U

def pseudo2pi(J):
    m = J[3, 3]
    h = J[:3, 3]
    Sigma = J[:3, :3]
    trace_sigma = np.trace(Sigma)
    I_bar = trace_sigma * np.eye(3) - Sigma
    I_xx, I_yy, I_zz = I_bar[0, 0], I_bar[1, 1], I_bar[2, 2]
    I_xy, I_xz, I_yz = -I_bar[0, 1], -I_bar[0, 2], -I_bar[1, 2]  # 부호
    return np.array([m, *h, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz])