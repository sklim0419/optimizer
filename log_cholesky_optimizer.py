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
        Y = compute_regressor(accel, vel)
        force = Y @ true_pi
        samples.append({'Y': Y, 'force': force})
    return samples, true_pi

def compute_regressor(accel, vel):
    """
    Slotine-Li 방식의 정확한 6x10 관성 파라미터 회귀자 행렬 계산
    accel: 6D 가속도 [v̇; ω̇]
    vel:   6D 속도 [v; ω]
    """
    a_v = accel[:3]     # 선속도 가속도
    a_w = accel[3:]     # 각속도 가속도
    v_v = vel[:3]       # 선속도
    v_w = vel[3:]       # 각속도

    def skew(x):
        return np.array([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ])

    I3 = np.eye(3)
    O3 = np.zeros((3, 3))

    # 질량에 대한 선형 부분
    Yf_m = np.zeros((3, 10))
    Yf_m[:, 0] = a_v  # 질량 m 항

    # 질량중심 위치 h (h = m·c) 에 대한 항
    Yf_h = np.zeros((3, 10))
    Yf_h[:, 1:4] = -skew(a_w) - skew(v_w) @ skew(v_w)

    # 회전 관성에 대한 항 (Euler equation)
    Yτ_I = np.zeros((3, 10))
    Yτ_I[:, 4:] = np.array([
        [a_w[0], -v_w[1]*v_w[2],  v_w[1]*v_w[2],  a_w[1],  a_w[2],  v_w[1]*v_w[2]],
        [v_w[2]*v_w[0], a_w[1], -v_w[2]*v_w[0],  a_w[0],  v_w[0]*v_w[2], a_w[2]],
        [-v_w[0]*v_w[1], v_w[0]*v_w[1], a_w[2],  v_w[0]*v_w[1], a_w[0],  a_w[1]]
    ])

    # 전체 회귀자 구성
    Y = np.zeros((6, 10))
    Y[:3, :] = Yf_m + Yf_h
    Y[3:, :] = Yτ_I

    return Y

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

def numerical_gradient(theta, samples, epsilon=1e-6):
    grad = np.zeros_like(theta)
    base_U = logchol2chol(theta)
    base_J = compute_J(base_U)
    base_pi = pseudo2pi(base_J)
    base_cost = sum(0.5 * np.sum((s['Y'] @ base_pi - s['force'])**2) for s in samples)

    print("\n[Selected θ[i] Numerical Gradient (Euler Forward)]")
    for i in range(len(theta)):
        theta_eps = np.copy(theta)
        theta_eps[i] += epsilon

        U_eps = logchol2chol(theta_eps)
        J_eps = compute_J(U_eps)
        pi_eps = pseudo2pi(J_eps)
        cost_eps = sum(0.5 * np.sum((s['Y'] @ pi_eps - s['force'])**2) for s in samples)

        grad[i] = (cost_eps - base_cost) / epsilon
        print(f"θ[{i}] → grad ≈ (f(θ+ε) - f(θ)) / ε = ({cost_eps:.6f} - {base_cost:.6f}) / {epsilon:.1e} = {grad[i]: .6e}")

    return grad

def estimate_inertial_parameters(samples, learning_rate=1e-4, max_iter=10000):
    theta = np.random.normal(0, 0.1, size=10)

    for it in range(max_iter):
        U = logchol2chol(theta)
        J = compute_J(U)
        pi = pseudo2pi(J)

        grad_pi = np.zeros(10)
        cost = 0
        for s in samples:
            Y, tau = s['Y'], s['force']
            e = Y @ pi - tau
            grad_pi += Y.T @ e
            cost += 0.5 * np.sum(e**2)

        J_pi = compute_jacobian_logcholesky(theta)
        grad_theta = J_pi.T @ grad_pi

        #num_grad = numerical_gradient(theta, samples)
        #   diff = grad_theta - num_grad
        #    norm_diff = np.linalg.norm(diff)
        print(f"\nIter {it+1} | Cost: {cost:.6f} ")

        theta -= learning_rate * grad_theta

    U = logchol2chol(theta)
    J = compute_J(U)
    return pseudo2pi(J)

if __name__ == "__main__":
    samples, true_pi = generate_sample_data()
    est_pi = estimate_inertial_parameters(samples)
    print("\n[True Inertial Parameters]\n", true_pi)
    print("\n[Estimated Inertial Parameters]\n", est_pi)
    print("\n[Relative Error (%)]\n", 100 * np.abs((est_pi - true_pi) / true_pi))