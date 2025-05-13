import numpy as np

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

