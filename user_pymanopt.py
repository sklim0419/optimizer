from pymanopt.manifolds import SymmetricPositiveDefinite
from pymanopt.optimizers import SteepestDescent
from pymanopt import Problem
from pymanopt.function import autograd
import autograd.numpy as anp

def pseudo2pi(J):
    m = J[3, 3]
    h = J[0:3, 3]
    I_c = J[0:3, 0:3]
    c = h / m
    I = I_c - m * anp.outer(c, c)
    pi = anp.zeros(10)
    pi = anp.concatenate((
        anp.array([m]),
        h,
        anp.array([
            I[0, 0],
            I[1, 1],
            I[2, 2],
            I[0, 1],
            I[0, 2],
            I[1, 2],
        ])
    ))
    return pi

def estimate_inertial_parameters(samples, true_pi=None, max_iterations=None):
    manifold = SymmetricPositiveDefinite(n=4)

    @autograd(manifold)
    def cost(J):
        pi = pseudo2pi(J)
        cost_sum = 0.0
        for s in samples:
            Y_i = s['Y']
            tau_i = s['force']
            e = Y_i @ pi - tau_i
            cost_sum += 0.5 * anp.sum(e**2)
        return cost_sum / len(samples)

    problem = Problem(manifold=manifold, cost=cost)
    # initialization
    J = anp.eye(4)

    solver = SteepestDescent(max_iterations=max_iterations)
    result = solver.run(problem, initial_point=J)
    J = result.point
    grad_norm = result.gradient_norm
    
    if grad_norm < 1e-8:  # 조기 종료 조건
        print(f"Terminated at iteration with grad_norm {grad_norm:.2e}")
        pi_est = pseudo2pi(J)
        print(pi_est)
        #break
        
    pi_est = pseudo2pi(J)

    
    print("\n[True Inertial Parameters]")
    print(true_pi)
    print("[Estimated Inertial Parameters]")
    print(pi_est)
    print("[Relative Error (%)]")
    print(100 * anp.abs((pi_est - true_pi) / true_pi))

    return pi_est