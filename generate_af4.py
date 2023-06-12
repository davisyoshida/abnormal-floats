from functools import partial
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.stats import norm, halfnorm, truncnorm

def integrand(block_size, x, m):
    p_z_less_than_mx = truncnorm.cdf(m * x, -m, m)
    pm = block_size * (halfnorm.cdf(m) ** (block_size - 1)) * 2 * norm.pdf(m)

    return p_z_less_than_mx * pm

def scaled_norm_cdf(block_size, x):
    result = quad(
        partial(integrand, block_size, x),
        0,
        np.inf,
        epsabs=1e-9,
    )
    return result[0]

def cdf(x, block_size):
    discrete_mass = 1 / (2 * block_size)
    cont_mass = scaled_norm_cdf(block_size, x) * (block_size - 1) / block_size
    result = discrete_mass + cont_mass
    result = np.where(x < -1, 0, result)
    result = np.where(x >= 1, 1, result)
    return result

def inv_cdf(val, block_size):
    edge_mass  = 1 / (2 * block_size)
    if val <= edge_mass:
        return -1
    if val >= 1 - edge_mass:
        return 1
    def search_fn(x):
        return cdf(x, block_size) - val

    return root_scalar(search_fn, bracket=[-1, 1]).root

def build_code(start, lower_bound, upper_bound, n_steps, bcdf, binv_cdf, lower_bound_is_code_point=True):
    code = [start]
    a = lower_bound if lower_bound_is_code_point else None
    b = start
    prev_midpoint_prob = None
    for _ in range(n_steps):
        if prev_midpoint_prob is None:
            prev_midpoint_prob = bcdf((a + b) / 2) if lower_bound_is_code_point else 0
        prev_mass = bcdf(b) - prev_midpoint_prob

        next_midpoint_prob = bcdf(b) + prev_mass
        if next_midpoint_prob > 1:
            c = 1
        else:
            next_midpoint = binv_cdf(next_midpoint_prob)
            prev_midpoint_prob = next_midpoint_prob

            c = next_midpoint * 2 - b

            a = b
            b = c

        if c >= upper_bound:
            code.extend(upper_bound for _ in range(n_steps + 1 - len(code)))
            break

        code.append(c)

    return np.stack(code, -1)

def interval_code_search(
    lower_bound,
    upper_bound,
    n_steps,
    block_size,
    bounds_are_code_points=True,
):
    bcdf = partial(cdf, block_size=block_size)
    binv_cdf = partial(inv_cdf, block_size=block_size)

    code_builder = partial(
        build_code,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        n_steps=n_steps,
        bcdf=bcdf,
        binv_cdf=binv_cdf,
        lower_bound_is_code_point=bounds_are_code_points
    )

    lower_bracket = lower_bound + (1e-5 if bounds_are_code_points else 0)

    lower_feasible = lower_bracket
    upper_feasible = upper_bound - 1e-5

    while upper_feasible - lower_feasible > 1e-4:
        mid = (lower_feasible + upper_feasible) / 2
        code = code_builder(mid)
        infeasible = np.any(code[1:] - code[:-1] <= 0) or any(code[:-1] >= upper_bound) or code[-1] > upper_bound
        if infeasible:
            upper_feasible = mid
        else:
            lower_feasible = mid

    upper_bracket = lower_feasible
    def search_fn(val):
        code = code_builder(val)
        top = code[-1]
        prev_split = (top + code[-2]) / 2

        top_prob = bcdf(top)
        target_prob = top_prob + (top_prob - bcdf(prev_split))
        return target_prob - 1/2

    opt_a2 = root_scalar(search_fn, bracket=[lower_bracket, upper_bracket]).root
    code = code_builder(opt_a2)
    return code

def construct_af4(block_size):
    lower =  interval_code_search(-1, 0, 5, block_size)
    upper = -interval_code_search(-1, 0, 6, block_size)[::-1]
    code = np.asarray([-1., *lower, 0., *upper, 1.], dtype=np.float64)
    assert code.shape == (16,)
    return code

def main():
    for block_size in (32, 64, 128, 256, 512, 1024, 4096):
        start = time.time()
        code = construct_af4(block_size)
        end = time.time()
        print(f'B = {block_size} - Runtime: {end - start:.2f} sec\n{code}\n')
        np.save(f'af4_{block_size}.npy', code)

if __name__ == '__main__':
    main()
