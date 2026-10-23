"""Reproduce the numerical illustration and figures in the article.

Dependencies: numpy, scipy, numba, matplotlib.
Run from the source directory with

    python numerical_example.py

The script writes numerical_results.npz, sample_path.csv, and the four PDF
figures referenced by the LaTeX source.
"""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import quad
from scipy.special import gammainc

OUT = Path(__file__).resolve().parent

ALPHA = 0.95
DELTA = 0.20
HCOEF = 5.0
KCOEF = 0.40

Y_GRID = np.arange(0.0, 30.0 + 1e-12, 0.25)
W_GRID = np.arange(0.0, 1.0 + 1e-12, 0.025)
A_GRID = np.arange(0.0, 15.0 + 1e-12, 0.125)
Z_GRID = np.arange(0.0, 30.0 + 1e-12, 0.05)
DZ = Z_GRID[1] - Z_GRID[0]

F_L = Z_GRID * np.exp(-Z_GRID)
F_H = Z_GRID**4 * np.exp(-Z_GRID) / 24.0
F_L /= np.trapezoid(F_L, Z_GRID)
F_H /= np.trapezoid(F_H, Z_GRID)
OMEGA = Z_GRID / (Z_GRID + 4.0)


def cumulative_trapezoid(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    out[1:] = np.cumsum(0.5 * DZ * (values[:-1] + values[1:]))
    return out


CUM_FL = cumulative_trapezoid(F_L)
CUM_FH = cumulative_trapezoid(F_H)
CUM_OL = cumulative_trapezoid(F_L * OMEGA)
CUM_OH = cumulative_trapezoid(F_H * OMEGA)


@njit(cache=True)
def interp_uniform_1d(arr, x, x0, dx):
    n = arr.size
    if x <= x0:
        return arr[0]
    xmax = x0 + dx * (n - 1)
    if x >= xmax:
        return arr[n - 1]
    u = (x - x0) / dx
    i = int(np.floor(u))
    t = u - i
    return (1.0 - t) * arr[i] + t * arr[i + 1]


@njit(cache=True)
def interp_value(V, y, w, y0, dy, w0, dw):
    ny, nw = V.shape
    if w <= w0:
        jw, tw = 0, 0.0
    else:
        wmax = w0 + dw * (nw - 1)
        if w >= wmax:
            jw, tw = nw - 2, 1.0
        else:
            uw = (w - w0) / dw
            jw = int(np.floor(uw))
            tw = uw - jw

    if y <= y0:
        iy, ty = 0, 0.0
    else:
        ymax = y0 + dy * (ny - 1)
        if y >= ymax:
            iy = ny - 2
            ty = (y - (y0 + dy * iy)) / dy
        else:
            uy = (y - y0) / dy
            iy = int(np.floor(uy))
            ty = uy - iy

    v00, v01 = V[iy, jw], V[iy, jw + 1]
    v10, v11 = V[iy + 1, jw], V[iy + 1, jw + 1]
    vlo = (1.0 - tw) * v00 + tw * v01
    vhi = (1.0 - tw) * v10 + tw * v11
    return (1.0 - ty) * vlo + ty * vhi


@njit(cache=True)
def omega_scalar(z):
    return z / (z + 4.0)


@njit(cache=True)
def tail_integrals(V, b, z_grid, f_l, f_h, omega, y0, dy, w0, dw):
    nz = z_grid.size
    dz = z_grid[1] - z_grid[0]
    bp = b if b > 0.0 else 0.0
    if bp >= z_grid[nz - 1]:
        return 0.0, 0.0

    k = int(np.floor((bp - z_grid[0]) / dz))
    k = max(0, min(k, nz - 2))
    t = (bp - z_grid[k]) / dz
    f_l_b = (1.0 - t) * f_l[k] + t * f_l[k + 1]
    f_h_b = (1.0 - t) * f_h[k] + t * f_h[k + 1]
    v_b = interp_value(V, 0.0, omega_scalar(bp), y0, dy, w0, dw)

    z1 = z_grid[k + 1]
    v1 = interp_value(V, z1 - b, omega[k + 1], y0, dy, w0, dw)
    s_l = 0.5 * (z1 - bp) * (f_l_b * v_b + f_l[k + 1] * v1)
    s_h = 0.5 * (z1 - bp) * (f_h_b * v_b + f_h[k + 1] * v1)
    prev_l, prev_h = f_l[k + 1] * v1, f_h[k + 1] * v1

    for iz in range(k + 2, nz):
        z = z_grid[iz]
        vv = interp_value(V, z - b, omega[iz], y0, dy, w0, dw)
        cur_l, cur_h = f_l[iz] * vv, f_h[iz] * vv
        s_l += 0.5 * dz * (prev_l + cur_l)
        s_h += 0.5 * dz * (prev_h + cur_h)
        prev_l, prev_h = cur_l, cur_h
    return s_l, s_h


@njit(cache=True)
def value_iteration(tol=1e-5, max_iter=800):
    ny, nw, na = Y_GRID.size, W_GRID.size, A_GRID.size
    dy, dw = Y_GRID[1] - Y_GRID[0], W_GRID[1] - W_GRID[0]
    V = np.zeros((ny, nw))
    policy = np.zeros((ny, nw))
    errors = np.empty(max_iter)

    for iteration in range(max_iter):
        V_new = np.empty_like(V)
        policy_new = np.empty_like(policy)
        for iy in range(ny):
            y = Y_GRID[iy]
            e_l, e_h = np.empty(na), np.empty(na)
            for ia in range(na):
                a = A_GRID[ia]
                b = a - (1.0 - DELTA) * y
                e_l[ia], e_h[ia] = tail_integrals(
                    V, b, Z_GRID, F_L, F_H, OMEGA,
                    Y_GRID[0], dy, W_GRID[0], dw
                )

            for iw in range(nw):
                w = W_GRID[iw]
                best, best_action = 1e300, 0.0
                for ia in range(na):
                    a = A_GRID[ia]
                    b = a - (1.0 - DELTA) * y
                    bp = max(b, 0.0)
                    f_l = interp_uniform_1d(CUM_FL, bp, Z_GRID[0], DZ)
                    f_h = interp_uniform_1d(CUM_FH, bp, Z_GRID[0], DZ)
                    o_l = interp_uniform_1d(CUM_OL, bp, Z_GRID[0], DZ)
                    o_h = interp_uniform_1d(CUM_OH, bp, Z_GRID[0], DZ)
                    mass = (1.0 - w) * f_l + w * f_h
                    censored = 0.0
                    if mass > 1e-14:
                        phi = ((1.0 - w) * o_l + w * o_h) / mass
                        censored = mass * interp_value(
                            V, 0.0, phi, Y_GRID[0], dy, W_GRID[0], dw
                        )
                    uncensored = (1.0 - w) * e_l[ia] + w * e_h[ia]
                    value = HCOEF * y + KCOEF * a * a + ALPHA * (
                        censored + uncensored
                    )
                    if value < best:
                        best, best_action = value, a
                V_new[iy, iw] = best
                policy_new[iy, iw] = best_action

        error = np.max(np.abs(V_new - V))
        errors[iteration] = error
        V, policy = V_new, policy_new
        if error < tol:
            return V, policy, errors[: iteration + 1]
    return V, policy, errors


def bilinear(A, y, w):
    y = np.clip(y, Y_GRID[0], Y_GRID[-1])
    w = np.clip(w, W_GRID[0], W_GRID[-1])
    iy = max(0, min(np.searchsorted(Y_GRID, y) - 1, len(Y_GRID) - 2))
    iw = max(0, min(np.searchsorted(W_GRID, w) - 1, len(W_GRID) - 2))
    ty = (y - Y_GRID[iy]) / (Y_GRID[iy + 1] - Y_GRID[iy])
    tw = (w - W_GRID[iw]) / (W_GRID[iw + 1] - W_GRID[iw])
    return (
        (1 - ty) * (1 - tw) * A[iy, iw]
        + (1 - ty) * tw * A[iy, iw + 1]
        + ty * (1 - tw) * A[iy + 1, iw]
        + ty * tw * A[iy + 1, iw + 1]
    )


def f_l_scalar(z):
    return z * np.exp(-z)


def f_h_scalar(z):
    return z**4 * np.exp(-z) / 24.0


def omega(z):
    return z / (z + 4.0)


def censoring_mass(w, b):
    if b <= 0.0:
        return 0.0
    return (1.0 - w) * gammainc(2, b) + w * gammainc(5, b)


def phi(w, b):
    mass = censoring_mass(w, b)
    if mass <= 1e-14:
        return w
    numerator = (1.0 - w) * quad(
        lambda z: omega(z) * f_l_scalar(z), 0.0, b, epsabs=1e-11
    )[0]
    numerator += w * quad(
        lambda z: omega(z) * f_h_scalar(z), 0.0, b, epsabs=1e-11
    )[0]
    return numerator / mass


def make_figures(V, policy, errors):
    selected_w = [0.0, 0.25, 0.5, 0.75, 1.0]
    displayed = Y_GRID <= 15.0

    plt.figure(figsize=(6.2, 4.1))
    for w in selected_w:
        iw = int(round(w / (W_GRID[1] - W_GRID[0])))
        plt.plot(Y_GRID[displayed], policy[displayed, iw], label=fr'$w={w:g}$')
    plt.xlabel('Pollution stock $y$')
    plt.ylabel('Optimal abatement $a^*(y,w)$')
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / 'optimal_policy_curves.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6.2, 4.1))
    for w in selected_w:
        iw = int(round(w / (W_GRID[1] - W_GRID[0])))
        plt.plot(Y_GRID[displayed], V[displayed, iw], label=fr'$w={w:g}$')
    plt.xlabel('Pollution stock $y$')
    plt.ylabel('Approximate value $v(y,w)$')
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / 'value_function_curves.pdf', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6.2, 4.1))
    plt.semilogy(np.arange(1, len(errors) + 1), errors)
    plt.xlabel('Value-iteration step')
    plt.ylabel(r'$\|v_{n+1}-v_n\|_\infty$')
    plt.tight_layout()
    plt.savefig(OUT / 'value_iteration_convergence.pdf', bbox_inches='tight')
    plt.close()

    rng = np.random.default_rng(1)
    periods = 20
    y, w = 6.0, 0.5
    stocks, beliefs, actions, disturbances, censored_times = [y], [w], [], [], []
    for t in range(periods):
        action = bilinear(policy, y, w)
        action = A_GRID[np.argmin(np.abs(A_GRID - action))]
        disturbance = rng.gamma(shape=5 if rng.random() < w else 2, scale=1.0)
        next_stock = max((1.0 - DELTA) * y + disturbance - action, 0.0)
        b = action - (1.0 - DELTA) * y
        if next_stock <= 1e-12:
            next_belief = phi(w, b)
            censored_times.append(t + 1)
        else:
            next_belief = omega(disturbance)
        actions.append(action)
        disturbances.append(disturbance)
        stocks.append(next_stock)
        beliefs.append(next_belief)
        y, w = next_stock, next_belief

    figure, axes = plt.subplots(3, 1, figsize=(6.4, 6.8), sharex=True)
    state_times = np.arange(periods + 1)
    action_times = np.arange(periods)
    axes[0].step(state_times, stocks, where='post')
    axes[0].set_ylabel('$Y_t$')
    axes[1].step(action_times, actions, where='post')
    axes[1].set_ylabel('$a_t$')
    axes[2].step(state_times, beliefs, where='post')
    axes[2].set_ylabel('$w_t$')
    axes[2].set_xlabel('Period $t$')
    for axis in axes:
        for time_index in censored_times:
            axis.axvline(time_index, linestyle=':', linewidth=0.8)
    figure.tight_layout()
    figure.savefig(OUT / 'sample_path.pdf', bbox_inches='tight')
    plt.close(figure)

    np.savetxt(
        OUT / 'sample_path.csv',
        np.column_stack(
            [
                np.arange(periods), stocks[:-1], beliefs[:-1], actions,
                disturbances, stocks[1:], beliefs[1:],
                np.isclose(stocks[1:], 0.0).astype(int),
            ]
        ),
        delimiter=',',
        header='t,y_t,w_t,a_t,zeta_t,y_next,w_next,censored',
        comments='',
    )


def main():
    start = time.time()
    V, policy, errors = value_iteration()
    np.savez(
        OUT / 'numerical_results.npz',
        V=V, policy=policy, errors=errors,
        y_grid=Y_GRID, w_grid=W_GRID, a_grid=A_GRID, z_grid=Z_GRID,
        fL=F_L, fH=F_H, omega=OMEGA,
        alpha=ALPHA, delta=DELTA, hcoef=HCOEF, kcoef=KCOEF,
    )
    make_figures(V, policy, errors)
    print(
        f'Completed in {time.time() - start:.2f} seconds; '
        f'{len(errors)} iterations; final error {errors[-1]:.3e}; '
        f'maximum selected action {policy.max():.3f}.'
    )


if __name__ == '__main__':
    main()
