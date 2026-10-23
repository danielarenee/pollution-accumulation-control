"""Phase 4: solve the calibrated model and measure the value of information.

Usage:
    python rama_solve.py <calibration.json> <output_dir>

For each delta (and each abatement-cost scenario) we solve two problems by
value iteration on the reduced state (y, w) of Section 7 (Prop. 7.1):

  * PARTIAL  - the model of the paper: a zero stock only reveals zeta <= b,
               and the belief moves to w' = Phi(w, b);
  * FULL     - benchmark in which the disturbance is always revealed, so the
               next belief is w' = omega(zeta) even when the stock hits zero.

The gap between the two is the cost of censored information. Both policies
are then simulated on the same long path of the *true* hidden disturbance
chain, which also gives a "belief vs. truth" figure.

Disturbances use the logistic persistence omega(z) = 1/(1+exp(-(z-m)/s))
calibrated in rama_calibrate.py. Units: ug/m3 and days.
Costs (not identified by the data): h(y) = y, k(a) = KAPPA * a^2.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.95
KAPPAS = [0.02, 0.05]
Y_GRID = np.arange(0.0, 80.0 + 1e-9, 0.5)
W_GRID = np.arange(0.0, 1.0 + 1e-9, 0.025)
A_GRID = np.arange(0.0, 60.0 + 1e-9, 0.5)
Z_GRID = np.arange(0.0, 80.0 + 1e-9, 0.2)
DZ = Z_GRID[1] - Z_GRID[0]
N_SIM, BURN = 60_000, 1_000


# ---------------------------------------------------------------- numerics
@njit(cache=True)
def interp1(arr, x, x0, dx):
    n = arr.size
    if x <= x0:
        return arr[0]
    if x >= x0 + dx * (n - 1):
        return arr[n - 1]
    u = (x - x0) / dx
    i = int(u)
    t = u - i
    return (1.0 - t) * arr[i] + t * arr[i + 1]


@njit(cache=True)
def interp_v(V, y, w, y0, dy, w0, dw):
    ny, nw = V.shape
    if w <= w0:
        jw, tw = 0, 0.0
    elif w >= w0 + dw * (nw - 1):
        jw, tw = nw - 2, 1.0
    else:
        u = (w - w0) / dw
        jw = int(u)
        tw = u - jw
    if y <= y0:
        iy, ty = 0, 0.0
    elif y >= y0 + dy * (ny - 1):
        iy = ny - 2
        ty = (y - (y0 + dy * iy)) / dy  # linear extrapolation in y
    else:
        u = (y - y0) / dy
        iy = int(u)
        ty = u - iy
    vlo = (1.0 - tw) * V[iy, jw] + tw * V[iy, jw + 1]
    vhi = (1.0 - tw) * V[iy + 1, jw] + tw * V[iy + 1, jw + 1]
    return (1.0 - ty) * vlo + ty * vhi


@njit(cache=True)
def expected_next(V, b, z, fl, fh, om, y0, dy, w0, dw, full_info):
    """Integrals over z of the next value, split by regime density.

    partial: returns  int_{z>b+} V(z-b, om(z)) f(z) dz   (uncensored part only)
    full:    returns  int_{z>=0} V((z-b)^+, om(z)) f(z) dz
    """
    nz = z.size
    sl, sh = 0.0, 0.0
    bp = b if b > 0.0 else 0.0
    for i in range(nz - 1):
        za, zb = z[i], z[i + 1]
        if not full_info:
            if zb <= bp:
                continue
            if za < bp:  # partial cell: integrate from bp to zb
                t = (bp - za) / (zb - za)
                fla = (1 - t) * fl[i] + t * fl[i + 1]
                fha = (1 - t) * fh[i] + t * fh[i + 1]
                oma = (1 - t) * om[i] + t * om[i + 1]
                va = interp_v(V, 0.0, oma, y0, dy, w0, dw)
                vb = interp_v(V, zb - b, om[i + 1], y0, dy, w0, dw)
                h = zb - bp
                sl += 0.5 * h * (fla * va + fl[i + 1] * vb)
                sh += 0.5 * h * (fha * va + fh[i + 1] * vb)
                continue
        ya = za - b if za - b > 0.0 else 0.0
        yb = zb - b if zb - b > 0.0 else 0.0
        va = interp_v(V, ya, om[i], y0, dy, w0, dw)
        vb = interp_v(V, yb, om[i + 1], y0, dy, w0, dw)
        sl += 0.5 * (zb - za) * (fl[i] * va + fl[i + 1] * vb)
        sh += 0.5 * (zb - za) * (fh[i] * va + fh[i + 1] * vb)
    return sl, sh


@njit(cache=True)
def value_iteration(yg, wg, ag, z, fl, fh, om, cfl, cfh, col, coh,
                    delta, kappa, alpha, full_info, tol, max_iter):
    ny, nw, na = yg.size, wg.size, ag.size
    dy, dw, dz = yg[1] - yg[0], wg[1] - wg[0], z[1] - z[0]
    V = np.zeros((ny, nw))
    pol = np.zeros((ny, nw))
    errs = np.empty(max_iter)
    el = np.empty(na)
    eh = np.empty(na)
    for it in range(max_iter):
        Vn = np.empty_like(V)
        for iy in range(ny):
            y = yg[iy]
            for ia in range(na):
                b = ag[ia] - (1.0 - delta) * y
                el[ia], eh[ia] = expected_next(V, b, z, fl, fh, om, yg[0], dy,
                                               wg[0], dw, full_info)
            for iw in range(nw):
                w = wg[iw]
                best, besta = 1e300, 0.0
                for ia in range(na):
                    a = ag[ia]
                    b = a - (1.0 - delta) * y
                    future = (1.0 - w) * el[ia] + w * eh[ia]
                    if (not full_info) and b > 0.0:
                        ml = interp1(cfl, b, z[0], dz)
                        mh = interp1(cfh, b, z[0], dz)
                        mass = (1.0 - w) * ml + w * mh
                        if mass > 1e-14:
                            phi = ((1.0 - w) * interp1(col, b, z[0], dz)
                                   + w * interp1(coh, b, z[0], dz)) / mass
                            future += mass * interp_v(V, 0.0, phi, yg[0], dy, wg[0], dw)
                    val = y + kappa * a * a + alpha * future
                    if val < best:
                        best, besta = val, a
                Vn[iy, iw] = best
                pol[iy, iw] = besta
        err = np.max(np.abs(Vn - V))
        errs[it] = err
        V = Vn
        if err < tol:
            return V, pol, errs[: it + 1]
    return V, pol, errs


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, fitpar, delta):
        self.delta = delta
        self.kL, self.thL = fitpar["kL"], fitpar["thL"]
        self.kH, self.thH = fitpar["kH"], fitpar["thH"]
        self.m, self.s = fitpar["q"]
        z = Z_GRID
        self.fl = stats.gamma.pdf(z, self.kL, scale=self.thL)
        self.fh = stats.gamma.pdf(z, self.kH, scale=self.thH)
        self.om = self.omega(z)
        cum = lambda v: np.r_[0.0, np.cumsum(0.5 * DZ * (v[:-1] + v[1:]))]
        self.cfl, self.cfh = cum(self.fl), cum(self.fh)
        self.col, self.coh = cum(self.fl * self.om), cum(self.fh * self.om)

    def omega(self, z):
        return 1.0 / (1.0 + np.exp(-(np.asarray(z) - self.m) / self.s))

    def phi(self, w, b):
        if b <= 0:
            return w
        ml, mh = np.interp(b, Z_GRID, self.cfl), np.interp(b, Z_GRID, self.cfh)
        mass = (1 - w) * ml + w * mh
        if mass < 1e-14:
            return w
        return ((1 - w) * np.interp(b, Z_GRID, self.col)
                + w * np.interp(b, Z_GRID, self.coh)) / mass

    def solve(self, kappa, full_info):
        return value_iteration(Y_GRID, W_GRID, A_GRID, Z_GRID, self.fl, self.fh,
                               self.om, self.cfl, self.cfh, self.col, self.coh,
                               self.delta, kappa, ALPHA, full_info, 1e-4, 2000)

    def true_path(self, n, rng):
        zeta = np.empty(n)
        z = self.kL * self.thL
        for t in range(n):
            high = rng.random() < self.omega(z)
            z = rng.gamma(self.kH, self.thH) if high else rng.gamma(self.kL, self.thL)
            zeta[t] = z
        return zeta


def bilinear(A, y, w):
    y = min(max(y, Y_GRID[0]), Y_GRID[-1])
    w = min(max(w, W_GRID[0]), W_GRID[-1])
    iy = min(int((y - Y_GRID[0]) / (Y_GRID[1] - Y_GRID[0])), len(Y_GRID) - 2)
    iw = min(int((w - W_GRID[0]) / (W_GRID[1] - W_GRID[0])), len(W_GRID) - 2)
    ty = (y - Y_GRID[iy]) / (Y_GRID[1] - Y_GRID[0])
    tw = (w - W_GRID[iw]) / (W_GRID[1] - W_GRID[0])
    return ((1 - ty) * (1 - tw) * A[iy, iw] + (1 - ty) * tw * A[iy, iw + 1]
            + ty * (1 - tw) * A[iy + 1, iw] + ty * tw * A[iy + 1, iw + 1])


def simulate(model, pol, zeta, kappa, full_info, y0):
    n = zeta.size
    y, w = y0, model.omega(model.kL * model.thL)
    Y, Wb, A, cens = np.empty(n), np.empty(n), np.empty(n), np.zeros(n, bool)
    cost = np.empty(n)
    for t in range(n):
        a = bilinear(pol, y, w)
        Y[t], Wb[t], A[t] = y, w, a
        cost[t] = y + kappa * a * a
        b = a - (1 - model.delta) * y
        y_next = max(zeta[t] - b, 0.0)
        if y_next <= 0.0:
            cens[t] = True
            w = model.omega(zeta[t]) if full_info else model.phi(w, b)
        else:
            w = model.omega(zeta[t])
        y = y_next
    return dict(Y=Y, w=Wb, a=A, censored=cens, cost=cost)


# ---------------------------------------------------------------- main
def main(cal_path, out_dir, kappas=None):
    global KAPPAS
    if kappas:
        KAPPAS = [float(k) for k in kappas.split(",")]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cal = json.load(open(cal_path))
    rows = []
    paths = {}
    for dkey, rec in cal.items():
        delta = float(dkey)
        model = Model(rec["fits"]["logistic"], delta)
        zeta = model.true_path(N_SIM + BURN, np.random.default_rng(2026))
        for kappa in KAPPAS:
            res = {}
            for full in (False, True):
                t0 = time.time()
                V, pol, errs = model.solve(kappa, full)
                sim = simulate(model, pol, zeta, kappa, full, y0=20.0)
                res[full] = (V, pol, errs, sim)
                print(f"delta={delta} kappa={kappa} full={full}: {len(errs)} it, "
                      f"err={errs[-1]:.1e}, {time.time() - t0:.0f}s")
            Vp, polp, _, sp = res[False]
            Vf, polf, _, sf = res[True]
            sl = slice(BURN, None)
            avg_p, avg_f = sp["cost"][sl].mean(), sf["cost"][sl].mean()
            # value-function gap at the median dry-season state
            y_med = 20.0
            w_grid_gap = (Vp - Vf)[int(y_med / 0.5)]
            pol_spread = (polp[:, -1] - polp[:, 0])[Y_GRID <= 40]
            rows.append(dict(
                delta=delta, kappa=kappa,
                avg_cost_partial=avg_p, avg_cost_full=avg_f,
                info_cost_pct=100 * (avg_p - avg_f) / avg_f,
                V_gap_pct_y20=float(100 * np.mean(w_grid_gap / Vf[int(y_med / 0.5)])),
                censor_freq=float(sp["censored"][sl].mean()),
                mean_stock=float(sp["Y"][sl].mean()),
                mean_action=float(sp["a"][sl].mean()),
                policy_spread_w=float(pol_spread.mean()),
            ))
            paths[(delta, kappa)] = (model, sp, zeta, polp)
            np.savez(out / f"solution_d{delta}_k{kappa}.npz", V_partial=Vp, pol_partial=polp,
                     V_full=Vf, pol_full=polf, y_grid=Y_GRID, w_grid=W_GRID)

    import pandas as pd
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "value_of_information.csv", index=False)
    print(tab.round(3).to_string(index=False))

    # figure 1: policy a*(y, w) for each delta (first kappa)
    kappa = KAPPAS[0]
    fig, axes = plt.subplots(1, len(cal), figsize=(12, 3.6), sharey=True)
    for ax, dkey in zip(axes, cal):
        d = np.load(out / f"solution_d{float(dkey)}_k{kappa}.npz")
        sel = Y_GRID <= 40
        for w, col in zip([0.0, 0.5, 1.0], ["#2c6fb5", "#777777", "#c0392b"]):
            iw = int(round(w / 0.025))
            ax.plot(Y_GRID[sel], d["pol_partial"][sel, iw], color=col, label=f"w = {w:g}")
        ax.set_title(f"δ = {dkey}")
        ax.set_xlabel("PM2.5 hoy, y (µg/m³)")
    axes[0].set_ylabel("abatimiento óptimo a*(y, w)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "rama_policy.png", dpi=150)

    # figure 2: belief vs truth, 60 days, delta=0.9, first kappa
    dmax = max(float(k) for k in cal)
    model, sp, zeta, _ = paths[(dmax, kappa)]
    t0, T = BURN + 200, 60
    tt = np.arange(T)
    truth_high = model.omega(zeta[t0 - 1:t0 + T - 1])  # P(today's regime is H | yesterday's true zeta)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    axes[0].plot(tt, sp["Y"][t0:t0 + T], color="#333333")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[1].plot(tt, sp["a"][t0:t0 + T], color="#333333")
    axes[1].set_ylabel("abatimiento")
    axes[2].plot(tt, truth_high, color="#999999", lw=2, label="con información completa: ω(ζ de ayer)")
    axes[2].plot(tt, sp["w"][t0:t0 + T], color="#c0392b", label="creencia w (observación censurada)")
    axes[2].set_ylabel("P(régimen alto)")
    axes[2].set_xlabel("día")
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")
    for c in np.where(sp["censored"][t0:t0 + T])[0]:
        for ax in axes:
            ax.axvline(c + 1, color="#2c6fb5", ls=":", lw=0.8)
    axes[0].set_title(f"Trayectoria simulada, δ = {dmax}, κ = {kappa} (líneas punteadas: días censurados)")
    fig.tight_layout()
    fig.savefig(out / "rama_belief_vs_truth.png", dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:4])
