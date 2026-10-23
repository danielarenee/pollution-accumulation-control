"""Option 1: optimal policy of the paper vs. simple, contingency-style rules.

Usage:
    python rama_contingency.py <calibration.json> <solutions_dir> <output_dir>

Rules compared on the calibrated PM2.5 model (same true disturbance path):
  constant     a_t = a0                                   (best a0)
  threshold    a_t = a0 if Y_t <= T else a0 + a1          (best a0, a1, T)
               -> stylised "environmental contingency": extra abatement is
                  activated only when the observed level exceeds a threshold
  optimal      a_t = a*(Y_t, w_t)  from the paper's Bellman equation
  full info    benchmark in which disturbances are always revealed

Rule parameters are tuned by grid search on one simulated path (seed 11) and
evaluated on an independent path (seed 2026), so the comparison does not
favour the rules through over-fitting. Cost: long-run average of y + kappa a^2.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rama_solve import Model, simulate, KAPPAS, BURN  # noqa: E402

N_TUNE, N_EVAL = 20_000, 60_000
A0_GRID = np.arange(0.0, 40.0 + 1e-9, 0.5)
A1_GRID = np.arange(0.0, 30.0 + 1e-9, 1.0)
T_GRID = np.arange(0.0, 40.0 + 1e-9, 1.0)


@njit(cache=True)
def avg_cost_threshold(zeta, delta, kappa, a0, a1, T, burn):
    y, tot = 20.0, 0.0
    for t in range(zeta.size):
        a = a0 + a1 if y > T else a0
        if t >= burn:
            tot += y + kappa * a * a
        y = (1.0 - delta) * y + zeta[t] - a
        if y < 0.0:
            y = 0.0
    return tot / (zeta.size - burn)


@njit(cache=True)
def tune(zeta, delta, kappa, a0g, a1g, tg, burn):
    best = (1e300, 0.0, 0.0, 0.0)
    for a0 in a0g:
        for a1 in a1g:
            for T in tg:
                c = avg_cost_threshold(zeta, delta, kappa, a0, a1, T, burn)
                if c < best[0]:
                    best = (c, a0, a1, T)
                if a1 == 0.0:  # threshold irrelevant when a1 = 0
                    break
    return best


def main(cal_path, sol_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cal = json.load(open(cal_path))
    rows = []
    for dkey, rec in cal.items():
        delta = float(dkey)
        model = Model(rec["fits"]["logistic"], delta)
        z_tune = model.true_path(N_TUNE + BURN, np.random.default_rng(11))
        z_eval = model.true_path(N_EVAL + BURN, np.random.default_rng(2026))
        for kappa in KAPPAS:
            # constant rule = threshold rule with a1 = 0
            c_best = min((avg_cost_threshold(z_tune, delta, kappa, a0, 0.0, 0.0, BURN), a0)
                         for a0 in A0_GRID)
            _, a0c = c_best
            _, a0, a1, T = tune(z_tune, delta, kappa, A0_GRID, A1_GRID, T_GRID, BURN)
            cost_const = avg_cost_threshold(z_eval, delta, kappa, a0c, 0.0, 0.0, BURN)
            cost_thr = avg_cost_threshold(z_eval, delta, kappa, a0, a1, T, BURN)
            sol = np.load(Path(sol_dir) / f"solution_d{delta}_k{kappa}.npz")
            sp = simulate(model, sol["pol_partial"], z_eval, kappa, False, 20.0)
            sf = simulate(model, sol["pol_full"], z_eval, kappa, True, 20.0)
            c_opt, c_full = sp["cost"][BURN:].mean(), sf["cost"][BURN:].mean()
            rows.append(dict(
                delta=delta, kappa=kappa,
                const_a0=a0c, thr_a0=a0, thr_a1=a1, thr_T=T,
                cost_const=cost_const, cost_threshold=cost_thr,
                cost_optimal=c_opt, cost_full=c_full,
                const_vs_opt_pct=100 * (cost_const - c_opt) / c_opt,
                thr_vs_opt_pct=100 * (cost_thr - c_opt) / c_opt,
                opt_vs_full_pct=100 * (c_opt - c_full) / c_full,
            ))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "contingency_comparison.csv", index=False)
    print(tab.round(2).to_string(index=False))

    # figure: extra cost of each rule relative to the paper's optimal policy
    fig, axes = plt.subplots(1, len(KAPPAS), figsize=(10, 3.6), sharey=True)
    for ax, kappa in zip(np.atleast_1d(axes), KAPPAS):
        sub = tab[tab.kappa == kappa]
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub.const_vs_opt_pct, 0.4, color="#999999", label="regla constante")
        ax.bar(x + 0.2, sub.thr_vs_opt_pct, 0.4, color="#c0392b", label="regla de umbral (contingencia)")
        ax.set_xticks(x, [f"δ = {d}" for d in sub.delta])
        ax.set_title(f"abatimiento {'barato' if kappa == min(KAPPAS) else 'caro'} (κ = {kappa})")
        ax.axhline(0, color="#333333", lw=0.8)
    np.atleast_1d(axes)[0].set_ylabel("costo extra vs. política óptima (%)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "contingency_comparison.png", dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:4])
