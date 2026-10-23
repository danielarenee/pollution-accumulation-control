"""Second benchmark: value of tracking the belief at all.

Usage:
    python rama_value_of_beliefs.py <calibration.json> <output_dir>

Re-uses the solutions saved by rama_solve.py and simulates, on the same true
disturbance path, three decision makers:
  partial  - optimal policy of the paper (belief w updated, censored info);
  full     - optimal policy when disturbances are always revealed;
  naive    - same policy table, but w frozen at its long-run average, i.e. a
             planner that ignores the persistence of emissions.
Reports long-run average daily cost and the % gaps relative to `full`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rama_solve import Model, simulate, bilinear, KAPPAS, N_SIM, BURN  # noqa: E402


def simulate_naive(model, pol, zeta, kappa, w_bar, y0):
    y, cost = y0, np.empty(zeta.size)
    for t in range(zeta.size):
        a = bilinear(pol, y, w_bar)
        cost[t] = y + kappa * a * a
        y = max(zeta[t] - (a - (1 - model.delta) * y), 0.0)
    return cost


def main(cal_path, out_dir, kappas=None):
    ks = [float(k) for k in kappas.split(',')] if kappas else KAPPAS
    out = Path(out_dir)
    cal = json.load(open(cal_path))
    rows = []
    for dkey, rec in cal.items():
        delta = float(dkey)
        model = Model(rec["fits"]["logistic"], delta)
        zeta = model.true_path(N_SIM + BURN, np.random.default_rng(2026))
        for kappa in ks:
            sol = np.load(out / f"solution_d{delta}_k{kappa}.npz")
            sp = simulate(model, sol["pol_partial"], zeta, kappa, False, 20.0)
            sf = simulate(model, sol["pol_full"], zeta, kappa, True, 20.0)
            w_bar = float(sf["w"][BURN:].mean())
            cn = simulate_naive(model, sol["pol_partial"], zeta, kappa, w_bar, 20.0)
            sl = slice(BURN, None)
            cp, cf, cnv = sp["cost"][sl].mean(), sf["cost"][sl].mean(), cn[sl].mean()
            rows.append(dict(delta=delta, kappa=kappa, cost_full=cf, cost_partial=cp,
                             cost_naive=cnv,
                             censoring_loss_pct=100 * (cp - cf) / cf,
                             ignoring_persistence_loss_pct=100 * (cnv - cf) / cf,
                             censor_freq=float(sp["censored"][sl].mean()),
                             w_sd=float(sf["w"][sl].std())))
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "value_of_beliefs.csv", index=False)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
