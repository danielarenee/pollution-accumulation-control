"""Phase 3: calibrate the two-regime disturbance model of Section 7 to RAMA PM2.5.

Usage:
    python rama_calibrate.py <pm25_daily.csv> <output_dir>

Disturbance model (Section 7 of the manuscript, with free parameters):
    zeta_{t+1} | zeta_t = z  ~  (1 - omega(z)) f_L + omega(z) f_H,
    f_L = Gamma(kL, thL),  f_H = Gamma(kH, thH)  (mean_L < mean_H).
Persistence functions compared:
    "paper":    omega(z) = z / (z + c)                 (form used in Sec. 7)
    "logistic": omega(z) = 1 / (1 + exp(-(z - m)/s))
    "iid":      omega(z) = w0 (constant)  -> no dependence across days
Proposition 7.1 (belief reduces to the scalar w) holds for any measurable
omega with values in [0, 1].

For each delta in DELTAS:
  1. recover zeta_t = Y_{t+1} - (1 - delta) Y_t on consecutive dry-season days
     (Nov-May; no abatement assumed);
  2. drop holiday days (Dec 24-26, Dec 31-Jan 2: fireworks) and transitions
     with recovered emissions <= EPS (reported);
  3. maximise the explicit transition likelihood for each omega;
  4. compare by AIC; the logistic-vs-iid comparison isolates dependence,
     since both models have the same marginal flexibility.
Outputs: calibration.json, calibration_fit.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DELTAS = [0.5, 0.7, 0.9]
DRY = [11, 12, 1, 2, 3, 4, 5]
EPS = 0.5  # recovered emissions at or below this (ug/m3) are treated as invalid


HOLIDAYS = [(12, 24), (12, 25), (12, 26), (12, 31), (1, 1), (1, 2)]  # fireworks/bonfires


def recovered_pairs(daily: pd.DataFrame, delta: float):
    md = list(zip(daily.index.month, daily.index.day))
    holiday = np.array([x in HOLIDAYS for x in md])
    y = daily.pm25.where(daily.index.month.isin(DRY) & ~holiday)
    z = y.shift(-1) - (1 - delta) * y          # zeta_t, indexed by t
    pairs = pd.DataFrame({"z": z, "zn": z.shift(-1)}).dropna()
    n_all = len(pairs)
    pairs = pairs[(pairs.z > EPS) & (pairs.zn > EPS)]
    return pairs.z.to_numpy(), pairs.zn.to_numpy(), n_all


def omega(kind, z, q):
    if kind == "paper":
        return z / (z + q[0])
    if kind == "logistic":
        return 1.0 / (1.0 + np.exp(-(z - q[0]) / q[1]))
    return np.full_like(z, 1.0 / (1.0 + np.exp(-q[0])))  # iid


def unpack(kind, p):
    kL, thL, kH, thH = np.exp(p[:4])
    if kind == "paper":
        q = [np.exp(p[4])]
    elif kind == "logistic":
        q = [p[4], np.exp(p[5])]
    else:
        q = [p[4]]
    return kL, thL, kH, thH, q


def negloglik(p, kind, z, zn):
    kL, thL, kH, thH, q = unpack(kind, p)
    w = omega(kind, z, q)
    dens = (1 - w) * stats.gamma.pdf(zn, kL, scale=thL) + w * stats.gamma.pdf(zn, kH, scale=thH)
    return -np.sum(np.log(np.maximum(dens, 1e-300)))


def fit(kind, z, zn):
    m, sd = zn.mean(), zn.std()
    lo, hi = m - 0.5 * sd, m + 0.5 * sd
    starts = []
    for kl, kh in [(4, 8), (3, 15), (8, 8)]:
        base = [kl, lo / kl, kh, hi / kh]
        if kind == "paper":
            extra = [[np.log(m)], [np.log(0.5 * m)], [np.log(2 * m)]]
            starts += [np.r_[np.log(base), e] for e in extra]
        elif kind == "logistic":
            extra = [[m + o * sd, np.log(f * sd)]
                     for o in (-0.5, 0.0, 0.5) for f in (0.1, 0.3, 1.0)]
            starts += [np.r_[np.log(base), e] for e in extra]
        else:
            starts.append(np.r_[np.log(base), 0.0])
    best = None
    for s in starts:
        r = optimize.minimize(negloglik, s, args=(kind, z, zn), method="Nelder-Mead",
                              options={"maxiter": 40000, "maxfev": 40000,
                                       "xatol": 1e-9, "fatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    kL, thL, kH, thH, q = unpack(kind, best.x)
    if kL * thL > kH * thH:  # relabel so that L is the low-mean regime
        kL, thL, kH, thH = kH, thH, kL, thL
        if kind == "logistic":
            q = [q[0], -q[1]]  # omega -> 1 - omega
        elif kind == "iid":
            q = [-q[0]]
        else:
            raise RuntimeError("paper-form omega with swapped labels")
    k = len(best.x)
    return dict(kL=kL, thL=thL, kH=kH, thH=thH, q=[float(v) for v in q],
                mean_L=kL * thL, mean_H=kH * thH,
                nll=float(best.fun), aic=float(2 * best.fun + 2 * k))


def cond_mean(par, kind, z):
    w = omega(kind, z, par["q"])
    return (1 - w) * par["mean_L"] + w * par["mean_H"]


def main(csv: str, out_dir: str, column: str = "pm25") -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(csv, parse_dates=["date"], index_col="date")
    if column != "pm25":
        # other pollutants: rescale so the mean matches PM2.5 (20.55 ug/m3);
        # the model is unit-free, and this keeps grids and costs comparable
        s = daily[column]
        daily = pd.DataFrame({"pm25": s * 20.55 / s.mean()})
    results, rows = {}, []
    fig, axes = plt.subplots(2, len(DELTAS), figsize=(12, 6.5))
    for j, d in enumerate(DELTAS):
        z, zn, n_all = recovered_pairs(daily, d)
        fits = {kind: fit(kind, z, zn) for kind in ("paper", "logistic", "iid")}
        results[str(d)] = dict(delta=d, n_pairs=int(len(z)), n_dropped=int(n_all - len(z)),
                               corr_zeta=float(np.corrcoef(z, zn)[0, 1]), fits=fits)
        for kind, f in fits.items():
            rows.append(dict(delta=d, model=kind, aic=f["aic"], mean_L=f["mean_L"],
                             mean_H=f["mean_H"], kL=f["kL"], kH=f["kH"],
                             omega_params=np.round(f["q"], 2).tolist()))

        lg = fits["logistic"]
        # top: marginal fit (logistic model)
        ax = axes[0, j]
        grid = np.linspace(0.1, np.percentile(zn, 99.5), 300)
        wbar = omega("logistic", z, lg["q"]).mean()
        fL = (1 - wbar) * stats.gamma.pdf(grid, lg["kL"], scale=lg["thL"])
        fH = wbar * stats.gamma.pdf(grid, lg["kH"], scale=lg["thH"])
        ax.hist(zn, bins=40, density=True, color="#cccccc")
        ax.plot(grid, fL + fH, color="#333333", lw=1.5, label="mezcla (logística)")
        ax.plot(grid, fL, color="#2c6fb5", lw=1, ls="--", label="régimen bajo")
        ax.plot(grid, fH, color="#c0392b", lw=1, ls="--", label="régimen alto")
        ax.set_title(f"δ = {d}: emisiones recuperadas ζ")
        ax.set_xlabel("µg/m³")
        if j == 0:
            ax.legend(frameon=False, fontsize=8)

        # bottom: conditional mean E[zeta' | zeta], data vs models
        ax = axes[1, j]
        bins = np.quantile(z, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(z, bins[1:-1]), 0, 9)
        mz = np.array([z[idx == i].mean() for i in range(10)])
        mzn = np.array([zn[idx == i].mean() for i in range(10)])
        zz = np.linspace(np.percentile(z, 2), np.percentile(z, 98), 200)
        ax.plot(mz, mzn, "o", color="#333333", ms=4, label="datos (deciles)")
        ax.plot(zz, cond_mean(fits["logistic"], "logistic", zz), color="#c0392b", label="ω logística")
        ax.plot(zz, cond_mean(fits["paper"], "paper", zz), color="#2c6fb5", ls="--", label="ω = z/(z+c)")
        ax.axhline(zn.mean(), color="#999999", lw=0.8, ls=":", label="i.i.d.")
        ax.set_xlabel("ζ hoy (µg/m³)")
        ax.set_ylabel("E[ζ mañana | ζ hoy]")
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "calibration_fit.png", dpi=150)

    with open(out / "calibration.json", "w") as f:
        json.dump(results, f, indent=2)
    tab = pd.DataFrame(rows)
    tab["dAIC_vs_iid"] = tab.aic - tab.groupby("delta").aic.transform(
        lambda s: s[tab.loc[s.index, "model"] == "iid"].iloc[0])
    print(tab.round(2).to_string(index=False))
    for d, r in results.items():
        print(d, "pairs", r["n_pairs"], "dropped", r["n_dropped"], "corr", round(r["corr_zeta"], 3))


if __name__ == "__main__":
    main(*sys.argv[1:4])
