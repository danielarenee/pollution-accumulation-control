"""Export the small data files used by the presentation notebook."""
import json, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rama_calibrate import recovered_pairs, omega, cond_mean
from rama_solve import Model, simulate, BURN

OUT = Path("notebook/datos"); OUT.mkdir(parents=True, exist_ok=True)
shutil.copy("output/pm25_daily.csv", OUT / "pm25_diario.csv")
daily = pd.read_csv("output/pm25_daily.csv", parse_dates=["date"], index_col="date")
cal = json.load(open("output/calibration.json"))

# 1) calibration: data deciles vs model curves
rows, curves = [], []
for dkey, rec in cal.items():
    d = float(dkey)
    z, zn, _ = recovered_pairs(daily, d)
    bins = np.quantile(z, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(z, bins[1:-1]), 0, 9)
    for i in range(10):
        rows.append(dict(delta=d, hoy=z[idx == i].mean(), manana=zn[idx == i].mean()))
    zz = np.linspace(np.percentile(z, 2), np.percentile(z, 98), 100)
    for kind, name in [("logistic", "logistica"), ("paper", "paper")]:
        f = rec["fits"][kind]
        for a, b in zip(zz, cond_mean(f, kind, zz)):
            curves.append(dict(delta=d, modelo=name, hoy=a, manana=b))
    curves += [dict(delta=d, modelo="sin_persistencia", hoy=a, manana=zn.mean()) for a in zz]
pd.DataFrame(rows).to_csv(OUT / "calibracion_datos.csv", index=False)
pd.DataFrame(curves).to_csv(OUT / "calibracion_modelos.csv", index=False)

# 2) results table
vb = pd.read_csv("output/value_of_beliefs.csv")
cc = pd.read_csv("output/contingency_comparison.csv")
res = vb.merge(cc[["delta", "kappa", "thr_vs_opt_pct", "const_vs_opt_pct"]], on=["delta", "kappa"])
res = res.rename(columns={"censoring_loss_pct": "costo_censura_pct",
                          "ignoring_persistence_loss_pct": "costo_ignorar_persistencia_pct",
                          "thr_vs_opt_pct": "costo_regla_umbral_pct",
                          "const_vs_opt_pct": "costo_regla_constante_pct",
                          "censor_freq": "frac_dias_censurados"})
res[["delta", "kappa", "costo_censura_pct", "costo_ignorar_persistencia_pct",
     "costo_regla_umbral_pct", "costo_regla_constante_pct", "frac_dias_censurados"]].round(3)\
    .to_csv(OUT / "resultados.csv", index=False)

# 3) belief vs full information, delta = 0.9, kappa = 0.02
d, k = 0.9, 0.02
m = Model(cal[str(d)]["fits"]["logistic"], d)
zeta = m.true_path(60_000 + BURN, np.random.default_rng(2026))
sol = np.load(f"output/solution_d{d}_k{k}.npz")
sp = simulate(m, sol["pol_partial"], zeta, k, False, 20.0)
t0, T = BURN + 200, 60
pd.DataFrame(dict(dia=np.arange(T), contaminacion=sp["Y"][t0:t0+T], limpieza=sp["a"][t0:t0+T],
                  creencia=sp["w"][t0:t0+T], info_completa=m.omega(zeta[t0-1:t0+T-1]),
                  censurado_ayer=np.r_[False, sp["censored"][t0:t0+T-1]]))\
    .round(4).to_csv(OUT / "trayectoria.csv", index=False)
print(open(OUT / "resultados.csv").read())
