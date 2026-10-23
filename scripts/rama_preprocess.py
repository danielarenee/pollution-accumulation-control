"""Phase 1: build a daily city-wide PM2.5 series from RAMA hourly files.

Usage:
    python rama_preprocess.py <dir_with_contaminantes_YYYY.csv> <output_dir>

Steps
1. Read the SIMAT hourly files (2022-2025), keep PM2.5 (units: ug/m3).
2. Align timestamps: 2022-2024 files label hours 01:00..24:00 (hour-ending),
   the 2025 file labels 00:00..23:00 (hour-beginning). We shift the former by
   one hour so that every record is labelled by the hour in which it starts.
3. Keep a fixed panel of stations with >= ~40% hourly coverage in every year,
   excluding the stations removed from the city index in 2015
   (ACO, AJU, INN, MON, MPA).
4. Station-day mean if >= 18 valid hours (75% completeness rule).
5. City-day value = mean over available panel stations of the station-day
   means, corrected for station level (so the series does not jump when a
   station is missing). Days with fewer than MIN_STATIONS valid stations are
   set to missing.

Outputs: pm25_daily.csv, pm25_daily.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEARS = [2022, 2023, 2024, 2025]
EXCLUDED = {"ACO", "AJU", "INN", "MON", "MPA"}
PANEL = ["BJU", "CAM", "CCA", "FAR", "MER", "NEZ", "PED", "SAC", "TLA", "UAX", "UIZ"]
MIN_HOURS = 18
MIN_STATIONS = 6


def read_year(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_csv(
        path, skiprows=10, header=None,
        names=["date", "station", "param", "value", "unit"], dtype=str,
    )
    df = df[df.param == "PM2.5"].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # midnight records sometimes appear as a bare date ("2024-01-02")
    df["time"] = pd.to_datetime(df["date"], format="ISO8601")
    if year <= 2024:  # hour-ending convention -> hour-beginning
        df["time"] -= pd.Timedelta(hours=1)
    return df[["time", "station", "value"]]


def main(raw_dir: str, out_dir: str) -> None:
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hourly = pd.concat(
        [read_year(raw / f"contaminantes_{y}.csv", y) for y in YEARS]
    )
    hourly = hourly[hourly.station.isin(PANEL)]
    assert not EXCLUDED & set(hourly.station.unique())
    hourly["day"] = hourly.time.dt.floor("D")
    hourly = hourly[hourly.day.dt.year.isin(YEARS)]

    # station-day means with completeness rule
    g = hourly.groupby(["station", "day"]).value
    sd = pd.DataFrame({"mean": g.mean(), "n": g.count()}).reset_index()
    sd = sd[sd.n >= MIN_HOURS]
    wide = sd.pivot(index="day", columns="station", values="mean")
    wide = wide.reindex(pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq="D"))

    # station-level correction: average anomalies, add back the panel mean
    level = wide.mean()
    anomaly = wide.sub(level, axis=1)
    n_st = wide.notna().sum(axis=1)
    city = anomaly.mean(axis=1) + level.mean()
    city[n_st < MIN_STATIONS] = np.nan

    daily = pd.DataFrame({"pm25": city.round(2), "n_stations": n_st})
    daily.index.name = "date"
    daily["season"] = np.where(daily.index.month.isin([6, 7, 8, 9, 10]), "rainy", "dry")
    daily.to_csv(out / "pm25_daily.csv")

    # figure
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for y in YEARS:
        ax.axvspan(pd.Timestamp(f"{y}-06-01"), pd.Timestamp(f"{y}-10-31"),
                   color="#4a90d9", alpha=0.10, lw=0)
    ax.plot(daily.index, daily.pm25, lw=0.8, color="#333333")
    ax.axhline(daily.pm25.mean(), ls="--", lw=0.8, color="#c0392b")
    ax.set_ylabel("PM2.5 diario (µg/m³)")
    ax.set_title("Promedio diario de PM2.5, CDMX (panel de 11 estaciones RAMA)")
    ax.text(0.01, 0.95, "sombreado: temporada de lluvias (jun–oct)",
            transform=ax.transAxes, fontsize=8, va="top", color="#2c6fb5")
    fig.tight_layout()
    fig.savefig(out / "pm25_daily.png", dpi=150)

    print(f"days: {len(daily)}  missing: {daily.pm25.isna().sum()}")
    print(daily.pm25.describe().round(2).to_string())
    print(daily.groupby("season").pm25.describe().round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
