#!/usr/bin/env python
"""
Generate a synthetic Strait-of-Hormuz chokepoint-transits CSV that mimics
the schema IMF PortWatch publishes. Lets the dashboard run offline for
lecture demos without depending on a live download.

To swap in the real data, place the IMF PortWatch chokepoint-transits CSV
at portwatch/data/chokepoint_transits.csv with these columns (or adapt
the loader in app.py):

    date, chokepoint_name, cargo_type, transits
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "chokepoint_transits.csv"

# Approximate average daily transits at Hormuz by ship class.
BASE_RATES = {
    "tanker":        40.0,
    "dry_bulk":      15.0,
    "container":     10.0,
    "gas":            8.0,
    "general_cargo":  6.0,
    "ro_ro":          2.0,
}


def main() -> None:
    rng = np.random.default_rng(42)
    days = pd.date_range(date.today() - timedelta(days=400), date.today(), freq="D")

    rows = []
    for day in days:
        seasonal = 1.0 + 0.08 * np.sin(2 * np.pi * day.dayofyear / 365.25)
        dow_bump = 1.0 - 0.05 * (day.dayofweek >= 5)  # slight weekend dip
        for ship_type, base in BASE_RATES.items():
            mean = base * seasonal * dow_bump
            noise = rng.normal(0.0, mean * 0.09)
            count = int(max(0, round(mean + noise)))
            rows.append({
                "date": day.date().isoformat(),
                "chokepoint_name": "Strait of Hormuz",
                "cargo_type": ship_type,
                "transits": count,
            })

    df = pd.DataFrame(rows)

    # Inject a tanker-traffic dip so the lecture has something to point at.
    t_start = (date.today() - timedelta(days=140)).isoformat()
    t_end   = (date.today() - timedelta(days=100)).isoformat()
    dip = (df.date >= t_start) & (df.date <= t_end) & (df.cargo_type == "tanker")
    df.loc[dip, "transits"] = (df.loc[dip, "transits"] * 0.72).round().astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows ({df.date.min()} → {df.date.max()}) to {OUT}")


if __name__ == "__main__":
    main()
