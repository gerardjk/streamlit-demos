#!/usr/bin/env python
"""
Generate a binary ephemeris covering Sun, Moon, Mercury-Pluto, Moon's true
ascending node, and Black Moon Lilith. Output format matches ephemeris.py's
reader: monthly .bin chunks + one ephemeris_metadata.json.

Edit the CONFIG block to change range/cadence. First run will auto-download
the JPL DE440s kernel (~30 MB) via Skyfield.
"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from skyfield.api import load

# ============================ CONFIG ============================
START_YEAR = 2020
END_YEAR = 2030
STEP_MINUTES = 60  # hourly cadence; drop to 1 for minute resolution
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
MISSING_VALUE = -999.0

# Canonical field layout — keep stable so the reader doesn't need to change.
FIELDS = [
    "timestamp",
    "sun_lon",
    "moon_lon",    "moon_lat",    "moon_asc_node_lon",
    "mercury_lon", "mercury_lat", "mercury_retrograde",
    "venus_lon",   "venus_lat",   "venus_retrograde",
    "mars_lon",    "mars_lat",    "mars_retrograde",
    "jupiter_lon", "jupiter_lat", "jupiter_retrograde",
    "saturn_lon",  "saturn_lat",  "saturn_retrograde",
    "uranus_lon",  "uranus_lat",  "uranus_retrograde",
    "neptune_lon", "neptune_lat", "neptune_retrograde",
    "pluto_lon",   "pluto_lat",   "pluto_retrograde",
    "chiron_lon",  "chiron_retrograde",
    "ceres_lon",   "ceres_retrograde",
    "pallas_lon",  "pallas_retrograde",
    "juno_lon",    "juno_retrograde",
    "vesta_lon",   "vesta_retrograde",
    "eris_lon",    "eris_retrograde",
    "lilith_lon",  "lilith_retrograde",
]
BYTES_PER_ENTRY = len(FIELDS) * 4

PLANET_BODIES = {
    "mercury": "mercury",
    "venus":   "venus",
    "mars":    "mars barycenter",
    "jupiter": "jupiter barycenter",
    "saturn":  "saturn barycenter",
    "uranus":  "uranus barycenter",
    "neptune": "neptune barycenter",
    "pluto":   "pluto barycenter",
}

EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def ecliptic_lonlat(earth, body, t):
    obs = earth.at(t).observe(body).apparent()
    lat, lon, _ = obs.ecliptic_latlon()
    return lat.degrees, lon.degrees


def true_node_deg(t) -> np.ndarray:
    """Vectorized Moon true (osculating) ascending node, degrees."""
    jd = t.ut1
    T = (jd - 2451545.0) / 36525.0
    omega_mean = 125.0445479 - 1934.1362891 * T + 0.0020754 * T**2 + T**3 / 467441.0
    D  = 297.8501921 + 445267.1114034 * T - 0.0018819 * T**2
    M  = 357.5291092 + 35999.0502909  * T - 0.0001536 * T**2
    Mp = 134.9633964 + 477198.8675055 * T + 0.0087414 * T**2
    F  =  93.2720950 + 483202.0175233 * T - 0.0036539 * T**2
    Dr, Mr, Mpr, Fr = map(np.deg2rad, (D, M, Mp, F))
    delta = (
        -1.274 * np.sin(Mpr - 2 * Dr)
        + 0.658 * np.sin(2 * Dr)
        - 0.186 * np.sin(Mr)
        - 0.059 * np.sin(2 * Mpr - 2 * Dr)
        - 0.057 * np.sin(Mpr - 2 * Dr + Mr)
        + 0.053 * np.sin(Mpr + 2 * Dr)
        + 0.046 * np.sin(2 * Dr - Mr)
        + 0.041 * np.sin(Mpr - Mr)
        - 0.035 * np.sin(Dr)
        - 0.031 * np.sin(Mpr + Mr)
        - 0.015 * np.sin(2 * Fr - 2 * Dr)
        + 0.011 * np.sin(Mpr - 4 * Dr)
    )
    return (omega_mean + delta) % 360.0


def lilith_deg(t) -> np.ndarray:
    """Mean lunar apogee (Black Moon Lilith), degrees."""
    jd = t.ut1
    T = (jd - 2451545.0) / 36525.0
    pi = (83.353 + 4069.0137 * T - 0.0103 * T**2 - T**3 / 80053.0) % 360.0
    return (pi + 180.0) % 360.0


def compute_retrograde(lons: np.ndarray, step_minutes: int) -> np.ndarray:
    """1.0 where longitude is decreasing over a ~24h window, else 0.0."""
    window = max(1, int(round(1440 / step_minutes)))
    retro = np.zeros(len(lons), dtype=np.float32)
    if len(lons) <= window:
        return retro
    delta = lons[window:] - lons[:-window]
    delta = np.where(delta > 180, delta - 360, delta)
    delta = np.where(delta < -180, delta + 360, delta)
    retro[window:] = (delta < -0.001).astype(np.float32)
    retro[:window] = retro[window]
    return retro


def generate_month(
    ts, earth, sun, moon, planets, year: int, month: int, step_minutes: int
) -> np.ndarray:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

    total_minutes = int((end - start).total_seconds() // 60)
    n = total_minutes // step_minutes

    offsets = np.arange(n) * step_minutes
    times = [start + timedelta(minutes=int(o)) for o in offsets]
    t = ts.utc(
        [d.year for d in times],
        [d.month for d in times],
        [d.day for d in times],
        [d.hour for d in times],
        [d.minute for d in times],
    )

    data: dict[str, np.ndarray] = {
        name: np.full(n, MISSING_VALUE, dtype=np.float32) for name in FIELDS
    }
    # Timestamp = minutes since 2000-01-01
    base_min = int((start - EPOCH).total_seconds() // 60)
    data["timestamp"] = (base_min + offsets).astype(np.int32)

    _, sun_lon = ecliptic_lonlat(earth, sun, t)
    data["sun_lon"] = sun_lon.astype(np.float32)

    moon_lat, moon_lon = ecliptic_lonlat(earth, moon, t)
    data["moon_lon"] = moon_lon.astype(np.float32)
    data["moon_lat"] = moon_lat.astype(np.float32)
    data["moon_asc_node_lon"] = true_node_deg(t).astype(np.float32)

    for name, body in planets.items():
        lat, lon = ecliptic_lonlat(earth, body, t)
        data[f"{name}_lon"] = lon.astype(np.float32)
        data[f"{name}_lat"] = lat.astype(np.float32)
        data[f"{name}_retrograde"] = compute_retrograde(lon, step_minutes)

    data["lilith_lon"] = lilith_deg(t).astype(np.float32)
    data["lilith_retrograde"] = np.zeros(n, dtype=np.float32)

    return data


def write_chunk(data: dict[str, np.ndarray], path: Path) -> int:
    n = len(data["timestamp"])
    with open(path, "wb") as f:
        for i in range(n):
            for name in FIELDS:
                val = data[name][i]
                if name == "timestamp":
                    f.write(struct.pack("<i", int(val)))
                else:
                    f.write(struct.pack("<f", float(val)))
    return n


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading JPL DE440s ephemeris (first run may download ~30 MB)...")
    ts = load.timescale(builtin=True)
    eph = load("de440s.bsp")
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    planets = {name: eph[target] for name, target in PLANET_BODIES.items()}

    chunks_meta = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            print(f"  {year}-{month:02d}...", end=" ", flush=True)
            data = generate_month(ts, earth, sun, moon, planets, year, month, STEP_MINUTES)
            filename = f"ephemeris_{year}_{month:02d}.bin"
            n = write_chunk(data, OUTPUT_DIR / filename)
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            if month == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(minutes=STEP_MINUTES)
            else:
                end = datetime(year, month + 1, 1, tzinfo=timezone.utc) - timedelta(minutes=STEP_MINUTES)
            chunks_meta.append({
                "year": year,
                "month": month,
                "filename": filename,
                "entries": n,
                "startTime": start.isoformat().replace("+00:00", "Z"),
                "endTime": end.isoformat().replace("+00:00", "Z"),
            })
            print(f"{n} entries")

    metadata = {
        "format": "chunked",
        "startTime": f"{START_YEAR}-01-01T00:00:00Z",
        "endTime": f"{END_YEAR}-12-31T23:59:00Z",
        "interval": STEP_MINUTES * 60_000,
        "fieldsPerEntry": len(FIELDS),
        "bytesPerEntry": BYTES_PER_ENTRY,
        "missingDataValue": MISSING_VALUE,
        "fields": FIELDS,
        "fieldIndices": {name: i for i, name in enumerate(FIELDS)},
        "chunks": chunks_meta,
    }
    with open(OUTPUT_DIR / "ephemeris_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote {len(chunks_meta)} chunks + metadata to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
