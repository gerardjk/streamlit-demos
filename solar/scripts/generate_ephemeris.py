#!/usr/bin/env python
"""
Generate binary ephemeris data covering Sun, Moon, and Mercury-Pluto.

This is a one-time data generation script — you run it once to build the
binary files that the Streamlit app reads at runtime. You only need to
re-run it if you want to extend the date range or change the time resolution.

How it works:
  1. Loads JPL's DE440s planetary ephemeris kernel via the Skyfield library.
     DE440s is a ~31 MB binary file from NASA's Jet Propulsion Laboratory
     that encodes precise positions and velocities of all solar system bodies
     as Chebyshev polynomial coefficients. JPL builds these by fitting
     decades of observational data (radar ranging, spacecraft tracking,
     lunar laser ranging) into a numerical integration of gravitational
     equations. "DE" = Development Ephemeris, "440" = version, "s" = short
     time span (1849-2150 vs the full multi-millennium version).

  2. For each hour from 2000 to 2030, asks Skyfield: "as seen from Earth,
     what is each body's ecliptic longitude and latitude?" Skyfield
     evaluates the DE440s polynomials at that instant to get the answer.

  3. Packs everything into compact binary chunk files — one per month,
     ~55 KB each. Each entry is a fixed-width row: one int32 timestamp
     followed by float32 fields for each body's longitude and latitude.
     Also writes an ephemeris_metadata.json index that tells the runtime
     reader (ephemeris.py) what fields exist and where each month's data
     lives.

Output: monthly .bin chunks + ephemeris_metadata.json in solar/data/,
readable by ephemeris.py's EphemerisLoader.

First run auto-downloads the DE440s kernel (~31 MB) from JPL's servers
via Skyfield. Subsequent runs reuse the local copy.

Run with:  python solar/scripts/generate_ephemeris.py
"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from skyfield.api import load

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

START_YEAR = 2000
END_YEAR = 2030
STEP_MINUTES = 60   # hourly cadence — one entry per hour per month
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
MISSING_VALUE = -999.0  # sentinel for fields that can't be computed

# Canonical field layout — the binary format is just these fields packed
# in order, one after another. The reader (ephemeris.py) uses this same
# list to build its numpy structured dtype. Keep this stable so existing
# .bin files remain readable.
#
# Each body gets an ecliptic longitude (degrees) and latitude (degrees).
# The Sun has no latitude field because the Sun defines the ecliptic plane
# (its ecliptic latitude is always 0 by definition).
FIELDS = [
    "timestamp",       # int32: minutes since 2000-01-01 UTC
    "sun_lon",         # float32: ecliptic longitude in degrees
    "moon_lon",    "moon_lat",
    "mercury_lon", "mercury_lat",
    "venus_lon",   "venus_lat",
    "mars_lon",    "mars_lat",
    "jupiter_lon", "jupiter_lat",
    "saturn_lon",  "saturn_lat",
    "uranus_lon",  "uranus_lat",
    "neptune_lon", "neptune_lat",
    "pluto_lon",   "pluto_lat",
]
BYTES_PER_ENTRY = len(FIELDS) * 4  # fields * 4 bytes each

# Mapping from our short names to the body identifiers that Skyfield/DE440s
# expects. Inner planets use their name directly; outer planets use
# "barycenter" (the center of mass of the planet + its moons) because
# DE440s stores those more precisely than individual planet centers.
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

# All timestamps in the binary files are stored as minutes since this epoch.
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Astronomical computations
# ---------------------------------------------------------------------------


def ecliptic_lonlat(earth, body, t):
    """Ask Skyfield: "from Earth, what ecliptic lon/lat does this body have?"

    Skyfield evaluates the DE440s Chebyshev polynomials at time t to get
    the body's 3D position, then projects it onto the ecliptic plane.
    The .apparent() call includes light-time correction and aberration.
    """
    obs = earth.at(t).observe(body).apparent()
    lat, lon, _ = obs.ecliptic_latlon()
    return lat.degrees, lon.degrees


# ---------------------------------------------------------------------------
# Month generation & binary I/O
# ---------------------------------------------------------------------------


def generate_month(
    ts, earth, sun, moon, planets, year: int, month: int, step_minutes: int,
) -> dict[str, np.ndarray]:
    """Compute all ephemeris fields for every hour in a given month.

    Returns a dict mapping field name to a numpy array of values.
    """
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)

    total_minutes = int((end - start).total_seconds() // 60)
    n = total_minutes // step_minutes  # ~720-744 entries per month at hourly cadence

    # Build an array of Skyfield Time objects for the whole month at once.
    # Skyfield's vectorized API computes all timestamps in a single call
    # to the DE440s evaluator, which is much faster than one-at-a-time.
    offsets = np.arange(n) * step_minutes
    times = [start + timedelta(minutes=int(o)) for o in offsets]
    t = ts.utc(
        [d.year for d in times],
        [d.month for d in times],
        [d.day for d in times],
        [d.hour for d in times],
        [d.minute for d in times],
    )

    # Initialize all fields to the missing sentinel.
    data: dict[str, np.ndarray] = {
        name: np.full(n, MISSING_VALUE, dtype=np.float32) for name in FIELDS
    }

    # Timestamps stored as minutes since the epoch (2000-01-01 00:00 UTC).
    base_min = int((start - EPOCH).total_seconds() // 60)
    data["timestamp"] = (base_min + offsets).astype(np.int32)

    # Sun — only longitude (latitude is 0 by definition of the ecliptic).
    _, sun_lon = ecliptic_lonlat(earth, sun, t)
    data["sun_lon"] = sun_lon.astype(np.float32)

    # Moon — longitude and latitude.
    moon_lat, moon_lon = ecliptic_lonlat(earth, moon, t)
    data["moon_lon"] = moon_lon.astype(np.float32)
    data["moon_lat"] = moon_lat.astype(np.float32)

    # Planets — longitude and latitude for each.
    for name, body in planets.items():
        lat, lon = ecliptic_lonlat(earth, body, t)
        data[f"{name}_lon"] = lon.astype(np.float32)
        data[f"{name}_lat"] = lat.astype(np.float32)

    return data


def write_chunk(data: dict[str, np.ndarray], path: Path) -> int:
    """Write one month's data to a flat binary file.

    Format: for each time step, write the fields in FIELDS order. The
    timestamp is packed as a little-endian int32, all other fields as
    little-endian float32. No header, no padding — just raw values.
    The reader (ephemeris.py) uses the same FIELDS list to build a numpy
    structured dtype and reads the file with np.fromfile().
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Skyfield's timescale (for UTC -> Julian Date conversion) and
    # the DE440s kernel. On first run, Skyfield downloads de440s.bsp from
    # JPL's servers and caches it locally.
    # Point Skyfield's download directory at scripts/ so the DE440s kernel
    # lives next to this script, not wherever the user ran it from.
    scripts_dir = Path(__file__).resolve().parent
    load.directory = str(scripts_dir)

    print("Loading JPL DE440s ephemeris (first run may download ~30 MB)...")
    ts = load.timescale(builtin=True)
    eph = load("de440s.bsp")

    # Extract the body objects we'll query from the kernel.
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    planets = {name: eph[target] for name, target in PLANET_BODIES.items()}

    # Generate one binary chunk per month for the full date range.
    chunks_meta = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            print(f"  {year}-{month:02d}...", end=" ", flush=True)
            data = generate_month(ts, earth, sun, moon, planets, year, month, STEP_MINUTES)
            filename = f"ephemeris_{year}_{month:02d}.bin"
            n = write_chunk(data, OUTPUT_DIR / filename)
            start = datetime(year, month, 1, tzinfo=timezone.utc)
            end = datetime(
                year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc,
            ) - timedelta(minutes=STEP_MINUTES)
            chunks_meta.append({
                "year": year,
                "month": month,
                "filename": filename,
                "entries": n,
                "startTime": start.isoformat().replace("+00:00", "Z"),
                "endTime": end.isoformat().replace("+00:00", "Z"),
            })
            print(f"{n} entries")

    # Write the metadata index that ephemeris.py uses to find chunks.
    metadata = {
        "format": "chunked",
        "startTime": f"{START_YEAR}-01-01T00:00:00Z",
        "endTime": f"{END_YEAR}-12-31T23:59:00Z",
        "interval": STEP_MINUTES * 60_000,       # in milliseconds
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
