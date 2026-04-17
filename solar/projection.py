"""
Project ecliptic ephemeris data onto Earth's surface.

Given a celestial body's ecliptic (longitude, latitude) and a UTC datetime,
this module computes:
  - the subpoint (lat, lon) where the body is at the zenith
  - the ascendant (rising ecliptic degree) for any geographic point
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

ZODIAC_SYMBOLS = [
    "\u2648", "\u2649", "\u264a", "\u264b", "\u264c", "\u264d",
    "\u264e", "\u264f", "\u2650", "\u2651", "\u2652", "\u2653",
]

OBLIQUITY_DEG = 23.4393


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def julian_date(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return 2440587.5 + dt.timestamp() / 86400.0


def gmst_deg(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time in degrees (IAU 1982)."""
    jd = julian_date(dt)
    t = (jd - 2451545.0) / 36525.0
    theta = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    return theta % 360.0


def ecliptic_to_equatorial(
    ecl_lon_deg, ecl_lat_deg, eps_deg: float = OBLIQUITY_DEG,
):
    """Return (RA, Dec) in degrees. Accepts scalars or numpy arrays."""
    lam = np.deg2rad(ecl_lon_deg)
    beta = np.deg2rad(ecl_lat_deg)
    eps = math.radians(eps_deg)
    sin_dec = np.sin(beta) * math.cos(eps) + np.cos(beta) * math.sin(eps) * np.sin(lam)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    y = np.sin(lam) * math.cos(eps) - np.tan(beta) * math.sin(eps)
    x = np.cos(lam)
    ra = np.arctan2(y, x)
    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def subpoint(ecl_lon_deg: float, ecl_lat_deg: float, dt: datetime) -> tuple[float, float]:
    """Geographic (lat, lon) where the body is at the zenith."""
    ra, dec = ecliptic_to_equatorial(ecl_lon_deg, ecl_lat_deg)
    gmst = gmst_deg(dt)
    sub_lon = ((float(ra) - gmst + 180.0) % 360.0) - 180.0
    return float(dec), sub_lon


# ---------------------------------------------------------------------------
# Zodiac helpers
# ---------------------------------------------------------------------------

def ascendant_deg(ramc_deg, lat_deg, eps_deg: float = OBLIQUITY_DEG):
    """Ecliptic longitude of the ascendant (rising point on the eastern horizon).

    Accepts scalars or numpy arrays.
    """
    ramc = np.deg2rad(ramc_deg)
    lat = np.deg2rad(lat_deg)
    eps = math.radians(eps_deg)
    y = -np.cos(ramc)
    x = np.sin(ramc) * math.cos(eps) + np.tan(lat) * math.sin(eps)
    asc = np.rad2deg(np.arctan2(y, x)) % 360.0
    diff = (asc - ramc_deg) % 360.0
    asc = np.where(diff < 180.0, asc, (asc + 180.0) % 360.0)
    return asc


def zodiac_sign_index(longitude_deg) -> int:
    return int((longitude_deg % 360.0) // 30.0)


def zodiac_sign_name(longitude_deg) -> str:
    return ZODIAC_SIGNS[zodiac_sign_index(longitude_deg)]


def ecliptic_sign_arcs(dt: datetime, n_per_sign: int = 16):
    """Split the ecliptic into 12 subpoint arcs (one per zodiac sign).

    Returns:
        arcs:   list of 12 (lats, lons) tuples (Aries -> Pisces)
        labels: list of 12 (lat, lon) midpoint anchors for sign glyphs
    """
    arcs: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[tuple[float, float]] = []
    for i in range(12):
        lon_start = i * 30.0
        lams = np.linspace(lon_start, lon_start + 30.0, n_per_sign + 1)
        lats = np.empty(len(lams), dtype=float)
        lons = np.empty(len(lams), dtype=float)
        for j, lam in enumerate(lams):
            lats[j], lons[j] = subpoint(float(lam), 0.0, dt)
        arcs.append((lats, lons))
        mid_lat, mid_lon = subpoint(lon_start + 15.0, 0.0, dt)
        labels.append((mid_lat, mid_lon))
    return arcs, labels


def rising_sign_grid(dt: datetime, lat_step: float = 4.0, lon_step: float = 4.0):
    """Rising sign sampled over a lat/lon grid at time *dt*.

    Returns (lats, lons, sign_indices) flat arrays.
    Latitude clipped to +/-66 deg to avoid polar singularities.
    """
    lats = np.arange(-64.0, 64.0 + lat_step / 2, lat_step)
    lons = np.arange(-180.0, 180.0 + lon_step / 2, lon_step)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    gmst = gmst_deg(dt)
    ramc = (gmst + lon_grid) % 360.0
    asc = ascendant_deg(ramc, lat_grid)
    sign_idx = (asc % 360.0) // 30.0

    return lat_grid.ravel(), lon_grid.ravel(), sign_idx.ravel().astype(int)
