"""
Project ecliptic ephemeris data onto Earth's surface.

Given a celestial body's ecliptic (longitude, latitude) and a UTC datetime,
this module computes:
  - the subpoint (lat, lon) where the body is at the zenith
  - the day/night terminator as a great-circle line
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

# Unicode glyphs for each sign, same order as ZODIAC_SIGNS.
ZODIAC_SYMBOLS = ["♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓"]

# Mean obliquity of the ecliptic — fine to 3 decimals for the modern era.
OBLIQUITY_DEG = 23.4393


def julian_date(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return 2440587.5 + dt.timestamp() / 86400.0


def gmst_deg(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time, degrees (IAU 1982)."""
    jd = julian_date(dt)
    t = (jd - 2451545.0) / 36525.0
    theta = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    return theta % 360.0


def ecliptic_to_equatorial(ecl_lon_deg, ecl_lat_deg, eps_deg: float = OBLIQUITY_DEG):
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


def terminator(sub_lat_deg: float, sub_lon_deg: float, n: int = 361):
    """
    Great-circle curve 90° from the subsolar point — the day/night boundary.
    Returns (lats, lons) arrays; points crossing the antimeridian are split
    with np.nan so plotting libraries draw a clean line.
    """
    lat0 = math.radians(sub_lat_deg)
    lon0 = math.radians(sub_lon_deg)
    bearings = np.linspace(0.0, 2.0 * math.pi, n)
    # Destination point formula with distance = 90°.
    lats = np.arcsin(np.cos(lat0) * np.cos(bearings))
    lons = lon0 + np.arctan2(
        np.sin(bearings) * np.cos(lat0),
        -np.sin(lat0) * np.sin(lats),
    )
    lats_deg = np.rad2deg(lats)
    lons_deg = (np.rad2deg(lons) + 180.0) % 360.0 - 180.0

    # Break the line where it jumps the antimeridian.
    jumps = np.where(np.abs(np.diff(lons_deg)) > 180.0)[0]
    if len(jumps) > 0:
        lats_deg = np.insert(lats_deg, jumps + 1, np.nan)
        lons_deg = np.insert(lons_deg, jumps + 1, np.nan)
    return lats_deg, lons_deg


def ascendant_deg(ramc_deg, lat_deg, eps_deg: float = OBLIQUITY_DEG):
    """
    Ecliptic longitude of the ascendant (rising point on the eastern horizon).
    Accepts scalars or numpy arrays for ramc_deg and lat_deg.
    """
    ramc = np.deg2rad(ramc_deg)
    lat = np.deg2rad(lat_deg)
    eps = math.radians(eps_deg)
    y = -np.cos(ramc)
    x = np.sin(ramc) * math.cos(eps) + np.tan(lat) * math.sin(eps)
    asc = np.rad2deg(np.arctan2(y, x))
    asc = asc % 360.0
    # Quadrant fix: atan2 returns the rising OR setting point; we want the
    # one in the eastern hemisphere, which by convention is the descendant
    # side flipped. If the result ends up on the wrong side of the meridian,
    # add 180°.
    diff = (asc - ramc_deg) % 360.0
    asc = np.where((diff < 180.0), asc, (asc + 180.0) % 360.0)
    return asc


def zodiac_sign_index(longitude_deg) -> int:
    return int((longitude_deg % 360.0) // 30.0)


def zodiac_sign_name(longitude_deg) -> str:
    return ZODIAC_SIGNS[zodiac_sign_index(longitude_deg)]


def ecliptic_sign_arcs(dt: datetime, n_per_sign: int = 16):
    """
    Split the ecliptic great circle into 12 subpoint arcs — one per
    zodiac sign — plus the subpoint of each sign's midpoint for labels.

    Each arc spans 30° of ecliptic longitude and is sampled with
    `n_per_sign + 1` points so that the end of arc i touches the start
    of arc i+1 and the band looks continuous.

    Returns:
        arcs:   list of 12 (lats, lons) tuples, in order Aries → Pisces
        labels: list of 12 (lat, lon) label anchors at the arc midpoints
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


def ecliptic_curve(dt: datetime, n: int = 361):
    """
    Subpoint of every ecliptic longitude 0°–360° at the moment `dt`.

    This traces the great circle where the plane of the ecliptic
    intersects Earth's surface (projected through the zenith). On an
    equirectangular map it draws a sinusoid with amplitude equal to the
    current obliquity — every planet's subpoint lies on this curve,
    because every planet sits on (or very near) the ecliptic.
    """
    lams = np.linspace(0.0, 360.0, n, endpoint=True)
    lats = np.empty(n, dtype=float)
    lons = np.empty(n, dtype=float)
    for i, lam in enumerate(lams):
        lats[i], lons[i] = subpoint(float(lam), 0.0, dt)
    # Split on antimeridian crossings so plotting libraries draw a clean
    # line rather than a streak across the whole map.
    jumps = np.where(np.abs(np.diff(lons)) > 180.0)[0]
    if len(jumps) > 0:
        lats = np.insert(lats, jumps + 1, np.nan)
        lons = np.insert(lons, jumps + 1, np.nan)
    return lats, lons


def rising_sign_grid(dt: datetime, lat_step: float = 4.0, lon_step: float = 4.0):
    """
    Sample the rising sign over a lat/lon grid at time dt.
    Returns (lats, lons, sign_indices) flat arrays, one entry per grid cell.
    Latitude is clipped to ±66° to avoid polar-circle singularities where
    certain signs never rise.
    """
    lats = np.arange(-64.0, 64.0 + lat_step / 2, lat_step)
    lons = np.arange(-180.0, 180.0 + lon_step / 2, lon_step)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    gmst = gmst_deg(dt)
    ramc = (gmst + lon_grid) % 360.0
    asc = ascendant_deg(ramc, lat_grid)
    sign_idx = (asc % 360.0) // 30.0

    return lat_grid.ravel(), lon_grid.ravel(), sign_idx.ravel().astype(int)
