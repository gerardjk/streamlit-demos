"""
Ephemeris Earth — project celestial bodies onto a world map.

Given a UTC datetime, this dashboard computes where each planet (Sun through
Pluto) sits at the zenith on Earth's surface, then plots those "subpoints"
on an orthographic globe with optional overlays: body tracks (fading trails
showing recent motion), the zodiac band (12 coloured ecliptic arcs), rising
signs (which zodiac sign is ascending at each lat/lon), and a day/night
terminator.

Data source: a pre-computed binary ephemeris covering 2000-2030, generated
by scripts/generate_ephemeris.py from JPL's DE440s planetary kernel via
the Skyfield library. The ephemeris stores ecliptic longitude and latitude
for each body at hourly resolution, packed into monthly binary chunk files
(~55 KB each). The EphemerisLoader in ephemeris.py reads these on demand
with an LRU cache, so the app never holds more than ~660 KB in memory.

The projection math (ecliptic -> equatorial -> geographic subpoint) lives
in projection.py and is pure numpy — no external astronomy library needed
at runtime.

Layout: positions table + controls on the left, orthographic globe with
animated scrub slider on the right.

Run with:  streamlit run solar/app.py
"""

import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ephemeris import EphemerisLoader
from projection import (
    ZODIAC_SIGNS,
    ZODIAC_SYMBOLS,
    ecliptic_sign_arcs,
    rising_sign_grid,
    subpoint,
    zodiac_sign_name,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"

# Each body is defined by (display_name, ecliptic_longitude_key, ecliptic_latitude_key).
# The keys reference fields in the binary ephemeris. lat_key is None for the
# Sun (which defines the ecliptic, so its latitude is always 0).
BODIES: list[tuple[str, str, str | None]] = [
    ("Sun",     "sun_lon",     None),
    ("Moon",    "moon_lon",    "moon_lat"),
    ("Mercury", "mercury_lon", "mercury_lat"),
    ("Venus",   "venus_lon",   "venus_lat"),
    ("Mars",    "mars_lon",    "mars_lat"),
    ("Jupiter", "jupiter_lon", "jupiter_lat"),
    ("Saturn",  "saturn_lon",  "saturn_lat"),
    ("Uranus",  "uranus_lon",  "uranus_lat"),
    ("Neptune", "neptune_lon", "neptune_lat"),
    ("Pluto",   "pluto_lon",   "pluto_lat"),
]

# Zodiac sign colours — one per 30-degree arc of the ecliptic.
# Each colour echoes the sign's classical elemental association
# (fire=warm, earth=green/brown, air=gold/blue, water=teal/cool).
SIGN_COLORS = [
    "#c92a2a",  # Aries
    "#5c8a3f",  # Taurus
    "#d4a017",  # Gemini
    "#75c0c0",  # Cancer
    "#e67e22",  # Leo
    "#3a6f5f",  # Virgo
    "#c77d99",  # Libra
    "#7b1538",  # Scorpio
    "#6b3fa0",  # Sagittarius
    "#6b4226",  # Capricorn
    "#2ba6cb",  # Aquarius
    "#4a9d82",  # Pisces
]

# Pale pastel palette for body markers — all around HSL(*, 60%, 85%)
# so they glow softly against the dark globe.
BODY_COLORS = {
    "Sun":     "#fff3b0",
    "Moon":    "#f5f7fb",
    "Mercury": "#d4d8e0",
    "Venus":   "#fcd5dc",
    "Mars":    "#ffc4a8",
    "Jupiter": "#fddab0",
    "Saturn":  "#fbe8b4",
    "Uranus":  "#c5f1ea",
    "Neptune": "#c4cffb",
    "Pluto":   "#dfd3fb",
}

# Traditional astrological/astronomical glyphs for the positions table.
BODY_SYMBOLS = {
    "Sun": "\u2609", "Moon": "\u263d", "Mercury": "\u263f", "Venus": "\u2640",
    "Mars": "\u2642", "Jupiter": "\u2643", "Saturn": "\u2644",
    "Uranus": "\u2645", "Neptune": "\u2646", "Pluto": "\u2647",
}

# Small city gazetteer for the "center globe on" selector. Not a full
# geocoder — just ~50 well-known places so the user can type a name
# and have the globe swing to that location.
CITIES: dict[str, tuple[float, float]] = {
    "london":         (51.51,   -0.13),
    "paris":          (48.86,    2.35),
    "berlin":         (52.52,   13.41),
    "madrid":         (40.42,   -3.70),
    "rome":           (41.90,   12.50),
    "amsterdam":      (52.37,    4.90),
    "copenhagen":     (55.68,   12.57),
    "stockholm":      (59.33,   18.07),
    "helsinki":        (60.17,   24.94),
    "dublin":         (53.35,   -6.26),
    "edinburgh":      (55.95,   -3.19),
    "reykjavik":      (64.15,  -21.94),
    "moscow":         (55.76,   37.62),
    "istanbul":       (41.01,   28.98),
    "cairo":          (30.04,   31.24),
    "dubai":          (25.20,   55.27),
    "mumbai":         (19.08,   72.88),
    "delhi":          (28.61,   77.21),
    "bangkok":        (13.76,  100.50),
    "singapore":      ( 1.35,  103.82),
    "kuala lumpur":   ( 3.14,  101.69),
    "jakarta":        (-6.21,  106.85),
    "manila":         (14.60,  120.98),
    "seoul":          (37.57,  126.98),
    "tokyo":          (35.68,  139.65),
    "beijing":        (39.90,  116.40),
    "hong kong":      (22.32,  114.17),
    "shanghai":       (31.23,  121.47),
    "sydney":         (-33.87, 151.21),
    "melbourne":      (-37.81, 144.96),
    "perth":          (-31.95, 115.86),
    "brisbane":       (-27.47, 153.02),
    "hobart":         (-42.88, 147.33),
    "wellington":     (-41.29, 174.78),
    "auckland":       (-36.85, 174.76),
    "honolulu":       ( 21.31, -157.86),
    "anchorage":      ( 61.22, -149.90),
    "vancouver":      ( 49.28, -123.12),
    "san francisco":  ( 37.77, -122.42),
    "los angeles":    ( 34.05, -118.24),
    "chicago":        ( 41.88,  -87.63),
    "toronto":        ( 43.65,  -79.38),
    "new york":       ( 40.71,  -74.01),
    "mexico city":    ( 19.43,  -99.13),
    "lima":           (-12.05,  -77.04),
    "rio de janeiro": (-22.91,  -43.17),
    "buenos aires":   (-34.61,  -58.38),
    "cape town":      (-33.92,   18.42),
    "johannesburg":   (-26.20,   28.05),
    "nairobi":        ( -1.29,   36.82),
    "lagos":          (  6.52,    3.38),
}

# Theme palette — dark navy with a cosmic feel.
BG      = "#060a16"
SURFACE = "#17223a"
BORDER  = "#2a3859"
MUTED   = "#8392b0"
FG      = "#e6edf7"

# Trail rendering constants.
TRAIL_HOURS = 24.0        # max trail length in hours of body motion
N_TRAIL_SEGMENTS = 16     # segments per trail (more = finer opacity fade)
N_TRACK_SAMPLES = 240     # total sample points across the track window

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _segment_bounds(start: int, end: int, n_seg: int) -> list[tuple[int, int]]:
    """Split [start, end) into *n_seg* contiguous slices with one-point
    overlap so adjacent segments touch visually."""
    out: list[tuple[int, int]] = []
    total = end - start
    if total <= 0:
        return out
    for si in range(n_seg):
        s0 = start + (total * si) // n_seg
        s1 = start + (total * (si + 1)) // n_seg + 1
        s1 = min(s1, end)
        if s1 - s0 >= 2:
            out.append((s0, s1))
    return out


def _segment_alpha(si: int) -> float:
    """Opacity for trail segment *si*: oldest (0) is nearly transparent,
    newest (N-1) is half-opaque. Trails are deliberately soft so the
    globe surface shows through."""
    return 0.04 + 0.46 * (si / max(1, N_TRAIL_SEGMENTS - 1))


def split_antimeridian(
    lats: np.ndarray, lons: np.ndarray, is_flat: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN breaks where a line crosses the +/-180 degree boundary.

    Only needed on flat (equirectangular) projections — on the orthographic
    globe the line wraps around the back of the sphere naturally.
    """
    if not is_flat:
        return lats, lons
    jumps = np.where(np.abs(np.diff(lons)) > 180.0)[0]
    if len(jumps) == 0:
        return lats, lons
    return np.insert(lats, jumps + 1, np.nan), np.insert(lons, jumps + 1, np.nan)


# ---------------------------------------------------------------------------
# Data loading
#
# Two caching layers work together:
#
# @st.cache_resource on get_loader() — creates ONE EphemerisLoader per
# server process and shares it across all reruns. The loader owns an
# internal LRU cache of monthly chunks, so jumping between months is
# fast after the first read.
#
# @st.cache_data on body_tracks() — memoises the computed track arrays
# by (dt_iso, window_hours). The _loader argument has a leading underscore
# so Streamlit skips it when building the cache key (EphemerisLoader is
# unhashable). This is safe because there's only one loader instance.
# ---------------------------------------------------------------------------


@st.cache_resource
def get_loader() -> EphemerisLoader:
    """Singleton ephemeris loader — cached across all reruns."""
    return EphemerisLoader(DATA_DIR)


@st.cache_data(show_spinner=False)
def body_tracks(
    _loader: EphemerisLoader,
    dt_iso: str,
    window_hours: float,
    n_samples: int = N_TRACK_SAMPLES,
) -> dict[str, list[tuple[float, float]]]:
    """Compute subpoint tracks for each body over [dt - window/2, dt + window/2].

    Returns a dict mapping body name to a list of (lat, lon) tuples.
    240 samples at a 24h window = ~6-minute cadence, finer than needed
    for a world map (1 degree of Earth rotation = ~4 minutes).
    """
    dt_center = datetime.fromisoformat(dt_iso)
    half = timedelta(hours=window_hours / 2)
    step = (2 * half) / max(1, n_samples - 1)
    tracks: dict[str, list[tuple[float, float]]] = {name: [] for name, _, _ in BODIES}
    for i in range(n_samples):
        t = dt_center - half + step * i
        entry = _loader.get(t)
        if not entry:
            continue
        for name, lon_key, lat_key in BODIES:
            ecl_lon = entry.get(lon_key)
            if ecl_lon is None:
                continue
            ecl_lat = entry.get(lat_key) if lat_key else 0.0
            if ecl_lat is None:
                ecl_lat = 0.0
            # Convert ecliptic coordinates to geographic subpoint using
            # the sidereal time at moment t.
            sub_lat, sub_lon = subpoint(ecl_lon, ecl_lat, t)
            tracks[name].append((sub_lat, sub_lon))
    return tracks


# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ephemeris \u2192 Earth",
    page_icon="\U0001f30d",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Hide Streamlit chrome for a clean dashboard look. */
      header[data-testid="stHeader"] { display: none !important; }
      div[data-testid="stToolbar"]   { display: none !important; }
      #MainMenu                      { display: none !important; }
      footer                         { visibility: hidden; }

      div.block-container {
          padding: 0.2rem 1.2rem 0.6rem 1.2rem;
      }
      h1, h2, h3, h4 {
          margin-top: 0.2rem !important;
          margin-bottom: 0.3rem !important;
          padding: 0 !important;
      }
      section[data-testid="stSidebar"] .stButton > button {
          padding: 0.25rem 0.25rem;
          font-size: 0.85rem;
      }

      /* Custom legend box (not currently used — Plotly's built-in legend
         is rendered instead, but the CSS is kept for potential future use). */
      .legend-box {
          background: #0e1628;
          border: 1px solid #1e2d4d;
          border-radius: 8px;
          padding: 0.55rem 0.6rem;
          font-size: 0.78rem;
          color: #cbd5e1;
          max-height: 520px;
          overflow-y: auto;
      }
      .legend-box .legend-section {
          font-size: 0.68rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #8392b0;
          margin-bottom: 0.25rem;
      }
      .legend-box .legend-row {
          display: flex; align-items: center; gap: 0.45rem;
          padding: 0.1rem 0; line-height: 1.15;
      }
      .legend-box .legend-swatch {
          display: inline-block; width: 10px; height: 10px;
          border-radius: 50%; flex-shrink: 0;
      }
      .legend-box .legend-gap { height: 0.55rem; }

      .control-strip-top { margin-top: 0.4rem; }

      /* Force dark styling on date/time/text inputs. */
      .stDateInput input, .stTimeInput input, .stTextInput input,
      div[data-testid="stDateInput"] input,
      div[data-testid="stTimeInput"] input,
      div[data-testid="stTextInput"] input,
      div[data-testid="stDateInputField"] input,
      div[data-testid="stTextInputRootElement"] input {
          background-color: #17223a !important;
          color: #ffffff !important;
          border: 1px solid #4b5d85 !important;
          caret-color: #ffffff !important;
      }
      .stTextInput input::placeholder,
      .stDateInput input::placeholder,
      div[data-testid="stTextInput"] input::placeholder {
          color: #8392b0 !important;
      }

      /* Compact step buttons (the +/- 1h/1d/1mo row). */
      .st-key-step_row .stButton > button {
          padding: 0.18rem 0.35rem !important;
          font-size: 0.72rem !important;
          min-height: 1.7rem !important;
          height: 1.7rem !important;
      }
      .st-key-bottom_ctrl div[data-testid="stSelectbox"] label,
      .st-key-bottom_ctrl div[data-testid="stSlider"] label {
          display: none;
      }

      /* Compressed toggle checkboxes so they stack tightly. */
      .st-key-key_toggles {
          margin-top: -68px !important;
          margin-right: -8px !important;
          position: relative;
          z-index: 20;
          transform: translateX(12px);
      }
      .st-key-key_toggles div[data-testid="stCheckbox"] label {
          font-size: 0.68rem !important;
          gap: 0.25rem !important;
      }
      .st-key-key_toggles div[data-testid="stCheckbox"] label > div:first-child {
          transform: scale(0.78);
          transform-origin: left center;
          margin-right: 0.15rem !important;
      }

      div[data-testid="stCheckbox"] {
          margin: 0 !important; padding: 0 !important;
      }
      div[data-testid="stCheckbox"] label {
          padding: 0 !important; gap: 0.35rem !important;
          font-size: 0.78rem !important; line-height: 1.0 !important;
      }
      div[data-testid="stCheckbox"] label > div:first-child {
          margin-right: 0.3rem !important;
      }
      div[data-testid="stCheckbox"] + div[data-testid="stCheckbox"] {
          margin-top: -4px !important;
      }

      /* Button styling — dark navy with amber highlight for active play. */
      .stButton > button,
      button[data-testid="stBaseButton-secondary"] {
          background-color: #1e2d4d !important;
          color: #e6edf7 !important;
          border: 1px solid #2a3859 !important;
          font-weight: 500 !important;
      }
      .stButton > button:hover,
      button[data-testid="stBaseButton-secondary"]:hover {
          background-color: #2a3859 !important;
          border-color: #4b5d85 !important;
          color: #ffffff !important;
      }
      .stButton > button[kind="primary"],
      button[data-testid="stBaseButton-primary"] {
          background-color: #fbbf24 !important;
          color: #0b1220 !important;
          border: 1px solid #fbbf24 !important;
          font-weight: 700 !important;
      }
      .stButton > button[kind="primary"]:hover,
      button[data-testid="stBaseButton-primary"]:hover {
          background-color: #f59e0b !important;
          border-color: #f59e0b !important;
          color: #0b1220 !important;
      }

      /* Positions table — dark cells with thin "wiring" borders. */
      div[data-testid="stDataFrame"] {
          background-color: #080f1e;
          border: 1px solid #1e2d4d;
      }
      div[data-testid="stDataFrame"] thead tr th {
          background-color: #080f1e !important;
          color: #8392b0 !important;
          font-weight: 600 !important;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          font-size: 0.7rem !important;
          border-bottom: 1px solid #2a3859 !important;
      }
      div[data-testid="stDataFrame"] tbody tr td {
          border-color: #1e2d4d !important;
      }

      /* Cosmic background gradient. */
      [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(ellipse at 20% 0%, #1a2950 0%, #0a1128 28%, #05080f 60%, #000000 100%) !important;
      }
      [data-testid="stAppViewContainer"] * { color: #e6edf7; }
      section[data-testid="stSidebar"] { background: #060a16 !important; }

      .legend-horiz {
          display: flex; flex-wrap: wrap; gap: 0.6rem 1.1rem;
          justify-content: center;
          padding: 0.5rem 0.75rem; margin-top: 0.4rem;
          background: rgba(14, 22, 40, 0.85);
          border: 1px solid #2a3859; border-radius: 8px;
          font-size: 0.78rem; color: #cbd5e1;
      }
      .legend-horiz .lh-item { display: inline-flex; align-items: center; gap: 0.35rem; }
      .legend-horiz .lh-sw {
          width: 10px; height: 10px; border-radius: 50%; display: inline-block;
      }

      /* Make bordered st.container panels match the dark theme. */
      div[data-testid="stVerticalBlockBorderWrapper"] {
          border-color: #2a3859 !important;
          background: rgba(14, 22, 40, 0.55) !important;
          border-radius: 8px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load ephemeris
# ---------------------------------------------------------------------------

try:
    loader = get_loader()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

start_iso, end_iso = loader.coverage

# ---------------------------------------------------------------------------
# Session state & callbacks
#
# Time navigation uses two levels:
#   anchor_dt  — set by the Date/Time inputs and step buttons ("big moves")
#   slider_dt  — fine-scrubbed within +/-(window/2) of the anchor
#
# The effective moment displayed everywhere is slider_dt. Moving a coarse
# control resets slider_dt to the new anchor so the slider re-centres.
# ---------------------------------------------------------------------------

if "anchor_dt" not in st.session_state:
    # Default to the 2024 summer solstice — a visually interesting moment
    # with the Sun at maximum declination.
    init = datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc)
    st.session_state.anchor_dt = init
    st.session_state.d_input = init.date()
    st.session_state.t_input = init.time().replace(microsecond=0)
    st.session_state.slider_dt = init
if "play_dir" not in st.session_state:
    st.session_state.play_dir = None  # None | "forward" | "backward"
if "globe_center" not in st.session_state:
    st.session_state.globe_center = (20.0, 0.0)  # (lat, lon)
if "location_query" not in st.session_state:
    st.session_state.location_query = None


def _on_location_change() -> None:
    """Callback when the user picks a city from the globe-centre selector."""
    pick = st.session_state.location_query
    if pick and pick in CITIES:
        st.session_state.globe_center = CITIES[pick]


def _toggle_play(direction: str) -> None:
    """Toggle auto-play in the given direction. Clicking the already-active
    direction stops playback."""
    st.session_state.play_dir = (
        None if st.session_state.play_dir == direction else direction
    )


def _shift(delta: timedelta) -> None:
    """Jump the anchor by *delta* (step buttons) and re-centre the slider."""
    new = st.session_state.anchor_dt + delta
    st.session_state.anchor_dt = new
    st.session_state.d_input = new.date()
    st.session_state.t_input = new.time().replace(microsecond=0)
    st.session_state.slider_dt = new


def _sync_from_widgets() -> None:
    """Called when the user edits the Date or Time input directly."""
    new = datetime.combine(
        st.session_state.d_input,
        st.session_state.t_input,
        tzinfo=timezone.utc,
    )
    st.session_state.anchor_dt = new
    st.session_state.slider_dt = new


# ---------------------------------------------------------------------------
# Auto-play advance
#
# Must run BEFORE the slider widget renders — Streamlit raises an error
# if you mutate a widget's session_state key after the widget has rendered.
# At the end of the script we sleep + st.rerun() to trigger the next tick.
# ---------------------------------------------------------------------------

if st.session_state.play_dir is not None:
    wh = st.session_state.get("window_hours", 168)
    half_td = timedelta(hours=wh / 2)
    smin = st.session_state.anchor_dt - half_td
    smax = st.session_state.anchor_dt + half_td
    # ~100 ticks across the full window for a ~12-second sweep.
    tick_td = timedelta(minutes=max(1, int(round(wh * 60 / 100))))
    cur = st.session_state.slider_dt

    nxt = cur + tick_td if st.session_state.play_dir == "forward" else cur - tick_td

    if nxt > smax or nxt < smin:
        # Overflowed the window — slide the whole window so the new
        # moment lands at its centre.
        st.session_state.anchor_dt = nxt
        st.session_state.slider_dt = nxt
        st.session_state.d_input = nxt.date()
        st.session_state.t_input = nxt.time().replace(microsecond=0)
    else:
        st.session_state.slider_dt = nxt

    # Stop if we've run off the end of the ephemeris coverage.
    cov_end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    cov_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    if st.session_state.slider_dt >= cov_end or st.session_state.slider_dt <= cov_start:
        st.session_state.play_dir = None

# ---------------------------------------------------------------------------
# Widget defaults
#
# Established before widgets render so the visualization code can read
# them on the first run (before the user has interacted with anything).
# ---------------------------------------------------------------------------

_defaults = {
    "window_hours": 168,          # track window: 7 days
    "show_zodiac_band": True,
    "show_rising": False,
    "show_body_tracks": True,
    "show_day_night": True,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

projection = "Globe"
layer = "Body tracks" if st.session_state["show_body_tracks"] else "None"
window_hours = st.session_state["window_hours"]
show_zodiac_band = st.session_state["show_zodiac_band"]
show_rising = st.session_state["show_rising"]
show_day_night = st.session_state["show_day_night"]

# Clamp scrub slider to current window bounds.
half = timedelta(hours=window_hours / 2.0)
slider_min_global = st.session_state.anchor_dt - half
slider_max_global = st.session_state.anchor_dt + half
st.session_state.slider_dt = max(
    slider_min_global, min(slider_max_global, st.session_state.slider_dt)
)

# The effective moment used by everything below.
dt = st.session_state.slider_dt

# ---------------------------------------------------------------------------
# Title row
# ---------------------------------------------------------------------------

title_cols = st.columns([3, 2])
title_cols[0].markdown("### \U0001f30d  Ephemeris \u2192 Earth")
title_cols[1].markdown(
    f"<div style='text-align:right; padding-top:0.55rem; color:#8392b0; font-size:0.95rem;'>"
    f"<b style='color:#e6edf7;'>{dt.strftime('%Y-%m-%d %H:%M')}</b> UTC \u00b7 "
    f"{dt.strftime('%A')}</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='margin:0.2rem 0 0.8rem 0; border-color:#1e2d4d;'>",
    unsafe_allow_html=True,
)

TABLE_COL, MAP_COL = st.columns([1.5, 2.5], gap="small")

# ---------------------------------------------------------------------------
# Fetch ephemeris for current moment
#
# loader.get(dt) looks up the monthly chunk for dt's year/month, binary-
# searches for the nearest entry, and returns a dict of field values.
# If the month's chunk is already in the loader's LRU cache, this is a
# pure in-memory lookup with zero disk I/O.
# ---------------------------------------------------------------------------

entry = loader.get(dt)
if entry is None:
    st.warning("No ephemeris data for that moment \u2014 pick a date inside the coverage window.")
    st.stop()

# Convert each body's ecliptic coordinates to a geographic subpoint
# (the lat/lon where the body is directly overhead).
subpoints: list[tuple[str, float, float, float]] = []  # (name, sub_lat, sub_lon, ecl_lon)
for name, lon_key, lat_key in BODIES:
    ecl_lon = entry.get(lon_key)
    if ecl_lon is None:
        continue
    ecl_lat = entry.get(lat_key) if lat_key else 0.0
    if ecl_lat is None:
        ecl_lat = 0.0
    sub_lat, sub_lon = subpoint(ecl_lon, ecl_lat, dt)
    subpoints.append((name, sub_lat, sub_lon, ecl_lon))

# ---------------------------------------------------------------------------
# Build the map figure
#
# Layers are added bottom-to-top:
#   1. Rising-sign grid (background wash)
#   2. Body tracks (fading trail segments)
#   3. Day/night terminator
#   4. Zodiac band (12 coloured arcs)
#   5. Current subpoint markers (topmost)
# ---------------------------------------------------------------------------

projection_type = "orthographic"
is_flat = False

fig = go.Figure()

# --- Rising-sign grid ---
# A lat/lon grid coloured by which zodiac sign is ascending on the eastern
# horizon at each location. Useful for seeing how the rising sign changes
# with geography and time.
if show_rising:
    grid_lats, grid_lons, sign_idx = rising_sign_grid(dt, lat_step=3.0, lon_step=3.0)
    for i, sign in enumerate(ZODIAC_SIGNS):
        mask = sign_idx == i
        if not mask.any():
            continue
        fig.add_trace(go.Scattergeo(
            lat=grid_lats[mask], lon=grid_lons[mask],
            mode="markers",
            marker=dict(size=6, color=SIGN_COLORS[i], opacity=0.35, line=dict(width=0)),
            name=f"\u2191 {sign}",
            hovertemplate="%{lat:.0f}\u00b0, %{lon:.0f}\u00b0<br>Rising: " + sign + "<extra></extra>",
            showlegend=False,
        ))

# --- Body tracks ---
# For each body, draw a fading trail showing where its subpoint has been
# over the last ~24 hours. The trail is split into N_TRAIL_SEGMENTS short
# line traces, each with decreasing opacity from newest to oldest, creating
# a "comet tail" effect. Plotly doesn't support per-vertex opacity on geo
# lines, so multi-trace segmentation is the only way to get a smooth fade.
track_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
trail_samples = 0

if layer == "Body tracks":
    tracks = body_tracks(loader, dt.isoformat(), float(window_hours))
    trail_samples = max(
        N_TRAIL_SEGMENTS + 1,
        int(round(N_TRACK_SAMPLES * TRAIL_HOURS / max(TRAIL_HOURS, window_hours))),
    )
    for name, _, _ in BODIES:
        pts = tracks.get(name, [])
        if not pts:
            continue
        lats = np.array([p[0] for p in pts], dtype=float)
        lons = np.array([p[1] for p in pts], dtype=float)
        track_arrays[name] = (lats, lons)

        # The trail ends at "now" (the middle of the track span) and
        # extends backward in time.
        now_idx = len(lats) // 2
        end = now_idx + 1
        start = max(0, end - trail_samples)
        seg_bounds = _segment_bounds(start, end, N_TRAIL_SEGMENTS)

        for si in range(N_TRAIL_SEGMENTS):
            if si < len(seg_bounds):
                s0, s1 = seg_bounds[si]
                s_lats, s_lons = split_antimeridian(lats[s0:s1], lons[s0:s1], is_flat)
            else:
                s_lats = np.array([np.nan])
                s_lons = np.array([np.nan])
            fig.add_trace(go.Scattergeo(
                lat=s_lats, lon=s_lons, mode="lines",
                line=dict(color=BODY_COLORS[name], width=1.8),
                opacity=_segment_alpha(si),
                name=name, showlegend=False, legendgroup=name,
                hoverinfo="skip",
            ))

        # Legend proxy: an invisible dot that gives the body a legend entry
        # styled as a solid circle (matching the "Now" markers).
        fig.add_trace(go.Scattergeo(
            lat=[None], lon=[None], mode="markers",
            marker=dict(size=11, color=BODY_COLORS[name], line=dict(color="#ffffff", width=1.5)),
            name=name, legendgroup=name, showlegend=True, hoverinfo="skip",
        ))

# --- Day/night terminator ---
# The terminator is a great circle 90 degrees from the subsolar point.
# On the orthographic globe we draw it as a filled polygon; on flat
# projections we fall back to a grid of marker dots (the polygon would
# wrap incorrectly across the antimeridian).
sun_sp = next((s for s in subpoints if s[0] == "Sun"), None)
if show_day_night and sun_sp is not None:
    sun_lat, sun_lon = sun_sp[1], sun_sp[2]

    N_TERM = 181
    lat0 = math.radians(sun_lat)
    lon0 = math.radians(sun_lon)
    bear = np.linspace(0.0, 2.0 * math.pi, N_TERM)
    tlat = np.arcsin(np.cos(lat0) * np.cos(bear))
    tlon = lon0 + np.arctan2(
        np.sin(bear) * np.cos(lat0),
        -np.sin(lat0) * np.sin(tlat),
    )
    tlat_deg = np.rad2deg(tlat)
    tlon_deg = (np.rad2deg(tlon) + 180.0) % 360.0 - 180.0

    if projection_type == "orthographic":
        fig.add_trace(go.Scattergeo(
            lat=tlat_deg, lon=tlon_deg, mode="lines",
            line=dict(color="#fbbf24", width=1.2),
            fill="toself", fillcolor="rgba(255, 240, 180, 0.16)",
            name="Day side", hoverinfo="skip", showlegend=False,
        ))
    else:
        # Flat-map fallback: scatter dots coloured by solar altitude.
        ws_lats = np.arange(-87.0, 87.1, 3.0)
        ws_lons = np.arange(-177.0, 177.1, 3.0)
        lon_g, lat_g = np.meshgrid(ws_lons, ws_lats)
        cos_d = (
            np.sin(math.radians(sun_lat)) * np.sin(np.deg2rad(lat_g))
            + np.cos(math.radians(sun_lat))
              * np.cos(np.deg2rad(lat_g))
              * np.cos(np.deg2rad(lon_g - sun_lon))
        )
        day_mask = cos_d > 0.0
        alpha = np.clip(cos_d[day_mask], 0.0, 1.0) ** 0.6 * 0.4
        fig.add_trace(go.Scattergeo(
            lat=lat_g[day_mask], lon=lon_g[day_mask], mode="markers",
            marker=dict(size=9, color="#fef3c7", opacity=alpha.tolist(), line=dict(width=0)),
            name="Day side", hoverinfo="skip", showlegend=False,
        ))

# --- Zodiac band ---
# The ecliptic great circle split into 12 coloured 30-degree arcs
# (Aries through Pisces). Each body's subpoint sits on the arc
# corresponding to its zodiac sign. Sign glyphs are placed at midpoints.
if show_zodiac_band:
    arcs, labels = ecliptic_sign_arcs(dt)
    for i, (a_lats, a_lons) in enumerate(arcs):
        s_lats, s_lons = split_antimeridian(a_lats, a_lons, is_flat)
        fig.add_trace(go.Scattergeo(
            lat=s_lats, lon=s_lons, mode="lines",
            line=dict(color=SIGN_COLORS[i], width=2.5),
            opacity=0.75,
            name=f"{ZODIAC_SYMBOLS[i]} {ZODIAC_SIGNS[i]}",
            hovertemplate=(
                f"{ZODIAC_SYMBOLS[i]} {ZODIAC_SIGNS[i]}<br>"
                "%{lat:.2f}\u00b0, %{lon:.2f}\u00b0<extra></extra>"
            ),
            legendgroup="zodiac",
            legendgrouptitle_text="Zodiac band" if i == 0 else None,
        ))
    fig.add_trace(go.Scattergeo(
        lat=[lbl[0] for lbl in labels],
        lon=[lbl[1] for lbl in labels],
        text=ZODIAC_SYMBOLS,
        mode="text",
        textfont=dict(color="#f8fafc", size=18),
        hoverinfo="skip", showlegend=False,
    ))

# --- Current subpoint markers ---
# One dot per body at the geographic point where that body is currently
# at the zenith. Coloured by the body's palette colour with a white
# outline for contrast against any overlay.
fig.add_trace(go.Scattergeo(
    lat=[s[1] for s in subpoints],
    lon=[s[2] for s in subpoints],
    text=[s[0] for s in subpoints],
    mode="markers+text",
    textposition="top center",
    textfont=dict(color=FG, size=12, family="Inter, -apple-system, system-ui"),
    marker=dict(
        size=15,
        color=[BODY_COLORS[s[0]] for s in subpoints],
        line=dict(color="#ffffff", width=2),
        opacity=0.95,
    ),
    name="Now",
    hovertemplate="%{text}<br>%{lat:.2f}\u00b0, %{lon:.2f}\u00b0<extra></extra>",
    showlegend=False,
))

# --- Globe projection & layout ---
ctr_lat, ctr_lon = st.session_state.globe_center
fig.update_geos(
    projection_type=projection_type,
    projection_rotation=dict(lon=ctr_lon, lat=ctr_lat, roll=0),
    showcoastlines=True,  coastlinecolor="#334773", coastlinewidth=0.6,
    showland=True,        landcolor="#0b1220",
    showocean=True,       oceancolor="#03070f",
    showlakes=False,
    showcountries=True,   countrycolor="#1a253f", countrywidth=0.4,
    showframe=False,
    bgcolor="#03070f",
)

MAP_HEIGHT = 540
fig.update_layout(
    height=MAP_HEIGHT,
    margin=dict(l=0, r=110, t=42, b=2),
    paper_bgcolor="#03070f",
    plot_bgcolor="#03070f",
    dragmode="pan",  # drag-to-rotate on orthographic projection
    legend=dict(
        orientation="v",
        yanchor="top", y=1.0, xanchor="left", x=1.01,
        bgcolor="rgba(3,7,15,0.85)",
        bordercolor=BORDER, borderwidth=1,
        font=dict(color=FG, size=9),
        itemsizing="constant", tracegroupgap=0,
    ),
    # uirevision keeps drag-rotation state across data updates, but resets
    # when the user picks a new city (changing the globe_center).
    uirevision=f"solar-{ctr_lat:.2f}-{ctr_lon:.2f}",
    showlegend=(layer == "Body tracks" or show_zodiac_band),
)

# ---------------------------------------------------------------------------
# Render: table column (left)
#
# Shows a styled positions table (body, ecliptic longitude, zodiac sign,
# geographic subpoint), date/time inputs, globe-centre selector, and
# four overlay toggle checkboxes.
# ---------------------------------------------------------------------------

with TABLE_COL:
    # Build the positions table from the computed subpoints.
    rows = []
    for name, sub_lat, sub_lon, ecl_lon in subpoints:
        rows.append({
            "Body":       f"{BODY_SYMBOLS.get(name, '')}  {name}",
            "Ecliptic \u03bb": f"{ecl_lon:7.3f}\u00b0",
            "Zodiac":     zodiac_sign_name(ecl_lon),
            "Sub-lat":    f"{sub_lat:+6.2f}\u00b0",
            "Sub-lon":    f"{sub_lon:+7.2f}\u00b0",
        })
    positions_df = pd.DataFrame(rows)

    # Cell styling: Body column tinted by body colour, Zodiac column
    # tinted by sign colour, other columns dark.
    DARK_CELL = "background-color: #0a1124; color: #e6edf7; border: 1px solid #1e2d4d;"

    def _style_row(row: pd.Series) -> list[str]:
        body_name = row["Body"].split()[-1] if row["Body"] else ""
        body_col = BODY_COLORS.get(body_name) or BODY_COLORS.get(row["Body"], "")
        try:
            sign_col = SIGN_COLORS[ZODIAC_SIGNS.index(row["Zodiac"])]
        except (ValueError, IndexError):
            sign_col = ""
        body_cell = (
            f"background-color: {body_col}; color: #0b1220; font-weight: 700;"
            f" border: 1px solid #1e2d4d;"
            if body_col else DARK_CELL
        )
        sign_cell = (
            f"background-color: {sign_col}; color: #ffffff; font-weight: 700;"
            f" border: 1px solid #1e2d4d;"
            if sign_col else DARK_CELL
        )
        return [body_cell, DARK_CELL, sign_cell, DARK_CELL, DARK_CELL]

    # Render the table via pandas Styler -> raw HTML (not st.dataframe)
    # so we have full control over header and cell styling.
    styled = (
        positions_df.style
        .hide(axis="index")
        .apply(_style_row, axis=1)
        .set_table_styles([
            {"selector": "",
             "props": [
                 ("border-collapse", "collapse"),
                 ("border", "1px solid #1e2d4d"),
                 ("width", "100%"),
                 ("background-color", "#080f1e"),
                 ("font-variant-numeric", "tabular-nums"),
                 ("font-size", "0.85rem"),
             ]},
            {"selector": "thead th",
             "props": [
                 ("background-color", "#c6cfe0"),
                 ("color", "#0b1220"),
                 ("font-weight", "700"),
                 ("text-transform", "uppercase"),
                 ("letter-spacing", "0.08em"),
                 ("font-size", "0.7rem"),
                 ("padding", "0.5rem 0.7rem"),
                 ("text-align", "left"),
                 ("border-bottom", "1px solid #2a3859"),
                 ("border-right", "1px solid #8892a8"),
             ]},
            {"selector": "tbody td",
             "props": [
                 ("padding", "0.35rem 0.7rem"),
                 ("border", "1px solid #1e2d4d"),
             ]},
        ])
    )

    # Date/time inputs — changing these calls _sync_from_widgets which
    # updates anchor_dt and slider_dt, triggering a full rerun.
    ti_cols = st.columns([1, 1])
    ti_cols[0].date_input("Date (UTC)", key="d_input", on_change=_sync_from_widgets)
    ti_cols[1].time_input("Time (UTC)", key="t_input", on_change=_sync_from_widgets, step=3600)

    # Globe centre selector — searchable dropdown of ~50 cities.
    city_names = sorted(CITIES.keys(), key=lambda s: s.title())
    st.selectbox(
        "Center globe on",
        options=city_names,
        index=None,
        placeholder="\u2014 pick a place \u2014",
        format_func=lambda s: s.title(),
        key="location_query",
        on_change=_on_location_change,
        help="Click and type to filter \u2014 the globe swings to the city you pick.",
    )

    with st.container(border=True):
        st.html(styled.to_html())

    # Overlay toggle checkboxes.
    tc = st.columns(4)
    tc[0].checkbox("Body tracks", key="show_body_tracks")
    tc[1].checkbox("Zodiac band", key="show_zodiac_band")
    tc[2].checkbox("Rising signs", key="show_rising")
    tc[3].checkbox("Day / night", key="show_day_night")

# ---------------------------------------------------------------------------
# Render: map column (right)
#
# The globe, scrub slider with play/pause buttons, and step buttons.
# ---------------------------------------------------------------------------

with MAP_COL:
    with st.container(border=True):
        st.plotly_chart(
            fig, width="stretch",
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                "doubleClick": "reset",
                "modeBarButtonsToRemove": [
                    "lasso2d", "select2d", "autoScale2d",
                    "toggleSpikelines", "hoverClosestGeo",
                ],
            },
        )

    # Scrub slider flanked by play-backward / play-forward buttons.
    anchor = st.session_state.anchor_dt
    slider_min = anchor - timedelta(hours=window_hours / 2)
    slider_max = anchor + timedelta(hours=window_hours / 2)
    step_minutes = max(1, int(round(window_hours * 60 / 200)))
    st.session_state.slider_dt = max(
        slider_min, min(slider_max, st.session_state.slider_dt)
    )

    pcols = st.columns([0.12, 1.0, 0.12], gap="small", vertical_alignment="center")
    pcols[0].button(
        "\u25c0", key="play_back",
        on_click=_toggle_play, args=("backward",),
        type="primary" if st.session_state.play_dir == "backward" else "secondary",
        width="stretch",
    )
    pcols[1].slider(
        "Moment (UTC)",
        min_value=slider_min, max_value=slider_max,
        step=timedelta(minutes=step_minutes),
        key="slider_dt", format="YYYY-MM-DD HH:mm",
        label_visibility="collapsed",
    )
    pcols[2].button(
        "\u25b6", key="play_fwd",
        on_click=_toggle_play, args=("forward",),
        type="primary" if st.session_state.play_dir == "forward" else "secondary",
        width="stretch",
    )

    # Step buttons for coarse time navigation.
    with st.container(key="step_row"):
        sb = st.columns(6, gap="small")
        sb[0].button("\u2212 1 mo", on_click=_shift, args=(timedelta(days=-30),), width="stretch")
        sb[1].button("\u2212 1 d",  on_click=_shift, args=(timedelta(days=-1),),  width="stretch")
        sb[2].button("\u2212 1 h",  on_click=_shift, args=(timedelta(hours=-1),), width="stretch")
        sb[3].button("+ 1 h",       on_click=_shift, args=(timedelta(hours=1),),  width="stretch")
        sb[4].button("+ 1 d",       on_click=_shift, args=(timedelta(days=1),),   width="stretch")
        sb[5].button("+ 1 mo",      on_click=_shift, args=(timedelta(days=30),),  width="stretch")

# ---------------------------------------------------------------------------
# Track window slider
#
# Controls how wide the body_tracks() sample window is. At 6h you see
# fine planetary motion; at 30d you see the full monthly sweep.
# ---------------------------------------------------------------------------

st.markdown('<div class="control-strip-top"></div>', unsafe_allow_html=True)

with st.container(key="bottom_ctrl"):
    st.select_slider(
        "Track window",
        options=[6, 24, 72, 168, 720],
        key="window_hours",
        format_func=lambda h: {6: "6 h", 24: "24 h", 72: "3 d", 168: "7 d", 720: "30 d"}[h],
        disabled=not st.session_state["show_body_tracks"],
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Auto-play loop
#
# The slider tick already happened at the top of the script. Here we just
# sleep and trigger another rerun. Stopping the loop is just play_dir
# being toggled back to None by a button click.
# ---------------------------------------------------------------------------

if st.session_state.play_dir is not None:
    time.sleep(0.12)
    st.rerun()
