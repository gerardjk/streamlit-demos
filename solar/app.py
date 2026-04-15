"""
Streamlit dashboard: project ephemeris bodies onto a world map.

Layout is deliberately split into the three zones that good dashboards
separate:

    1. Sidebar  — all controls (time navigation, overlay, window).
                  Keeping inputs out of the main column prevents them
                  competing with the primary visualization for attention.
    2. Header   — title, coverage caption, and a KPI strip summarising
                  the current state at a glance (what moment are we
                  looking at, which overlay is active, how wide is the
                  tracks window).
    3. Canvas   — the map (primary), then the details table and raw
                  ephemeris (secondary, collapsible).

Interaction defaults are tuned for an embedded dashboard: the Plotly
mode bar is hidden, scroll-wheel zoom is disabled (so page scroll keeps
working when the cursor happens to be over the map), and drag-pan is
turned off — the equirectangular world view has no useful zoom state
to preserve anyway.
"""

import math
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
    ecliptic_curve,
    ecliptic_sign_arcs,
    rising_sign_grid,
    subpoint,
    terminator,
    zodiac_sign_name,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

# Ecliptic longitude of each body, plus the ecliptic-latitude key if the
# binary schema has one. Only the Moon meaningfully departs from the
# ecliptic; treating the other planets as lat=0 is < 0.1° off at world-
# map scale, which is invisible.
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

# Traditional zodiac-sign colors — each chosen to echo the sign's
# classical association (element + ruling planet + folklore), while
# staying saturated enough to read on a dark background and distinct
# from the pale BODY_COLORS (which live in the 85% lightness band).
# Sign colors live in the 40–55% lightness band, ~60–80% saturation,
# so there's no visual collision between a body dot and its sign arc.
SIGN_COLORS = [
    "#c92a2a",  # Aries        — fire / Mars          : crimson red
    "#5c8a3f",  # Taurus       — earth / Venus        : olive green
    "#d4a017",  # Gemini       — air / Mercury        : goldenrod
    "#75c0c0",  # Cancer       — water / Moon         : sea glass
    "#e67e22",  # Leo          — fire / Sun           : regal orange
    "#3a6f5f",  # Virgo        — earth / Mercury      : forest teal
    "#c77d99",  # Libra        — air / Venus          : mauve rose
    "#7b1538",  # Scorpio      — water / Mars, Pluto  : deep maroon
    "#6b3fa0",  # Sagittarius  — fire / Jupiter       : royal purple
    "#6b4226",  # Capricorn    — earth / Saturn       : dark amber
    "#2ba6cb",  # Aquarius     — air / Saturn, Uranus : electric blue
    "#4a9d82",  # Pisces       — water / Jupiter, Nep : sea green
]

# Pale, ethereal body palette — every color sits around HSL(·, 60%, 85%)
# so the whole set reads as a family of soft pastels glowing against the
# deep-navy map. Picked for distinguishability and traditional flavor.
BODY_COLORS = {
    "Sun":     "#fff3b0",  # pale gold
    "Moon":    "#f5f7fb",  # moonlight white
    "Mercury": "#d4d8e0",  # pale slate
    "Venus":   "#fcd5dc",  # pale rose
    "Mars":    "#ffc4a8",  # pale peach
    "Jupiter": "#fddab0",  # pale cream
    "Saturn":  "#fbe8b4",  # pale butter
    "Uranus":  "#c5f1ea",  # pale aqua
    "Neptune": "#c4cffb",  # pale periwinkle
    "Pluto":   "#dfd3fb",  # pale lilac
}

# Small city gazetteer for the "orient globe at location" text input.
# Not a full geocoder — just a lookup of ~50 well-known places so
# the user can type "sydney" or "tokyo" and have the globe swing to
# center on it. Lowercase key, (lat, lon) in degrees.
CITIES: dict[str, tuple[float, float]] = {
    "london":         (51.51,   -0.13),
    "paris":          (48.86,    2.35),
    "berlin":         (52.52,   13.41),
    "madrid":         (40.42,   -3.70),
    "rome":           (41.90,   12.50),
    "amsterdam":      (52.37,    4.90),
    "copenhagen":     (55.68,   12.57),
    "stockholm":      (59.33,   18.07),
    "helsinki":       (60.17,   24.94),
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


def _match_city(text: str) -> tuple[float, float] | None:
    """Case-insensitive substring match against CITIES. Returns the
    first hit as (lat, lon) or None."""
    if not text:
        return None
    needle = text.strip().lower()
    if not needle:
        return None
    if needle in CITIES:
        return CITIES[needle]
    for name, coords in CITIES.items():
        if needle in name or name.startswith(needle):
            return coords
    return None


# Traditional astrological / astronomical glyphs for each body, shown
# next to the name in the positions table.
BODY_SYMBOLS = {
    "Sun":     "☉",
    "Moon":    "☽",
    "Mercury": "☿",
    "Venus":   "♀",
    "Mars":    "♂",
    "Jupiter": "♃",
    "Saturn":  "♄",
    "Uranus":  "♅",
    "Neptune": "♆",
    "Pluto":   "♇",
}

# Dark palette used consistently across map, legend, and page chrome.
# Slightly bluer/darker than pure slate for a more cosmic feel.
BG       = "#060a16"  # near-black, hint of navy
SURFACE  = "#17223a"  # land fill
BORDER   = "#2a3859"
MUTED    = "#8392b0"
FG       = "#e6edf7"


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

# =============================================================================
# CACHING LAYER 1 — @st.cache_resource: memoise a long-lived OBJECT
# =============================================================================
#
# Remember the core Streamlit rule: every widget interaction reruns the whole
# script from line 1 to the end. Without caching, that would mean rebuilding
# our EphemerisLoader on every click — reopening file handles, reparsing the
# metadata JSON, throwing away the internal chunk cache. The app would be
# unusable.
#
# @st.cache_resource says: "run this function exactly ONCE per server process,
# keep the return value forever, give the same instance to every caller."
# It does NOT pickle, it does NOT copy — every rerun sees the exact same
# Python object by identity. That's what we want for a loader, because the
# loader has internal state (its own LRU cache of recently-read monthly
# chunks) that we want to persist.
#
# Use cache_resource for:
#   • database connections / connection pools
#   • ML model weights loaded into memory
#   • file-handle-wrapping loaders (like this one)
#   • anything expensive to construct that should be shared
#
# Contrast with the other decorator, @st.cache_data (see portwatch/app.py
# load_data). cache_data memoises a VALUE — a DataFrame, a list, a dict. The
# return value is pickled and each caller gets their own copy, safe to mutate.
#
# The decision rule is one question:
#
#     Is the return value DATA (DataFrame/dict/list/array)?  →  cache_data
#     Or is it a THING with internal state (connection/loader/model)?  →  cache_resource
#
# If you pick wrong: cache_data will try to pickle your loader, fail, and
# crash. cache_resource will return a shared DataFrame that any caller can
# accidentally mutate, corrupting it for everyone else.
@st.cache_resource
def get_loader() -> EphemerisLoader:
    """Binary ephemeris loader — cached across reruns (it holds chunks in RAM)."""
    return EphemerisLoader(DATA_DIR)


# =============================================================================
# CACHING LAYER 2 — @st.cache_data: memoise a derived RESULT
#                   + the underscore-argument rule
# =============================================================================
#
# body_tracks computes, for a given instant, the path each body sweeps across
# the Earth's surface over a window of time (e.g. a 24-hour window centred on
# `dt`). It's not free — 240 samples, each requiring an ephemeris lookup and
# some trig. If we recomputed it on every widget interaction the app would
# feel sluggish.
#
# @st.cache_data says: "hash the arguments, and if you've already computed a
# result for this exact set of arguments, return the cached copy instead of
# running the function again." On a cache hit the function body never runs.
#
# BUT — and this is the gotcha every Streamlit student hits at least once —
# cache_data builds its cache key by HASHING the arguments. Look at the first
# argument below: `_loader`. An EphemerisLoader cannot be hashed (there's no
# sensible way to turn an open file handle into a number). If we wrote
# `loader` instead of `_loader`, the very first call would crash with
# UnhashableParamError.
#
# The leading underscore is a Streamlit convention — not a Python one — and
# it is LOAD-BEARING. It tells cache_data: "do not try to hash this argument,
# pretend it is not part of the cache key." The effective cache key for this
# function is therefore just (dt_iso, window_hours, n_samples). The loader
# still gets passed in; cache_data just ignores it for cache-key purposes.
#
# That's safe here because there is only one loader per process (it's behind
# @st.cache_resource above), so "the loader" is always the same instance —
# we aren't hiding a real dependency from the cache.
#
# The one-line rule to remember:
#
#     underscore prefix = skip this arg when hashing
#
# Prefix any un-hashable argument this way: connections, loaders, open files,
# class instances, anything custom. This is THE most common "why is my cache
# broken?" question in all of Streamlit.
#
# Note the second argument is `dt_iso: str`, not `dt: datetime`. Datetimes
# are hashable, so that isn't forced, but taking an ISO string gives us a
# stable, human-readable cache key that's easy to inspect when debugging.
@st.cache_data(show_spinner=False)
def body_tracks(
    _loader: EphemerisLoader,
    dt_iso: str,
    window_hours: float,
    n_samples: int = 240,
) -> dict[str, list[tuple[float, float]]]:
    """
    Subpoint track for each body over [dt − window/2, dt + window/2].

    240 samples is plenty: at a 24-hour window it's ~6-minute cadence,
    finer than needed for a world map (1° of Earth rotation ≈ 4 min).
    Keyed on dt_iso so time-navigation reruns hit the cache.
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
            sub_lat, sub_lon = subpoint(ecl_lon, ecl_lat, t)
            tracks[name].append((sub_lat, sub_lon))
    return tracks


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ephemeris → Earth",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tight CSS so the map is the hero at default viewport size (~720–
# 800px tall). We kill Streamlit's default top padding, compress
# heading margins, and shrink sidebar buttons so everything above
# the map costs ~90px of vertical real estate instead of ~280px.
st.markdown(
    """
    <style>
      /* Kill Streamlit's entire sticky header — it was floating over
         the "Ephemeris → Earth" title. Safe now that all controls
         are in the main page body (no sidebar toggle to preserve). */
      header[data-testid="stHeader"] { display: none !important; }
      div[data-testid="stToolbar"]   { display: none !important; }
      #MainMenu                      { display: none !important; }
      footer                         { visibility: hidden; }

      /* With the header gone, pull the main column right up to the
         top of the viewport so the map starts where the eye expects. */
      div.block-container {
          padding-top: 0.2rem;
          padding-bottom: 0.6rem;
          padding-left: 1.2rem;
          padding-right: 1.2rem;
      }
      /* Compress all heading margins — the tiny gaps between h3 and
         the next element add up fast in a dense dashboard. */
      h1, h2, h3, h4 {
          margin-top: 0.2rem !important;
          margin-bottom: 0.3rem !important;
          padding-top: 0 !important;
          padding-bottom: 0 !important;
      }
      /* Compact sidebar buttons for the time-nav row. */
      section[data-testid="stSidebar"] .stButton > button {
          padding: 0.25rem 0.25rem;
          font-size: 0.85rem;
      }

      /* Custom legend column that sits between the table and the
         globe, replacing Plotly's in-chart legend. */
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
          display: flex;
          align-items: center;
          gap: 0.45rem;
          padding: 0.1rem 0;
          line-height: 1.15;
      }
      .legend-box .legend-swatch {
          display: inline-block;
          width: 10px;
          height: 10px;
          border-radius: 50%;
          flex-shrink: 0;
      }
      .legend-box .legend-gap { height: 0.55rem; }

      /* Compact control strip at the bottom: trim widget spacing. */
      .control-strip-top { margin-top: 0.4rem; }

      /* Date / time / text inputs — force visible foreground and
         a dark surface so the text doesn't blend into the
         background. Use broad selectors so both legacy and newer
         Streamlit widget DOM structures are covered. */
      .stDateInput input,
      .stTimeInput input,
      .stTextInput input,
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

      /* Compact step buttons (six ± buttons under the slider) — but
         NOT the play buttons flanking the slider, which stay normal
         size so they're easy to hit. */
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

      /* Pull the zodiac/rising toggle container UP so it sits just
         below the last legend entry (Pisces) at the bottom-right
         of the plot area, and shift it right to align with the
         legend box's left edge (which hangs ~115 px in from the
         right edge of the map column). Also shrink its checkboxes
         via the nested rules below. */
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

      /* Tight stack for the zodiac/rising toggles under the legend.
         Streamlit wraps each checkbox in ~20 px of vertical padding
         + 8 px of gap by default — we strip all of that so the two
         boxes together cost ~36 px instead of ~80 px, which brings
         the slider bar back up above the fold. */
      div[data-testid="stCheckbox"] {
          margin: 0 !important;
          padding: 0 !important;
      }
      div[data-testid="stCheckbox"] label {
          padding: 0 !important;
          gap: 0.35rem !important;
          font-size: 0.78rem !important;
          line-height: 1.0 !important;
      }
      div[data-testid="stCheckbox"] label > div:first-child {
          margin-right: 0.3rem !important;
      }
      /* And squeeze the block-level wrapper Streamlit puts around
         every widget so the two checkboxes sit almost flush. */
      div[data-testid="stCheckbox"] + div[data-testid="stCheckbox"] {
          margin-top: -4px !important;
      }

      /* Force readable button styling — Streamlit's default in some
         themes renders as white-on-white, which was the complaint.
         Dark navy surface, pale foreground, subtle border; primary
         (active) buttons get a bright amber fill on dark text so the
         ◀ / ▶ play buttons are obviously "on". */
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

      /* Night-themed positions table — dark cells, light hairlines
         ("wiring") between them, muted uppercase header. */
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

      /* Cosmic page background — deep navy radial that hints at a
         starfield without being literal. Applied to the root app
         container so it sits behind everything Streamlit renders. */
      [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(ellipse at 20% 0%, #1a2950 0%, #0a1128 28%, #05080f 60%, #000000 100%) !important;
      }
      [data-testid="stAppViewContainer"] * { color: #e6edf7; }
      section[data-testid="stSidebar"] { background: #060a16 !important; }

      /* Horizontal legend strip beneath the map. */
      .legend-horiz {
          display: flex;
          flex-wrap: wrap;
          gap: 0.6rem 1.1rem;
          justify-content: center;
          padding: 0.5rem 0.75rem;
          margin-top: 0.4rem;
          background: rgba(14, 22, 40, 0.85);
          border: 1px solid #2a3859;
          border-radius: 8px;
          font-size: 0.78rem;
          color: #cbd5e1;
      }
      .legend-horiz .lh-item { display: inline-flex; align-items: center; gap: 0.35rem; }
      .legend-horiz .lh-sw {
          width: 10px; height: 10px; border-radius: 50%;
          display: inline-block;
      }

      /* Make st.container(border=True) panels match the legend & map
         borders so every panel's outer edge lines up vertically. */
      div[data-testid="stVerticalBlockBorderWrapper"] {
          border-color: #2a3859 !important;
          background: rgba(14, 22, 40, 0.55) !important;
          border-radius: 8px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    loader = get_loader()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

start_iso, end_iso = loader.coverage


# ---------------------------------------------------------------------------
# Session state + time-navigation callbacks
# ---------------------------------------------------------------------------
# The "effective moment" used by the table, map, and legend is:
#
#     dt = anchor_dt + scrub_hours hours
#
# `anchor_dt` is set by the Date/Time inputs and the step buttons —
# the "big move" controls. `scrub_hours` is a live slider that lets
# the user fine-scrub within ±(window/2) hours of the anchor, and
# every move triggers a full Streamlit rerun so the positions table
# + every other view updates in lockstep. Moving a coarse control
# resets scrub_hours to 0 so the slider always re-centers on its
# new anchor.

if "anchor_dt" not in st.session_state:
    _init = datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc)
    st.session_state.anchor_dt = _init
    st.session_state.d_input = _init.date()
    st.session_state.t_input = _init.time().replace(microsecond=0)
    st.session_state.slider_dt = _init
# Auto-play state: None (paused), "forward", or "backward".
if "play_dir" not in st.session_state:
    st.session_state.play_dir = None
# Globe center — (lat, lon). Default leaves the view as-was.
if "globe_center" not in st.session_state:
    st.session_state.globe_center = (20.0, 0.0)
if "location_query" not in st.session_state:
    st.session_state.location_query = None


def _on_location_change() -> None:
    # The selectbox returns an exact city key from CITIES (or None).
    pick = st.session_state.location_query
    if pick and pick in CITIES:
        st.session_state.globe_center = CITIES[pick]


def _toggle_play(direction: str) -> None:
    """Click the ◀ / ▶ button — toggles auto-play in that direction.
    Clicking the already-active direction stops it."""
    st.session_state.play_dir = None if st.session_state.play_dir == direction else direction


# ---------------------------------------------------------------------------
# Auto-play advance — MUST run before the slider widget renders
# ---------------------------------------------------------------------------
# Streamlit raises a silent error if you mutate a widget's session-state
# key after the widget has already rendered in the same run, so we tick
# slider_dt here at the top, then at the very end of the script we
# call st.rerun() to trigger the next tick. The script body in between
# reads the new slider_dt normally and everything refreshes.
if st.session_state.play_dir is not None:
    _wh = st.session_state.get("window_hours", 168)
    _half_td = timedelta(hours=_wh / 2)
    _smin = st.session_state.anchor_dt - _half_td
    _smax = st.session_state.anchor_dt + _half_td
    # ~100 ticks across the full window → 12-second sweep at 0.12 s/tick.
    _tick_td = timedelta(minutes=max(1, int(round(_wh * 60 / 100))))
    cur = st.session_state.slider_dt
    if st.session_state.play_dir == "forward":
        nxt = cur + _tick_td
    else:
        nxt = cur - _tick_td

    if nxt > _smax or nxt < _smin:
        # Overflowed the window — instead of stopping, slide the
        # whole window so the new moment lands at its centre. The
        # date/time inputs follow the anchor too.
        st.session_state.anchor_dt = nxt
        st.session_state.slider_dt = nxt
        st.session_state.d_input = nxt.date()
        st.session_state.t_input = nxt.time().replace(microsecond=0)
    else:
        st.session_state.slider_dt = nxt

    # Stop only if we've run off the end of the whole ephemeris
    # coverage, not just the current window.
    _cov_end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    _cov_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    if st.session_state.slider_dt >= _cov_end or st.session_state.slider_dt <= _cov_start:
        st.session_state.play_dir = None


def _shift(delta: timedelta) -> None:
    """Jump anchor_dt by `delta` (via the step buttons) and re-center
    the scrub slider on the new anchor."""
    new = st.session_state.anchor_dt + delta
    st.session_state.anchor_dt = new
    st.session_state.d_input = new.date()
    st.session_state.t_input = new.time().replace(microsecond=0)
    st.session_state.slider_dt = new


def _sync_from_widgets() -> None:
    """Called when the user edits Date or Time directly — anchor jumps,
    slider re-centers."""
    new = datetime.combine(
        st.session_state.d_input,
        st.session_state.t_input,
        tzinfo=timezone.utc,
    )
    st.session_state.anchor_dt = new
    st.session_state.slider_dt = new


# ---------------------------------------------------------------------------
# Sidebar (controls)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Three-column top row: positions table | legend strip | square globe
# ---------------------------------------------------------------------------
# Controls live in a compact strip at the bottom (see the end of the
# file). The top row is just the data + its companion viz:
#
#   LEFT  (wide)   — positions table
#   MID   (narrow) — custom HTML legend between the table and the
#                    globe, replacing Plotly's in-chart legend
#   RIGHT (wide)   — the globe (orthographic, roughly square)
#
# We *create* the three columns here but only write the table into
# LEFT now — the legend and the plotly chart are written later once
# we know which layers are active. The control strip at the bottom
# writes the widgets and defines projection / layer / window_hours /
# show_zodiac_band / show_rising.

# Controls must be rendered BEFORE we reference their values. We build
# a bottom-of-page container that Streamlit renders in order (top to
# bottom), but we execute the widget code first so variables are
# defined. Trick: put widget code inside an st.empty() placeholder
# that we'll fill at the end, while the widget values are read from
# session_state via their keys.
#
# Simpler approach: just build the control widgets first, capture
# their values, and — in a final pass — use st.columns at the very
# bottom of the page to render the control strip. But since the
# widgets themselves produce the DOM nodes, we can't render them
# twice. So instead we render the TOP row first (positions + legend
# + map) with *computed* values, and the BOTTOM row second (control
# strip with the actual widgets).
#
# That works as long as we know what defaults to use on a given run
# and can read back the live widget values from session_state.

# Establish widget keys with defaults so the top row can read them
# on the first run before the widgets below have rendered.
_defaults = {
    "window_hours": 168,
    "show_zodiac_band": True,
    "show_rising": False,
    "show_body_tracks": True,
    "show_day_night": True,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Projection is fixed to Globe; Overlay is replaced by a single
# body-tracks checkbox.
projection       = "Globe"
layer            = "Body tracks" if st.session_state["show_body_tracks"] else "None"
window_hours     = st.session_state["window_hours"]
show_zodiac_band = st.session_state["show_zodiac_band"]
show_rising      = st.session_state["show_rising"]
show_day_night   = st.session_state["show_day_night"]

# Clamp the scrub slider to the current window so widening/narrowing
# the window doesn't leave it out of range.
_half = timedelta(hours=window_hours / 2.0)
_slider_min_global = st.session_state.anchor_dt - _half
_slider_max_global = st.session_state.anchor_dt + _half
if "slider_dt" not in st.session_state:
    st.session_state.slider_dt = st.session_state.anchor_dt
elif st.session_state.slider_dt < _slider_min_global:
    st.session_state.slider_dt = _slider_min_global
elif st.session_state.slider_dt > _slider_max_global:
    st.session_state.slider_dt = _slider_max_global

# Effective moment: whatever the scrub slider is pointing at.
dt = st.session_state.slider_dt


# ---------------------------------------------------------------------------
# Title row — one simple line, no fancy HTML
# ---------------------------------------------------------------------------
# Plain Streamlit markdown beats custom flex CSS for reliability. Two
# columns: title on the left, current-moment readout on the right.
_title_cols = st.columns([3, 2])
_title_cols[0].markdown("### 🌍  Ephemeris → Earth")
_title_cols[1].markdown(
    f"<div style='text-align:right; padding-top:0.55rem; color:#8392b0; font-size:0.95rem;'>"
    f"<b style='color:#e6edf7;'>{dt.strftime('%Y-%m-%d %H:%M')}</b> UTC · "
    f"{dt.strftime('%A')}"
    + "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='margin:0.2rem 0 0.8rem 0; border-color:#1e2d4d;'>",
    unsafe_allow_html=True,
)

TABLE_COL, MAP_COL = st.columns([1.5, 2.5], gap="small")


# =============================================================================
# CACHING LAYER 3 — lazy load under the Streamlit layer
# =============================================================================
#
# This line is where 46 MB of ephemeris data on disk becomes ~128 KB in RAM.
# Watch what actually happens:
#
#   1. The user picks a date in the widgets above. `dt` is a Python datetime.
#   2. loader.get(dt) computes which month `dt` falls in.
#   3. If that month's chunk is already in the loader's internal dict → cache
#      hit inside the loader → zero disk I/O, return one row.
#   4. If not → read ONE monthly binary file (~128 KB) off disk, store it in
#      the dict, evict the oldest entry if the dict already holds 12, return
#      one row.
#
# So: changing the date but staying within the same month costs nothing.
# Jumping to a new month costs one 128 KB file read. The process never holds
# more than ~12 × 128 KB ≈ 1.5 MB of chunk data in memory, no matter how long
# the app runs or how many dates the user clicks through.
#
# This is the headline pattern for large-data Streamlit apps:
#
#     Streamlit caches the entry point.
#     Your code does its own smart paging underneath.
#
# @st.cache_resource on get_loader() guarantees exactly one EphemerisLoader
# per server process. That loader owns its own per-month LRU cache. Streamlit
# knows nothing about that internal cache — it just trusts the loader to be
# fast on repeated calls. Two complementary caching layers, collaborating.
#
# If the data were too big for even this pattern, you'd move it behind a
# query engine (DuckDB + Parquet, or a warehouse like Postgres/BigQuery),
# wrap the connection in @st.cache_resource, and wrap individual query
# results in @st.cache_data. Same architecture, different storage layer.
# ---------------------------------------------------------------------------
# Fetch the entry for the selected moment (bail out cleanly if missing)
# ---------------------------------------------------------------------------

entry = loader.get(dt)
if entry is None:
    st.warning("No ephemeris data for that moment — pick a date inside the coverage window.")
    st.stop()

# Current subpoints for every body — used for the markers on the map
# and the details table below it.
subpoints: list[tuple[str, float, float, float]] = []  # name, sub_lat, sub_lon, ecl_lon
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
# Compact header strip
# ---------------------------------------------------------------------------
# A single-line flex row: title on the left, current moment + overlay
# pills on the right. This replaces the old title + caption + 4-wide
# KPI stack, cutting ~200px of vertical chrome so the map fits inside
# a default laptop viewport without scrolling.

_meta_pills = ""  # overlay / window pills removed from the title bar

# (Header moved to the global title bar above.)


# ---------------------------------------------------------------------------
# Build the map
# ---------------------------------------------------------------------------
# Resolve the projection choice up-front because the track-splitting
# logic below depends on it: on a flat (equirectangular / Natural
# Earth) projection we must split lines where they cross the
# antimeridian, otherwise Plotly draws a horizontal streak across the
# whole map. On the orthographic globe the line wraps around the back
# of the sphere naturally, so splitting would leave an ugly gap right
# at the dateline — exactly the glitch we're fixing.
_projection_types = {
    "Globe":                  "orthographic",
    "Natural Earth":          "natural earth",
    "Flat (equirectangular)": "equirectangular",
}
_projection_type = _projection_types[projection]
_is_flat = _projection_type != "orthographic"


def split_antimeridian(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs where a line crosses the antimeridian — only on flat
    projections. On the globe we let Plotly route the line around the
    back of the sphere."""
    if not _is_flat:
        return lats, lons
    jumps = np.where(np.abs(np.diff(lons)) > 180.0)[0]
    if len(jumps) == 0:
        return lats, lons
    return (
        np.insert(lats, jumps + 1, np.nan),
        np.insert(lons, jumps + 1, np.nan),
    )


fig = go.Figure()

# Zodiac layer 1: rising-sign grid. Each lat/lon cell colored by
# which sign is rising on the eastern horizon there at time `dt`.
# Drawn first (furthest back) with reduced opacity so it reads as a
# background wash — tracks, terminator, and zodiac band sit on top.
if show_rising:
    grid_lats, grid_lons, sign_idx = rising_sign_grid(dt, lat_step=3.0, lon_step=3.0)
    for i, sign in enumerate(ZODIAC_SIGNS):
        mask = sign_idx == i
        if not mask.any():
            continue
        # showlegend=False: the rising-sign grid reuses the same 12
        # colours as the Zodiac band, which are already in the legend.
        # Duplicating them as "↑ Aries" etc. doubled the legend and
        # made it jump when the user toggled Rising signs.
        fig.add_trace(go.Scattergeo(
            lat=grid_lats[mask],
            lon=grid_lons[mask],
            mode="markers",
            marker=dict(size=6, color=SIGN_COLORS[i], opacity=0.35, line=dict(width=0)),
            name=f"↑ {sign}",
            hovertemplate="%{lat:.0f}°, %{lon:.0f}°<br>Rising: " + sign + "<extra></extra>",
            showlegend=False,
        ))

# Overlay: subpoint tracks with a time-fading trail.
#
# We cap each trail at ~24 h of body motion (one rotation) so planets
# don't stack into overlapping loops (see earlier comment). Within
# that trail we draw N_TRAIL_SEGMENTS short line traces per body,
# each with its own opacity — newest segment ≈ fully opaque, oldest
# ≈ nearly transparent — so the trail fades from a bright "comet
# head" back into the night. Plotly scattergeo lines don't support
# per-vertex opacity, so a multi-trace segmentation is the only way
# to get a smooth alpha gradient.
TRAIL_HOURS = 24.0
# More segments = finer opacity gradient. 16 gives a visibly smooth
# fade from transparent to full-opacity along the ~24 h trail.
N_TRAIL_SEGMENTS = 16
_n_samples_total = 240
track_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_trail_samples = 0
# Map body name → list of trace indices (one per segment) so
# animation frames can update them in lockstep.
trail_indices: dict[str, list[int]] = {}

def _segment_bounds(start: int, end: int, n_seg: int) -> list[tuple[int, int]]:
    """Split [start, end) into `n_seg` contiguous slices (with one-point
    overlap so adjacent segments touch and look continuous)."""
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
    """Opacity for segment `si` — 0 = oldest (~invisible), N-1 =
    newest (~half-opaque). Trails are deliberately soft so the
    globe surface reads through them."""
    return 0.04 + 0.46 * (si / max(1, N_TRAIL_SEGMENTS - 1))


if layer == "Body tracks":
    tracks = body_tracks(loader, dt.isoformat(), float(window_hours))
    # How many raw samples correspond to TRAIL_HOURS at the current
    # window's sample cadence. Always ≥ N_TRAIL_SEGMENTS + 1.
    _trail_samples = max(
        N_TRAIL_SEGMENTS + 1,
        int(round(_n_samples_total * TRAIL_HOURS / max(TRAIL_HOURS, window_hours))),
    )
    for name, _, _ in BODIES:
        pts = tracks.get(name, [])
        if not pts:
            continue
        lats = np.array([p[0] for p in pts], dtype=float)
        lons = np.array([p[1] for p in pts], dtype=float)
        track_arrays[name] = (lats, lons)

        # Static view trail = the most-recent _trail_samples points
        # ENDING at the "now" index (middle of the body_tracks span,
        # which corresponds to dt). That way the line is purely the
        # past, and its head sits exactly under the planet marker.
        now_idx = len(lats) // 2
        end = now_idx + 1
        start = max(0, end - _trail_samples)
        trail_indices[name] = []
        seg_bounds = _segment_bounds(start, end, N_TRAIL_SEGMENTS)
        # Always emit exactly N_TRAIL_SEGMENTS traces per body so the
        # animation frames can index them in lockstep. If a segment
        # would be empty/too short, we add a NaN placeholder trace.
        for si in range(N_TRAIL_SEGMENTS):
            if si < len(seg_bounds):
                s0, s1 = seg_bounds[si]
                s_lats, s_lons = split_antimeridian(lats[s0:s1], lons[s0:s1])
            else:
                s_lats = np.array([np.nan])
                s_lons = np.array([np.nan])
            fig.add_trace(go.Scattergeo(
                lat=s_lats, lon=s_lons, mode="lines",
                line=dict(color=BODY_COLORS[name], width=1.8),
                opacity=_segment_alpha(si),
                name=name,
                # Real track segments never appear in the legend — a
                # dedicated dot-style "legend proxy" trace below does
                # that so the legend symbol matches the Now marker.
                showlegend=False,
                legendgroup=name,
                hoverinfo="skip",
            ))
            trail_indices[name].append(len(fig.data) - 1)

        # Legend proxy — one invisible point at (nan, nan) per body,
        # drawn as a solid dot with the body's color + white outline.
        # Plotly renders this in the legend exactly as it's drawn in
        # the figure, so the legend entry matches the "Now" marker.
        fig.add_trace(go.Scattergeo(
            lat=[None], lon=[None], mode="markers",
            marker=dict(
                size=11,
                color=BODY_COLORS[name],
                line=dict(color="#ffffff", width=1.5),
            ),
            name=name,
            legendgroup=name,
            showlegend=True,
            hoverinfo="skip",
        ))

# Day / night — gated by a checkbox. Filled polygon bounded by
# the terminator great circle, sampled densely enough (181 points)
# that it reads as a smooth curve. `fill="toself"` fills the
# polygon interior in one pass — no stacked markers.
_sun = next((s for s in subpoints if s[0] == "Sun"), None)
if show_day_night and _sun is not None:
    _sun_lat, _sun_lon = _sun[1], _sun[2]

    # Build a closed terminator polygon WITHOUT the antimeridian
    # NaN split — `terminator()` splits for clean line rendering on
    # flat maps, but a polygon fill needs one continuous loop.
    _N = 181
    _lat0 = math.radians(_sun_lat)
    _lon0 = math.radians(_sun_lon)
    _bear = np.linspace(0.0, 2.0 * math.pi, _N)
    _tlat = np.arcsin(np.cos(_lat0) * np.cos(_bear))
    _tlon = _lon0 + np.arctan2(
        np.sin(_bear) * np.cos(_lat0),
        -np.sin(_lat0) * np.sin(_tlat),
    )
    _tlat = np.rad2deg(_tlat)
    _tlon = (np.rad2deg(_tlon) + 180.0) % 360.0 - 180.0

    # On flat projections the antimeridian crossing would drag the
    # fill across the whole map in a horizontal streak. Only use
    # the filled polygon on orthographic (globe) where the ring
    # sits on the sphere cleanly. For flat maps fall back to the
    # existing grid wash (still better than nothing visually).
    if _projection_type == "orthographic":
        fig.add_trace(go.Scattergeo(
            lat=_tlat, lon=_tlon,
            mode="lines",
            line=dict(color="#fbbf24", width=1.2),
            fill="toself",
            fillcolor="rgba(255, 240, 180, 0.16)",
            name="Day side",
            hoverinfo="skip",
            showlegend=False,
        ))
    else:
        # Flat-map fallback: the grid wash still works because
        # there's no polygon winding to worry about.
        _ws_lats = np.arange(-87.0, 87.1, 3.0)
        _ws_lons = np.arange(-177.0, 177.1, 3.0)
        _lon_g, _lat_g = np.meshgrid(_ws_lons, _ws_lats)
        _cos_d = (
            np.sin(math.radians(_sun_lat)) * np.sin(np.deg2rad(_lat_g))
            + np.cos(math.radians(_sun_lat))
              * np.cos(np.deg2rad(_lat_g))
              * np.cos(np.deg2rad(_lon_g - _sun_lon))
        )
        _day_mask = _cos_d > 0.0
        _alpha = np.clip(_cos_d[_day_mask], 0.0, 1.0) ** 0.6 * 0.4
        fig.add_trace(go.Scattergeo(
            lat=_lat_g[_day_mask], lon=_lon_g[_day_mask],
            mode="markers",
            marker=dict(size=9, color="#fef3c7", opacity=_alpha.tolist(), line=dict(width=0)),
            name="Day side",
            hoverinfo="skip",
            showlegend=False,
        ))

# Optional ecliptic-great-circle reference. Drawn *before* the markers
# so the body glyphs sit on top of it. This is the sinusoid that a
# line of constant latitude is NOT — every planet's subpoint lies on
# it, which is a nice visual confirmation.
# Zodiac layer 2: the ecliptic band itself, split into 12 colored
# 30° arcs (Aries → Pisces) with the sign symbol at the midpoint
# of each arc. This is the answer to "where are the signs on Earth
# right now": the sign that contains a planet's ecliptic longitude
# is the arc its subpoint sits on.
if show_zodiac_band:
    arcs, labels = ecliptic_sign_arcs(dt)
    for i, (a_lats, a_lons) in enumerate(arcs):
        s_lats, s_lons = split_antimeridian(a_lats, a_lons)
        fig.add_trace(go.Scattergeo(
            lat=s_lats, lon=s_lons, mode="lines",
            line=dict(color=SIGN_COLORS[i], width=2.5),
            opacity=0.75,
            name=f"{ZODIAC_SYMBOLS[i]} {ZODIAC_SIGNS[i]}",
            hovertemplate=(
                f"{ZODIAC_SYMBOLS[i]} {ZODIAC_SIGNS[i]}<br>"
                "%{lat:.2f}°, %{lon:.2f}°<extra></extra>"
            ),
            legendgroup="zodiac",
            legendgrouptitle_text="Zodiac band" if i == 0 else None,
        ))
    # Glyph labels at each sign's midpoint — one trace so they share
    # a single (hidden) legend entry.
    fig.add_trace(go.Scattergeo(
        lat=[lbl[0] for lbl in labels],
        lon=[lbl[1] for lbl in labels],
        text=ZODIAC_SYMBOLS,
        mode="text",
        textfont=dict(color="#f8fafc", size=18),
        hoverinfo="skip",
        showlegend=False,
    ))

# Current subpoint markers. We color each marker by its body's own
# color (BODY_COLORS) rather than a single uniform cream, so the Moon
# reads as a light-gray dot, Mars as red, etc. A white outline
# guarantees contrast against any overlay beneath. Recording the
# trace index lets us point animation frames at it later.
now_marker_names = [s[0] for s in subpoints]
fig.add_trace(go.Scattergeo(
    lat=[s[1] for s in subpoints],
    lon=[s[2] for s in subpoints],
    text=now_marker_names,
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
    hovertemplate="%{text}<br>%{lat:.2f}°, %{lon:.2f}°<extra></extra>",
    showlegend=False,
))
_now_marker_index = len(fig.data) - 1


# Plotly frame-based animation removed — the in-figure slider was
# client-side only and couldn't trigger a Streamlit rerun, so the
# positions table never refreshed. Scrubbing is now handled by the
# Streamlit slider beneath the map, which reruns the whole script on
# every change. Each rerun rebuilds the trails and positions table
# for the new effective `dt`, so everything updates in lockstep.

# Projection framing. The orthographic globe gets a gentle northward
# tilt, rotated to center on the current subsolar longitude so the
# action stays in view as you step through time.
geo_kwargs = dict(
    projection_type=_projection_type,
    showcoastlines=True,  coastlinecolor="#334773", coastlinewidth=0.6,
    showland=True,        landcolor="#0b1220",
    showocean=True,       oceancolor="#03070f",
    showlakes=False,
    showcountries=True,   countrycolor="#1a253f",  countrywidth=0.4,
    showframe=False,
    bgcolor="#03070f",
)
if _projection_type == "orthographic":
    # Rotation center — set by the location text input below the
    # table. Default is (20, 0) which looks "Atlantic-centric".
    _ctr_lat, _ctr_lon = st.session_state.globe_center
    geo_kwargs["projection_rotation"] = dict(lon=_ctr_lon, lat=_ctr_lat, roll=0)
else:
    geo_kwargs["lataxis_range"] = [-80, 80]
    geo_kwargs["lonaxis_range"] = [-180, 180]

fig.update_geos(**geo_kwargs)

# Map sizing — the right column is ~half the page width, so an
# orthographic globe naturally wants a square canvas. Plotly can't
# measure the column directly, so we pick a fixed height that looks
# roughly square on a standard monitor.
# Taller map; the stacked zodiac/rising checkboxes under the key
# are compressed via CSS (see .compact-checks below) so they barely
# add any vertical height.
_map_height = 540
_bottom_margin = 2

# Legend placement: overlaid in the top-right of the map plot area so
# it doesn't eat vertical space. Semi-transparent background so the
# ocean still shows through.
fig.update_layout(
    height=_map_height,
    # Top margin leaves headroom for Plotly's modebar icons so they
    # don't overlap the globe. Right margin makes space for the
    # vertical legend that hangs just outside the plot area.
    margin=dict(l=0, r=110, t=42, b=_bottom_margin),
    paper_bgcolor="#03070f",
    plot_bgcolor="#03070f",
    # "pan" on an orthographic projection = drag-to-rotate the
    # globe. Scroll-wheel zoom is still off (see the config dict on
    # st.plotly_chart below), so the page still scrolls when the
    # mouse is over the map — no accidental flipping from that.
    dragmode="pan",
    legend=dict(
        orientation="v",
        yanchor="top",   y=1.0,
        xanchor="left",  x=1.01,  # just outside the plot area, to the right
        bgcolor="rgba(3,7,15,0.85)",
        bordercolor=BORDER,
        borderwidth=1,
        # 9-px font + zero tracegroupgap + constant item sizing means
        # each row is about 16 px tall, so 22 rows fit in ~360 px —
        # comfortably inside the 540 px plot area.
        font=dict(color=FG, size=9),
        itemsizing="constant",
        tracegroupgap=0,
    ),
    # uirevision pinned to the current globe center: drag-rotations
    # and pinch-zooms persist across data updates, but when the
    # user types a new location the revision string changes and
    # Plotly snaps to the new projection_rotation.
    uirevision=f"solar-{st.session_state.globe_center[0]:.2f}-{st.session_state.globe_center[1]:.2f}",
    showlegend=(layer == "Body tracks" or show_zodiac_band),
)

# ---------------------------------------------------------------------------
# Render the top row: table, legend, map
# ---------------------------------------------------------------------------

with TABLE_COL:
    rows = []
    for name, sub_lat, sub_lon, ecl_lon in subpoints:
        rows.append({
            "Body":       f"{BODY_SYMBOLS.get(name, '')}  {name}",
            "Ecliptic λ": f"{ecl_lon:7.3f}°",
            "Zodiac":     zodiac_sign_name(ecl_lon),
            "Sub-lat":    f"{sub_lat:+6.2f}°",
            "Sub-lon":    f"{sub_lon:+7.2f}°",
        })
    _df = pd.DataFrame(rows)

    # Dark, night-themed cell for any column we aren't coloring by
    # body/zodiac. Thin lighter "wiring" border between cells.
    DARK_CELL = (
        "background-color: #0a1124; color: #e6edf7;"
        " border: 1px solid #1e2d4d;"
    )

    def _style_row(row: pd.Series) -> list[str]:
        # Body cell is "☉  Sun" etc. — split off the glyph to look
        # up the color (and fall back to a whole-cell lookup just
        # in case the glyph is missing).
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

    # st.dataframe renders headers on a canvas, so CSS on `thead th`
    # can't reach them. Workaround: attach the header styles via
    # pandas Styler.set_table_styles(), emit full HTML via
    # styled.to_html(), and render with st.markdown — giving us
    # total control over every thead/tbody/td rule.
    styled = (
        _df.style
           .hide(axis="index")
           .apply(_style_row, axis=1)
           .set_table_styles([
               # Outer frame — dark surface, thin border "wiring".
               {"selector": "",
                "props": [
                    ("border-collapse", "collapse"),
                    ("border", "1px solid #1e2d4d"),
                    ("width", "100%"),
                    ("background-color", "#080f1e"),
                    ("font-variant-numeric", "tabular-nums"),
                    ("font-size", "0.85rem"),
                ]},
               # Header row — silver background with near-black
               # text so the column labels read as the distinct
               # "metal strip" at the top of a dark table.
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
               # Body cells — padding + numeric alignment for the
               # position columns.
               {"selector": "tbody td",
                "props": [
                    ("padding", "0.35rem 0.7rem"),
                    ("border", "1px solid #1e2d4d"),
                ]},
           ])
    )
    # --- Inputs (rendered ABOVE the table HTML so the styler's
    #     <style> block can't affect their event handlers) -----------
    _ti_cols = st.columns([1, 1])
    _ti_cols[0].date_input(
        "Date (UTC)",
        key="d_input",
        on_change=_sync_from_widgets,
    )
    _ti_cols[1].time_input(
        "Time (UTC)",
        key="t_input",
        on_change=_sync_from_widgets,
        step=3600,
    )
    # Selectbox gives us a proper searchable dropdown — click it,
    # type a few letters, pick the match. More discoverable than a
    # free-text input that only works for cities in our gazetteer.
    _city_names = sorted(CITIES.keys(), key=lambda s: s.title())
    st.selectbox(
        "Center globe on",
        options=_city_names,
        index=None,
        placeholder="— pick a place —",
        format_func=lambda s: s.title(),
        key="location_query",
        on_change=_on_location_change,
        help="Click and type to filter — the globe swings to the city you pick.",
    )

    # Table rendered via st.html (raw-HTML element, not markdown) so
    # the styler's inline <style> block doesn't poison later widgets.
    with st.container(border=True):
        st.html(styled.to_html())

    # Four display toggles immediately under the table, right next
    # to it in the left column.
    _lc = st.columns(4)
    _lc[0].checkbox(
        "Body tracks",
        key="show_body_tracks",
        help="Draw a fading subpoint trail behind each body over the last ~24 h.",
    )
    _lc[1].checkbox(
        "Zodiac band",
        key="show_zodiac_band",
        help="The 12 colored 30° arcs of the ecliptic on the globe.",
    )
    _lc[2].checkbox(
        "Rising signs",
        key="show_rising",
        help="Wash each lat/lon by the sign rising on its eastern horizon.",
    )
    _lc[3].checkbox(
        "Day / night",
        key="show_day_night",
        help="Fill the sunlit hemisphere — bounded by the Sun's terminator great circle.",
    )


with MAP_COL:
    map_container = st.container(border=True)
    with map_container:
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                "doubleClick": "reset",
                "modeBarButtonsToRemove": [
                    "lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
                    "hoverClosestGeo",
                ],
            },
        )
        # (Legend is now rendered by Plotly itself on the right edge
        # of the map — see fig.update_layout(legend=...) above.)
    # Zodiac / Rising toggles live INLINE with the slider row (see
    # below) so they don't add vertical space above the slider
    # bar. Compact CSS on the checkboxes trims their default gap.

    # Streamlit-native scrub slider — a datetime slider flanked by
    # play-backward / play-forward buttons plus the two zodiac
    # toggles on the far right under the map's legend. Dragging
    # the slider reruns; clicking a play button sets play_dir and
    # the auto-advance block at the top of the file ticks dt.
    _anchor = st.session_state.anchor_dt
    _slider_min = _anchor - timedelta(hours=window_hours / 2)
    _slider_max = _anchor + timedelta(hours=window_hours / 2)
    _step_minutes = max(1, int(round(window_hours * 60 / 200)))
    if ("slider_dt" not in st.session_state
            or st.session_state.slider_dt < _slider_min
            or st.session_state.slider_dt > _slider_max):
        st.session_state.slider_dt = _anchor
    # (Toggles moved to the left column directly under the table.)

    _back_active = (st.session_state.play_dir == "backward")
    _fwd_active  = (st.session_state.play_dir == "forward")
    pcols = st.columns([0.12, 1.0, 0.12], gap="small", vertical_alignment="center")
    pcols[0].button(
        "◀",
        key="play_back",
        on_click=_toggle_play, args=("backward",),
        type="primary" if _back_active else "secondary",
        width="stretch",
        help="Play backward through time — click again to stop.",
    )
    pcols[1].slider(
        "Moment (UTC)",
        min_value=_slider_min,
        max_value=_slider_max,
        step=timedelta(minutes=_step_minutes),
        key="slider_dt",
        format="YYYY-MM-DD HH:mm",
        label_visibility="collapsed",
    )
    pcols[2].button(
        "▶",
        key="play_fwd",
        on_click=_toggle_play, args=("forward",),
        type="primary" if _fwd_active else "secondary",
        width="stretch",
        help="Play forward through time — click again to stop.",
    )

    # Six step buttons sit directly under the slider, spanning only
    # the map column's width — tighter than the full-page bottom
    # strip they used to live in.
    with st.container(key="step_row"):
        sb = st.columns(6, gap="small")
        sb[0].button("− 1 mo", on_click=_shift, args=(timedelta(days=-30),), width="stretch")
        sb[1].button("− 1 d",  on_click=_shift, args=(timedelta(days=-1),),  width="stretch")
        sb[2].button("− 1 h",  on_click=_shift, args=(timedelta(hours=-1),), width="stretch")
        sb[3].button("+ 1 h",  on_click=_shift, args=(timedelta(hours=1),),  width="stretch")
        sb[4].button("+ 1 d",  on_click=_shift, args=(timedelta(days=1),),   width="stretch")
        sb[5].button("+ 1 mo", on_click=_shift, args=(timedelta(days=30),),  width="stretch")


# ---------------------------------------------------------------------------
# Compact control strip at the bottom of the page
# ---------------------------------------------------------------------------
# Date/time/location live under the table (left column), so this
# bottom strip only holds the step buttons + display options. Wrapped
# in a keyed container so CSS can target .st-key-bottom_ctrl and
# shrink all its buttons without affecting the play buttons above.

st.markdown('<div class="control-strip-top"></div>', unsafe_allow_html=True)

with st.container(key="bottom_ctrl"):
    # Only the Track-window slider lives in this strip now. Step
    # buttons are under the slider in MAP_COL; projection is fixed
    # to the Globe; the body-tracks / zodiac / rising toggles live
    # under the table; date/time/location live above the table.
    st.select_slider(
        "Track window",
        options=[6, 24, 72, 168, 720],
        key="window_hours",
        format_func=lambda h: {6: "6 h", 24: "24 h", 72: "3 d", 168: "7 d", 720: "30 d"}[h],
        disabled=not st.session_state["show_body_tracks"],
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------------
# Auto-play loop driver
# ---------------------------------------------------------------------------
# The slider tick already happened at the TOP of the script (before
# the widget rendered). Here at the bottom we just sleep and trigger
# another rerun, which lands us back at the top for the next tick.
# Stopping the loop is just play_dir being toggled back to None by
# a button click — at that point the top block no-ops and the end
# block no-ops, so the script runs normally and the loop terminates.
if st.session_state.play_dir is not None:
    import time as _time
    _time.sleep(0.12)
    st.rerun()
