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

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
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

# Twelve evenly-spaced hues around the color wheel — 30° apart, starting
# at Aries = red. Each entry is HSL(hue, 75%, 55%) converted to hex, so
# the saturation and lightness are constant and the only thing changing
# is hue. This gives a true rainbow ring that maps 1:1 onto the zodiac
# ring, with warm signs (Aries/Leo) in the red-orange band, cool signs
# (Libra/Aquarius) in the cyan-blue band, and the ring closing back at
# Pisces → Aries.
SIGN_COLORS = [
    "#e23636",  # Aries        hue   0° — red
    "#e28c36",  # Taurus       hue  30° — orange
    "#e2e236",  # Gemini       hue  60° — yellow
    "#8ce236",  # Cancer       hue  90° — lime
    "#36e236",  # Leo          hue 120° — green
    "#36e28c",  # Virgo        hue 150° — spring green
    "#36e2e2",  # Libra        hue 180° — cyan
    "#368ce2",  # Scorpio      hue 210° — azure
    "#3636e2",  # Sagittarius  hue 240° — blue
    "#8c36e2",  # Capricorn    hue 270° — violet
    "#e236e2",  # Aquarius     hue 300° — magenta
    "#e2368c",  # Pisces       hue 330° — pink
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

# ─── DEMO [Hour 3]: @st.cache_resource — memoises an OBJECT (singleton) ─────
# EphemerisLoader opens file handles and keeps its own LRU cache of recently-
# read monthly chunks. We do NOT want to rebuild it on every rerun. @st.cache_
# resource guarantees ONE instance per server process, shared across reruns
# and across users. Use it for DB connections, ML model weights, loaders —
# anything with internal state you want shared.
#
# Contrast with portwatch's @st.cache_data on load_data:
#   • cache_data   → memoises a VALUE (pickled, copied per caller). For data.
#   • cache_resource → memoises a THING (identity, shared). For objects.
# The question to ask yourself: "is this a DataFrame/dict/list, or a THING
# with internal state?" That picks the decorator.
@st.cache_resource
def get_loader() -> EphemerisLoader:
    """Binary ephemeris loader — cached across reruns (it holds chunks in RAM)."""
    return EphemerisLoader(DATA_DIR)


# ─── DEMO [Hour 3]: @st.cache_data + the UNDERSCORE-ARG GOTCHA ──────────────
# cache_data memoises the return value. It decides cache hits by HASHING the
# arguments. An EphemerisLoader instance can't be hashed (file handles),
# so without the underscore you'd get UnhashableParamError at first call.
# The leading underscore on `_loader` is a Streamlit convention — it tells
# cache_data "don't hash this argument." The cache key becomes just
# (dt_iso, window_hours, n_samples). This is THE most common "my cache
# doesn't work" question — write "_ = skip hashing" on the board.
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
      /* Hide only the Deploy button / main menu — LEAVE Streamlit's
         header intact, because the sidebar collapse/expand toggle
         lives inside it and hiding the header was taking the toggle
         with it. Small price of a thin header bar is worth having
         the sidebar always reachable. */
      div[data-testid="stToolbar"]    { display: none !important; }
      #MainMenu                       { display: none !important; }
      footer                          { visibility: hidden; }
      /* Make sure the collapsed-sidebar indicator (the chevron that
         appears when the sidebar is closed) is always visible. */
      [data-testid="collapsedControl"] { display: block !important; }

      /* With the header gone, pull the main column right up to the
         top of the viewport so the map starts where the eye expects. */
      div.block-container {
          padding-top: 0.6rem;
          padding-bottom: 1rem;
          padding-left: 1.5rem;
          padding-right: 1.5rem;
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
      /* Compact global title bar — spans full width, fixed 40px,
         title on the left and current moment + overlay pills on the
         right. Sets the "app" frame without wasting vertical space. */
      .app-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0.45rem 0.9rem;
          background: linear-gradient(90deg, #0a1226 0%, #0c1631 100%);
          border: 1px solid #1e2d4d;
          border-radius: 10px;
          margin-bottom: 0.9rem;
      }
      .app-bar .app-title {
          font-size: 0.95rem;
          font-weight: 600;
          letter-spacing: 0.02em;
          color: #e6edf7;
          display: flex; align-items: center; gap: 0.45rem;
      }
      .app-bar .app-title .glyph { font-size: 1.1rem; }
      .app-bar .app-meta {
          font-size: 0.82rem;
          color: #8392b0;
          font-variant-numeric: tabular-nums;
      }
      .app-bar .app-meta b { color: #e6edf7; font-weight: 600; }
      .app-bar .pill {
          display: inline-block;
          padding: 0.08rem 0.6rem;
          margin-left: 0.45rem;
          border: 1px solid #2a3859;
          border-radius: 999px;
          color: #cbd5e1;
          font-size: 0.75rem;
      }

      /* Pane headers — small, muted, uppercase labels for each column
         so everything lines up at the same baseline. */
      .pane-title {
          font-size: 0.68rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #8392b0;
          margin: 0 0 0.4rem 0;
          padding-bottom: 0.35rem;
          border-bottom: 1px solid #1e2d4d;
          height: 22px;
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
if "scrub_h" not in st.session_state:
    st.session_state.scrub_h = 0.0


def _shift(delta: timedelta) -> None:
    """Jump anchor_dt by `delta` (via the step buttons) and reset the
    scrub slider so it re-centers on the new anchor."""
    new = st.session_state.anchor_dt + delta
    st.session_state.anchor_dt = new
    st.session_state.d_input = new.date()
    st.session_state.t_input = new.time().replace(microsecond=0)
    st.session_state.scrub_h = 0.0


def _sync_from_widgets() -> None:
    """Called when the user edits Date or Time directly — anchor jumps,
    scrub resets."""
    st.session_state.anchor_dt = datetime.combine(
        st.session_state.d_input,
        st.session_state.t_input,
        tzinfo=timezone.utc,
    )
    st.session_state.scrub_h = 0.0


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
    "projection": "Globe",
    "layer": "Body tracks",
    "window_hours": 168,
    "show_zodiac_band": True,
    "show_rising": False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

projection       = st.session_state["projection"]
layer            = st.session_state["layer"]
window_hours     = st.session_state["window_hours"]
show_zodiac_band = st.session_state["show_zodiac_band"]
show_rising      = st.session_state["show_rising"]

# Clamp the scrub slider to the current window so widening/narrowing
# the window doesn't leave scrub_h out of range.
_half = window_hours / 2.0
if st.session_state.scrub_h < -_half:
    st.session_state.scrub_h = -_half
elif st.session_state.scrub_h > _half:
    st.session_state.scrub_h = _half

# Effective moment: anchor shifted by the scrub offset.
dt = st.session_state.anchor_dt + timedelta(hours=float(st.session_state.scrub_h))


# ---------------------------------------------------------------------------
# Global title bar — compact, spans full width, sits above everything
# ---------------------------------------------------------------------------
_window_label = {6: "6 h", 24: "24 h", 72: "3 d", 168: "7 d", 720: "30 d"}[int(window_hours)]
_meta_pills = f'<span class="pill">{layer}</span>'
if layer == "Body tracks":
    _meta_pills += f'<span class="pill">{_window_label}</span>'

st.markdown(
    f"""
    <div class="app-bar">
      <div class="app-title"><span class="glyph">🌍</span> Ephemeris → Earth</div>
      <div class="app-meta">
        <b>{dt.strftime('%Y-%m-%d %H:%M')}</b> UTC · {dt.strftime('%A')}
        {_meta_pills}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

TABLE_COL, LEGEND_COL, MAP_COL = st.columns([1.2, 0.5, 1.6], gap="small")


# ─── DEMO [Hour 3]: the lazy-load moment ────────────────────────────────────
# This is where 46 MB on disk becomes ~128 KB in RAM. loader.get(dt) reads
# ONE monthly chunk file (the one containing dt), keeps it in the loader's
# internal dict (hard-capped at 12 chunks, LRU eviction), and returns one
# row. Change the date but stay in the same month → zero disk I/O, cache hit.
# Change to a new month → one 128 KB file read. You never pay more than
# ~1.5 MB of RAM even after running the app all day. This is the "Streamlit
# caches the entry point; your code does its own smart paging underneath"
# pattern.
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

_window_label = {6: "6 h", 24: "24 h", 72: "3 d", 168: "7 d", 720: "30 d"}[int(window_hours)]
_meta_pills = f'<span class="pill">{layer}</span>'
if layer == "Body tracks":
    _meta_pills += f'<span class="pill">window {_window_label}</span>'

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
        fig.add_trace(go.Scattergeo(
            lat=grid_lats[mask],
            lon=grid_lons[mask],
            mode="markers",
            marker=dict(size=6, color=SIGN_COLORS[i], opacity=0.35, line=dict(width=0)),
            name=f"↑ {sign}",
            hovertemplate="%{lat:.0f}°, %{lon:.0f}°<br>Rising: " + sign + "<extra></extra>",
            legendgroup="rising",
            legendgrouptitle_text="Rising sign" if i == 0 else None,
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
N_TRAIL_SEGMENTS = 8
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
    """Opacity for segment `si` — 0 = oldest (faint), N-1 = newest."""
    return 0.08 + 0.92 * (si / max(1, N_TRAIL_SEGMENTS - 1))


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

        # Static view trail = the _trail_samples points centered on dt
        # (window midpoint). Animation frames will slide this window
        # to the tail ending at each frame's moment.
        center_idx = len(lats) // 2
        start = max(0, center_idx - _trail_samples // 2)
        end = min(len(lats), start + _trail_samples)
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
                showlegend=(si == N_TRAIL_SEGMENTS - 1),
                legendgroup=name,
                hoverinfo="skip",
            ))
            trail_indices[name].append(len(fig.data) - 1)

# Overlay: day/night terminator — great circle 90° from the subsolar
# point. Also drop a marker at the antisolar point (local midnight).
if layer == "Day / night":
    sun = next((s for s in subpoints if s[0] == "Sun"), None)
    if sun is not None:
        sun_lat, sun_lon = sun[1], sun[2]
        term_lats, term_lons = terminator(sun_lat, sun_lon)
        fig.add_trace(go.Scattergeo(
            lat=term_lats, lon=term_lons, mode="lines",
            line=dict(color="#fbbf24", width=2, dash="dot"),
            name="Terminator",
            hoverinfo="skip",
        ))
        anti_lat = -sun_lat
        anti_lon = ((sun_lon + 180.0 + 180.0) % 360.0) - 180.0
        fig.add_trace(go.Scattergeo(
            lat=[anti_lat], lon=[anti_lon], mode="markers",
            marker=dict(size=8, color=SURFACE, line=dict(color=MUTED, width=1)),
            name="Antisolar",
            hovertemplate="Antisolar (midnight)<extra></extra>",
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
    showcoastlines=True,  coastlinecolor="#4b5d85", coastlinewidth=0.6,
    showland=True,        landcolor=SURFACE,
    showocean=True,       oceancolor=BG,
    showlakes=False,
    showcountries=True,   countrycolor="#223353",  countrywidth=0.4,
    showframe=False,
    bgcolor=BG,
)
if _projection_type == "orthographic":
    # Rotate so the current subsolar longitude faces the camera — the
    # view follows the action as you step through time.
    sun_sub = next((s for s in subpoints if s[0] == "Sun"), None)
    center_lon = sun_sub[2] if sun_sub else 0.0
    geo_kwargs["projection_rotation"] = dict(lon=center_lon, lat=20, roll=0)
else:
    geo_kwargs["lataxis_range"] = [-80, 80]
    geo_kwargs["lonaxis_range"] = [-180, 180]

fig.update_geos(**geo_kwargs)

# Map sizing — the right column is ~half the page width, so an
# orthographic globe naturally wants a square canvas. Plotly can't
# measure the column directly, so we pick a fixed height that looks
# roughly square on a standard monitor.
_map_height = 520
_bottom_margin = 6

# Legend placement: overlaid in the top-right of the map plot area so
# it doesn't eat vertical space. Semi-transparent background so the
# ocean still shows through.
fig.update_layout(
    height=_map_height,
    margin=dict(l=0, r=0, t=6, b=_bottom_margin),
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    dragmode="zoom",
    legend=dict(
        orientation="v",
        yanchor="top",   y=0.98,
        xanchor="right", x=0.99,
        bgcolor="rgba(11,18,32,0.75)",
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(color=FG, size=11),
    ),
    showlegend=False,  # legend lives in its own middle column now
)

# ---------------------------------------------------------------------------
# Render the top row: table, legend, map
# ---------------------------------------------------------------------------

with TABLE_COL:
    st.markdown('<div class="pane-title">Positions</div>', unsafe_allow_html=True)
    rows = []
    for name, sub_lat, sub_lon, ecl_lon in subpoints:
        rows.append({
            "Body":       name,
            "Ecliptic λ": f"{ecl_lon:7.3f}°",
            "Zodiac":     zodiac_sign_name(ecl_lon),
            "Sub-lat":    f"{sub_lat:+6.2f}°",
            "Sub-lon":    f"{sub_lon:+7.2f}°",
        })
    st.dataframe(rows, hide_index=True, use_container_width=True, height=360)


# --- Custom HTML legend in the middle column -------------------------------
# Plotly's built-in legend sits inside the figure frame; the user asked
# to pull it out between the table and the globe. We build it as a
# styled HTML block that lists only the overlays currently active.
with LEGEND_COL:
    st.markdown('<div class="pane-title">Key</div>', unsafe_allow_html=True)
    legend_items: list[str] = []
    if layer == "Body tracks":
        legend_items.append('<div class="legend-section">Bodies</div>')
        for bname, _, _ in BODIES:
            color = BODY_COLORS[bname]
            legend_items.append(
                f'<div class="legend-row">'
                f'<span class="legend-swatch" style="background:{color};"></span>'
                f'<span class="legend-label">{bname}</span>'
                f'</div>'
            )
    if show_zodiac_band:
        if legend_items:
            legend_items.append('<div class="legend-gap"></div>')
        legend_items.append('<div class="legend-section">Zodiac</div>')
        for i, sign in enumerate(ZODIAC_SIGNS):
            legend_items.append(
                f'<div class="legend-row">'
                f'<span class="legend-swatch" style="background:{SIGN_COLORS[i]};"></span>'
                f'<span class="legend-label">{ZODIAC_SYMBOLS[i]} {sign}</span>'
                f'</div>'
            )
    if show_rising:
        if legend_items:
            legend_items.append('<div class="legend-gap"></div>')
        legend_items.append('<div class="legend-section">Rising</div>')
        for i, sign in enumerate(ZODIAC_SIGNS):
            legend_items.append(
                f'<div class="legend-row">'
                f'<span class="legend-swatch" style="background:{SIGN_COLORS[i]};opacity:0.5;"></span>'
                f'<span class="legend-label">↑ {sign}</span>'
                f'</div>'
            )
    if not legend_items:
        legend_items.append('<div class="legend-section">—</div>')
    st.markdown(
        '<div class="legend-box">' + ''.join(legend_items) + '</div>',
        unsafe_allow_html=True,
    )


with MAP_COL:
    st.markdown('<div class="pane-title">View</div>', unsafe_allow_html=True)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": False,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": [
                "lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
                "hoverClosestGeo",
            ],
        },
    )
    # Streamlit-native scrub slider. Every drag fires a full rerun,
    # so the positions table and the map both update in lockstep.
    st.slider(
        "Scrub within window (hours from anchor)",
        min_value=float(-window_hours / 2),
        max_value=float(window_hours / 2),
        step=max(0.25, float(window_hours) / 200.0),
        key="scrub_h",
        label_visibility="collapsed",
        format="%+.2f h",
    )


# ---------------------------------------------------------------------------
# Compact control strip at the bottom of the page
# ---------------------------------------------------------------------------
# All widgets live down here so the top row (table + legend + globe)
# is uncluttered. Two tight rows: time nav on top, display on bottom.

st.markdown('<div class="control-strip-top"></div>', unsafe_allow_html=True)

c1 = st.columns([1.2, 1, 0.6, 0.6, 0.6, 0.2, 0.6, 0.6, 0.6])
c1[0].date_input("Date (UTC)", key="d_input", on_change=_sync_from_widgets, label_visibility="collapsed")
c1[1].time_input("Time (UTC)", key="t_input", on_change=_sync_from_widgets, step=3600, label_visibility="collapsed")
c1[2].button("− 1 mo", on_click=_shift, args=(timedelta(days=-30),), use_container_width=True)
c1[3].button("− 1 d",  on_click=_shift, args=(timedelta(days=-1),),  use_container_width=True)
c1[4].button("− 1 h",  on_click=_shift, args=(timedelta(hours=-1),), use_container_width=True)
c1[6].button("+ 1 h",  on_click=_shift, args=(timedelta(hours=1),),  use_container_width=True)
c1[7].button("+ 1 d",  on_click=_shift, args=(timedelta(days=1),),   use_container_width=True)
c1[8].button("+ 1 mo", on_click=_shift, args=(timedelta(days=30),),  use_container_width=True)

c2 = st.columns([1, 1, 2, 0.8, 0.8])
c2[0].selectbox(
    "Projection",
    ["Globe", "Natural Earth", "Flat (equirectangular)"],
    key="projection",
    label_visibility="collapsed",
)
c2[1].selectbox(
    "Overlay",
    ["Body tracks", "Day / night", "None"],
    key="layer",
    label_visibility="collapsed",
)
c2[2].select_slider(
    "Track window",
    options=[6, 24, 72, 168, 720],
    key="window_hours",
    format_func=lambda h: {6: "6 h", 24: "24 h", 72: "3 d", 168: "7 d", 720: "30 d"}[h],
    disabled=(layer != "Body tracks"),
    label_visibility="collapsed",
)
c2[3].checkbox("Zodiac band",   key="show_zodiac_band")
c2[4].checkbox("Rising signs",  key="show_rising")
