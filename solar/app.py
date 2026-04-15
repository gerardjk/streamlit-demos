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
      /* Hide the Deploy button, status toast, and menu — but LEAVE the
         header element itself in place so the sidebar collapse/expand
         toggle (which lives inside stHeader) stays clickable. Making
         the header transparent + zero-height reclaims its space
         without removing the toggle. */
      header[data-testid="stHeader"] {
          background: transparent !important;
          height: 0 !important;
      }
      div[data-testid="stToolbar"]    { display: none !important; }
      #MainMenu                       { display: none !important; }
      footer                          { visibility: hidden; }

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
      /* The compact header strip uses these classes. */
      .dash-header {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 1rem;
          padding: 0 0 0.5rem 0;
          border-bottom: 1px solid #1e293b;
          margin-bottom: 0.6rem;
      }
      .dash-title {
          font-size: 1.15rem;
          font-weight: 600;
          color: #e2e8f0;
          letter-spacing: 0.01em;
      }
      .dash-title .dim { color: #94a3b8; font-weight: 400; }
      .dash-meta {
          font-size: 0.9rem;
          color: #94a3b8;
          font-variant-numeric: tabular-nums;
      }
      .dash-meta b { color: #e2e8f0; font-weight: 600; }
      .dash-meta .pill {
          display: inline-block;
          padding: 0.05rem 0.55rem;
          margin-left: 0.5rem;
          border: 1px solid #334155;
          border-radius: 999px;
          color: #cbd5e1;
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
# The canonical "current moment" lives in st.session_state.dt. The date/
# time widgets and the step buttons both read and write through it, so
# they stay in sync no matter which one the user touches.

if "dt" not in st.session_state:
    init_dt = datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc)
    st.session_state.dt = init_dt
    st.session_state.d_input = init_dt.date()
    st.session_state.t_input = init_dt.time().replace(microsecond=0)


def _shift(delta: timedelta) -> None:
    """Nudge the current moment by `delta` and push the new values into
    the widget keys so the date/time inputs reflect the change."""
    new = st.session_state.dt + delta
    st.session_state.dt = new
    st.session_state.d_input = new.date()
    st.session_state.t_input = new.time().replace(microsecond=0)


def _sync_from_widgets() -> None:
    """Called when the user edits the date or time widget directly."""
    st.session_state.dt = datetime.combine(
        st.session_state.d_input,
        st.session_state.t_input,
        tzinfo=timezone.utc,
    )


# ---------------------------------------------------------------------------
# Sidebar (controls)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    st.caption(f"Coverage {start_iso[:10]} → {end_iso[:10]}")

    st.subheader("Time", divider="gray")
    st.date_input("Date (UTC)", key="d_input", on_change=_sync_from_widgets)
    st.time_input("Time (UTC)", key="t_input", on_change=_sync_from_widgets, step=3600)

    # Step buttons laid out as two symmetric rows — back on top, forward
    # below. Using equal-width columns so each button is uniform.
    st.caption("Step backward")
    back = st.columns(3)
    back[0].button("− 1 mo", on_click=_shift, args=(timedelta(days=-30),),  use_container_width=True)
    back[1].button("− 1 d",  on_click=_shift, args=(timedelta(days=-1),),   use_container_width=True)
    back[2].button("− 1 h",  on_click=_shift, args=(timedelta(hours=-1),),  use_container_width=True)

    st.caption("Step forward")
    fwd = st.columns(3)
    fwd[0].button("+ 1 h",  on_click=_shift, args=(timedelta(hours=1),),    use_container_width=True)
    fwd[1].button("+ 1 d",  on_click=_shift, args=(timedelta(days=1),),     use_container_width=True)
    fwd[2].button("+ 1 mo", on_click=_shift, args=(timedelta(days=30),),    use_container_width=True)

    st.subheader("Display", divider="gray")
    # Projection choice. Orthographic is the default because it reads
    # instantly as a globe — and lines of constant declination become
    # visible curves, which matches people's intuition of "the Sun's
    # path across the sky" better than a flat equirectangular band.
    projection = st.selectbox(
        "Projection",
        ["Globe", "Natural Earth", "Flat (equirectangular)"],
        index=0,
        help="Globe rotates in 3D; flat is best for comparing longitudes.",
    )
    # Mutually-exclusive primary layer — owns the bulk of the visual.
    layer = st.selectbox(
        "Overlay",
        ["Body tracks", "Day / night", "None"],
        help=(
            "Body tracks — subpoint path of each body over the selected window.\n"
            "Day / night — the solar terminator."
        ),
    )
    window_hours = st.select_slider(
        "Track window",
        options=[6, 24, 72, 168, 720],
        value=168,
        format_func=lambda h: {6: "6 h", 24: "24 h", 72: "3 d", 168: "7 d", 720: "30 d"}[h],
        disabled=(layer != "Body tracks"),
        help=(
            "Width of the time window drawn for the Body tracks overlay. "
            "Longer windows reveal the declination drift — especially for "
            "the Moon, whose path becomes a clear sinusoid at 7 days or more."
        ),
    )

    # Zodiac layers are independent toggles — they can stack on top of
    # any primary overlay. This replaces the old mutually-exclusive
    # "Rising sign" option in the Overlay dropdown.
    st.caption("Zodiac")
    show_zodiac_band = st.checkbox(
        "Zodiac band",
        value=True,
        help=(
            "The ecliptic great circle split into 12 colored 30° arcs — "
            "each is where that sign's longitudes currently project onto "
            "Earth. Every planet's subpoint lies on its own sign's arc."
        ),
    )
    show_rising = st.checkbox(
        "Rising signs",
        value=False,
        help=(
            "Color each lat/lon by which zodiac sign is rising on the "
            "eastern horizon there right now. Sweeps westward ~360°/day."
        ),
    )

    with st.expander("About"):
        st.markdown(
            "Each body's **subpoint** is the spot on Earth where it is at "
            "the zenith. As Earth rotates, the subpoint sweeps westward — "
            "one lap per day for the Sun. The **Body tracks** overlay "
            "draws the path that each body's subpoint traces over the "
            "selected time window."
        )

dt = st.session_state.dt


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

st.markdown(
    f"""
    <div class="dash-header">
      <div class="dash-title">🌍 Ephemeris → Earth <span class="dim">· subpoints &amp; tracks</span></div>
      <div class="dash-meta">
        <b>{dt.strftime('%Y-%m-%d %H:%M')}</b> UTC · {dt.strftime('%A')}
        {_meta_pills}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


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
        for si, (s0, s1) in enumerate(_segment_bounds(start, end, N_TRAIL_SEGMENTS)):
            s_lats, s_lons = split_antimeridian(lats[s0:s1], lons[s0:s1])
            fig.add_trace(go.Scattergeo(
                lat=s_lats, lon=s_lons, mode="lines",
                line=dict(color=BODY_COLORS[name], width=1.8),
                opacity=_segment_alpha(si),
                # Only the newest, brightest segment gets a legend
                # entry; the rest share its legendgroup so toggling
                # the legend toggles the whole trail.
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


# ---------------------------------------------------------------------------
# Animation frames (Body tracks only)
# ---------------------------------------------------------------------------
# We build one frame per animation step. Each frame carries:
#   (a) one trail trace per body — the track from the start of the
#       window up to the frame's moment (growing trail), split on the
#       antimeridian for flat projections,
#   (b) the "Now" markers repositioned to the frame's moment.
#
# Plotly's frame mechanism requires that every frame lists the
# `traces` indices it is updating. We grabbed those indices above
# (trail_indices for the body lines, _now_marker_index for markers).
# Initial figure state = the final frame (so the static render equals
# what you see at the end of playback), which means nothing visually
# changes until the user presses ▶ Play.
if layer == "Body tracks" and track_arrays:
    # trail_indices already populated above (dict[name, list[int]],
    # one entry per segment). ~60 animation frames feels smooth at
    # the 80ms/frame default and keeps the payload reasonable.
    _n_points = len(next(iter(track_arrays.values()))[0])
    _n_frames = min(60, _n_points)
    _frame_indices = np.linspace(0, _n_points - 1, _n_frames, dtype=int)

    _half = timedelta(hours=window_hours / 2)
    _step = (2 * _half) / max(1, _n_points - 1)

    # Each frame shows a sliding N-segment fading trail ending at
    # that frame's moment. Segments are updated by index (via the
    # `traces=` parameter of go.Frame) so Plotly swaps their data
    # without rebuilding the whole figure.
    frames: list[go.Frame] = []
    for fi, pi in enumerate(_frame_indices):
        frame_data = []
        frame_traces = []
        tail_start = max(0, int(pi) - _trail_samples + 1)
        tail_end = int(pi) + 1
        for bname, (b_lats, b_lons) in track_arrays.items():
            seg_bounds = _segment_bounds(tail_start, tail_end, N_TRAIL_SEGMENTS)
            # If the tail is too short for a full N-segment split,
            # pad with empty slots so we still match trail_indices.
            for si in range(N_TRAIL_SEGMENTS):
                if si < len(seg_bounds):
                    s0, s1 = seg_bounds[si]
                    s_lats, s_lons = split_antimeridian(b_lats[s0:s1], b_lons[s0:s1])
                else:
                    s_lats = np.array([np.nan])
                    s_lons = np.array([np.nan])
                frame_data.append(go.Scattergeo(lat=s_lats, lon=s_lons))
                frame_traces.append(trail_indices[bname][si])

        # Markers repositioned to frame moment
        frame_data.append(go.Scattergeo(
            lat=[track_arrays[n][0][pi] for n in now_marker_names if n in track_arrays],
            lon=[track_arrays[n][1][pi] for n in now_marker_names if n in track_arrays],
            text=[n for n in now_marker_names if n in track_arrays],
        ))
        frame_traces.append(_now_marker_index)

        # Short YY-MM-DD label for the slider tick / currentvalue.
        t_i = dt - _half + _step * int(pi)
        label = t_i.strftime("%y-%m-%d")
        frames.append(go.Frame(data=frame_data, traces=frame_traces, name=label))

    fig.frames = frames
    _frame_labels = [f.name for f in frames]

    # Scrubber slider only — no play/pause buttons. User drags to
    # step through time. scattergeo animation requires redraw=True
    # because the projection re-renders the geo canvas each frame.
    fig.update_layout(
        sliders=[dict(
            active=len(_frame_labels) - 1,
            x=0.0, y=-0.08, len=1.0,
            pad=dict(t=0, b=0),
            currentvalue=dict(
                prefix="t = ",
                font=dict(color=FG, size=12),
            ),
            bgcolor="rgba(30,41,59,0.6)",
            activebgcolor="#fbbf24",
            bordercolor=BORDER,
            tickcolor=MUTED,
            font=dict(color=MUTED, size=10),
            steps=[
                dict(
                    method="animate",
                    label=lbl,
                    args=[[lbl], dict(
                        frame=dict(duration=0, redraw=True),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                )
                for lbl in _frame_labels
            ],
        )],
    )

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

# Map sizing is tuned for a default laptop viewport: header strip
# (~70px) + map (540px) + animation controls when present (~60px) +
# browser chrome fits inside ~720px. Below that, the positions table
# is intentionally below-the-fold — a common pattern for hero-viz
# dashboards (map-first, details-on-scroll).
_has_animation = bool(layer == "Body tracks" and track_arrays)
_map_height = 540 if _has_animation else 560
_bottom_margin = 60 if _has_animation else 6

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
    showlegend=(layer == "Body tracks" or show_zodiac_band or show_rising),
)

# Plotly config: keep the modebar (for deliberate zoom-in/zoom-out/
# reset clicks) but disable scroll-wheel zoom so the mouse wheel
# still scrolls the page. Strip the buttons that are useless for a
# geo chart — lasso, box-select, autoscale etc.
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


# ---------------------------------------------------------------------------
# Secondary panel (below the fold)
# ---------------------------------------------------------------------------
# Positions table and raw JSON live here. Both are inside an expander
# so the initial viewport is the map-and-header only — a classic
# "hero viz first, details on demand" pattern. The expander starts
# collapsed; users who want the data just click.

with st.expander("📊 Positions & raw data", expanded=False):
    left, right = st.columns([3, 2])

    with left:
        rows = []
        for name, sub_lat, sub_lon, ecl_lon in subpoints:
            rows.append({
                "Body":       name,
                "Ecliptic λ": f"{ecl_lon:7.3f}°",
                "Zodiac":     zodiac_sign_name(ecl_lon),
                "Sub-lat":    f"{sub_lat:+6.2f}°",
                "Sub-lon":    f"{sub_lon:+7.2f}°",
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

    with right:
        st.json({k: v for k, v in entry.items() if v is not None})
