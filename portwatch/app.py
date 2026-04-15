"""
Strait of Hormuz — Daily Transit Monitor.

A real-world operational dashboard built on (synthetic, by default)
IMF PortWatch chokepoint data. Demonstrates KPI-first layout,
filter-upstream-of-views, and cross-chart context filtering via
Plotly click events.
"""

import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _lighten(hex_color: str, amount: float = 0.55) -> str:
    """Blend a hex colour toward white by `amount` (0–1). 0 = original, 1 = white.
    Used for the per-country halo rings behind each port bubble — a light
    tint of the country's own colour reads better than hard white."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"

DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "chokepoint_transits.csv"
DATA_PARQUET = DATA_DIR / "chokepoint_transits.parquet"
PORTS_PATH = DATA_DIR / "gulf_ports.csv"
PORTS_PARQUET = DATA_DIR / "gulf_ports.parquet"
DAILY_PORTS_PATH = DATA_DIR / "daily_gulf_ports.parquet"
HORMUZ_LAT, HORMUZ_LON = 26.566, 56.25  # rough center of the strait

SHIP_TYPE_COLORS = {
    "tanker":        "#ef4444",
    "dry_bulk":      "#a16207",
    "container":     "#3b82f6",
    "gas":           "#eab308",
    "general_cargo": "#14b8a6",
    "ro_ro":         "#a855f7",
}

# Country palette for port markers — picked to sit clearly on top of a
# satellite basemap (no blues, no greens that read as vegetation).
COUNTRY_COLORS = {
    "Saudi Arabia":         "#166534",  # dark green
    "United Arab Emirates": "#86efac",  # light green
    "Iran":                 "#facc15",  # yellow
    "Qatar":                "#f97316",  # orange
    "Kuwait":               "#92400e",  # brown
    "Iraq":                 "#d2b48c",  # tan
    "Bahrain":              "#f472b6",  # pink
    "Oman":                 "#7dd3fc",  # light blue
}

# Short legend labels so the horizontal legend fits on one row.
COUNTRY_LABELS = {
    "Saudi Arabia":         "KSA",
    "United Arab Emirates": "UAE",
}


DAILY_PORT_CARGO_COLS = {
    "tanker":    "portcalls_tanker",
    "container": "portcalls_container",
    "dry_bulk":  "portcalls_dry_bulk",
}


# =============================================================================
# @st.cache_data — memoise a VALUE (a DataFrame, list, dict, anything picklable)
# + the mtime-keyed cache-invalidation trick
# =============================================================================
#
# Core idea: every widget interaction reruns this whole script top-to-bottom.
# Without caching, we'd reread the Parquet and rebuild the 7-day rolling
# averages on every click — wasted work. @st.cache_data fixes that: hash the
# arguments, and if we've seen that hash before, return the cached DataFrame
# instantly instead of running the function again.
#
# Compare with @st.cache_resource (see solar/app.py get_loader):
#   • cache_data     → memoises a VALUE. Pickled, copied per caller, safe to
#                      mutate. Use for DataFrames, arrays, dicts, lists.
#   • cache_resource → memoises an OBJECT. Singleton, shared by identity,
#                      must not be mutated. Use for connections, loaders,
#                      ML models.
#
# The clever bit on this specific function is the `mtime` argument. We don't
# actually use it inside the function body — but because cache_data builds
# its cache key by hashing the arguments, making mtime part of the signature
# means the cache key changes whenever the Parquet file on disk is rewritten.
# The load_daily_ports wrapper below passes DAILY_PORTS_PATH.stat().st_mtime,
# so:
#
#   • File unchanged → same mtime → same cache key → cache hit, instant.
#   • File rewritten → new mtime  → new cache key → cache miss, fresh read.
#
# This is event-driven invalidation instead of time-based. It's more precise
# than `ttl=3600` (which we also set as a belt-and-braces safety net). If
# your data updates on a schedule you don't control, this pattern is gold.
@st.cache_data(ttl=86400, show_spinner=False)
def load_coastline_geojson() -> dict:
    """Natural Earth 50m coastline, fetched once and cached on disk.

    We keep a local copy under data/ so the app still works offline after the
    first successful fetch. The mapbox layer expects an inline GeoJSON dict."""
    local = DATA_DIR / "ne_50m_coastline.geojson"
    if not local.exists():
        url = (
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_50m_coastline.geojson"
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                local.write_bytes(resp.read())
        except Exception:
            return {"type": "FeatureCollection", "features": []}
    try:
        return json.loads(local.read_text())
    except Exception:
        return {"type": "FeatureCollection", "features": []}


@st.cache_data(ttl=86400, show_spinner=False)
def load_country_polygons() -> dict[str, dict]:
    """Natural Earth 50m country polygons, keyed by country name.

    Returns a dict like {"Iran": <FeatureCollection>, ...} containing only the
    Gulf-region countries we actually shade on the map. Splitting per-country
    at load time lets us add one mapbox fill layer per country with its own
    colour — the alternative (one big geojson + feature-state expressions)
    is more complex and no faster for 8 features."""
    local = DATA_DIR / "ne_50m_admin_0_countries.geojson"
    if not local.exists():
        url = (
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_50m_admin_0_countries.geojson"
        )
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                local.write_bytes(resp.read())
        except Exception:
            return {}
    try:
        full = json.loads(local.read_text())
    except Exception:
        return {}
    # Natural Earth stores the country name in properties.NAME or ADMIN —
    # different names for the same countries in different NE vintages, so
    # we check both and also use a few known aliases.
    wanted = set(COUNTRY_COLORS.keys())
    aliases = {
        "United Arab Emirates": {"United Arab Emirates", "UAE"},
        "Saudi Arabia": {"Saudi Arabia"},
    }
    out: dict[str, dict] = {}
    for feat in full.get("features", []):
        props = feat.get("properties", {}) or {}
        names = {props.get("NAME"), props.get("ADMIN"), props.get("NAME_EN")}
        names.discard(None)
        for target in wanted:
            if target in names or names & aliases.get(target, set()):
                out.setdefault(target, {"type": "FeatureCollection", "features": []})
                out[target]["features"].append(feat)
                break
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _load_daily_ports_cached(mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(DAILY_PORTS_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["portid", "date"])
    # Pre-compute 7-day rolling averages for each cargo column AND the total
    # so map lookups are instant regardless of which ship-type filter is active.
    roll_cols = ["portcalls"] + [c for c in DAILY_PORT_CARGO_COLS.values() if c in df.columns]
    for col in roll_cols:
        df[f"{col}_roll7"] = (
            df.groupby("portid", observed=True)[col]
              .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )
    return df


def load_daily_ports() -> pd.DataFrame:
    """Thin wrapper: guards against a missing file, then passes the current
    mtime into the cached loader so disk rewrites auto-invalidate the cache."""
    if not DAILY_PORTS_PATH.exists():
        return pd.DataFrame()
    return _load_daily_ports_cached(DAILY_PORTS_PATH.stat().st_mtime)


@st.cache_data(ttl=3600, show_spinner=False)
def load_ports() -> pd.DataFrame:
    if PORTS_PARQUET.exists():
        return pd.read_parquet(PORTS_PARQUET)
    if PORTS_PATH.exists():
        df = pd.read_csv(PORTS_PATH)
        df.to_parquet(PORTS_PARQUET, index=False)
        return df
    return pd.DataFrame()


# =============================================================================
# CSV-once, Parquet-forever — a two-layer caching pattern
# =============================================================================
#
# The first time this function runs, there's no Parquet file yet, so we do
# the expensive thing: read the CSV, filter to Hormuz rows, narrow the dtypes
# (category for repeated strings, int32 for small ints — drops memory ~4×
# and makes the Parquet file tiny), and write the result back out as Parquet.
#
# Every run after that, the Parquet file exists, so we short-circuit straight
# to pd.read_parquet(). Parquet is columnar and compressed — reading it is
# roughly an order of magnitude faster than parsing a CSV.
#
# Stacked caching layers, from shortest lifetime to longest:
#
#   1. @st.cache_data       → within a running session, the DataFrame lives
#                             in memory. No disk read at all on cache hit.
#   2. The Parquet file     → survives across Streamlit reruns, app restarts,
#                             even server reboots. Next cold start is still
#                             ~10× faster than parsing the CSV.
#   3. The CSV (fallback)   → the authoritative source. Only read when we
#                             have to regenerate the Parquet.
#
# The `ttl=3600` on the decorator means the in-memory cache expires after an
# hour, at which point we'll re-enter the function and hit layer 2 (Parquet).
# That's the right trade-off for data that refreshes daily — we pick up a
# fresh CSV eventually without hammering the disk on every session.
@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> pd.DataFrame:
    if DATA_PARQUET.exists():
        df = pd.read_parquet(DATA_PARQUET)
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        df = df[df["chokepoint_name"].str.contains("Hormuz", case=False, na=False)]
        # Narrow dtypes: category for repeated strings, int32 for small ints.
        # Drops memory ~4x and makes the Parquet tiny.
        df["cargo_type"] = df["cargo_type"].astype("category")
        df["chokepoint_name"] = df["chokepoint_name"].astype("category")
        df["transits"] = df["transits"].astype("int32")
        df.to_parquet(DATA_PARQUET, index=False)
    else:
        # st.error renders a red box; st.stop halts the whole script so we
        # don't run downstream code on missing data.
        st.error(
            f"No data at `{DATA_PATH}`. Run "
            "`python portwatch/scripts/fetch_real_data.py` to download real IMF "
            "PortWatch data."
        )
        st.stop()
    return df


# set_page_config must be the first Streamlit call in the script.
# Calling it after any other st.* function raises an error. Sets the browser
# tab title and the wide layout (default is centered).
st.set_page_config(
    page_title="Hormuz Transit Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Compact CSS: tighten padding without clipping the header.
st.markdown(
    """
    <style>
      /* Page background — slate blue to match the map */
      [data-testid="stAppViewContainer"] { background: #0b1220 !important; }
      [data-testid="stAppViewContainer"] * { color: #e2e8f0; }
      header[data-testid="stHeader"] { height: 0; min-height: 0; visibility: hidden; }
      .block-container {
          padding-top: 0 !important;
          padding-bottom: 0.4rem !important;
          padding-left: 3rem !important;
          padding-right: 3rem !important;
          max-width: 100% !important;
      }
      /* Thin coloured header bar — three-slot grid so the middle truly centers */
      .pw-header {
          background: linear-gradient(90deg, #1e3a8a 0%, #0f172a 100%);
          color: #f8fafc;
          padding: 0.5rem 3rem;
          margin: 0 -3rem 0.7rem -3rem;
          font-size: 1.0rem;
          font-weight: 600;
          letter-spacing: 0.01em;
          border-bottom: 1px solid #1e293b;
          display: grid;
          grid-template-columns: 1fr auto 1fr;
          align-items: center;
      }
      .pw-header .pw-left  { justify-self: start; }
      .pw-header .pw-mid   { justify-self: center; color: #93c5fd; font-weight: 500; font-size: 0.95rem; }
      .pw-header .pw-right { justify-self: end; }
      /* Compact metric cards */
      [data-testid="stMetric"] {
          background: #0f172a;
          border: 1px solid #1e293b;
          border-radius: 4px;
          padding: 0.15rem 0.35rem !important;
          min-width: 0 !important;
      }
      /* Streamlit wraps label/value/delta in an inner div inside stMetric,
         so the grid has to live on that wrapper, not on stMetric itself. */
      [data-testid="stMetric"] > div {
          display: grid !important;
          grid-template-columns: 1fr auto !important;
          grid-template-rows: auto auto !important;
          column-gap: 0.35rem !important;
          align-items: end !important;
          width: 100% !important;
      }
      [data-testid="stMetricLabel"] {
          grid-column: 1 / 3 !important;
          grid-row: 1 !important;
          font-size: 0.55rem !important;
          color: #94a3b8 !important;
          line-height: 1.0 !important;
      }
      [data-testid="stMetricLabel"] p { font-size: 0.55rem !important; }
      [data-testid="stMetricValue"] {
          grid-column: 1 !important;
          grid-row: 2 !important;
          font-size: 0.82rem !important;
          line-height: 1.0 !important;
          color: #f1f5f9 !important;
      }
      [data-testid="stMetricDelta"] {
          grid-column: 2 !important;
          grid-row: 2 !important;
          align-self: end !important;
          white-space: nowrap !important;
          font-size: 0.52rem !important;
          line-height: 1.0 !important;
      }
      [data-testid="stMetricDelta"] svg { width: 0.6rem !important; height: 0.6rem !important; }
      [data-testid="stCaptionContainer"] { margin-top: 0 !important; }
      /* Input widgets keep the dark background */
      [data-baseweb="input"], [data-baseweb="select"] > div { background: #0f172a !important; }
      label[data-baseweb="form-control-label"], .stDateInput label, .stMultiSelect label, .stSlider label {
          color: #cbd5e1 !important; font-size: 0.72rem !important; margin-bottom: 0.1rem !important;
      }
      div[data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
      [data-testid="stSlider"] { padding: 0 !important; }
      section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
      /* Ship-types multiselect: colour each chip to match the stacked bar
         chart. Alphabetical-default order from ship_types:
           1=container, 2=dry_bulk, 3=gas, 4=general_cargo, 5=ro_ro, 6=tanker.
         Toggling items reorders the chips, but the common case (everything
         selected) matches the chart legend. */
      [data-testid="stMultiSelect"] [data-baseweb="tag"] {
          color: #0b1220 !important;
          border: none !important;
      }
      [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
      [data-testid="stMultiSelect"] [data-baseweb="tag"] div {
          color: inherit !important;
      }
      [data-testid="stMultiSelect"] [data-baseweb="tag"]:nth-of-type(1) { background: #3b82f6 !important; color: #f8fafc !important; }
      [data-testid="stMultiSelect"] [data-baseweb="tag"]:nth-of-type(2) { background: #a16207 !important; color: #f8fafc !important; }
      [data-testid="stMultiSelect"] [data-baseweb="tag"]:nth-of-type(3) { background: #eab308 !important; color: #0b1220 !important; }
      [data-testid="stMultiSelect"] [data-baseweb="tag"]:nth-of-type(4) { background: #14b8a6 !important; color: #0b1220 !important; }
      [data-testid="stMultiSelect"] [data-baseweb="tag"]:nth-of-type(5) { background: #a855f7 !important; color: #f8fafc !important; }
      [data-testid="stMultiSelect"] [data-baseweb="tag"]:nth-of-type(6) { background: #ef4444 !important; color: #f8fafc !important; }
      /* Map wrapper: a span.pw-map-anchor sits in an stElementContainer
         immediately before the plotly map. The :has() + sibling rule paints
         the next element-container (the map) with a thin white border. */
      [data-testid="stElementContainer"]:has(.pw-map-anchor) {
          display: none !important;
      }
      [data-testid="stElementContainer"]:has(.pw-map-anchor) + [data-testid="stElementContainer"] {
          border: 2px solid #ffffff !important;
          border-radius: 4px !important;
          padding: 3px !important;
          background: #0b1220 !important;
          box-sizing: border-box !important;
      }
      /* Chart column section headings — centered, a bit larger */
      .chart-heading {
          color: #e2e8f0;
          font-size: 1.05rem;
          font-weight: 700;
          letter-spacing: 0.03em;
          text-transform: uppercase;
          text-align: center;
          border-bottom: 1px solid #1e293b;
          padding: 0.15rem 0 0.2rem 0;
          margin: 0.35rem 0 0.3rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
# ──────────────────────────────── Thin coloured header ──────────────────────
_today_str = date.today().strftime("%A, %d %B %Y")
st.markdown(
    f"<div class='pw-header'>"
    f"<span class='pw-left'>Strait of Hormuz · Daily Transit Monitor</span>"
    f"<span class='pw-mid'>{_today_str}</span>"
    f"<span class='pw-right'></span>"
    f"</div>",
    unsafe_allow_html=True,
)

# Cached calls — on rerun these are cheap dictionary lookups, not file reads.
df = load_data()
ports_df = load_ports()
daily_ports = load_daily_ports()

min_d, max_d = df.date.min().date(), df.date.max().date()
default_start = max(min_d, max_d - timedelta(days=365))
ship_types = sorted(df.cargo_type.dropna().unique().tolist())

# Main row: two columns. Each uses st.empty() placeholders so we can fill in
# two passes. The top-of-chart widgets (date range + smoothing) render RIGHT
# NOW so their values feed the filter/aggregate code below. The rest of
# chart_col (Map-date slider, time series) and map_col are filled later, once
# aggregates + the selected day are known.
map_col, chart_col = st.columns([1, 1], gap="medium")
with chart_col:
    top_widgets_slot = st.empty()
    rest_slot = st.empty()
with map_col:
    map_slot = st.empty()
    ship_types_slot = st.empty()
    kpi_slot = st.empty()

# Fixed 7-day rolling window for every "rolling / avg" visual on the page.
# We pre-compute this once at load time inside _load_daily_ports_cached
# (see portcalls_*_roll7 columns) so the map and the trend line use the
# exact same window. If you ever want to make it configurable again, turn
# this into a slider and remove the pre-compute from the cached loader.
ROLLING_WINDOW = 7

with top_widgets_slot.container():
    # Narrow, centred date-range picker that sits directly above the red
    # Map-date slider. Flanking empty columns do the centring.
    _dl, _dm, _dr = st.columns([2, 3, 2], gap="small")
    with _dm:
        date_range = st.date_input(
            "Date range",
            value=(default_start, max_d),
            min_value=min_d,
            max_value=max_d,
        )

# Ship-types filter renders into a slot below the map (map_col). It sits in
# the empty space under the map and to the left of the bar chart. We fill
# the slot RIGHT NOW so selected_types is available to the filter code below.
with ship_types_slot.container():
    selected_types = st.multiselect(
        "Ship types",
        options=ship_types,
        default=ship_types,
    )


# Guard against the mid-selection single-date blip st.date_input can return.
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = default_start, max_d

mask = (
    (df.date.dt.date >= start)
    & (df.date.dt.date <= end)
    & (df.cargo_type.isin(selected_types))
)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# ──────────────────────────────── Aggregate once ────────────────────────────────
daily = (
    fdf.groupby("date", as_index=False)["transits"].sum()
       .sort_values("date")
       .reset_index(drop=True)
)
daily["rolling"] = daily["transits"].rolling(ROLLING_WINDOW, min_periods=1).mean()

# Per-day cargo-type breakdown embedded as a monospaced Unicode bar chart in
# the hover tooltip of the Daily trace. Plotly hovers are text-only (no SVG),
# but with hoverlabel.font.family = "Courier New" the block characters align
# cleanly. Each day's tooltip shows every cargo type in the current filter,
# with a 10-char █ bar scaled to the GLOBAL max (so day-to-day comparison is
# meaningful) and the integer count on the right.
day_pivot = fdf.pivot_table(
    index="date", columns="cargo_type", values="transits",
    aggfunc="sum", fill_value=0, observed=True,
)
cargo_order = day_pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
day_pivot = day_pivot[cargo_order]
_bar_max = float(day_pivot.to_numpy().max()) or 1.0
_BAR_W = 10
_LABEL_W = max((len(c) for c in cargo_order), default=8) + 2


def _breakdown_text(row) -> str:
    lines = []
    for col in cargo_order:
        val = int(row[col])
        filled = int(round(_BAR_W * val / _bar_max))
        bar = ("█" * filled).ljust(_BAR_W)
        lines.append(f"{col:<{_LABEL_W}}{bar} {val:>4}")
    return "<br>".join(lines)


daily["breakdown"] = daily["date"].map(
    lambda d: _breakdown_text(day_pivot.loc[d]) if d in day_pivot.index else ""
)

latest_val = float(daily["transits"].iloc[-1])
avg_7 = float(daily["transits"].tail(7).mean())
avg_30 = float(daily["transits"].tail(30).mean())

latest_date = daily["date"].iloc[-1]
ya_target = latest_date - pd.Timedelta(days=365)
ya_window = daily[(daily.date >= ya_target - pd.Timedelta(days=3)) &
                  (daily.date <= ya_target + pd.Timedelta(days=3))]
year_ago = float(ya_window["transits"].mean()) if not ya_window.empty else float("nan")

slider_min = daily["date"].iloc[0].to_pydatetime()
slider_max = daily["date"].iloc[-1].to_pydatetime()

# Fill rest_slot (chart_col below the widgets row) with the blue time series,
# the Map-date slider, and the cargo-type bar chart. Each section is wrapped
# with a chart-heading so the two panels read as distinct graphs.
with rest_slot.container():
    # Map-date slider FIRST (above the heading), then the centered heading,
    # then the chart itself. A small play/pause button sits to the left of
    # the slider so students can watch the map animate through time.
    #
    # Animation pattern: the play button flips st.session_state.playing.
    # If playing, we advance the stored date by one step BEFORE the slider
    # renders (so the widget picks up the new value), then at the very
    # bottom of the script we sleep briefly and call st.rerun() to tick.
    SLIDER_KEY = "map_date_slider"
    if SLIDER_KEY not in st.session_state:
        st.session_state[SLIDER_KEY] = slider_max
    if "playing" not in st.session_state:
        st.session_state.playing = False

    # Clamp any stale value against the current filter bounds.
    if st.session_state[SLIDER_KEY] < slider_min:
        st.session_state[SLIDER_KEY] = slider_min
    if st.session_state[SLIDER_KEY] > slider_max:
        st.session_state[SLIDER_KEY] = slider_max

    # If we're in a play tick, advance the stored date by one day.
    if st.session_state.playing and slider_min != slider_max:
        nxt = st.session_state[SLIDER_KEY] + timedelta(days=1)
        if nxt > slider_max:
            nxt = slider_min  # wrap around so it loops forever
        st.session_state[SLIDER_KEY] = nxt

    if slider_min == slider_max:
        selected_day = slider_max
    else:
        play_col, slider_col = st.columns([1, 10], gap="small")
        with play_col:
            label = "⏸" if st.session_state.playing else "▶"
            if st.button(label, key="play_btn", use_container_width=True,
                         help="Auto-advance the map date. Click again to pause."):
                st.session_state.playing = not st.session_state.playing
                st.rerun()
        with slider_col:
            selected_day = st.slider(
                "Map date",
                min_value=slider_min,
                max_value=slider_max,
                format="YYYY-MM-DD",
                step=timedelta(days=1),
                key=SLIDER_KEY,
                label_visibility="collapsed",
            )
    st.markdown("<div class='chart-heading'>Daily transits through Hormuz</div>", unsafe_allow_html=True)

    selected_day_ts = pd.Timestamp(selected_day).normalize()
    day_row = daily[daily["date"] == selected_day_ts]
    day_val = float(day_row["transits"].iloc[0]) if not day_row.empty else 0.0
    day_rolling = float(day_row["rolling"].iloc[0]) if not day_row.empty else 0.0
    total_transits = int(fdf["transits"].sum())
    days_covered = int(daily.shape[0])

    fig_ts = px.line(
        daily, x="date", y=["transits", "rolling"],
        labels={"value": "Transits / day", "date": "", "variable": ""},
    )
    fig_ts.data[0].name = "Daily"
    fig_ts.data[0].mode = "lines+markers"
    fig_ts.data[0].marker = dict(size=5, color="#60a5fa")
    # Attach the pre-computed per-day breakdown text as customdata on the
    # Daily trace, then reference it from the hover template. The Rolling
    # trace keeps its simple text hover.
    fig_ts.data[0].customdata = daily["breakdown"].to_numpy().reshape(-1, 1)
    fig_ts.data[0].hovertemplate = (
        "<b>%{x|%Y-%m-%d}</b>  ·  %{y:.0f} transits<br>"
        "<span style='color:#94a3b8'>─────────────────</span><br>"
        "%{customdata[0]}"
        "<extra></extra>"
    )
    fig_ts.data[1].name = f"{ROLLING_WINDOW}-day avg (fixed)"
    fig_ts.data[1].hovertemplate = (
        "%{x|%Y-%m-%d}<br>%{y:.1f}<extra>%{fullData.name}</extra>"
    )
    fig_ts.add_vline(
        x=selected_day_ts, line_width=1, line_dash="dot", line_color="#f87171"
    )
    # Shared x-axis range + fixed left/right margins so this chart and the
    # bar chart below line up pixel-for-pixel (and the red vlines match).
    _x_min = daily["date"].min()
    _x_max = daily["date"].max()
    fig_ts.update_layout(
        height=185,
        margin=dict(l=52, r=12, t=5, b=20),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#cbd5e1", size=11),
        xaxis=dict(
            gridcolor="#1e293b",
            showgrid=True,
            tickfont=dict(size=12, color="#e2e8f0"),
            tickformat="%b %Y",
            nticks=6,
            range=[_x_min, _x_max],
        ),
        yaxis=dict(gridcolor="#1e293b", showgrid=True, title=None, tickfont=dict(size=11)),
        legend=dict(orientation="h", y=1.2, x=1, xanchor="right", font=dict(size=10)),
        # Monospace hover so the per-day bar chart in the tooltip aligns.
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#475569",
            font=dict(family="Courier New, monospace", size=12, color="#e2e8f0"),
            align="left",
        ),
    )
    st.plotly_chart(
        fig_ts,
        use_container_width=True,
        key="ts_chart",
        config={"displayModeBar": False, "scrollZoom": False},
    )

    st.markdown("<div class='chart-heading'>Transits by cargo type</div>", unsafe_allow_html=True)

    by_type = fdf.groupby(["date", "cargo_type"], as_index=False, observed=True)["transits"].sum()
    fig_bar = px.bar(
        by_type, x="date", y="transits", color="cargo_type",
        color_discrete_map=SHIP_TYPE_COLORS,
        labels={"transits": "Transits", "date": "", "cargo_type": "Ship type"},
    )
    fig_bar.add_vline(
        x=selected_day_ts, line_width=1, line_dash="dot", line_color="#f87171"
    )
    fig_bar.update_layout(
        height=185,
        margin=dict(l=52, r=12, t=5, b=20),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(color="#cbd5e1", size=11),
        xaxis=dict(
            gridcolor="#1e293b",
            showgrid=False,
            tickfont=dict(size=12, color="#e2e8f0"),
            tickformat="%b %Y",
            nticks=6,
            range=[_x_min, _x_max],
        ),
        yaxis=dict(gridcolor="#1e293b", showgrid=True, title=None, tickfont=dict(size=11)),
        barmode="stack",
        legend=dict(
            orientation="h", y=1.4, x=1, xanchor="right",
            font=dict(size=10, color="#f8fafc"),
            title=None,
        ),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ───────────────────────── Map (left half of main row) ─────────────────
with map_slot.container():
    import numpy as np

    # Push the map down so it lines up visually with the chart-column content
    # (widgets row + first chart heading).
    st.markdown("<div style='height: 1.95rem'></div>", unsafe_allow_html=True)

    # Drop a marker anchor right before the plotly chart. CSS below uses
    # :has() to target the *next* stElementContainer — the map — and paint
    # a thin white border around it.
    st.markdown("<span class='pw-map-anchor'></span>", unsafe_allow_html=True)

    fig_map = go.Figure()

    if not ports_df.empty:
        if not daily_ports.empty:
            available = set(DAILY_PORT_CARGO_COLS)
            matched = set(selected_types) & available
            day_frame = daily_ports[daily_ports["date"] == selected_day_ts]
            if matched:
                sum_cols = [f"{DAILY_PORT_CARGO_COLS[t]}_roll7" for t in matched]
                day_snapshot = day_frame.set_index("portid")[sum_cols].sum(axis=1)
                metric_label = (
                    f"{ROLLING_WINDOW}-day avg calls (fixed) · "
                    f"{', '.join(sorted(matched))} · {selected_day_ts.date()}"
                )
            else:
                day_snapshot = day_frame.set_index("portid")["portcalls_roll7"] * 0.0
                metric_label = "No per-port data for the selected ship types"
            port_metric = ports_df["portid"].map(day_snapshot).fillna(0.0).to_numpy()
        else:
            port_metric = ports_df["vessel_count_total"].fillna(0).to_numpy()
            metric_label = "Total vessel count (all time)"

        max_metric = max(float(port_metric.max()), 1.0)
        sizes = 5.0 + 26.0 * np.sqrt(port_metric / max_metric)

        # Attach computed metric + size onto ports_df so we can slice by
        # country for the per-country traces (one trace = one legend row).
        ports_plot = ports_df.reset_index(drop=True).copy()
        ports_plot["__metric"] = port_metric
        ports_plot["__size"] = sizes
        top_idx = set(np.argsort(-port_metric)[:10].tolist())
        ports_plot["__label"] = [
            row["portname"] if i in top_idx else ""
            for i, row in ports_plot.iterrows()
        ]

        # Route spokes: one line per Gulf-side port → Hormuz, width scaled
        # by that port's metric, coloured by its country so the flow map
        # reads the same colour-key as the bubbles.
        gulf = ports_plot[ports_plot["lon"] < 56.3]
        for _, row in gulf.iterrows():
            m = float(row["__metric"])
            width = 0.6 + 5.5 * np.sqrt(m / max_metric)
            colour_line = COUNTRY_COLORS.get(row["country"], "#94a3b8")
            fig_map.add_trace(go.Scattermapbox(
                lat=[row["lat"], HORMUZ_LAT],
                lon=[row["lon"], HORMUZ_LON],
                mode="lines",
                line=dict(color=colour_line, width=width),
                opacity=0.65,
                hoverinfo="skip",
                showlegend=False,
            ))

        # Stable country ordering: put the ones we have colours for first,
        # then any stragglers — so the legend looks predictable.
        country_order = [c for c in COUNTRY_COLORS if c in ports_plot["country"].unique()]
        country_order += [
            c for c in ports_plot["country"].dropna().unique()
            if c not in COUNTRY_COLORS
        ]

        # Outline halo underneath every port bubble. Scattermapbox markers
        # have no `marker.line` support, so we fake an outline by plotting
        # every port twice: first as a slightly larger dot in a LIGHTER
        # shade of the country's colour, then the normal coloured dot on
        # top. We loop per-country so each halo matches its bubble, giving
        # a soft pastel ring around each one instead of a hard white stroke.
        for country in country_order:
            group = ports_plot[ports_plot["country"] == country]
            if group.empty:
                continue
            halo_colour = _lighten(COUNTRY_COLORS.get(country, "#94a3b8"), 0.55)
            fig_map.add_trace(go.Scattermapbox(
                lat=group["lat"],
                lon=group["lon"],
                mode="markers",
                marker=dict(
                    size=(group["__size"] + 2.6).tolist(),
                    color=halo_colour,
                    opacity=0.95,
                ),
                hoverinfo="skip",
                showlegend=False,
            ))

        for country in country_order:
            group = ports_plot[ports_plot["country"] == country]
            if group.empty:
                continue
            colour = COUNTRY_COLORS.get(country, "#94a3b8")
            customdata = list(zip(
                group["portname"].fillna("—"),
                group["country"].fillna("—"),
                group["__metric"],
                group["vessel_count_total"].fillna(0).astype(int),
                group["industry_top1"].fillna("—"),
            ))
            fig_map.add_trace(go.Scattermapbox(
                lat=group["lat"],
                lon=group["lon"],
                text=group["__label"],
                customdata=customdata,
                mode="markers+text",
                textposition="top right",
                textfont=dict(color="#f8fafc", size=11),
                marker=dict(
                    size=group["__size"].tolist(),
                    color=colour,
                    opacity=0.95,
                ),
                name=COUNTRY_LABELS.get(country, country),
                legendgroup=country,
                hovertemplate=(
                    "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                    "Port calls (7-day avg): %{customdata[2]:.1f}<br>"
                    "All-time vessel count: %{customdata[3]:,}<br>"
                    "Top industry: %{customdata[4]}"
                    "<extra></extra>"
                ),
            ))

    # Hormuz chokepoint marker — bullseye with glow.
    #
    # Scattermapbox markers can't have outline strokes, so we build the
    # bullseye out of multiple lat/lon circle traces:
    #   • Four soft filled halos at increasing radii fading to transparent
    #     (this creates the radial "glow")
    #   • Three sharp concentric ring line traces in fiery red shades
    #   • A small bright center dot
    import math

    def _latlon_ring(center_lat: float, center_lon: float, radius_deg: float, n: int = 96):
        """Return (lats, lons) for a closed circle around a lat/lon center.
        Corrects longitude spacing for latitude so the ring looks round on the map."""
        theta = np.linspace(0, 2 * np.pi, n)
        lon_scale = 1.0 / math.cos(math.radians(center_lat))
        lats = (center_lat + radius_deg * np.cos(theta)).tolist()
        lons = (center_lon + radius_deg * np.sin(theta) * lon_scale).tolist()
        return lats, lons

    # Soft glow halos — filled circles at increasing radii. Dark red-950 at
    # the outermost radius fading into hot red-500 toward the core, so the
    # glow reads as heat building up around the centre instead of a flat
    # pink wash. Drawn first so everything else sits on top.
    glow_layers = [
        (1.30, "rgba(69, 10, 10, 0.05)"),    # red-950 — deepest outer smoke
        (1.05, "rgba(127, 29, 29, 0.07)"),   # red-900
        (0.82, "rgba(185, 28, 28, 0.09)"),   # red-700
        (0.62, "rgba(220, 38, 38, 0.12)"),   # red-600
        (0.44, "rgba(239, 68, 68, 0.16)"),   # red-500 — hot inner halo
    ]
    for radius_deg, rgba in glow_layers:
        g_lats, g_lons = _latlon_ring(HORMUZ_LAT, HORMUZ_LON, radius_deg)
        fig_map.add_trace(go.Scattermapbox(
            lat=g_lats,
            lon=g_lons,
            mode="lines",
            fill="toself",
            fillcolor=rgba,
            line=dict(width=0, color=rgba),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Three sharp concentric rings — proper flame gradient from deep crimson
    # outer → vivid red middle → hot orange inner. Widths thicken toward
    # the centre so the eye reads a clean bullseye.
    concentric_rings = [
        (0.48, "#991b1b", 1.2),  # outer  — deep crimson (red-800)
        (0.30, "#ef4444", 1.8),  # middle — vivid red    (red-500)
        (0.15, "#f97316", 2.4),  # inner  — hot orange   (orange-500)
    ]
    for radius_deg, color, width in concentric_rings:
        r_lats, r_lons = _latlon_ring(HORMUZ_LAT, HORMUZ_LON, radius_deg)
        fig_map.add_trace(go.Scattermapbox(
            lat=r_lats,
            lon=r_lons,
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Stacked centre dots so the core looks white-hot. A real flame's
    # hottest visible core is yellow-white, wrapped in orange, wrapped in
    # red — three marker traces at the same lat/lon build that up. The
    # topmost (brightest) dot carries the hover tooltip.
    # Multiplicative sizing so all three dots scale *together* with day volume.
    # Additive offsets (+4, +8) would skew the visual: a small core with a
    # fixed +8 halo looks cartoonish; a big core with the same +8 looks
    # cropped. Proportional ratios keep the flame shape stable at every size.
    _core_size = max(6, min(14, 4 + day_val * 0.10))
    _mid_size = _core_size * 1.55
    _outer_size = _core_size * 2.10
    fig_map.add_trace(go.Scattermapbox(  # deep red background dot
        lat=[HORMUZ_LAT], lon=[HORMUZ_LON], mode="markers",
        marker=dict(size=_outer_size, color="#dc2626", opacity=0.38),
        hoverinfo="skip", showlegend=False,
    ))
    fig_map.add_trace(go.Scattermapbox(  # hot orange mid layer
        lat=[HORMUZ_LAT], lon=[HORMUZ_LON], mode="markers",
        marker=dict(size=_mid_size, color="#f97316", opacity=0.55),
        hoverinfo="skip", showlegend=False,
    ))
    fig_map.add_trace(go.Scattermapbox(  # white-hot core + tooltip
        lat=[HORMUZ_LAT], lon=[HORMUZ_LON], mode="markers",
        marker=dict(size=_core_size, color="#fff7ed", opacity=0.85),
        name="Hormuz chokepoint",
        showlegend=False,
        hovertemplate=(
            "<b>Strait of Hormuz</b><br>"
            f"{selected_day_ts.date()}: {day_val:.0f} transits<br>"
            f"{ROLLING_WINDOW}-day avg here: {day_rolling:.1f}<br>"
            f"Filter total: {total_transits:,} over {days_covered} days"
            "<extra></extra>"
        ),
    ))

    # Esri World Imagery — free satellite raster tiles, no token required.
    fig_map.update_layout(
        mapbox=dict(
            # Dark carto base shows through the tinted satellite raster,
            # giving us a darker map so overlaid bubbles/lines pop.
            style="carto-darkmatter",
            layers=[
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/"
                        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ],
                    "opacity": 0.38,
                },
                # Translucent blue wash over the satellite imagery, still
                # below the trace layer so bubbles/lines stay crisp.
                {
                    "below": "traces",
                    "sourcetype": "geojson",
                    "type": "fill",
                    "color": "#0b1a3a",
                    "opacity": 0.35,
                    "source": {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [20, -10], [80, -10],
                                [80, 50], [20, 50], [20, -10],
                            ]],
                        },
                    },
                },
                # Esri reference tiles: transparent raster with country
                # borders + place labels. Drawn above the blue wash so
                # the borders "pop" against the darkened map.
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/"
                        "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
                    ],
                    "opacity": 0.85,
                },
                # Crisp vector coastline (Natural Earth 50m) so the
                # land/water edge reads even at low raster opacity.
                {
                    "below": "traces",
                    "sourcetype": "geojson",
                    "type": "line",
                    "color": "#f8fafc",
                    "opacity": 0.9,
                    "line": {"width": 1.2},
                    "source": load_coastline_geojson(),
                },
                # Very faint per-country fill tinted with each country's
                # palette colour. This gives students a geographic anchor
                # ("oh, that yellow cluster is in Iran, the green cluster
                # is in Saudi") without overpowering the bubbles on top.
                # Drawn below the coastline so the coast still pops.
                *[
                    {
                        "below": "traces",
                        "sourcetype": "geojson",
                        "type": "fill",
                        "color": COUNTRY_COLORS[_country],
                        "opacity": 0.10,
                        "source": _geojson,
                    }
                    for _country, _geojson in load_country_polygons().items()
                    if _country in COUNTRY_COLORS
                ],
            ],
            center=dict(lat=25.5, lon=55.0),
            zoom=4.4,
        ),
        height=420,
        margin=dict(l=0, r=0, t=0, b=24),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5, y=-0.02, xanchor="center", yanchor="top",
            bgcolor="rgba(11,18,32,0.0)",
            bordercolor="#334155",
            borderwidth=0,
            font=dict(color="#e2e8f0", size=9),
            itemsizing="constant",
            itemwidth=30,
            tracegroupgap=2,
            title=dict(text="", font=dict(size=9, color="#cbd5e1")),
        ),
    )
    st.plotly_chart(
        fig_map,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "doubleClick": "reset",
        },
    )

# KPI cards render into the slot below the ship-types filter (map_col). The
# .pw-mini-metrics wrapper shrinks the metric component via CSS so all four
# fit under the narrow left column instead of spanning the page.
with kpi_slot.container():
    st.markdown("<div class='pw-mini-metrics'>", unsafe_allow_html=True)
    mk1, mk2, mk3, mk4 = st.columns(4, gap="small")
    mk1.metric(
        f"{selected_day_ts.date()}",
        f"{day_val:.0f}",
        delta=f"{day_val - avg_7:+.0f}",
    )
    mk2.metric("7-day avg", f"{avg_7:.1f}")
    mk3.metric("30-day avg", f"{avg_30:.1f}")
    if pd.isna(year_ago) or year_ago == 0:
        mk4.metric("Year-ago", "—")
    else:
        mk4.metric(
            "Year-ago",
            f"{year_ago:.1f}",
            delta=f"{((latest_val / year_ago) - 1) * 100:+.1f}%",
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ───────────────────────── Animation tick ─────────────────────────────────
# If the user hit the play button, pause briefly and rerun. Streamlit has
# no native animation loop — we build one by ending the script with
# time.sleep + st.rerun, so every tick is a full top-to-bottom rerun that
# advances the stored slider date by one day (see the top of rest_slot).
# Adjust the sleep to taste: 0.15s ≈ ~6 fps, fine for a teaching demo.
if st.session_state.get("playing", False):
    time.sleep(0.15)
    st.rerun()
