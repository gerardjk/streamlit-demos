"""
Strait of Hormuz — Daily Transit Monitor.

Second example for the lecture: a real-world operational dashboard built
on (synthetic, by default) IMF PortWatch chokepoint data. Demonstrates
KPI-first layout, filter-upstream-of-views, and cross-chart context
filtering via Plotly click events.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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


# ─── DEMO [Hour 3]: @st.cache_data memoises VALUES ──────────────────────────
# First call actually reads the file. Later calls with the same arguments get
# the cached DataFrame — instantly. `ttl=3600` expires the entry after an hour
# so a daily refresh eventually gets picked up. Use cache_data for DataFrames,
# lists, dicts — anything picklable. For the *loader object* pattern, see
# solar/app.py and @st.cache_resource.
@st.cache_data(ttl=3600, show_spinner=False)
def _load_daily_ports_cached(mtime: float) -> pd.DataFrame:
    # mtime is part of the cache key — when the parquet file is rewritten,
    # Streamlit invalidates automatically.
    df = pd.read_parquet(DAILY_PORTS_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["portid", "date"])
    df["portcalls_roll7"] = (
        df.groupby("portid", observed=True)["portcalls"]
          .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    return df


def load_daily_ports() -> pd.DataFrame:
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


# ─── DEMO [Hour 3]: the "CSV-once, Parquet-forever" trick ───────────────────
# First run: read the CSV, narrow the dtypes, save as Parquet. Every future
# run: read the Parquet directly — columnar, smaller, ~10x faster to load.
# This is on top of @st.cache_data — the decorator caches within a session,
# the Parquet file caches across sessions. Two layers, different lifetimes.
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


# ─── DEMO [Hour 1]: set_page_config MUST be the first Streamlit call ────────
# Call it after any other st.* and you get an error. Sets the browser tab
# title and the wide layout (default is centered).
st.set_page_config(
    page_title="Hormuz Transit Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Compact CSS: tighten padding without clipping the header.
st.markdown(
    """
    <style>
      /* Kill Streamlit's default top padding and the hidden toolbar */
      header[data-testid="stHeader"] { height: 0; min-height: 0; visibility: hidden; }
      .block-container {
          padding-top: 0.6rem !important;
          padding-bottom: 0.6rem !important;
          padding-left: 1.5rem !important;
          padding-right: 1.5rem !important;
          max-width: 100% !important;
      }
      /* Compact heading */
      .pw-title {
          font-size: 1.3rem;
          font-weight: 700;
          letter-spacing: -0.01em;
          margin: 0 0 0.1rem 0;
          color: #f1f5f9;
      }
      .pw-title .pw-sub { color: #94a3b8; font-weight: 400; font-size: 0.85rem; margin-left: 0.6rem; }
      /* Tighter metric cards */
      [data-testid="stMetricValue"] { font-size: 1.25rem !important; line-height: 1.1 !important; }
      [data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #94a3b8 !important; }
      [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }
      [data-testid="stCaptionContainer"] { margin-top: 0 !important; }
      /* Zero out vertical gaps between stacked blocks */
      div[data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
      [data-testid="stSlider"] { padding: 0 !important; }
      [data-testid="stSlider"] label { display: none; }
      /* Compact form controls */
      [data-baseweb="input"] input, [data-baseweb="select"] { font-size: 0.85rem !important; }
      label { font-size: 0.75rem !important; margin-bottom: 0.1rem !important; }
      section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='pw-title'>Strait of Hormuz · Daily Transit Monitor"
    "<span class='pw-sub'>IMF PortWatch · 45 Gulf ports</span></div>",
    unsafe_allow_html=True,
)

# Cached calls — on rerun these are cheap dictionary lookups, not file reads.
df = load_data()
ports_df = load_ports()
daily_ports = load_daily_ports()

# ──────────────────────────────── Inline filter row ────────────────────────────────
min_d, max_d = df.date.min().date(), df.date.max().date()
default_start = max(min_d, max_d - timedelta(days=365))

ship_types = sorted(df.cargo_type.dropna().unique().tolist())

fc1, fc2, fc3 = st.columns([2, 3, 1])
with fc1:
    date_range = st.date_input(
        "Date range",
        value=(default_start, max_d),
        min_value=min_d,
        max_value=max_d,
    )
with fc2:
    selected_types = st.multiselect(
        "Ship types",
        options=ship_types,
        default=ship_types,
    )
with fc3:
    rolling_window = st.slider("Smoothing (days)", 1, 30, 7)

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
daily["rolling"] = daily["transits"].rolling(rolling_window, min_periods=1).mean()

latest_val = float(daily["transits"].iloc[-1])
avg_7 = float(daily["transits"].tail(7).mean())
avg_30 = float(daily["transits"].tail(30).mean())

latest_date = daily["date"].iloc[-1]
ya_target = latest_date - pd.Timedelta(days=365)
ya_window = daily[(daily.date >= ya_target - pd.Timedelta(days=3)) &
                  (daily.date <= ya_target + pd.Timedelta(days=3))]
year_ago = float(ya_window["transits"].mean()) if not ya_window.empty else float("nan")

# Day slider — placed above the time series so time series + map both react.
slider_min = daily["date"].iloc[0].to_pydatetime()
slider_max = daily["date"].iloc[-1].to_pydatetime()
if slider_min == slider_max:
    selected_day = slider_max
else:
    selected_day = st.slider(
        "Date shown on the map",
        min_value=slider_min,
        max_value=slider_max,
        value=slider_max,
        format="YYYY-MM-DD",
        step=timedelta(days=1),
        label_visibility="collapsed",
    )
selected_day_ts = pd.Timestamp(selected_day).normalize()
day_row = daily[daily["date"] == selected_day_ts]
day_val = float(day_row["transits"].iloc[0]) if not day_row.empty else 0.0
day_rolling = float(day_row["rolling"].iloc[0]) if not day_row.empty else 0.0
total_transits = int(fdf["transits"].sum())
days_covered = int(daily.shape[0])

# ──────────────────────────────── KPI strip (compact, full-width) ────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    f"{selected_day_ts.date()}",
    f"{day_val:.0f}",
    delta=f"{day_val - avg_7:+.0f} vs 7-day avg",
)
k2.metric("7-day avg", f"{avg_7:.1f}")
k3.metric("30-day avg", f"{avg_30:.1f}")
if pd.isna(year_ago) or year_ago == 0:
    k4.metric("Year-ago (±3d)", "—")
else:
    k4.metric(
        "Year-ago (±3d)",
        f"{year_ago:.1f}",
        delta=f"{((latest_val / year_ago) - 1) * 100:+.1f}%",
    )

# ──────────────────────────────── Time-series (above map) ────────────────────────
fig_ts = px.line(
    daily, x="date", y=["transits", "rolling"],
    labels={"value": "Transits / day", "date": "", "variable": ""},
)
fig_ts.data[0].name = "Daily"
fig_ts.data[1].name = f"{rolling_window}-day avg"
fig_ts.data[1].update(line=dict(width=3, color="#fbbf24"))
fig_ts.data[0].update(line=dict(color="#38bdf8", width=1))
fig_ts.update_traces(
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}<extra>%{fullData.name}</extra>"
)
fig_ts.add_vline(
    x=selected_day_ts, line_width=1, line_dash="dot", line_color="#f87171"
)
fig_ts.update_layout(
    height=170,
    margin=dict(l=0, r=0, t=5, b=0),
    paper_bgcolor="#0b1220",
    plot_bgcolor="#0b1220",
    font=dict(color="#cbd5e1", size=10),
    xaxis=dict(gridcolor="#1e293b", showgrid=True),
    yaxis=dict(gridcolor="#1e293b", showgrid=True, title=None),
    legend=dict(orientation="h", y=1.18, x=1, xanchor="right", font=dict(size=9)),
)
ts_event = st.plotly_chart(
    fig_ts,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="ts_chart",
    config={"displayModeBar": False, "scrollZoom": False},
)

fig_map = go.Figure()

# Connection lines from Persian Gulf ports to the chokepoint. Any port west
# of ~56.3°E is inside the Gulf and its traffic must transit Hormuz. Ports
# east of that (Fujairah, Khor Fakkan, Sohar, Chabahar) sit on the Gulf of
# Oman side and bypass the strait, so we leave them unconnected.
if not ports_df.empty:
    gulf_ports = ports_df[ports_df["lon"] < 56.3]
    conn_lats: list[float | None] = []
    conn_lons: list[float | None] = []
    for _, row in gulf_ports.iterrows():
        conn_lats += [row["lat"], HORMUZ_LAT, None]
        conn_lons += [row["lon"], HORMUZ_LON, None]
    fig_map.add_trace(go.Scattergeo(
        lat=conn_lats,
        lon=conn_lons,
        mode="lines",
        line=dict(color="rgba(148,163,184,0.35)", width=1),
        name=f"Must transit Hormuz (n={len(gulf_ports)})",
        hoverinfo="skip",
    ))

# Real Gulf-region ports from the IMF PortWatch ports database.
if not ports_df.empty:
    import numpy as np

    # For each port, look up the 7-day rolling avg of port calls at the
    # selected day. If we don't have daily data (or the day is outside the
    # daily-data range), fall back to the static vessel_count_total.
    if not daily_ports.empty:
        day_snapshot = (
            daily_ports[daily_ports["date"] == selected_day_ts]
            .set_index("portid")["portcalls_roll7"]
        )
        port_metric = ports_df["portid"].map(day_snapshot).fillna(0.0).to_numpy()
        metric_label = f"7-day avg port calls on {selected_day_ts.date()}"
    else:
        port_metric = ports_df["vessel_count_total"].fillna(0).to_numpy()
        metric_label = "Total vessel count (all time)"

    max_metric = max(float(port_metric.max()), 1.0)
    sizes = 5.0 + 26.0 * np.sqrt(port_metric / max_metric)

    # Label the top-N by *current* metric so labels track the slider too.
    top_n = 10
    top_idx = set(np.argsort(-port_metric)[:top_n].tolist())
    labels = [
        row["portname"] if i in top_idx else ""
        for i, row in ports_df.reset_index(drop=True).iterrows()
    ]

    customdata = list(zip(
        ports_df["portname"].fillna("—"),
        ports_df["country"].fillna("—"),
        port_metric,
        ports_df["vessel_count_total"].fillna(0).astype(int),
        ports_df["industry_top1"].fillna("—"),
    ))

    fig_map.add_trace(go.Scattergeo(
        lat=ports_df["lat"],
        lon=ports_df["lon"],
        text=labels,
        customdata=customdata,
        mode="markers+text",
        textposition="top center",
        textfont=dict(color="#cbd5e1", size=10),
        marker=dict(
            size=sizes.tolist(),
            color="#38bdf8",
            opacity=0.85,
            line=dict(color="#0c4a6e", width=1),
            symbol="circle",
        ),
        name=metric_label,
        hovertemplate=(
            "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
            "Port calls (7-day avg): %{customdata[2]:.1f}<br>"
            "All-time vessel count: %{customdata[3]:,}<br>"
            "Top industry: %{customdata[4]}"
            "<extra></extra>"
        ),
    ))

# Hormuz chokepoint marker — sized by day volume, no text label.
fig_map.add_trace(go.Scattergeo(
    lat=[HORMUZ_LAT],
    lon=[HORMUZ_LON],
    mode="markers",
    marker=dict(
        size=max(22, min(70, day_val * 0.7)),
        color="rgba(239,68,68,0.55)",
        line=dict(color="#fecaca", width=2),
        symbol="circle",
    ),
    name="Chokepoint",
    hovertemplate=(
        "<b>Strait of Hormuz</b><br>"
        f"{selected_day_ts.date()}: {day_val:.0f} transits<br>"
        f"{rolling_window}-day avg here: {day_rolling:.1f}<br>"
        f"Filter total: {total_transits:,} over {days_covered} days"
        "<extra></extra>"
    ),
))

fig_map.update_geos(
    projection_type="natural earth",
    center=dict(lat=26.0, lon=54.0),
    lataxis_range=[22, 32],
    lonaxis_range=[46, 62],
    showcountries=True, countrycolor="#64748b", countrywidth=0.8,
    showcoastlines=True, coastlinecolor="#94a3b8", coastlinewidth=0.8,
    showland=True, landcolor="#1e293b",
    showocean=True, oceancolor="#0b1220",
    showlakes=False,
    resolution=50,
    bgcolor="#0b1220",
)
fig_map.update_layout(
    height=340,
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor="#0b1220",
    plot_bgcolor="#0b1220",
    showlegend=False,
)
st.plotly_chart(
    fig_map,
    use_container_width=True,
    config={
        "scrollZoom": False,          # page scroll no longer hijacks the map
        "displayModeBar": True,       # show toolbar so users can still zoom
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "doubleClick": "reset",       # double-click snaps back to the region
    },
)
# Drill-down: shown only when the user clicks a point on the time series.
picked_date = None
try:
    points = ts_event["selection"]["points"]  # type: ignore[index]
    if points:
        picked_date = pd.Timestamp(points[0]["x"]).normalize()
except (KeyError, TypeError, IndexError):
    picked_date = None

if picked_date is not None:
    day_rows = fdf[fdf.date.dt.normalize() == picked_date]
    if not day_rows.empty:
        breakdown = (
            day_rows.groupby("cargo_type", as_index=False, observed=True)["transits"].sum()
                    .sort_values("transits", ascending=False)
        )
        fig_day = px.bar(
            breakdown, x="cargo_type", y="transits", color="cargo_type",
            color_discrete_map=SHIP_TYPE_COLORS,
        )
        fig_day.update_layout(
            title=dict(
                text=f"{picked_date.date()} — {int(day_rows.transits.sum())} transits by ship type",
                x=0, font=dict(size=12, color="#e2e8f0"),
            ),
            height=180,
            showlegend=False,
            margin=dict(l=0, r=0, t=28, b=0),
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            font=dict(color="#cbd5e1", size=10),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_day, use_container_width=True, config={"displayModeBar": False})
