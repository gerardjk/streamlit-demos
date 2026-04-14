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
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "chokepoint_transits.csv"
HORMUZ_COORDS = (26.566, 56.25)  # rough center of the strait

SHIP_TYPE_COLORS = {
    "tanker":        "#ef4444",
    "dry_bulk":      "#a16207",
    "container":     "#3b82f6",
    "gas":           "#eab308",
    "general_cargo": "#14b8a6",
    "ro_ro":         "#a855f7",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(
            f"No data at `{path}`.\n\n"
            "Run `python portwatch/scripts/generate_sample_data.py` to create "
            "a synthetic dataset, or drop the real IMF PortWatch chokepoint CSV "
            "there with columns: date, chokepoint_name, cargo_type, transits."
        )
        st.stop()
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["chokepoint_name"].str.contains("Hormuz", case=False, na=False)]
    return df


st.set_page_config(page_title="Hormuz Transit Monitor", layout="wide")
st.title("Strait of Hormuz — Daily Transit Monitor")
st.caption("Data: IMF PortWatch (chokepoint transits). Synthetic sample shown when the real CSV is absent.")

df = load_data(DATA_PATH)

# ──────────────────────────────── Sidebar filters ────────────────────────────────
st.sidebar.header("Filters")
min_d, max_d = df.date.min().date(), df.date.max().date()
default_start = max(min_d, max_d - timedelta(days=180))
date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, max_d),
    min_value=min_d,
    max_value=max_d,
)
# st.date_input may return a single date if the user picks only one end.
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = default_start, max_d

ship_types = sorted(df.cargo_type.dropna().unique().tolist())
selected_types = st.sidebar.multiselect(
    "Ship types",
    options=ship_types,
    default=ship_types,
)

rolling_window = st.sidebar.slider("Rolling average (days)", 1, 30, 7)

# Filter once, upstream of every view. This is the pattern.
mask = (
    (df.date.dt.date >= start)
    & (df.date.dt.date <= end)
    & (df.cargo_type.isin(selected_types))
)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No rows match the current filters.")
    st.stop()

# ──────────────────────────────── KPI row ────────────────────────────────
daily = (
    fdf.groupby("date", as_index=False)["transits"].sum()
       .sort_values("date")
       .reset_index(drop=True)
)

latest_val = float(daily["transits"].iloc[-1])
avg_7 = float(daily["transits"].tail(7).mean())
avg_30 = float(daily["transits"].tail(30).mean())

# Year-ago comparison (±3 day window around the date one year before latest)
latest_date = daily["date"].iloc[-1]
ya_target = latest_date - pd.Timedelta(days=365)
ya_window = daily[(daily.date >= ya_target - pd.Timedelta(days=3)) &
                  (daily.date <= ya_target + pd.Timedelta(days=3))]
year_ago = float(ya_window["transits"].mean()) if not ya_window.empty else float("nan")

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Latest day",
    f"{latest_val:.0f}",
    delta=f"{latest_val - avg_7:+.0f} vs 7-day avg",
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

# ──────────────────────────────── Time-series ────────────────────────────────
daily["rolling"] = daily["transits"].rolling(rolling_window, min_periods=1).mean()

fig_ts = px.line(
    daily,
    x="date",
    y=["transits", "rolling"],
    labels={"value": "Transits / day", "date": "", "variable": ""},
)
fig_ts.data[0].name = "Daily"
fig_ts.data[1].name = f"{rolling_window}-day avg"
fig_ts.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}<extra>%{fullData.name}</extra>")
fig_ts.update_layout(
    height=320,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", y=1.1),
)

# Capture click events → feed the drill-down below.
ts_event = st.plotly_chart(
    fig_ts,
    use_container_width=True,
    on_select="rerun",
    selection_mode="points",
    key="ts_chart",
)

# ──────────────────────────────── Stacked breakdown ────────────────────────────────
by_type = fdf.groupby(["date", "cargo_type"], as_index=False)["transits"].sum()
fig_bar = px.bar(
    by_type,
    x="date",
    y="transits",
    color="cargo_type",
    color_discrete_map=SHIP_TYPE_COLORS,
    labels={"transits": "Transits", "date": "", "cargo_type": "Ship type"},
)
fig_bar.update_layout(
    height=300,
    margin=dict(l=0, r=0, t=10, b=0),
    barmode="stack",
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ──────────────────────────────── Context filtering drill-down ─────────────────
st.subheader("Drill-down")

picked_date = None
try:
    points = ts_event["selection"]["points"]  # type: ignore[index]
    if points:
        picked_date = pd.Timestamp(points[0]["x"]).normalize()
except (KeyError, TypeError, IndexError):
    picked_date = None

if picked_date is None:
    st.caption("Click a point on the daily time-series above to drill into that day.")
else:
    day_rows = fdf[fdf.date.dt.normalize() == picked_date]
    if day_rows.empty:
        st.caption(f"No data for {picked_date.date()}.")
    else:
        st.caption(f"Breakdown for **{picked_date.date()}** ({int(day_rows.transits.sum())} total transits)")
        breakdown = (
            day_rows.groupby("cargo_type", as_index=False)["transits"].sum()
                    .sort_values("transits", ascending=False)
        )
        fig_day = px.bar(
            breakdown,
            x="cargo_type",
            y="transits",
            color="cargo_type",
            color_discrete_map=SHIP_TYPE_COLORS,
        )
        fig_day.update_layout(
            height=260,
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_day, use_container_width=True)

# ──────────────────────────────── Geo context ────────────────────────────────
with st.expander("Location"):
    st.map(
        pd.DataFrame({"lat": [HORMUZ_COORDS[0]], "lon": [HORMUZ_COORDS[1]]}),
        zoom=5,
    )

with st.expander("Raw rows (latest 50)"):
    st.dataframe(fdf.sort_values("date").tail(50), use_container_width=True)
