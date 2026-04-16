"""
Hormuz Transit Monitor — barebones teaching version.

Strip-down of app.py that shows the core Streamlit building blocks a beginner
needs before touching the full dashboard:

  1. set_page_config    — one-time page setup
  2. @st.cache_data     — load the CSV once, not on every rerun
  3. st.sidebar widgets — user input
  4. st.metric / columns — KPIs
  5. st.plotly_chart    — a chart driven by the filtered DataFrame
  6. st.map             — the simplest possible map

Run with:  streamlit run portwatch/app_basic.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "chokepoint_transits.parquet"
PORTS_PATH = Path(__file__).parent / "data" / "gulf_ports.parquet"


# ── 1. Page setup ────────────────────────────────────────────────────────────
# Must be the FIRST Streamlit call in the script. Sets the browser tab title
# and uses the full window width instead of the narrow default.
st.set_page_config(page_title="Hormuz Monitor (basic)", layout="wide")


# ── 2. Cached data loading ───────────────────────────────────────────────────
# Streamlit reruns this whole file top-to-bottom on every widget interaction.
# @st.cache_data memoises the return value so we don't reread the file each
# time. Key rule: the function must be deterministic given its arguments.
@st.cache_data
def load_transits() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def load_ports() -> pd.DataFrame:
    return pd.read_parquet(PORTS_PATH)


df = load_transits()
ports = load_ports()


# ── 3. Sidebar filters ───────────────────────────────────────────────────────
# Widgets return plain Python values. Assign them to variables and use them
# like any other variable — Streamlit handles the reactive rerun for you.
st.sidebar.header("Filters") 

min_date, max_date = df["date"].min().date(), df["date"].max().date()
start, end = st.sidebar.date_input(
    "Date range",
    value=(max_date - pd.Timedelta(days=30), max_date),
    min_value=min_date,
    max_value=max_date,
)

cargo_types = sorted(df["cargo_type"].unique())
selected = st.sidebar.multiselect("Cargo types", cargo_types, default=cargo_types)

# Filter the DataFrame using the widget values. This is the "filter upstream
# of views" pattern — one filtered df feeds every chart below.
mask = (
    (df["date"] >= pd.Timestamp(start))
    & (df["date"] <= pd.Timestamp(end))
    & (df["cargo_type"].isin(selected))
)
view = df.loc[mask]


# ── 4. KPIs ──────────────────────────────────────────────────────────────────
st.title("Strait of Hormuz — Transit Monitor")

col1, col2, col3 = st.columns(3)
col1.metric("Total transits", f"{view['transits'].sum():,}")
col2.metric("Avg / day", f"{view.groupby('date')['transits'].sum().mean():,.0f}")
col3.metric("Days in view", view["date"].nunique())


# ── 5. Chart ─────────────────────────────────────────────────────────────────
daily = view.groupby(["date", "cargo_type"], observed=True)["transits"].sum().reset_index()
fig = px.area(daily, x="date", y="transits", color="cargo_type",
              title="Daily transits by cargo type")
st.plotly_chart(fig, use_container_width=True)


# ── 6. Map ───────────────────────────────────────────────────────────────────
# st.map wants columns literally named "lat" and "lon" — this file already has them.
st.subheader("Gulf ports")
st.map(ports[["lat", "lon"]])
