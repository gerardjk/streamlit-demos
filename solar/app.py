"""Streamlit app: project ephemeris bodies onto a world map."""

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ephemeris import EphemerisLoader
from projection import (
    OBLIQUITY_DEG,
    ZODIAC_SIGNS,
    rising_sign_grid,
    subpoint,
    terminator,
    zodiac_sign_name,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

# Ecliptic longitude of each body, plus the ecliptic-latitude key if we have
# one in the binary schema (only the Moon actually stores latitude — planets
# are assumed on the ecliptic, which is a small error for a world map).
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

SIGN_COLORS = [
    "#e74c3c",  # Aries
    "#8b6914",  # Taurus
    "#f1c40f",  # Gemini
    "#3498db",  # Cancer
    "#e67e22",  # Leo
    "#16a085",  # Virgo
    "#ec87c0",  # Libra
    "#34495e",  # Scorpio
    "#9b59b6",  # Sagittarius
    "#7f8c8d",  # Capricorn
    "#1abc9c",  # Aquarius
    "#2980b9",  # Pisces
]


@st.cache_resource
def get_loader() -> EphemerisLoader:
    return EphemerisLoader(DATA_DIR)


@st.cache_data(show_spinner=False)
def annual_sun_path(_loader: EphemerisLoader, year: int):
    """
    Subsolar point sampled at 12:00 UTC every day of `year`.
    Returns list of (iso, sub_lat, sub_lon, sun_ecl_lon).
    """
    out = []
    dt = datetime(year, 1, 1, 12, 0, tzinfo=timezone.utc)
    step = timedelta(days=1)
    while dt.year == year:
        entry = _loader.get(dt)
        if entry and entry.get("sun_lon") is not None:
            sl = entry["sun_lon"]
            sub_lat, sub_lon = subpoint(sl, 0.0, dt)
            out.append((dt.isoformat(), sub_lat, sub_lon, sl))
        dt += step
    return out


st.set_page_config(page_title="Ephemeris → Earth", layout="wide")
st.title("Ephemeris → Earth projection")

try:
    loader = get_loader()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

start_iso, end_iso = loader.coverage
st.caption(f"Coverage: {start_iso} — {end_iso}")

col_d, col_t, col_layer = st.columns([1, 1, 1])
d = col_d.date_input("Date (UTC)", value=date(2024, 6, 21))
t = col_t.time_input("Time (UTC)", value=time(12, 0))
layer = col_layer.selectbox(
    "Overlay",
    ["Day / night", "Rising sign", "Annual sun path", "None"],
)

dt = datetime.combine(d, t, tzinfo=timezone.utc)

entry = loader.get(dt)
if entry is None:
    st.warning("No ephemeris data for that moment.")
    st.stop()

# --- Compute subpoints for each body ---
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

# --- Build the map ---
fig = go.Figure()

if layer == "Rising sign":
    grid_lats, grid_lons, sign_idx = rising_sign_grid(dt, lat_step=3.0, lon_step=3.0)
    for i, sign in enumerate(ZODIAC_SIGNS):
        mask = sign_idx == i
        if not mask.any():
            continue
        fig.add_trace(go.Scattergeo(
            lat=grid_lats[mask],
            lon=grid_lons[mask],
            mode="markers",
            marker=dict(size=7, color=SIGN_COLORS[i], opacity=0.55, line=dict(width=0)),
            name=sign,
            hovertemplate="%{lat:.0f}°, %{lon:.0f}°<br>Rising: " + sign + "<extra></extra>",
        ))

if layer == "Annual sun path":
    path = annual_sun_path(loader, dt.year)
    if path:
        lats = np.array([p[1] for p in path])
        lons = np.array([p[2] for p in path])
        # Break the line where it wraps across the antimeridian so Plotly
        # doesn't draw a horizontal streak across the whole map.
        jumps = np.where(np.abs(np.diff(lons)) > 180.0)[0]
        plat = lats.astype(float)
        plon = lons.astype(float)
        if len(jumps) > 0:
            plat = np.insert(plat, jumps + 1, np.nan)
            plon = np.insert(plon, jumps + 1, np.nan)

        # Analemma trace.
        fig.add_trace(go.Scattergeo(
            lat=plat, lon=plon, mode="lines+markers",
            line=dict(color="#fbbf24", width=1.5),
            marker=dict(size=3, color="#fbbf24"),
            name=f"Subsolar point (daily, {dt.year})",
            hovertemplate="%{lat:.2f}°, %{lon:.2f}°<extra></extra>",
        ))

        # Tropic lines.
        for tlat, tname in [
            (OBLIQUITY_DEG,  "Tropic of Cancer"),
            (-OBLIQUITY_DEG, "Tropic of Capricorn"),
        ]:
            fig.add_trace(go.Scattergeo(
                lat=[tlat, tlat], lon=[-180, 180], mode="lines",
                line=dict(color="#f87171", width=1.2, dash="dash"),
                name=tname,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scattergeo(
                lat=[tlat], lon=[-150], mode="text",
                text=[f"{tname}  ({tlat:+.2f}°)"],
                textposition="top right" if tlat > 0 else "bottom right",
                textfont=dict(color="#fca5a5", size=11),
                showlegend=False, hoverinfo="skip",
            ))

        # Solstice highlights — the extreme latitude points.
        i_north = int(np.argmax([p[1] for p in path]))
        i_south = int(np.argmin([p[1] for p in path]))
        for idx, label in [
            (i_north, "☀ enters ♋ Cancer (June solstice)"),
            (i_south, "☀ enters ♑ Capricorn (December solstice)"),
        ]:
            _, plat_i, plon_i, _ = path[idx]
            fig.add_trace(go.Scattergeo(
                lat=[plat_i], lon=[plon_i], mode="markers+text",
                text=[label],
                textposition="middle right",
                textfont=dict(color="#fde68a", size=11),
                marker=dict(size=13, color="#f59e0b",
                            line=dict(color="#7c2d12", width=2),
                            symbol="star"),
                showlegend=False,
                hovertemplate=label + "<extra></extra>",
            ))

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
        # Mark the antisolar point (darkest night).
        anti_lat, anti_lon = -sun_lat, ((sun_lon + 180.0 + 180.0) % 360.0) - 180.0
        fig.add_trace(go.Scattergeo(
            lat=[anti_lat], lon=[anti_lon], mode="markers",
            marker=dict(size=8, color="#1e293b", line=dict(color="#94a3b8", width=1)),
            name="Antisolar",
            hovertemplate="Antisolar (midnight)<extra></extra>",
        ))

# Body subpoint markers — drawn last so they sit on top.
fig.add_trace(go.Scattergeo(
    lat=[s[1] for s in subpoints],
    lon=[s[2] for s in subpoints],
    text=[s[0] for s in subpoints],
    mode="markers+text",
    textposition="top center",
    textfont=dict(color="#f8fafc", size=11),
    marker=dict(size=11, color="#fef3c7", line=dict(color="#0f172a", width=1.5)),
    name="Subpoints",
    hovertemplate="%{text}<br>%{lat:.2f}°, %{lon:.2f}°<extra></extra>",
    showlegend=False,
))

fig.update_geos(
    projection_type="equirectangular",
    showcoastlines=True, coastlinecolor="#94a3b8", coastlinewidth=0.7,
    showland=True, landcolor="#1e293b",
    showocean=True, oceancolor="#0b1220",
    showlakes=False,
    lataxis_range=[-80, 80],
    lonaxis_range=[-180, 180],
    bgcolor="#0b1220",
)
fig.update_layout(
    height=560,
    margin=dict(l=0, r=0, t=10, b=10),
    paper_bgcolor="#0b1220",
    plot_bgcolor="#0b1220",
    legend=dict(
        orientation="h", yanchor="bottom", y=-0.12,
        bgcolor="rgba(11,18,32,0.6)", font=dict(color="#e2e8f0"),
    ),
)

st.plotly_chart(fig, use_container_width=True)

if layer == "Annual sun path":
    st.info(
        "The subsolar point — where the Sun sits directly overhead — drifts "
        "between **+23.44°N and −23.44°S** over the year. It reaches its "
        "northernmost point on the **June solstice**, when the Sun enters "
        "**♋ Cancer** (0° ecliptic longitude 90°). That latitude was named the "
        "**Tropic of Cancer**. Six months later the Sun enters **♑ Capricorn** "
        "(ecliptic longitude 270°) and the subsolar point touches its southern "
        "extreme — the **Tropic of Capricorn**. The names come from the zodiac "
        "sign the Sun is *in* when it turns back ('tropos' = 'turning'). "
        "The angle ±23.44° is just Earth's axial tilt."
    )

# --- Details table ---
rows = []
for name, sub_lat, sub_lon, ecl_lon in subpoints:
    rows.append({
        "Body": name,
        "Ecliptic λ": f"{ecl_lon:7.3f}°",
        "Zodiac": zodiac_sign_name(ecl_lon),
        "Sub-lat": f"{sub_lat:+6.2f}°",
        "Sub-lon": f"{sub_lon:+7.2f}°",
    })
st.table(rows)

with st.expander("Raw ephemeris entry"):
    st.json({k: v for k, v in entry.items() if v is not None})
