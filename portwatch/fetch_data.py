#!/usr/bin/env python
"""
Download IMF PortWatch data for the Persian Gulf / Strait of Hormuz region.

IMF PortWatch (https://portwatch.imf.org) tracks global maritime shipping
using AIS (Automatic Identification System) satellite transponder data.
The IMF publishes the processed data through two channels:
  - ArcGIS REST FeatureServer APIs (for port metadata and daily port calls)
  - Downloadable CSVs via ArcGIS Hub (for chokepoint transit counts)

This script fetches three datasets and writes them to portwatch/data/:
  1. gulf_ports.csv          — static port metadata (location, vessel counts)
  2. daily_gulf_ports.parquet — daily port-call time series per port
  3. chokepoint_transits.csv  — daily Strait of Hormuz transits by cargo type

Run with:  python portwatch/fetch_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# URLs — all hosted by the IMF via Esri's ArcGIS Online platform
# ---------------------------------------------------------------------------

# A full CSV export of chokepoint transit data for every global chokepoint
# (Suez, Panama, Hormuz, Malacca, etc.). We download the whole file and
# filter to Hormuz rows locally.
CHOKEPOINT_CSV_URL = (
    "https://hub.arcgis.com/api/download/v1/items/"
    "42132aa4e2fc4d41bdaf9a445f688931/csv?redirect=true&layers=0"
)

# ArcGIS FeatureServer endpoint for the port metadata layer. Supports
# spatial queries (bounding-box intersection) so we can ask "give me
# all ports inside the Persian Gulf rectangle."
PORTS_QUERY_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
    "PortWatch_ports_database/FeatureServer/0/query"
)

# ArcGIS FeatureServer for daily port-call time series. One row per
# port per day, with columns for total calls and per-cargo-type calls.
DAILY_PORTS_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
PORTS_OUT = DATA_DIR / "gulf_ports.csv"
DAILY_PORTS_OUT = DATA_DIR / "daily_gulf_ports.parquet"
CHOKEPOINT_OUT = DATA_DIR / "chokepoint_transits.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Geographic bounding box that covers the Persian Gulf, Gulf of Oman, and
# the Strait of Hormuz. Used as a spatial filter when querying port metadata
# so we only get ports in the region of interest.
GULF_BBOX = {"lon_min": 46.0, "lat_min": 22.0, "lon_max": 62.0, "lat_max": 32.0}

# The chokepoint CSV stores transit counts in wide-format columns like
# "n_tanker", "n_dry_bulk", etc. This mapping renames them to the short
# cargo-type labels the dashboard expects after we pivot to long format.
CARGO_COLS = {
    "n_tanker": "tanker",
    "n_dry_bulk": "dry_bulk",
    "n_container": "container",
    "n_general_cargo": "general_cargo",
    "n_roro": "ro_ro",
}

# Which columns to request from the port metadata API. Requesting only
# the fields we need keeps the response small and avoids hitting ArcGIS
# transfer limits.
PORT_FIELDS = [
    "portid", "portname", "country", "ISO3", "lat", "lon",
    "vessel_count_total", "vessel_count_tanker", "vessel_count_container",
    "vessel_count_dry_bulk", "industry_top1", "industry_top2",
    "share_country_maritime_export",
]

# Which columns to request from the daily port-calls API.
DAILY_FIELDS = "date,portid,portcalls,portcalls_tanker,portcalls_container,portcalls_dry_bulk"

# ArcGIS caps query responses at this many features per request. We page
# through results using resultOffset to get the full dataset.
ARCGIS_PAGE_SIZE = 1000


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_gulf_ports() -> pd.DataFrame:
    """Query the PortWatch ports database for ports inside the Gulf bounding box.

    Uses an ArcGIS spatial query: we pass a bounding-box envelope in WGS84
    (EPSG:4326) coordinates, and the server returns all port features whose
    geometry intersects that rectangle. Only ports with at least one vessel
    are kept (filters out decommissioned or inactive entries).
    """
    print("Querying Gulf-region ports ...")
    bbox = f"{GULF_BBOX['lon_min']},{GULF_BBOX['lat_min']},{GULF_BBOX['lon_max']},{GULF_BBOX['lat_max']}"
    resp = requests.get(PORTS_QUERY_URL, timeout=60, params={
        "where": "1=1",                              # no attribute filter — get all rows
        "geometry": bbox,                             # spatial filter: this bounding box
        "geometryType": "esriGeometryEnvelope",       # interpret 'geometry' as a rectangle
        "inSR": "4326",                               # coordinate system: WGS84 lat/lon
        "spatialRel": "esriSpatialRelIntersects",     # keep features that touch the box
        "outFields": ",".join(PORT_FIELDS),           # only return these columns
        "returnGeometry": "false",                    # skip the geometry blob — we have lat/lon
        "f": "json",                                  # response format
    })
    resp.raise_for_status()

    # ArcGIS wraps each row in {"attributes": {...}, "geometry": {...}}.
    # We only asked for attributes (returnGeometry=false).
    ports = pd.DataFrame([f["attributes"] for f in resp.json()["features"]])
    ports = ports[ports["vessel_count_total"].fillna(0) > 0].reset_index(drop=True)
    print(f"  {len(ports)} ports with non-zero vessel counts")
    return ports


def fetch_daily_port_calls(portids: list[str]) -> pd.DataFrame:
    """Fetch daily port-call time series for each port, one port at a time.

    ArcGIS FeatureServer pagination requires a stable orderByFields, and
    long IN clauses (WHERE portid IN ('A','B','C',...)) fail inconsistently
    on this particular service. Querying one port at a time with pagination
    is slower but reliable.

    For each port we page through all available history in chunks of
    ARCGIS_PAGE_SIZE rows, advancing resultOffset until we get back fewer
    rows than the page size (meaning we've reached the end).
    """
    all_rows: list[dict] = []

    for i, portid in enumerate(portids, start=1):
        offset, port_rows = 0, 0
        while True:
            resp = requests.post(DAILY_PORTS_URL, timeout=120, data={
                "where": f"portid='{portid}'",
                "outFields": DAILY_FIELDS,
                "orderByFields": "date ASC",          # stable sort for pagination
                "returnGeometry": "false",
                "f": "json",
                "resultRecordCount": ARCGIS_PAGE_SIZE, # max rows per response
                "resultOffset": offset,                # skip this many rows (pagination)
            })
            resp.raise_for_status()
            body = resp.json()
            if "error" in body:
                raise RuntimeError(f"ArcGIS error on {portid}: {body['error']}")

            feats = body.get("features", [])
            if not feats:
                break

            all_rows.extend(f["attributes"] for f in feats)
            port_rows += len(feats)

            # Keep paging if the server says there's more, OR if we got a
            # full page (some ArcGIS tiers don't set exceededTransferLimit).
            if not body.get("exceededTransferLimit") and len(feats) < ARCGIS_PAGE_SIZE:
                break
            offset += len(feats)

        print(f"  [{i:>2}/{len(portids)}] {portid}: {port_rows:>5} rows")

    if not all_rows:
        raise RuntimeError("Daily_Ports_Data returned zero rows")

    df = pd.DataFrame(all_rows)
    # ArcGIS returns dates as epoch milliseconds — convert to datetime.
    df["date"] = pd.to_datetime(df["date"], unit="ms").dt.normalize()
    for col in ("portcalls", "portcalls_tanker", "portcalls_container", "portcalls_dry_bulk"):
        df[col] = df[col].fillna(0).astype("int32")
    df["portid"] = df["portid"].astype("category")
    return df.sort_values(["portid", "date"]).reset_index(drop=True)


def fetch_chokepoint_transits() -> pd.DataFrame:
    """Download the global chokepoint CSV and extract Strait of Hormuz rows.

    The CSV from ArcGIS Hub contains daily transit counts for every major
    chokepoint worldwide. Each row is one chokepoint on one day, with
    wide-format columns like n_tanker, n_dry_bulk, n_container, etc.

    We filter to Hormuz, then pivot from wide to long format so the
    dashboard gets one row per (date, cargo_type) with a single 'transits'
    column — easier to group, filter, and chart.
    """
    print("Downloading chokepoint transits CSV ...")
    df = pd.read_csv(CHOKEPOINT_CSV_URL)
    print(f"  {len(df):,} rows across {df.portname.nunique()} chokepoints")

    # Keep only Strait of Hormuz rows.
    hormuz = df[df.portname.str.contains("Hormuz", case=False, na=False)].copy()
    hormuz["date"] = pd.to_datetime(hormuz["date"], utc=True).dt.tz_convert(None).dt.normalize()

    # Pivot wide → long: one row per (date, cargo_type) instead of one
    # row with multiple n_tanker / n_dry_bulk / ... columns.
    long = hormuz.melt(
        id_vars=["date", "portname"],
        value_vars=list(CARGO_COLS),       # the wide columns to unpivot
        var_name="cargo_col",              # temporary column holding the old col name
        value_name="transits",             # the values become this column
    )
    long["cargo_type"] = long["cargo_col"].map(CARGO_COLS)  # rename n_tanker → tanker, etc.
    long = long.rename(columns={"portname": "chokepoint_name"})
    long = long[["date", "chokepoint_name", "cargo_type", "transits"]]
    long["transits"] = long["transits"].fillna(0).astype(int)
    return long.sort_values(["date", "cargo_type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Port metadata — static snapshot of every active port in the Gulf.
    ports = fetch_gulf_ports()
    ports.to_csv(PORTS_OUT, index=False)
    print(f"Wrote {len(ports)} ports to {PORTS_OUT}")

    # 2. Daily port calls — historical time series for each port above.
    print(f"Querying daily port calls for {len(ports)} Gulf ports ...")
    daily = fetch_daily_port_calls(ports["portid"].tolist())
    daily.to_parquet(DAILY_PORTS_OUT, index=False)
    print(f"Wrote {len(daily):,} rows to {DAILY_PORTS_OUT}")

    # 3. Chokepoint transits — daily ship counts through the Strait of Hormuz.
    transits = fetch_chokepoint_transits()
    transits.to_csv(CHOKEPOINT_OUT, index=False)
    print(f"Wrote {len(transits):,} rows to {CHOKEPOINT_OUT}")


if __name__ == "__main__":
    main()
