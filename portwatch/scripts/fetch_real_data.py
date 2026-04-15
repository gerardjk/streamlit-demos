#!/usr/bin/env python
"""
Download the real IMF PortWatch daily chokepoint transit dataset, keep only
the Strait of Hormuz rows, and write them to portwatch/data/chokepoint_transits.csv
in the long-format schema the dashboard expects:

    date, chokepoint_name, cargo_type, transits

Source: https://portwatch.imf.org/datasets/42132aa4e2fc4d41bdaf9a445f688931_0/about
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://hub.arcgis.com/api/download/v1/items/"
    "42132aa4e2fc4d41bdaf9a445f688931/csv?redirect=true&layers=0"
)
PORTS_QUERY_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
    "PortWatch_ports_database/FeatureServer/0/query"
)
DAILY_PORTS_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT = DATA_DIR / "chokepoint_transits.csv"
PORTS_OUT = DATA_DIR / "gulf_ports.csv"
DAILY_PORTS_OUT = DATA_DIR / "daily_gulf_ports.parquet"

# Bounding box covering the Persian Gulf, Gulf of Oman, and the Strait of Hormuz.
GULF_BBOX = dict(lon_min=46.0, lat_min=22.0, lon_max=62.0, lat_max=32.0)

CARGO_COLS = {
    "n_tanker":        "tanker",
    "n_dry_bulk":      "dry_bulk",
    "n_container":     "container",
    "n_general_cargo": "general_cargo",
    "n_roro":          "ro_ro",
}


def fetch_gulf_ports() -> pd.DataFrame:
    print(f"Querying Gulf-region ports from {PORTS_QUERY_URL} ...")
    params = {
        "where": "1=1",
        "geometry": f"{GULF_BBOX['lon_min']},{GULF_BBOX['lat_min']},"
                    f"{GULF_BBOX['lon_max']},{GULF_BBOX['lat_max']}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": ",".join([
            "portid", "portname", "country", "ISO3", "lat", "lon",
            "vessel_count_total", "vessel_count_tanker", "vessel_count_container",
            "vessel_count_dry_bulk", "industry_top1", "industry_top2",
            "share_country_maritime_export",
        ]),
        "returnGeometry": "false",
        "f": "json",
    }
    import requests
    r = requests.get(PORTS_QUERY_URL, params=params, timeout=60)
    r.raise_for_status()
    feats = r.json()["features"]
    ports = pd.DataFrame([f["attributes"] for f in feats])
    ports = ports[ports["vessel_count_total"].fillna(0) > 0].reset_index(drop=True)
    print(f"  got {len(ports)} ports with non-zero vessel counts")
    return ports


def fetch_daily_port_calls(portids: list[str]) -> pd.DataFrame:
    """
    Query Daily_Ports_Data one port at a time. ArcGIS pagination needs a
    stable orderByFields, and long IN clauses fail inconsistently — so we
    just ask per-portid and page by date.
    """
    import requests

    page_size = 1000  # ArcGIS Daily_Ports_Data maxRecordCount
    all_rows: list[dict] = []
    for i, portid in enumerate(portids, start=1):
        offset = 0
        port_rows = 0
        while True:
            r = requests.post(
                DAILY_PORTS_URL,
                data={
                    "where": f"portid='{portid}'",
                    "outFields": "date,portid,portcalls,portcalls_tanker,portcalls_container,portcalls_dry_bulk",
                    "orderByFields": "date ASC",
                    "returnGeometry": "false",
                    "f": "json",
                    "resultRecordCount": page_size,
                    "resultOffset": offset,
                },
                timeout=120,
            )
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                raise RuntimeError(f"ArcGIS error on {portid}: {j['error']}")
            feats = j.get("features", [])
            if not feats:
                break
            all_rows.extend(f["attributes"] for f in feats)
            port_rows += len(feats)
            # Keep paging while the server says there's more, OR while it's
            # returning full pages (some tiers don't set exceededTransferLimit).
            if not j.get("exceededTransferLimit") and len(feats) < page_size:
                break
            offset += len(feats)
        print(f"  [{i:>2}/{len(portids)}] {portid}: {port_rows:>5} rows")

    if not all_rows:
        raise RuntimeError("Daily_Ports_Data returned zero rows")

    df = pd.DataFrame(all_rows)
    # ArcGIS date fields come through as epoch millis.
    df["date"] = pd.to_datetime(df["date"], unit="ms").dt.normalize()
    for col in ("portcalls", "portcalls_tanker", "portcalls_container", "portcalls_dry_bulk"):
        df[col] = df[col].fillna(0).astype("int32")
    df["portid"] = df["portid"].astype("category")
    return df.sort_values(["portid", "date"]).reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ports = fetch_gulf_ports()
    ports.to_csv(PORTS_OUT, index=False)
    print(f"Wrote {len(ports)} ports to {PORTS_OUT}")

    print(f"Querying daily port calls for {len(ports)} Gulf ports ...")
    daily = fetch_daily_port_calls(ports["portid"].tolist())
    daily.to_parquet(DAILY_PORTS_OUT, index=False)
    print(
        f"Wrote {len(daily):,} rows "
        f"({daily.date.min().date()} → {daily.date.max().date()}) to {DAILY_PORTS_OUT}"
    )

    print(f"Downloading {SOURCE_URL} ...")
    df = pd.read_csv(SOURCE_URL)
    print(f"  got {len(df):,} rows across {df.portname.nunique()} chokepoints")

    hormuz = df[df.portname.str.contains("Hormuz", case=False, na=False)].copy()
    hormuz["date"] = pd.to_datetime(hormuz["date"], utc=True).dt.tz_convert(None).dt.normalize()

    long = hormuz.melt(
        id_vars=["date", "portname"],
        value_vars=list(CARGO_COLS),
        var_name="cargo_col",
        value_name="transits",
    )
    long["cargo_type"] = long["cargo_col"].map(CARGO_COLS)
    long = long.rename(columns={"portname": "chokepoint_name"})
    long = long[["date", "chokepoint_name", "cargo_type", "transits"]]
    long["transits"] = long["transits"].fillna(0).astype(int)
    long = long.sort_values(["date", "cargo_type"]).reset_index(drop=True)

    long.to_csv(OUT, index=False)
    print(
        f"Wrote {len(long):,} rows ({long.date.min().date()} → {long.date.max().date()}) "
        f"to {OUT}"
    )


if __name__ == "__main__":
    main()
