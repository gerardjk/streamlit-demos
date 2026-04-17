"""
Binary ephemeris loader.

This module reads the pre-computed ephemeris data that generate_ephemeris.py
created. The data lives in monthly binary chunk files (ephemeris_YYYY_MM.bin)
described by a single ephemeris_metadata.json index.

Binary format (per entry, 76 bytes):
  - 1 x int32:   timestamp (minutes since 2000-01-01 00:00 UTC)
  - 18 x float32: ecliptic longitudes and latitudes for each body

Each monthly file contains ~720-744 entries (one per hour). At ~55 KB per
file, the full 2000-2030 dataset is ~21 MB on disk but the loader only
keeps up to 12 months in memory at a time (~660 KB) via an LRU cache.

The app (solar/app.py) uses this loader to answer questions like "where is
Mars at 2024-06-21 14:00 UTC?" without needing the Skyfield library or the
31 MB JPL kernel at runtime — all the heavy astronomy was done once at
generation time.

Missing values are encoded as -999.0.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

# All timestamps in the binary files are minutes since this epoch.
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
METADATA_FILENAME = "ephemeris_metadata.json"


class EphemerisLoader:
    """Lazy-loading ephemeris reader backed by monthly binary chunks.

    On init, reads the metadata JSON to learn what fields exist, how many
    bytes per entry, and which chunk files are available. Actual binary
    data is loaded on demand when get() or range() is called.

    Internal LRU cache: keeps the 12 most recently accessed months in
    memory. When a 13th month is requested, the oldest cached month is
    evicted. This means browsing within a year costs zero disk I/O after
    the first pass, and the process never holds more than ~660 KB of
    chunk data.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / METADATA_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No ephemeris metadata at {meta_path}. "
                f"Run `python scripts/generate_ephemeris.py` first."
            )
        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.fields: list[str] = self.metadata["fields"]
        self.missing: float = self.metadata["missingDataValue"]
        self.bytes_per_entry: int = self.metadata["bytesPerEntry"]

        # Build a numpy structured dtype matching the binary layout:
        # first field is int32 (timestamp), all others are float32.
        # Little-endian ("<") to match the struct.pack("<i") / "<f" used
        # by generate_ephemeris.py's write_chunk().
        dtype_fields = [("timestamp", "<i4")]
        for name in self.fields[1:]:
            dtype_fields.append((name, "<f4"))
        self._dtype = np.dtype(dtype_fields)
        assert self._dtype.itemsize == self.bytes_per_entry

        # Index: (year, month) -> chunk metadata dict.
        self._chunk_index = {
            (c["year"], c["month"]): c for c in self.metadata["chunks"]
        }
        # LRU cache: (year, month) -> numpy structured array.
        self._chunk_cache: dict[tuple[int, int], np.ndarray] = {}

    def _load_chunk(self, year: int, month: int) -> np.ndarray | None:
        """Load a monthly chunk from disk (or return cached). Returns None
        if the month isn't in the metadata or the file is missing."""
        key = (year, month)
        if key in self._chunk_cache:
            return self._chunk_cache[key]
        meta = self._chunk_index.get(key)
        if meta is None:
            return None
        path = self.data_dir / meta["filename"]
        if not path.exists():
            return None
        # np.fromfile reads the flat binary directly into a structured array
        # using the dtype we built from the metadata. No parsing needed.
        arr = np.fromfile(path, dtype=self._dtype)
        # Simple LRU: evict the oldest entry if we're at capacity.
        if len(self._chunk_cache) >= 12:
            self._chunk_cache.pop(next(iter(self._chunk_cache)))
        self._chunk_cache[key] = arr
        return arr

    def get(self, dt: datetime) -> dict[str, Any] | None:
        """Return the ephemeris entry nearest to *dt* (UTC).

        Loads the month's chunk if needed, then binary-searches the
        timestamp column for the closest entry. Returns a dict with
        all field values, or None if the date is outside coverage.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        chunk = self._load_chunk(dt.year, dt.month)
        if chunk is None or len(chunk) == 0:
            return None

        # Binary search for the nearest timestamp.
        target_min = int((dt - EPOCH).total_seconds() // 60)
        idx = int(np.searchsorted(chunk["timestamp"], target_min))
        if idx >= len(chunk):
            idx = len(chunk) - 1
        elif idx > 0 and abs(int(chunk["timestamp"][idx - 1]) - target_min) < abs(
            int(chunk["timestamp"][idx]) - target_min
        ):
            idx -= 1

        # Unpack the structured array row into a plain dict.
        row = chunk[idx]
        out: dict[str, Any] = {}
        for name in self.fields:
            val = row[name]
            if name == "timestamp":
                minutes = int(val)
                out["timestamp"] = minutes
                out["iso"] = (EPOCH + timedelta(minutes=minutes)).isoformat()
            elif float(val) == self.missing:
                out[name] = None  # -999.0 → None for missing data
            else:
                out[name] = float(val)
        return out

    def range(self, start: datetime, end: datetime) -> np.ndarray:
        """Return a concatenated structured array over [start, end] (inclusive).

        Loads all monthly chunks that overlap the range and concatenates them,
        then filters to the exact timestamp bounds. Useful for bulk analysis
        but not used by the Streamlit app (which queries one moment at a time).
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        parts: list[np.ndarray] = []
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            chunk = self._load_chunk(y, m)
            if chunk is not None:
                parts.append(chunk)
            m += 1
            if m > 12:
                m = 1
                y += 1
        if not parts:
            return np.empty(0, dtype=self._dtype)

        all_data = np.concatenate(parts)
        start_min = int((start - EPOCH).total_seconds() // 60)
        end_min = int((end - EPOCH).total_seconds() // 60)
        mask = (all_data["timestamp"] >= start_min) & (all_data["timestamp"] <= end_min)
        return all_data[mask]

    @property
    def coverage(self) -> tuple[str, str]:
        """Return (start_iso, end_iso) for the full ephemeris date range."""
        return self.metadata["startTime"], self.metadata["endTime"]
