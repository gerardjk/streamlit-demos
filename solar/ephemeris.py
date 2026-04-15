"""
Binary ephemeris loader.

Reads monthly chunk files (ephemeris_YYYY_MM.bin) described by
ephemeris_metadata.json. Each entry is 172 bytes: one int32 timestamp
(minutes since 2000-01-01 UTC) followed by 42 float32 fields.
Samples are hourly, so ~744 entries per month, ~128 KB per file.
Missing values are encoded as -999.0.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
METADATA_FILENAME = "ephemeris_metadata.json"


class EphemerisLoader:
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

        # Structured dtype: first field int32, rest float32, little-endian.
        dtype_fields = [("timestamp", "<i4")]
        for name in self.fields[1:]:
            dtype_fields.append((name, "<f4"))
        self._dtype = np.dtype(dtype_fields)
        assert self._dtype.itemsize == self.bytes_per_entry

        self._chunk_index = {
            (c["year"], c["month"]): c for c in self.metadata["chunks"]
        }
        self._chunk_cache: dict[tuple[int, int], np.ndarray] = {}

    def _load_chunk(self, year: int, month: int) -> np.ndarray | None:
        key = (year, month)
        if key in self._chunk_cache:
            return self._chunk_cache[key]
        meta = self._chunk_index.get(key)
        if meta is None:
            return None
        path = self.data_dir / meta["filename"]
        if not path.exists():
            return None
        arr = np.fromfile(path, dtype=self._dtype)
        if len(self._chunk_cache) >= 12:
            self._chunk_cache.pop(next(iter(self._chunk_cache)))
        self._chunk_cache[key] = arr
        return arr

    def get(self, dt: datetime) -> dict[str, Any] | None:
        """Return the ephemeris entry for the given UTC datetime (nearest minute)."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        chunk = self._load_chunk(dt.year, dt.month)
        if chunk is None or len(chunk) == 0:
            return None

        target_min = int((dt - EPOCH).total_seconds() // 60)
        idx = int(np.searchsorted(chunk["timestamp"], target_min))
        if idx >= len(chunk):
            idx = len(chunk) - 1
        elif idx > 0 and abs(int(chunk["timestamp"][idx - 1]) - target_min) < abs(
            int(chunk["timestamp"][idx]) - target_min
        ):
            idx -= 1

        row = chunk[idx]
        out: dict[str, Any] = {}
        for name in self.fields:
            val = row[name]
            if name == "timestamp":
                minutes = int(val)
                out["timestamp"] = minutes
                out["iso"] = (EPOCH + timedelta(minutes=minutes)).isoformat()
            elif float(val) == self.missing:
                out[name] = None
            elif name.endswith("_retrograde"):
                out[name] = bool(val == 1.0)
            else:
                out[name] = float(val)
        return out

    def range(self, start: datetime, end: datetime) -> np.ndarray:
        """Return a concatenated structured array over [start, end] (inclusive)."""
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
        return self.metadata["startTime"], self.metadata["endTime"]
