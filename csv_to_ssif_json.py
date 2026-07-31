# -*- coding: utf-8 -*-
"""Convert a tabular CSV archive into the event-JSON schema used by SSIF_V3.

Supported layouts
-----------------
1. wide: one row per event/station, with one intensity column per second.
2. sequence: one row per event/station, with a JSON/Python-list sequence column.
3. long: one row per event/station/second/intensity observation.

The converter is intentionally explicit and audit-oriented. It writes one JSON
file per event plus conversion_summary.json and event_index.csv. Automatic
column detection is provided, but users should inspect the reported mapping
before a formal conversion.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EVENT_CANDIDATES = (
    "event_id", "eq_id", "earthquake_id", "event", "number", "isnumber", "event_no"
)
STATION_CANDIDATES = (
    "station_id", "station", "station_code", "stationid", "stid", "sta", "site"
)
SEQUENCE_CANDIDATES = (
    "intensity_series", "intensity_sequence", "intensities", "sequence", "series",
    "values", "intensity"
)
SECOND_CANDIDATES = (
    "second", "sec", "elapsed_second", "time_index", "second_index", "t"
)
VALUE_CANDIDATES = (
    "intensity_class", "cwa_intensity", "intensity_value", "value", "shindo", "intensity"
)
ORIGIN_CANDIDATES = ("origin_time", "origin", "ot", "event_time")
MAG_CANDIDATES = ("magnitude", "mag", "ml", "mw")
DEPTH_CANDIDATES = ("depth", "depth_km", "dep")
LON_CANDIDATES = ("longitude", "lon", "event_lon", "eq_lon")
LAT_CANDIDATES = ("latitude", "lat", "event_lat", "eq_lat")
DIST_CANDIDATES = ("epicentral_distance", "distance", "distance_km", "epi_dist", "dist")

WIDE_PATTERNS = (
    re.compile(r"^(?:t|sec|second|s)[_-]?(\d{1,3})$", re.I),
    re.compile(r"^(?:intensity|shindo)[_-]?(\d{1,3})$", re.I),
    re.compile(r"^(\d{1,3})$"),
)


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _column_lookup(columns: Sequence[str]) -> Dict[str, str]:
    return {_norm(c): c for c in columns}


def _first_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = _column_lookup(columns)
    for candidate in candidates:
        if _norm(candidate) in lookup:
            return lookup[_norm(candidate)]
    return None


def detect_wide_columns(columns: Sequence[str], horizon: int) -> List[Tuple[int, str]]:
    found: Dict[int, str] = {}
    for column in columns:
        name = str(column).strip()
        for pattern in WIDE_PATTERNS:
            match = pattern.match(name)
            if not match:
                continue
            second = int(match.group(1))
            if 0 <= second <= horizon:
                found.setdefault(second, column)
            break
    if not found:
        return []
    seconds = sorted(found)
    if seconds and seconds[0] == 0 and seconds[-1] <= horizon - 1:
        return [(second + 1, found[second]) for second in seconds if second + 1 <= horizon]
    return [(second, found[second]) for second in seconds if 1 <= second <= horizon]


def inspect_schema(csv_path: Path, rows: int, horizon: int, encoding: Optional[str]) -> Dict[str, Any]:
    frame = pd.read_csv(csv_path, nrows=rows, low_memory=False, encoding=encoding)
    columns = list(frame.columns)
    event_col = _first_column(columns, EVENT_CANDIDATES)
    station_col = _first_column(columns, STATION_CANDIDATES)
    sequence_col = _first_column(columns, SEQUENCE_CANDIDATES)
    second_col = _first_column(columns, SECOND_CANDIDATES)
    value_col = _first_column(columns, VALUE_CANDIDATES)
    wide_cols = detect_wide_columns(columns, horizon)

    if wide_cols and len(wide_cols) >= min(10, horizon):
        layout = "wide"
    elif sequence_col and not (sequence_col == value_col and second_col):
        layout = "sequence"
    elif second_col and value_col:
        layout = "long"
    else:
        layout = "unknown"

    return {
        "csv_path": str(csv_path),
        "file_size_bytes": csv_path.stat().st_size,
        "columns": columns,
        "preview": frame.head(rows).where(pd.notna(frame), None).to_dict(orient="records"),
        "detected": {
            "layout": layout,
            "event_col": event_col,
            "station_col": station_col,
            "sequence_col": sequence_col,
            "second_col": second_col,
            "value_col": value_col,
            "origin_col": _first_column(columns, ORIGIN_CANDIDATES),
            "magnitude_col": _first_column(columns, MAG_CANDIDATES),
            "depth_col": _first_column(columns, DEPTH_CANDIDATES),
            "longitude_col": _first_column(columns, LON_CANDIDATES),
            "latitude_col": _first_column(columns, LAT_CANDIDATES),
            "distance_col": _first_column(columns, DIST_CANDIDATES),
            "wide_time_columns": [{"second": s, "column": c} for s, c in wide_cols],
        },
    }


def _missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else False
    except Exception:
        return value is None


def sanitize_intensity(value: Any) -> float:
    if _missing(value) or isinstance(value, bool):
        return -99.0
    try:
        x = float(value)
    except (TypeError, ValueError):
        return -99.0
    if not math.isfinite(x) or x < 0 or x > 9:
        return -99.0
    return float(int(round(x)))


def parse_sequence(value: Any, horizon: int) -> List[float]:
    if _missing(value):
        raw: Sequence[Any] = []
    elif isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            raw = []
        else:
            parsed = None
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception as exc:
                    raise ValueError(f"cannot parse sequence: {text[:80]!r}") from exc
            if not isinstance(parsed, (list, tuple)):
                raise ValueError("sequence column must contain a list")
            raw = list(parsed)
    out = [sanitize_intensity(x) for x in raw[:horizon]]
    if len(out) < horizon:
        out.extend([-99.0] * (horizon - len(out)))
    return out


def _safe_text(value: Any, fallback: str = "") -> str:
    if _missing(value):
        return fallback
    return str(value).strip() or fallback


def _safe_float(value: Any) -> Optional[float]:
    if _missing(value):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@dataclass
class EventBuffer:
    event_id: str
    origin_time: str
    eq_info: Dict[str, Any] = field(default_factory=dict)
    intensity: Dict[str, List[float]] = field(default_factory=dict)
    distance: Dict[str, float] = field(default_factory=dict)


def _resolve_column(explicit: Optional[str], detected: Optional[str], columns: Sequence[str], label: str) -> Optional[str]:
    value = explicit or detected
    if value is not None and value not in columns:
        raise ValueError(f"{label} column not found: {value}")
    return value


def convert_csv(args: argparse.Namespace) -> Dict[str, Any]:
    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = inspect_schema(csv_path, args.preview_rows, args.horizon, args.encoding)
    columns = schema["columns"]
    detected = schema["detected"]
    layout = args.layout if args.layout != "auto" else detected["layout"]
    if layout == "unknown":
        raise ValueError(
            "Could not detect CSV layout. Run the inspect command and supply --layout and column mappings."
        )

    event_col = _resolve_column(args.event_col, detected["event_col"], columns, "event")
    station_col = _resolve_column(args.station_col, detected["station_col"], columns, "station")
    if not event_col or not station_col:
        raise ValueError("event and station columns are required")

    sequence_col = _resolve_column(args.sequence_col, detected["sequence_col"], columns, "sequence")
    second_col = _resolve_column(args.second_col, detected["second_col"], columns, "second")
    value_col = _resolve_column(args.value_col, detected["value_col"], columns, "value")
    origin_col = _resolve_column(args.origin_col, detected["origin_col"], columns, "origin")
    magnitude_col = _resolve_column(args.magnitude_col, detected["magnitude_col"], columns, "magnitude")
    depth_col = _resolve_column(args.depth_col, detected["depth_col"], columns, "depth")
    longitude_col = _resolve_column(args.longitude_col, detected["longitude_col"], columns, "longitude")
    latitude_col = _resolve_column(args.latitude_col, detected["latitude_col"], columns, "latitude")
    distance_col = _resolve_column(args.distance_col, detected["distance_col"], columns, "distance")

    wide_cols = [(int(x["second"]), x["column"]) for x in detected["wide_time_columns"]]
    if layout == "wide" and len(wide_cols) < min(10, args.horizon):
        raise ValueError("wide layout requires detectable time columns; rename columns or use another layout")
    if layout == "sequence" and not sequence_col:
        raise ValueError("sequence layout requires --sequence-col")
    if layout == "long" and (not second_col or not value_col):
        raise ValueError("long layout requires --second-col and --value-col")

    events: Dict[str, EventBuffer] = {}
    counters = defaultdict(int)

    usecols = {event_col, station_col}
    for col in (
        sequence_col, second_col, value_col, origin_col, magnitude_col, depth_col,
        longitude_col, latitude_col, distance_col,
    ):
        if col:
            usecols.add(col)
    if layout == "wide":
        usecols.update(col for _, col in wide_cols)

    def event_buffer(row: Mapping[str, Any]) -> EventBuffer:
        event_id = _safe_text(row[event_col])
        if not event_id:
            raise ValueError("empty event ID")
        origin = _safe_text(row.get(origin_col) if origin_col else None, fallback=event_id)
        if event_id not in events:
            eq_info: Dict[str, Any] = {"number": event_id, "origin_time": origin}
            mappings = (
                ("magnitude", magnitude_col), ("depth", depth_col),
                ("longitude", longitude_col), ("latitude", latitude_col),
            )
            for target, source in mappings:
                if source:
                    value = _safe_float(row.get(source))
                    if value is not None:
                        eq_info[target] = value
            events[event_id] = EventBuffer(event_id=event_id, origin_time=origin, eq_info=eq_info)
        return events[event_id]

    for chunk in pd.read_csv(
        csv_path,
        chunksize=args.chunk_size,
        usecols=list(usecols),
        low_memory=False,
        encoding=args.encoding,
    ):
        counters["rows_read"] += len(chunk)
        for row in chunk.to_dict(orient="records"):
            try:
                event = event_buffer(row)
                station_id = _safe_text(row[station_col])
                if not station_id:
                    raise ValueError("empty station ID")

                if layout == "wide":
                    series = [-99.0] * args.horizon
                    for second, column in wide_cols:
                        if 1 <= second <= args.horizon:
                            series[second - 1] = sanitize_intensity(row.get(column))
                    if station_id in event.intensity:
                        counters["duplicate_station_rows"] += 1
                        if args.duplicate_policy == "error":
                            raise ValueError(f"duplicate event/station row: {event.event_id}/{station_id}")
                    event.intensity[station_id] = series

                elif layout == "sequence":
                    series = parse_sequence(row.get(sequence_col), args.horizon)
                    if station_id in event.intensity:
                        counters["duplicate_station_rows"] += 1
                        if args.duplicate_policy == "error":
                            raise ValueError(f"duplicate event/station row: {event.event_id}/{station_id}")
                    event.intensity[station_id] = series

                else:
                    second = int(float(row.get(second_col)))
                    if args.second_origin == 0:
                        second += 1
                    if not 1 <= second <= args.horizon:
                        counters["long_rows_outside_horizon"] += 1
                        continue
                    series = event.intensity.setdefault(station_id, [-99.0] * args.horizon)
                    if series[second - 1] != -99.0:
                        counters["duplicate_second_values"] += 1
                        if args.duplicate_policy == "error":
                            raise ValueError(
                                f"duplicate second: {event.event_id}/{station_id}/second={second}"
                            )
                    series[second - 1] = sanitize_intensity(row.get(value_col))

                if distance_col:
                    distance = _safe_float(row.get(distance_col))
                    if distance is not None:
                        event.distance[station_id] = distance
                counters["rows_converted"] += 1
            except Exception as exc:
                counters["row_errors"] += 1
                if args.row_error_policy == "error":
                    raise
                if counters["row_errors"] <= 20:
                    print(f"[row warning] {type(exc).__name__}: {exc}")

    index_rows: List[Dict[str, Any]] = []
    for i, event_id in enumerate(sorted(events), start=1):
        event = events[event_id]
        digest = hashlib.sha1(event_id.encode("utf-8")).hexdigest()[:10]
        filename = f"event_{i:06d}_{digest}.json"
        payload = {
            "eq_info": event.eq_info,
            "intensity": event.intensity,
            "epicenter_distance": event.distance,
        }
        with (output_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        valid_count = sum(
            1 for series in event.intensity.values() for value in series if value != -99.0
        )
        index_rows.append({
            "event_id": event_id,
            "origin_time": event.origin_time,
            "filename": filename,
            "n_stations": len(event.intensity),
            "n_valid_intensity_values": valid_count,
        })

    with (output_dir / "event_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["event_id", "origin_time", "filename", "n_stations", "n_valid_intensity_values"],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    mapping = {
        "layout": layout,
        "event_col": event_col,
        "station_col": station_col,
        "sequence_col": sequence_col,
        "second_col": second_col,
        "value_col": value_col,
        "origin_col": origin_col,
        "magnitude_col": magnitude_col,
        "depth_col": depth_col,
        "longitude_col": longitude_col,
        "latitude_col": latitude_col,
        "distance_col": distance_col,
        "wide_time_columns": [{"second": s, "column": c} for s, c in wide_cols],
    }
    summary = {
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "horizon": args.horizon,
        "mapping": mapping,
        "n_events": len(events),
        "n_station_records": sum(len(e.intensity) for e in events.values()),
        "counters": dict(counters),
    }
    with (output_dir / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CSV data to SSIF event JSON files")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect CSV columns and auto-detected layout")
    inspect.add_argument("--csv", required=True)
    inspect.add_argument("--rows", type=int, default=5)
    inspect.add_argument("--horizon", type=int, default=120)
    inspect.add_argument("--encoding", default=None)

    convert = sub.add_parser("convert", help="Convert CSV to event JSON archive")
    convert.add_argument("--csv", required=True)
    convert.add_argument("--output-dir", required=True)
    convert.add_argument("--layout", choices=["auto", "wide", "sequence", "long"], default="auto")
    convert.add_argument("--event-col", default=None)
    convert.add_argument("--station-col", default=None)
    convert.add_argument("--sequence-col", default=None)
    convert.add_argument("--second-col", default=None)
    convert.add_argument("--value-col", default=None)
    convert.add_argument("--origin-col", default=None)
    convert.add_argument("--magnitude-col", default=None)
    convert.add_argument("--depth-col", default=None)
    convert.add_argument("--longitude-col", default=None)
    convert.add_argument("--latitude-col", default=None)
    convert.add_argument("--distance-col", default=None)
    convert.add_argument("--horizon", type=int, default=120)
    convert.add_argument("--chunk-size", type=int, default=5000)
    convert.add_argument("--preview-rows", type=int, default=5)
    convert.add_argument("--encoding", default=None)
    convert.add_argument("--second-origin", type=int, choices=[0, 1], default=1)
    convert.add_argument("--duplicate-policy", choices=["error", "last"], default="error")
    convert.add_argument("--row-error-policy", choices=["error", "skip"], default="error")
    convert.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if args.command == "inspect":
        print(json.dumps(inspect_schema(csv_path, args.rows, args.horizon, args.encoding), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(convert_csv(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
