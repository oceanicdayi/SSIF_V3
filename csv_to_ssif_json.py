# -*- coding: utf-8 -*-
"""Convert CSV archives into the event-JSON schema used by SSIF_V3.

Supported layouts
-----------------
* event_json: one CSV row is one complete event. Nested columns such as
  ``eq_info``, ``intensity``, ``epicenter_distance``, ``stids`` and ``times``
  contain JSON or Python-literal strings. This is the layout used by
  ``combined_data.csv``.
* wide: one row per event/station, with one intensity column per second.
* sequence: one row per event/station, with a list-valued sequence column.
* long: one row per event/station/second/value.

The converter writes one JSON file per event, ``event_index.csv`` and
``conversion_summary.json``. ``--overwrite`` cleans stale generated files before
conversion, so an interrupted earlier run cannot contaminate a new archive.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import shutil
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

EVENT_JSON_REQUIRED = ("eq_info", "intensity")
EVENT_JSON_OPTIONAL = ("times", "stids", "epicenter_distance", "variables", "source_file")

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
        found = lookup.get(_norm(candidate))
        if found is not None:
            return found
    return None


def _has_columns(columns: Sequence[str], required: Sequence[str]) -> bool:
    lookup = _column_lookup(columns)
    return all(_norm(name) in lookup for name in required)


def _missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else False
    except Exception:
        return value is None


def parse_nested(value: Any, *, expected: Optional[type] = None, field_name: str = "value") -> Any:
    """Parse an already-materialized object, JSON text, or Python-literal text."""
    if _missing(value):
        parsed: Any = None
    elif isinstance(value, (dict, list, tuple, int, float, bool)):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            parsed = None
        else:
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception as exc:
                    raise ValueError(f"cannot parse {field_name}: {text[:120]!r}") from exc
    if expected is not None and not isinstance(parsed, expected):
        raise ValueError(
            f"{field_name} must decode to {expected.__name__}, got {type(parsed).__name__}"
        )
    return parsed


def _preview_value(value: Any, limit: int = 180) -> Any:
    if _missing(value):
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "..."


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
    if seconds[0] == 0 and seconds[-1] <= horizon - 1:
        return [(s + 1, found[s]) for s in seconds if s + 1 <= horizon]
    return [(s, found[s]) for s in seconds if 1 <= s <= horizon]


def inspect_schema(csv_path: Path, rows: int, horizon: int, encoding: Optional[str]) -> Dict[str, Any]:
    frame = pd.read_csv(csv_path, nrows=rows, low_memory=False, encoding=encoding)
    columns = list(frame.columns)
    event_col = _first_column(columns, EVENT_CANDIDATES)
    station_col = _first_column(columns, STATION_CANDIDATES)
    sequence_col = _first_column(columns, SEQUENCE_CANDIDATES)
    second_col = _first_column(columns, SECOND_CANDIDATES)
    value_col = _first_column(columns, VALUE_CANDIDATES)
    wide_cols = detect_wide_columns(columns, horizon)

    if _has_columns(columns, EVENT_JSON_REQUIRED):
        layout = "event_json"
    elif wide_cols and len(wide_cols) >= min(10, horizon):
        layout = "wide"
    elif event_col and station_col and sequence_col and not (sequence_col == value_col and second_col):
        layout = "sequence"
    elif event_col and station_col and second_col and value_col:
        layout = "long"
    else:
        layout = "unknown"

    preview = []
    for row in frame.head(rows).to_dict(orient="records"):
        preview.append({str(k): _preview_value(v) for k, v in row.items()})

    detected = {
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
        "event_json_columns": {
            name: _first_column(columns, (name,))
            for name in (*EVENT_JSON_REQUIRED, *EVENT_JSON_OPTIONAL)
        },
    }
    return {
        "csv_path": str(csv_path),
        "file_size_bytes": csv_path.stat().st_size,
        "columns": columns,
        "preview": preview,
        "detected": detected,
    }


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


def normalize_sequence(value: Any, horizon: int) -> List[float]:
    raw = parse_nested(value, field_name="intensity sequence")
    if not isinstance(raw, (list, tuple, np.ndarray)):
        raise ValueError(f"intensity sequence must be a list, got {type(raw).__name__}")
    out = [sanitize_intensity(x) for x in list(raw)[:horizon]]
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


def _event_identity(eq_info: Mapping[str, Any], source_file: Any, row_number: int) -> Tuple[str, str]:
    origin = _safe_text(eq_info.get("origin_time"))
    number = _safe_text(eq_info.get("number") or eq_info.get("isnumber"))
    source = _safe_text(source_file)
    if number and origin:
        event_id = f"{number}|{origin}"
    elif origin:
        event_id = origin
    elif source:
        event_id = source
    else:
        event_id = f"csv_row_{row_number:08d}"
    return event_id, origin


def _filename(event_id: str, row_number: int) -> str:
    digest = hashlib.sha1(event_id.encode("utf-8")).hexdigest()[:12]
    return f"event_{row_number:06d}_{digest}.json"


def _clean_output(output_dir: Path) -> None:
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_event_payload(payload: Mapping[str, Any], horizon: int) -> Dict[str, int]:
    eq_info = payload.get("eq_info")
    intensity = payload.get("intensity")
    distance = payload.get("epicenter_distance", {})
    if not isinstance(eq_info, dict):
        raise ValueError("output eq_info must be a dictionary")
    if not isinstance(intensity, dict) or not intensity:
        raise ValueError("output intensity must be a non-empty station dictionary")
    if distance is not None and not isinstance(distance, dict):
        raise ValueError("output epicenter_distance must be a dictionary")
    valid_values = 0
    missing_values = 0
    for station_id, series in intensity.items():
        if not str(station_id).strip():
            raise ValueError("empty station ID")
        if not isinstance(series, list) or len(series) != horizon:
            raise ValueError(f"station {station_id}: expected {horizon} values")
        for value in series:
            if value == -99.0:
                missing_values += 1
            elif isinstance(value, (int, float)) and math.isfinite(float(value)) and 0 <= float(value) <= 9:
                valid_values += 1
            else:
                raise ValueError(f"station {station_id}: invalid intensity value {value!r}")
    return {
        "n_stations": len(intensity),
        "n_valid_intensity_values": valid_values,
        "n_missing_intensity_values": missing_values,
    }


def _convert_event_json(
    args: argparse.Namespace,
    csv_path: Path,
    output_dir: Path,
    columns: Sequence[str],
) -> Dict[str, Any]:
    lookup = _column_lookup(columns)
    col = {name: lookup.get(_norm(name)) for name in (*EVENT_JSON_REQUIRED, *EVENT_JSON_OPTIONAL)}
    for required in EVENT_JSON_REQUIRED:
        if not col[required]:
            raise ValueError(f"event_json layout requires column: {required}")

    usecols = [c for c in col.values() if c]
    seen_event_ids: set[str] = set()
    counters: Dict[str, int] = defaultdict(int)
    index_rows: List[Dict[str, Any]] = []
    output_files: List[Path] = []
    row_number = 0

    try:
        for chunk in pd.read_csv(
            csv_path,
            chunksize=args.chunk_size,
            usecols=usecols,
            low_memory=False,
            encoding=args.encoding,
        ):
            counters["rows_read"] += len(chunk)
            for row in chunk.to_dict(orient="records"):
                row_number += 1
                try:
                    eq_info = parse_nested(row.get(col["eq_info"]), expected=dict, field_name="eq_info")
                    raw_intensity = parse_nested(
                        row.get(col["intensity"]), expected=dict, field_name="intensity"
                    )
                    intensity: Dict[str, List[float]] = {}
                    for station_id, sequence in raw_intensity.items():
                        station = str(station_id).strip()
                        if not station:
                            raise ValueError("intensity contains an empty station ID")
                        intensity[station] = normalize_sequence(sequence, args.horizon)
                    if not intensity:
                        raise ValueError("intensity dictionary is empty")

                    raw_distance = (
                        parse_nested(
                            row.get(col["epicenter_distance"]),
                            expected=dict,
                            field_name="epicenter_distance",
                        )
                        if col["epicenter_distance"] and not _missing(row.get(col["epicenter_distance"]))
                        else {}
                    )
                    distance: Dict[str, float] = {}
                    for station_id, value in raw_distance.items():
                        x = _safe_float(value)
                        if x is not None:
                            distance[str(station_id)] = x

                    source_file = row.get(col["source_file"]) if col["source_file"] else ""
                    event_id, origin_time = _event_identity(eq_info, source_file, row_number)
                    if event_id in seen_event_ids:
                        counters["duplicate_events"] += 1
                        if args.duplicate_policy == "error":
                            raise ValueError(f"duplicate event ID: {event_id}")
                    seen_event_ids.add(event_id)

                    payload: Dict[str, Any] = {
                        "eq_info": eq_info,
                        "intensity": intensity,
                        "epicenter_distance": distance,
                    }
                    if col["times"] and not _missing(row.get(col["times"])):
                        times = parse_nested(row.get(col["times"]), expected=list, field_name="times")
                        payload["times"] = list(times[: args.horizon])
                    if col["stids"] and not _missing(row.get(col["stids"])):
                        payload["stids"] = parse_nested(
                            row.get(col["stids"]), expected=dict, field_name="stids"
                        )
                    if col["variables"] and not _missing(row.get(col["variables"])):
                        payload["variables"] = parse_nested(row.get(col["variables"]), field_name="variables")
                    if source_file and not _missing(source_file):
                        payload["source_file"] = str(source_file)

                    validation = _validate_event_payload(payload, args.horizon)
                    filename = _filename(event_id, row_number)
                    fp = output_dir / filename
                    with fp.open("w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    output_files.append(fp)
                    index_rows.append({
                        "event_id": event_id,
                        "origin_time": origin_time,
                        "filename": filename,
                        **validation,
                    })
                    counters["rows_converted"] += 1
                except Exception as exc:
                    counters["row_errors"] += 1
                    if args.row_error_policy == "error":
                        raise ValueError(f"CSV row {row_number}: {exc}") from exc
                    if counters["row_errors"] <= 20:
                        print(f"[row warning] row={row_number} {type(exc).__name__}: {exc}")
    except Exception:
        for fp in output_files:
            fp.unlink(missing_ok=True)
        raise

    return {
        "mapping": {"layout": "event_json", "columns": col},
        "index_rows": index_rows,
        "counters": dict(counters),
    }


@dataclass
class EventBuffer:
    event_id: str
    origin_time: str
    eq_info: Dict[str, Any] = field(default_factory=dict)
    intensity: Dict[str, List[float]] = field(default_factory=dict)
    distance: Dict[str, float] = field(default_factory=dict)


def _resolve_column(
    explicit: Optional[str], detected: Optional[str], columns: Sequence[str], label: str
) -> Optional[str]:
    value = explicit or detected
    if value is not None and value not in columns:
        raise ValueError(f"{label} column not found: {value}")
    return value


def _convert_station_table(
    args: argparse.Namespace,
    csv_path: Path,
    output_dir: Path,
    columns: Sequence[str],
    detected: Mapping[str, Any],
    layout: str,
) -> Dict[str, Any]:
    event_col = _resolve_column(args.event_col, detected.get("event_col"), columns, "event")
    station_col = _resolve_column(args.station_col, detected.get("station_col"), columns, "station")
    if not event_col or not station_col:
        raise ValueError(f"{layout} layout requires event and station columns")

    sequence_col = _resolve_column(args.sequence_col, detected.get("sequence_col"), columns, "sequence")
    second_col = _resolve_column(args.second_col, detected.get("second_col"), columns, "second")
    value_col = _resolve_column(args.value_col, detected.get("value_col"), columns, "value")
    origin_col = _resolve_column(args.origin_col, detected.get("origin_col"), columns, "origin")
    magnitude_col = _resolve_column(args.magnitude_col, detected.get("magnitude_col"), columns, "magnitude")
    depth_col = _resolve_column(args.depth_col, detected.get("depth_col"), columns, "depth")
    longitude_col = _resolve_column(args.longitude_col, detected.get("longitude_col"), columns, "longitude")
    latitude_col = _resolve_column(args.latitude_col, detected.get("latitude_col"), columns, "latitude")
    distance_col = _resolve_column(args.distance_col, detected.get("distance_col"), columns, "distance")
    wide_cols = [(int(x["second"]), x["column"]) for x in detected.get("wide_time_columns", [])]

    if layout == "wide" and len(wide_cols) < min(10, args.horizon):
        raise ValueError("wide layout requires detectable time columns")
    if layout == "sequence" and not sequence_col:
        raise ValueError("sequence layout requires --sequence-col")
    if layout == "long" and (not second_col or not value_col):
        raise ValueError("long layout requires --second-col and --value-col")

    events: Dict[str, EventBuffer] = {}
    counters: Dict[str, int] = defaultdict(int)
    usecols = {event_col, station_col}
    for c in (
        sequence_col, second_col, value_col, origin_col, magnitude_col, depth_col,
        longitude_col, latitude_col, distance_col,
    ):
        if c:
            usecols.add(c)
    if layout == "wide":
        usecols.update(c for _, c in wide_cols)

    def get_event(row: Mapping[str, Any]) -> EventBuffer:
        event_id = _safe_text(row[event_col])
        if not event_id:
            raise ValueError("empty event ID")
        origin = _safe_text(row.get(origin_col) if origin_col else None, event_id)
        if event_id not in events:
            eq_info: Dict[str, Any] = {"number": event_id, "origin_time": origin}
            for target, source in (
                ("magnitude", magnitude_col), ("depth", depth_col),
                ("longitude", longitude_col), ("latitude", latitude_col),
            ):
                if source:
                    value = _safe_float(row.get(source))
                    if value is not None:
                        eq_info[target] = value
            events[event_id] = EventBuffer(event_id, origin, eq_info)
        return events[event_id]

    for chunk in pd.read_csv(
        csv_path, chunksize=args.chunk_size, usecols=list(usecols),
        low_memory=False, encoding=args.encoding,
    ):
        counters["rows_read"] += len(chunk)
        for row in chunk.to_dict(orient="records"):
            try:
                event = get_event(row)
                station_id = _safe_text(row[station_col])
                if not station_id:
                    raise ValueError("empty station ID")
                if layout == "wide":
                    series = [-99.0] * args.horizon
                    for second, column in wide_cols:
                        if 1 <= second <= args.horizon:
                            series[second - 1] = sanitize_intensity(row.get(column))
                elif layout == "sequence":
                    series = normalize_sequence(row.get(sequence_col), args.horizon)
                else:
                    second = int(float(row.get(second_col))) + (1 if args.second_origin == 0 else 0)
                    if not 1 <= second <= args.horizon:
                        counters["long_rows_outside_horizon"] += 1
                        continue
                    series = event.intensity.setdefault(station_id, [-99.0] * args.horizon)
                    if series[second - 1] != -99.0 and args.duplicate_policy == "error":
                        raise ValueError(f"duplicate second: {event.event_id}/{station_id}/{second}")
                    series[second - 1] = sanitize_intensity(row.get(value_col))
                    counters["rows_converted"] += 1
                    continue

                if station_id in event.intensity and args.duplicate_policy == "error":
                    raise ValueError(f"duplicate event/station row: {event.event_id}/{station_id}")
                event.intensity[station_id] = series
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
        payload = {
            "eq_info": event.eq_info,
            "intensity": event.intensity,
            "epicenter_distance": event.distance,
        }
        validation = _validate_event_payload(payload, args.horizon)
        filename = _filename(event_id, i)
        with (output_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        index_rows.append({
            "event_id": event_id,
            "origin_time": event.origin_time,
            "filename": filename,
            **validation,
        })

    return {
        "mapping": {
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
        },
        "index_rows": index_rows,
        "counters": dict(counters),
    }


def convert_csv(args: argparse.Namespace) -> Dict[str, Any]:
    csv_path = Path(args.csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; use --overwrite")
    if args.overwrite:
        _clean_output(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    schema = inspect_schema(csv_path, args.preview_rows, args.horizon, args.encoding)
    columns = schema["columns"]
    detected = schema["detected"]
    layout = args.layout if args.layout != "auto" else detected["layout"]
    if layout == "unknown":
        raise ValueError("Could not detect CSV layout; run inspect and provide explicit mappings")

    if layout == "event_json":
        result = _convert_event_json(args, csv_path, output_dir, columns)
    else:
        result = _convert_station_table(args, csv_path, output_dir, columns, detected, layout)

    index_rows = result.pop("index_rows")
    with (output_dir / "event_index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "event_id", "origin_time", "filename", "n_stations",
            "n_valid_intensity_values", "n_missing_intensity_values",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)

    summary = {
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "horizon": args.horizon,
        "mapping": result["mapping"],
        "n_events": len(index_rows),
        "n_station_records": sum(int(row["n_stations"]) for row in index_rows),
        "counters": result["counters"],
    }
    with (output_dir / "conversion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def validate_archive(data_dir: Path, horizon: int) -> Dict[str, Any]:
    files = sorted(
        fp for fp in data_dir.rglob("*.json")
        if fp.name not in {"conversion_summary.json"}
    )
    errors: List[str] = []
    events = 0
    stations = 0
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            stats = _validate_event_payload(payload, horizon)
            events += 1
            stations += stats["n_stations"]
        except Exception as exc:
            errors.append(f"{fp.name}: {type(exc).__name__}: {exc}")
    result = {
        "data_dir": str(data_dir),
        "horizon": horizon,
        "n_event_json": events,
        "n_station_records": stations,
        "n_errors": len(errors),
        "errors": errors[:50],
    }
    if errors:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CSV data to SSIF event JSON files")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect CSV columns and auto-detected layout")
    inspect.add_argument("--csv", required=True)
    inspect.add_argument("--rows", type=int, default=3)
    inspect.add_argument("--horizon", type=int, default=120)
    inspect.add_argument("--encoding", default=None)

    convert = sub.add_parser("convert", help="Convert CSV to event JSON archive")
    convert.add_argument("--csv", required=True)
    convert.add_argument("--output-dir", required=True)
    convert.add_argument(
        "--layout", choices=["auto", "event_json", "wide", "sequence", "long"], default="auto"
    )
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
    convert.add_argument("--chunk-size", type=int, default=200)
    convert.add_argument("--preview-rows", type=int, default=3)
    convert.add_argument("--encoding", default=None)
    convert.add_argument("--second-origin", type=int, choices=[0, 1], default=1)
    convert.add_argument("--duplicate-policy", choices=["error", "last"], default="error")
    convert.add_argument("--row-error-policy", choices=["error", "skip"], default="error")
    convert.add_argument("--overwrite", action="store_true")

    validate = sub.add_parser("validate", help="Validate a converted SSIF event-JSON archive")
    validate.add_argument("--data-dir", required=True)
    validate.add_argument("--horizon", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        result = inspect_schema(Path(args.csv).expanduser().resolve(), args.rows, args.horizon, args.encoding)
    elif args.command == "convert":
        result = convert_csv(args)
    else:
        result = validate_archive(Path(args.data_dir).expanduser().resolve(), args.horizon)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
