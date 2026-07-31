# -*- coding: utf-8 -*-
"""Stream combined_data.csv into SSIF event JSON files.

The expected CSV schema is one complete earthquake event per row with nested
Python/JSON literals in the columns: eq_info, intensity, and optionally times,
stids, epicenter_distance, variables, source_file.

Commands:
  inspect   inspect columns and parse sample rows
  scan      validate every row without writing event JSON files
  convert   stream rows into one JSON file per event
  validate  validate a converted JSON archive

The implementation avoids loading the ~728 MB CSV into memory and produces
explicit error and duplicate reports suitable for research audit trails.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

MISSING_VALUE = -99.0
REQUIRED_COLUMNS = ("eq_info", "intensity")
OPTIONAL_COLUMNS = ("times", "stids", "epicenter_distance", "variables", "source_file")


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_set_csv_field_limit()


class _SafeNameTransformer(ast.NodeTransformer):
    """Replace common non-literal scalar names before ast.literal_eval."""

    _values = {
        "nan": None,
        "NaN": None,
        "NAN": None,
        "null": None,
        "None": None,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
        "inf": None,
        "Inf": None,
        "Infinity": None,
    }

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._values:
            return ast.copy_location(ast.Constant(self._values[node.id]), node)
        raise ValueError(f"unsupported name in nested literal: {node.id}")

    def visit_Call(self, node: ast.Call) -> ast.AST:
        raise ValueError("function calls are not allowed in nested literals")

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        raise ValueError("attributes are not allowed in nested literals")


def parse_nested(value: Any, *, expected: Optional[type] = None, field: str = "value") -> Any:
    if value is None:
        parsed: Any = None
    elif isinstance(value, (dict, list, tuple, int, float, bool)):
        parsed = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            parsed = None
        else:
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    try:
                        tree = ast.parse(text, mode="eval")
                        tree = _SafeNameTransformer().visit(tree)
                        ast.fix_missing_locations(tree)
                        parsed = ast.literal_eval(tree)
                    except Exception as exc:
                        raise ValueError(f"cannot parse {field}: {text[:160]!r}") from exc
    if expected is not None and not isinstance(parsed, expected):
        raise ValueError(f"{field} must decode to {expected.__name__}, got {type(parsed).__name__}")
    return parsed


def _finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def sanitize_intensity(value: Any) -> float:
    x = _finite_float(value)
    if x is None or x < 0 or x > 9:
        return MISSING_VALUE
    return float(int(round(x)))


def normalize_sequence(value: Any, horizon: int) -> Tuple[List[float], Dict[str, int]]:
    sequence = parse_nested(value, field="intensity sequence")
    if not isinstance(sequence, (list, tuple)):
        raise ValueError(f"intensity sequence must be list/tuple, got {type(sequence).__name__}")
    original_length = len(sequence)
    out = [sanitize_intensity(item) for item in sequence[:horizon]]
    if len(out) < horizon:
        out.extend([MISSING_VALUE] * (horizon - len(out)))
    return out, {
        "short_sequence": int(original_length < horizon),
        "long_sequence": int(original_length > horizon),
        "missing_values": sum(item == MISSING_VALUE for item in out),
        "valid_values": sum(item != MISSING_VALUE for item in out),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _rounded(value: Any, digits: int) -> str:
    x = _finite_float(value)
    return "" if x is None else f"{x:.{digits}f}"


def event_identity(eq_info: Mapping[str, Any], row_number: int) -> str:
    number = _text(eq_info.get("number") or eq_info.get("isnumber"))
    origin = _text(eq_info.get("origin_time"))
    lon = _rounded(eq_info.get("longitude"), 4)
    lat = _rounded(eq_info.get("latitude"), 4)
    depth = _rounded(eq_info.get("depth"), 1)
    magnitude = _rounded(eq_info.get("magnitude"), 1)
    if number and origin:
        return f"{number}|{origin}"
    if origin:
        return "|".join([origin, lon, lat, depth, magnitude])
    if number:
        return f"number:{number}"
    return f"csv_row:{row_number:08d}"


def payload_fingerprint(payload: Mapping[str, Any]) -> str:
    # source_file is archive provenance, not scientific event content. The same
    # event copied between folders should still be recognized as an exact
    # duplicate rather than an identity conflict.
    scientific_payload = {key: value for key, value in payload.items() if key != "source_file"}
    canonical = json.dumps(
        scientific_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _filename(identity: str, first_row: int) -> str:
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return f"event_{first_row:06d}_{digest}.json"


def parse_event_row(row: Mapping[str, str], row_number: int, horizon: int) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    eq_info = parse_nested(row.get("eq_info"), expected=dict, field="eq_info")
    raw_intensity = parse_nested(row.get("intensity"), expected=dict, field="intensity")
    if not raw_intensity:
        raise ValueError("intensity dictionary is empty")

    counters = Counter()
    intensity: Dict[str, List[float]] = {}
    for station_id, raw_sequence in raw_intensity.items():
        station = _text(station_id)
        if not station:
            raise ValueError("intensity contains an empty station ID")
        sequence, stats = normalize_sequence(raw_sequence, horizon)
        intensity[station] = sequence
        counters.update(stats)

    distance: Dict[str, float] = {}
    raw_distance = parse_nested(row.get("epicenter_distance"), field="epicenter_distance")
    if raw_distance is not None:
        if not isinstance(raw_distance, dict):
            raise ValueError("epicenter_distance must decode to dict or null")
        for station_id, value in raw_distance.items():
            x = _finite_float(value)
            if x is not None:
                distance[_text(station_id)] = x

    payload: Dict[str, Any] = {
        "eq_info": _json_safe(eq_info),
        "intensity": intensity,
        "epicenter_distance": distance,
    }

    raw_times = parse_nested(row.get("times"), field="times")
    if raw_times is not None:
        if not isinstance(raw_times, (list, tuple)):
            raise ValueError("times must decode to list/tuple or null")
        payload["times"] = [_json_safe(item) for item in list(raw_times)[:horizon]]

    raw_stids = parse_nested(row.get("stids"), field="stids")
    if raw_stids is not None:
        if not isinstance(raw_stids, dict):
            raise ValueError("stids must decode to dict or null")
        payload["stids"] = _json_safe(raw_stids)

    raw_variables = parse_nested(row.get("variables"), field="variables")
    if raw_variables is not None:
        payload["variables"] = _json_safe(raw_variables)

    source_file = _text(row.get("source_file"))
    if source_file:
        payload["source_file"] = source_file

    identity = event_identity(eq_info, row_number)
    counters["stations"] = len(intensity)
    counters["distance_stations"] = len(distance)
    return identity, payload, dict(counters)


def open_reader(csv_path: Path, encoding: str) -> Tuple[Any, csv.DictReader]:
    handle = csv_path.open("r", encoding=encoding, newline="")
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        handle.close()
        raise ValueError("CSV has no header")
    reader.fieldnames = [str(name).lstrip("\ufeff") for name in reader.fieldnames]
    missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
    if missing:
        handle.close()
        raise ValueError(f"missing required columns: {missing}; found={reader.fieldnames}")
    return handle, reader


def inspect_csv(csv_path: Path, rows: int, horizon: int, encoding: str) -> Dict[str, Any]:
    handle, reader = open_reader(csv_path, encoding)
    parsed = []
    try:
        for row_number, row in enumerate(reader, start=1):
            identity, payload, stats = parse_event_row(row, row_number, horizon)
            parsed.append({
                "row": row_number,
                "identity": identity,
                "origin_time": payload["eq_info"].get("origin_time"),
                "stations": stats["stations"],
                "short_sequences": stats.get("short_sequence", 0),
                "long_sequences": stats.get("long_sequence", 0),
            })
            if len(parsed) >= rows:
                break
    finally:
        handle.close()
    return {
        "csv_path": str(csv_path),
        "file_size_bytes": csv_path.stat().st_size,
        "columns": reader.fieldnames,
        "layout": "event_json",
        "sample_rows": parsed,
    }


def scan_csv(csv_path: Path, horizon: int, encoding: str, max_errors: int) -> Dict[str, Any]:
    handle, reader = open_reader(csv_path, encoding)
    counters = Counter()
    errors: List[Dict[str, Any]] = []
    duplicate_examples: List[Dict[str, Any]] = []
    seen: Dict[str, Tuple[str, int]] = {}
    origin_counts = Counter()
    try:
        for row_number, row in enumerate(reader, start=1):
            counters["rows_read"] += 1
            try:
                identity, payload, stats = parse_event_row(row, row_number, horizon)
                fingerprint = payload_fingerprint(payload)
                origin = _text(payload["eq_info"].get("origin_time"))
                if origin:
                    origin_counts[origin] += 1
                if identity in seen:
                    previous_fingerprint, previous_row = seen[identity]
                    if previous_fingerprint == fingerprint:
                        counters["exact_duplicate_rows"] += 1
                        kind = "exact_duplicate"
                    else:
                        counters["identity_conflicts"] += 1
                        kind = "identity_conflict"
                    if len(duplicate_examples) < max_errors:
                        duplicate_examples.append({
                            "kind": kind,
                            "identity": identity,
                            "first_row": previous_row,
                            "row": row_number,
                            "source_file": row.get("source_file", ""),
                        })
                else:
                    seen[identity] = (fingerprint, row_number)
                counters["rows_parsed"] += 1
                counters["station_records"] += stats.get("stations", 0)
                counters["short_sequences"] += stats.get("short_sequence", 0)
                counters["long_sequences"] += stats.get("long_sequence", 0)
                counters["valid_values"] += stats.get("valid_values", 0)
                counters["missing_values"] += stats.get("missing_values", 0)
            except Exception as exc:
                counters["row_errors"] += 1
                if len(errors) < max_errors:
                    errors.append({
                        "row": row_number,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "source_file": row.get("source_file", ""),
                    })
    finally:
        handle.close()

    duplicate_origins = [
        {"origin_time": origin, "count": count}
        for origin, count in origin_counts.most_common()
        if count > 1
    ]
    return {
        "csv_path": str(csv_path),
        "horizon": horizon,
        "columns": reader.fieldnames,
        "counters": dict(counters),
        "errors": errors,
        "duplicate_examples": duplicate_examples,
        "duplicate_origin_times": duplicate_origins[:max_errors],
        "n_duplicate_origin_groups": len(duplicate_origins),
    }


def _merge_payload(existing: MutableMapping[str, Any], incoming: Mapping[str, Any]) -> Dict[str, int]:
    stats = Counter()
    existing_intensity = existing["intensity"]
    for station, sequence in incoming["intensity"].items():
        if station not in existing_intensity:
            existing_intensity[station] = sequence
            stats["stations_added"] += 1
        elif existing_intensity[station] == sequence:
            stats["stations_identical"] += 1
        else:
            raise ValueError(f"conflicting intensity sequence for station {station}")

    existing_distance = existing.setdefault("epicenter_distance", {})
    for station, value in incoming.get("epicenter_distance", {}).items():
        if station not in existing_distance:
            existing_distance[station] = value
        elif not math.isclose(float(existing_distance[station]), float(value), rel_tol=0, abs_tol=1e-6):
            stats["distance_conflicts"] += 1

    existing_stids = existing.setdefault("stids", {}) if "stids" in existing or "stids" in incoming else None
    if existing_stids is not None:
        for station, metadata in incoming.get("stids", {}).items():
            existing_stids.setdefault(station, metadata)

    sources: List[str] = []
    for source in (existing.get("source_file"), incoming.get("source_file")):
        if isinstance(source, list):
            sources.extend(_text(item) for item in source if _text(item))
        elif _text(source):
            sources.append(_text(source))
    if sources:
        existing["source_file"] = sorted(set(sources))
    return dict(stats)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def convert_csv(
    csv_path: Path,
    output_dir: Path,
    horizon: int,
    encoding: str,
    overwrite: bool,
    duplicate_policy: str,
    error_policy: str,
    max_errors: int,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handle, reader = open_reader(csv_path, encoding)
    counters = Counter()
    errors: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    index_rows: Dict[str, Dict[str, Any]] = {}

    try:
        for row_number, row in enumerate(reader, start=1):
            counters["rows_read"] += 1
            try:
                identity, payload, stats = parse_event_row(row, row_number, horizon)
                fingerprint = payload_fingerprint(payload)
                if identity in seen:
                    prior = seen[identity]
                    same = prior["fingerprint"] == fingerprint
                    kind = "exact_duplicate" if same else "identity_conflict"
                    counters[kind] += 1
                    if len(duplicates) < max_errors:
                        duplicates.append({
                            "kind": kind,
                            "identity": identity,
                            "first_row": prior["row"],
                            "row": row_number,
                            "source_file": row.get("source_file", ""),
                        })
                    if same and duplicate_policy in {"skip-identical", "merge"}:
                        counters["rows_skipped_duplicate"] += 1
                        continue
                    if duplicate_policy == "error" or (duplicate_policy == "skip-identical" and not same):
                        raise ValueError(f"{kind}: {identity}; first row={prior['row']}")
                    if duplicate_policy == "merge":
                        event_path = Path(prior["path"])
                        existing = json.loads(event_path.read_text(encoding="utf-8"))
                        merge_stats = _merge_payload(existing, payload)
                        _atomic_json_write(event_path, existing)
                        counters.update(merge_stats)
                        counters["rows_merged"] += 1
                        prior["fingerprint"] = payload_fingerprint(existing)
                        index_rows[identity]["n_stations"] = len(existing["intensity"])
                        continue
                    if duplicate_policy == "suffix":
                        identity = f"{identity}|duplicate_row:{row_number}"
                        fingerprint = payload_fingerprint(payload)

                filename = _filename(identity, row_number)
                event_path = output_dir / filename
                _atomic_json_write(event_path, payload)
                seen[identity] = {
                    "fingerprint": fingerprint,
                    "row": row_number,
                    "path": str(event_path),
                }
                index_rows[identity] = {
                    "event_id": identity,
                    "origin_time": _text(payload["eq_info"].get("origin_time")),
                    "filename": filename,
                    "csv_row": row_number,
                    "n_stations": stats.get("stations", 0),
                    "source_file": row.get("source_file", ""),
                }
                counters["rows_converted"] += 1
                counters["station_records"] += stats.get("stations", 0)
                counters["short_sequences"] += stats.get("short_sequence", 0)
                counters["long_sequences"] += stats.get("long_sequence", 0)
                counters["valid_values"] += stats.get("valid_values", 0)
                counters["missing_values"] += stats.get("missing_values", 0)
            except Exception as exc:
                counters["row_errors"] += 1
                error = {
                    "row": row_number,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "source_file": row.get("source_file", ""),
                }
                if len(errors) < max_errors:
                    errors.append(error)
                if error_policy == "error":
                    raise ValueError(f"CSV row {row_number}: {exc}") from exc
    except Exception:
        report = {
            "status": "failed",
            "csv_path": str(csv_path),
            "output_dir": str(output_dir),
            "horizon": horizon,
            "duplicate_policy": duplicate_policy,
            "error_policy": error_policy,
            "counters": dict(counters),
            "errors": errors,
            "duplicates": duplicates,
        }
        (output_dir / "conversion_error_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    finally:
        handle.close()

    with (output_dir / "event_index.csv").open("w", encoding="utf-8-sig", newline="") as handle_out:
        fieldnames = ["event_id", "origin_time", "filename", "csv_row", "n_stations", "source_file"]
        writer = csv.DictWriter(handle_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows.values())

    summary = {
        "status": "ok" if not errors else "completed_with_skipped_errors",
        "csv_path": str(csv_path),
        "output_dir": str(output_dir),
        "horizon": horizon,
        "duplicate_policy": duplicate_policy,
        "error_policy": error_policy,
        "n_events": len(index_rows),
        "counters": dict(counters),
        "errors": errors,
        "duplicates": duplicates,
    }
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def validate_archive(data_dir: Path, horizon: int, max_errors: int) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    counters = Counter()
    event_files = sorted(data_dir.glob("event_*.json"))
    for path in event_files:
        counters["event_json"] += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload.get("eq_info"), dict):
                raise ValueError("eq_info missing/not dict")
            intensity = payload.get("intensity")
            if not isinstance(intensity, dict) or not intensity:
                raise ValueError("intensity missing/empty")
            counters["station_records"] += len(intensity)
            for station, sequence in intensity.items():
                if not isinstance(sequence, list) or len(sequence) != horizon:
                    raise ValueError(f"station {station}: length != {horizon}")
                for value in sequence:
                    if value == MISSING_VALUE:
                        counters["missing_values"] += 1
                    elif _finite_float(value) is None or not 0 <= float(value) <= 9:
                        raise ValueError(f"station {station}: invalid value {value!r}")
                    else:
                        counters["valid_values"] += 1
        except Exception as exc:
            counters["errors"] += 1
            if len(errors) < max_errors:
                errors.append({"file": path.name, "error_type": type(exc).__name__, "error": str(exc)})
    return {
        "data_dir": str(data_dir),
        "horizon": horizon,
        "counters": dict(counters),
        "errors": errors,
    }


def write_json(path: Optional[str], payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert combined_data.csv to SSIF event JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--csv", required=True)
    common.add_argument("--horizon", type=int, default=120)
    common.add_argument("--encoding", default="utf-8-sig")
    common.add_argument("--max-errors", type=int, default=20)
    common.add_argument("--report", default=None)

    inspect = sub.add_parser("inspect", parents=[common])
    inspect.add_argument("--rows", type=int, default=2)

    sub.add_parser("scan", parents=[common])

    convert = sub.add_parser("convert", parents=[common])
    convert.add_argument("--output-dir", required=True)
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument(
        "--duplicate-policy",
        choices=["error", "skip-identical", "merge", "suffix"],
        default="skip-identical",
    )
    convert.add_argument("--error-policy", choices=["error", "skip"], default="error")

    validate = sub.add_parser("validate")
    validate.add_argument("--data-dir", required=True)
    validate.add_argument("--horizon", type=int, default=120)
    validate.add_argument("--max-errors", type=int, default=20)
    validate.add_argument("--report", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        result = inspect_csv(Path(args.csv), args.rows, args.horizon, args.encoding)
    elif args.command == "scan":
        result = scan_csv(Path(args.csv), args.horizon, args.encoding, args.max_errors)
    elif args.command == "convert":
        result = convert_csv(
            Path(args.csv), Path(args.output_dir), args.horizon, args.encoding,
            args.overwrite, args.duplicate_policy, args.error_policy, args.max_errors,
        )
    else:
        result = validate_archive(Path(args.data_dir), args.horizon, args.max_errors)
    write_json(args.report, result)


if __name__ == "__main__":
    main()
