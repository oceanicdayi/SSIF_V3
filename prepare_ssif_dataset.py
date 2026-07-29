# -*- coding: utf-8 -*-
"""Audit a CWA SSIF archive and create a reproducible event-level split.

This program is deliberately separate from model training.  It freezes the
scientific definition of the dataset before any model is fitted:

* final labels are computed only within a fixed label horizon (default 120 s),
* event duplication and malformed files are reported,
* EW10--EW40 share one station-event cohort when requested,
* train/validation/calibration/test are mutually exclusive at event level,
* candidate splits are searched for balanced event distributions,
* all outputs include a deterministic data fingerprint.

The generated ``split_manifest.json`` can be passed directly to
``train_ssif_v3.py --split-manifest ...``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ssif_core import (
    ALERT_CLASS_THRESHOLD,
    CWA_CLASSES,
    DEFAULT_WINDOWS,
    StationRecord,
    iter_json_files,
    load_station_records,
    save_json,
)

SPLIT_NAMES = ("train", "validation", "calibration", "test")


@dataclass
class EventAudit:
    event_id: str
    source_file: str
    origin_time: str
    year: Optional[int]
    longitude: Optional[float]
    latitude: Optional[float]
    depth_km: Optional[float]
    magnitude: Optional[float]
    n_station_records: int
    n_common_records: int
    n_positive_records: int
    positive_fraction: float
    max_final_class: int
    max_final_label: str
    has_event_positive: bool


def _float_or_none(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _year_from_origin(origin: Any) -> Optional[int]:
    text = str(origin or "").strip()
    if len(text) >= 4 and text[:4].isdigit():
        y = int(text[:4])
        if 1900 <= y <= 2200:
            return y
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).year
    except Exception:
        return None


def _event_id(event: Mapping[str, Any], rel: str) -> str:
    eq = event.get("eq_info") or {}
    number = str(eq.get("number") or eq.get("isnumber") or "").strip()
    origin = str(eq.get("origin_time") or "").strip()
    if number and origin:
        return f"{number}|{origin}"
    if origin:
        return f"{origin}|{rel}"
    return rel


def _record_key(record: StationRecord) -> str:
    return f"{record.event_id}\t{record.station_id}"


def common_cohort_keys(
    records: Sequence[StationRecord],
    windows: Sequence[int],
    min_window_valid_fraction: float,
) -> set[str]:
    windows = tuple(sorted(set(int(w) for w in windows)))
    max_window = max(windows)
    keys: set[str] = set()
    for rec in records:
        if len(rec.values) < max_window:
            continue
        if all(float(rec.valid[:w].mean()) >= min_window_valid_fraction for w in windows):
            keys.add(_record_key(rec))
    return keys


def _read_event_metadata(root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], List[Dict[str, Any]]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, List[str]] = defaultdict(list)
    file_audit: List[Dict[str, Any]] = []
    for fp in iter_json_files(root):
        rel = str(fp.relative_to(root)).replace("\\", "/")
        row: Dict[str, Any] = {"source_file": rel, "status": "ok", "error": ""}
        try:
            with fp.open("r", encoding="utf-8") as f:
                event = json.load(f)
            eid = _event_id(event, rel)
            eq = event.get("eq_info") or {}
            intensity = event.get("intensity")
            row.update({
                "event_id": eid,
                "n_raw_stations": len(intensity) if isinstance(intensity, dict) else 0,
                "origin_time": str(eq.get("origin_time") or ""),
            })
            sources[eid].append(rel)
            metadata.setdefault(eid, {
                "event_id": eid,
                "source_file": rel,
                "origin_time": str(eq.get("origin_time") or ""),
                "year": _year_from_origin(eq.get("origin_time")),
                "longitude": _float_or_none(eq.get("longitude")),
                "latitude": _float_or_none(eq.get("latitude")),
                "depth_km": _float_or_none(eq.get("depth")),
                "magnitude": _float_or_none(eq.get("magnitude")),
            })
            if not isinstance(intensity, dict) or not intensity:
                row["status"] = "no_intensity_dictionary"
        except Exception as exc:
            row.update({"status": "parse_error", "error": f"{type(exc).__name__}: {exc}"})
        file_audit.append(row)
    return metadata, sources, file_audit


def _event_audits(
    records: Sequence[StationRecord],
    metadata: Mapping[str, Mapping[str, Any]],
    common_keys: set[str],
) -> List[EventAudit]:
    grouped: Dict[str, List[StationRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.event_id].append(rec)
    rows: List[EventAudit] = []
    for eid, recs in grouped.items():
        meta = dict(metadata.get(eid, {}))
        positive = sum(r.final_class >= ALERT_CLASS_THRESHOLD for r in recs)
        common = sum(_record_key(r) in common_keys for r in recs)
        max_cls = max(r.final_class for r in recs)
        rows.append(EventAudit(
            event_id=eid,
            source_file=str(meta.get("source_file") or recs[0].source_file),
            origin_time=str(meta.get("origin_time") or ""),
            year=meta.get("year"),
            longitude=meta.get("longitude"),
            latitude=meta.get("latitude"),
            depth_km=meta.get("depth_km"),
            magnitude=meta.get("magnitude"),
            n_station_records=len(recs),
            n_common_records=common,
            n_positive_records=positive,
            positive_fraction=float(positive / len(recs)),
            max_final_class=max_cls,
            max_final_label=CWA_CLASSES[max_cls],
            has_event_positive=bool(positive > 0),
        ))
    return sorted(rows, key=lambda r: r.event_id)


def _largest_remainder_counts(n: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    normalized = {k: ratios[k] / total for k in SPLIT_NAMES}
    raw = {k: n * normalized[k] for k in SPLIT_NAMES}
    counts = {k: int(math.floor(raw[k])) for k in SPLIT_NAMES}
    remaining = n - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[:remaining]:
        counts[k] += 1
    if n >= len(SPLIT_NAMES):
        for k in SPLIT_NAMES:
            if counts[k] == 0:
                donor = max(SPLIT_NAMES, key=lambda x: counts[x])
                if counts[donor] <= 1:
                    raise ValueError("not enough events for four non-empty splits")
                counts[donor] -= 1
                counts[k] += 1
    return counts


def _mag_bin(x: Optional[float]) -> str:
    if x is None:
        return "missing"
    if x < 4.0:
        return "<4"
    if x < 5.0:
        return "4-<5"
    if x < 6.0:
        return "5-<6"
    return ">=6"


def _depth_bin(x: Optional[float]) -> str:
    if x is None:
        return "missing"
    if x < 15:
        return "<15"
    if x < 30:
        return "15-<30"
    if x < 70:
        return "30-<70"
    return ">=70"


def _intensity_bin(x: int) -> str:
    if x < 4:
        return "0-3"
    if x == 4:
        return "4"
    if x <= 6:
        return "5-/5+"
    return "6-/6+/7"


def _numeric_matrix(events: Sequence[EventAudit]) -> np.ndarray:
    cols: List[List[float]] = []
    attrs = ("magnitude", "depth_km", "latitude", "longitude", "year", "positive_fraction", "max_final_class")
    for attr in attrs:
        vals = np.asarray([
            float(getattr(e, attr)) if getattr(e, attr) is not None else np.nan
            for e in events
        ], dtype=np.float64)
        finite = np.isfinite(vals)
        fill = float(np.median(vals[finite])) if finite.any() else 0.0
        vals[~finite] = fill
        sd = float(vals.std())
        vals = (vals - float(vals.mean())) / (sd if sd > 1e-9 else 1.0)
        cols.append(vals.tolist())
    station = np.log1p(np.asarray([e.n_common_records for e in events], dtype=np.float64))
    sd = float(station.std())
    station = (station - float(station.mean())) / (sd if sd > 1e-9 else 1.0)
    cols.append(station.tolist())
    return np.asarray(cols, dtype=np.float64).T


def _split_score(events: Sequence[EventAudit], assignment: Mapping[str, Sequence[int]], X: np.ndarray) -> float:
    global_mean = X.mean(axis=0)
    score = 0.0
    n = len(events)
    global_cats = {
        "positive": Counter(str(e.has_event_positive) for e in events),
        "magnitude": Counter(_mag_bin(e.magnitude) for e in events),
        "depth": Counter(_depth_bin(e.depth_km) for e in events),
        "intensity": Counter(_intensity_bin(e.max_final_class) for e in events),
    }
    for split, idxs_seq in assignment.items():
        idxs = np.asarray(idxs_seq, dtype=np.int64)
        if idxs.size == 0:
            return 1e12
        weight = math.sqrt(idxs.size / n)
        score += weight * float(np.mean(np.abs(X[idxs].mean(axis=0) - global_mean)))
        subset = [events[int(i)] for i in idxs]
        cats = {
            "positive": Counter(str(e.has_event_positive) for e in subset),
            "magnitude": Counter(_mag_bin(e.magnitude) for e in subset),
            "depth": Counter(_depth_bin(e.depth_km) for e in subset),
            "intensity": Counter(_intensity_bin(e.max_final_class) for e in subset),
        }
        for family in global_cats:
            keys = set(global_cats[family]) | set(cats[family])
            for key in keys:
                pg = global_cats[family][key] / n
                ps = cats[family][key] / len(subset)
                score += 0.35 * weight * abs(ps - pg)

    # Hard scientific safeguards when the archive is large enough.
    n_pos = sum(e.has_event_positive for e in events)
    n_neg = n - n_pos
    if n_pos >= len(SPLIT_NAMES):
        for idxs in assignment.values():
            if not any(events[i].has_event_positive for i in idxs):
                score += 100.0
    if n_neg >= len(SPLIT_NAMES):
        for idxs in assignment.values():
            if not any(not events[i].has_event_positive for i in idxs):
                score += 100.0
    return score


def balanced_random_split(
    events: Sequence[EventAudit],
    ratios: Mapping[str, float],
    *,
    seed: int,
    candidates: int,
) -> Tuple[Dict[str, List[str]], float]:
    if len(events) < len(SPLIT_NAMES):
        raise ValueError("at least four events are needed for train/validation/calibration/test")
    counts = _largest_remainder_counts(len(events), ratios)
    X = _numeric_matrix(events)
    rng = random.Random(seed)
    indices = list(range(len(events)))
    best_score = float("inf")
    best: Optional[Dict[str, List[int]]] = None
    for _ in range(max(1, candidates)):
        rng.shuffle(indices)
        pos = 0
        candidate: Dict[str, List[int]] = {}
        for name in SPLIT_NAMES:
            candidate[name] = indices[pos: pos + counts[name]].copy()
            pos += counts[name]
        score = _split_score(events, candidate, X)
        if score < best_score:
            best_score = score
            best = candidate
    assert best is not None
    return {
        name: sorted(events[i].event_id for i in best[name])
        for name in SPLIT_NAMES
    }, float(best_score)


def validate_split(split: Mapping[str, Sequence[str]], all_event_ids: Sequence[str]) -> Dict[str, Any]:
    expected = set(all_event_ids)
    seen: Dict[str, str] = {}
    overlaps: List[Dict[str, str]] = []
    for name in SPLIT_NAMES:
        if name not in split:
            raise ValueError(f"split is missing required group: {name}")
        for eid in split[name]:
            if eid in seen:
                overlaps.append({"event_id": eid, "first": seen[eid], "second": name})
            seen[eid] = name
    assigned = set(seen)
    missing = sorted(expected - assigned)
    unknown = sorted(assigned - expected)
    return {
        "valid": not overlaps and not missing and not unknown,
        "n_expected": len(expected),
        "n_assigned": len(assigned),
        "overlaps": overlaps,
        "missing_events": missing,
        "unknown_events": unknown,
    }


def _fingerprint(events: Sequence[EventAudit], records: Sequence[StationRecord], config: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for event in sorted(events, key=lambda x: x.event_id):
        h.update(f"{event.event_id}|{event.source_file}|{event.n_station_records}|{event.n_common_records}\n".encode("utf-8"))
    for rec in sorted(records, key=lambda x: (x.event_id, x.station_id)):
        h.update(f"{rec.event_id}|{rec.station_id}|{rec.final_class}|{int(rec.valid.sum())}\n".encode("utf-8"))
    return h.hexdigest()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _distribution_report(events: Sequence[EventAudit], split: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    by_id = {e.event_id: e for e in events}
    out: Dict[str, Any] = {}
    for name in SPLIT_NAMES:
        subset = [by_id[eid] for eid in split[name]]
        mags = [e.magnitude for e in subset if e.magnitude is not None]
        depths = [e.depth_km for e in subset if e.depth_km is not None]
        out[name] = {
            "n_events": len(subset),
            "n_station_records": sum(e.n_station_records for e in subset),
            "n_common_records": sum(e.n_common_records for e in subset),
            "n_event_positive": sum(e.has_event_positive for e in subset),
            "event_positive_fraction": float(np.mean([e.has_event_positive for e in subset])) if subset else 0.0,
            "magnitude_mean": float(np.mean(mags)) if mags else None,
            "magnitude_median": float(np.median(mags)) if mags else None,
            "depth_mean_km": float(np.mean(depths)) if depths else None,
            "max_class_counts": dict(sorted(Counter(e.max_final_label for e in subset).items())),
            "magnitude_bin_counts": dict(sorted(Counter(_mag_bin(e.magnitude) for e in subset).items())),
            "depth_bin_counts": dict(sorted(Counter(_depth_bin(e.depth_km) for e in subset).items())),
            "year_counts": dict(sorted(Counter(str(e.year) for e in subset).items())),
        }
    return out


def command_audit_split(args: argparse.Namespace) -> None:
    root = Path(args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    windows = tuple(sorted(set(args.windows)))
    if any(w < 10 for w in windows):
        raise ValueError("EW05 is excluded; all windows must be >=10")
    if args.label_horizon < max(windows):
        raise ValueError("label_horizon must be at least the largest early window")

    metadata, event_sources, file_audit = _read_event_metadata(root)
    duplicate_ids = {eid: files for eid, files in event_sources.items() if len(files) > 1}
    records, load_stats = load_station_records(
        root,
        min_full_valid_fraction=args.min_label_valid_fraction,
        min_series_length=args.label_horizon,
        label_horizon=args.label_horizon,
        require_label_horizon=True,
    )
    common_keys = common_cohort_keys(records, windows, args.min_window_valid_fraction)
    events = _event_audits(records, metadata, common_keys)
    if args.require_common_records:
        events = [e for e in events if e.n_common_records > 0]
        allowed_events = {e.event_id for e in events}
        records = [r for r in records if r.event_id in allowed_events]
        common_keys = {k for k in common_keys if k.split("\t", 1)[0] in allowed_events}

    if duplicate_ids and args.fail_on_duplicate_event:
        save_json(output / "duplicate_events.json", duplicate_ids)
        raise RuntimeError(
            f"found {len(duplicate_ids)} duplicate event IDs; see {output / 'duplicate_events.json'}"
        )

    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "calibration": args.calibration_ratio,
        "test": args.test_ratio,
    }
    if any(v < 0 for v in ratios.values()) or sum(ratios.values()) <= 0:
        raise ValueError("split ratios must be non-negative and sum to a positive value")
    split, split_score = balanced_random_split(
        events, ratios, seed=args.seed, candidates=args.split_candidates
    )
    validation = validate_split(split, [e.event_id for e in events])
    if not validation["valid"]:
        raise RuntimeError(f"internal split validation failed: {validation}")

    config = {
        "format_version": 3,
        "windows": list(windows),
        "label_horizon": args.label_horizon,
        "min_label_valid_fraction": args.min_label_valid_fraction,
        "min_window_valid_fraction": args.min_window_valid_fraction,
        "cohort": "common_EW10_to_EW40",
        "split_ratios": ratios,
        "split_seed": args.seed,
        "split_candidates": args.split_candidates,
    }
    fingerprint = _fingerprint(events, records, config)
    manifest = {
        **config,
        "data_root": str(root.resolve()),
        "data_fingerprint_sha256": fingerprint,
        "split_search_score": split_score,
        "splits": split,
        "validation": validation,
        "n_events": len(events),
        "n_station_records": len(records),
        "n_common_cohort_records": len(common_keys),
    }

    origin_groups: Dict[str, List[str]] = defaultdict(list)
    for event in events:
        if event.origin_time:
            origin_groups[event.origin_time].append(event.event_id)
    possible_origin_duplicates = {k: v for k, v in origin_groups.items() if len(v) > 1}

    save_json(output / "split_manifest.json", manifest)
    save_json(output / "audit_summary.json", {
        "configuration": config,
        "loader": load_stats,
        "n_parsed_event_metadata": len(metadata),
        "n_duplicate_event_ids": len(duplicate_ids),
        "duplicate_event_ids": duplicate_ids,
        "n_possible_duplicate_origin_times": len(possible_origin_duplicates),
        "possible_duplicate_origin_times": possible_origin_duplicates,
        "n_events_used": len(events),
        "n_records_used": len(records),
        "n_common_cohort_records": len(common_keys),
        "common_cohort_fraction": len(common_keys) / len(records) if records else 0.0,
        "data_fingerprint_sha256": fingerprint,
    })
    save_json(output / "split_distribution.json", _distribution_report(events, split))
    save_json(output / "duplicate_events.json", duplicate_ids)
    save_json(output / "possible_duplicate_origin_times.json", possible_origin_duplicates)

    _write_csv(output / "file_audit.csv", file_audit,
               ["source_file", "status", "error", "event_id", "origin_time", "n_raw_stations"])
    _write_csv(output / "event_audit.csv", (asdict(e) for e in events), list(EventAudit.__dataclass_fields__))
    _write_csv(output / "common_cohort.csv", (
        {"event_id": key.split("\t", 1)[0], "station_id": key.split("\t", 1)[1]}
        for key in sorted(common_keys)
    ), ["event_id", "station_id"])
    split_rows = []
    by_id = {e.event_id: e for e in events}
    for name in SPLIT_NAMES:
        for eid in split[name]:
            row = asdict(by_id[eid])
            row["split"] = name
            split_rows.append(row)
    _write_csv(output / "event_split.csv", split_rows, ["split", *EventAudit.__dataclass_fields__.keys()])

    print(json.dumps({
        "output_dir": str(output),
        "n_events": len(events),
        "n_records": len(records),
        "n_common_records": len(common_keys),
        "split_counts": {k: len(v) for k, v in split.items()},
        "split_score": split_score,
        "fingerprint": fingerprint,
        "duplicates": len(duplicate_ids),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit SSIF data and freeze a reproducible event split")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit-split", help="Audit archive and create train/validation/calibration/test manifest")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    p.add_argument("--label-horizon", type=int, default=120)
    p.add_argument("--min-label-valid-fraction", type=float, default=0.80)
    p.add_argument("--min-window-valid-fraction", type=float, default=0.80)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--validation-ratio", type=float, default=0.10)
    p.add_argument("--calibration-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--split-candidates", type=int, default=5000)
    p.add_argument("--require-common-records", action="store_true", default=True)
    p.add_argument("--keep-events-without-common-records", dest="require_common_records", action="store_false")
    p.add_argument("--fail-on-duplicate-event", action="store_true", default=True)
    p.add_argument("--allow-duplicate-event", dest="fail_on_duplicate_event", action="store_false")
    p.set_defaults(func=command_audit_split)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
