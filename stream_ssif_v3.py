# -*- coding: utf-8 -*-
"""Event-aligned real-time streaming inference for SSIF v3.

The training archive contains origin/event-aligned prefixes. Therefore this
program performs real-time inference on an active event session and emits
predictions at elapsed seconds 10, 15, ..., 40. It does not pretend that a
model trained on event-aligned prefixes is already validated for arbitrary
continuous rolling windows.

JSONL protocol
--------------
Start an event::

    {"type":"start_event","event_id":"E001","origin_time":"2026-07-26T10:00:00+08:00"}

Send one network tick per elapsed second::

    {"type":"tick","event_id":"E001","second":1,
     "observations":{"A001":0,"A002":-99}}

End the event::

    {"type":"end_event","event_id":"E001"}

Predictions are emitted as JSON lines at EW10--EW40.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ssif_core import (
    CWA_CLASSES,
    DEFAULT_WINDOWS,
    load_model_checkpoint,
    parse_cwa_intensity,
)


@dataclass
class LoadedWindowModel:
    window: int
    model: Any
    threshold: float
    payload: Mapping[str, Any]


@dataclass
class EventSession:
    event_id: str
    origin_time: Optional[str]
    current_second: int = 0
    station_values: Dict[str, List[float]] = field(default_factory=dict)
    station_valid: Dict[str, List[bool]] = field(default_factory=dict)
    emitted: set[Tuple[str, int]] = field(default_factory=set)

    def append_tick(self, second: int, observations: Mapping[str, Any], *, allow_gaps: bool) -> None:
        if second <= self.current_second:
            raise ValueError(
                f"event {self.event_id}: second must increase; got {second} after {self.current_second}"
            )
        if second > self.current_second + 1:
            if not allow_gaps:
                raise ValueError(
                    f"event {self.event_id}: missing ticks {self.current_second + 1}..{second - 1}"
                )
            for missing_second in range(self.current_second + 1, second):
                self._append_one(missing_second, {})
        self._append_one(second, observations)

    def _append_one(self, second: int, observations: Mapping[str, Any]) -> None:
        known = set(self.station_values) | {str(k) for k in observations}
        for station_id in known:
            if station_id not in self.station_values:
                self.station_values[station_id] = [0.0] * (second - 1)
                self.station_valid[station_id] = [False] * (second - 1)
            raw = observations.get(station_id)
            value, valid = parse_cwa_intensity(raw)
            self.station_values[station_id].append(value)
            self.station_valid[station_id].append(valid)
        self.current_second = second


class StreamingEngine:
    def __init__(
        self,
        model_root: Path,
        *,
        windows: Sequence[int],
        device: str,
        min_valid_fraction: float,
        batch_size: int,
        allow_gaps: bool,
    ) -> None:
        self.device = device
        self.min_valid_fraction = float(min_valid_fraction)
        self.batch_size = int(batch_size)
        self.allow_gaps = bool(allow_gaps)
        self.models: Dict[int, LoadedWindowModel] = {}
        for window in sorted(set(windows)):
            ckpt = model_root / f"EW{window:02d}" / "best.pt"
            model, payload = load_model_checkpoint(ckpt, device=device)
            self.models[window] = LoadedWindowModel(
                window=window,
                model=model,
                threshold=float(payload["alert_probability_threshold"]),
                payload=payload,
            )
        self.sessions: Dict[str, EventSession] = {}

    def start_event(self, event_id: str, origin_time: Optional[str]) -> Dict[str, Any]:
        if event_id in self.sessions:
            raise ValueError(f"event already active: {event_id}")
        self.sessions[event_id] = EventSession(event_id=event_id, origin_time=origin_time)
        return {"type": "event_started", "event_id": event_id, "origin_time": origin_time}

    def end_event(self, event_id: str) -> Dict[str, Any]:
        session = self.sessions.pop(event_id, None)
        if session is None:
            raise ValueError(f"unknown event: {event_id}")
        return {
            "type": "event_ended",
            "event_id": event_id,
            "last_second": session.current_second,
            "n_stations": len(session.station_values),
        }

    def tick(self, event_id: str, second: int, observations: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if event_id not in self.sessions:
            raise ValueError(f"unknown event: {event_id}; send start_event first")
        session = self.sessions[event_id]
        previous_second = session.current_second
        session.append_tick(second, observations, allow_gaps=self.allow_gaps)
        # When --allow-gaps fills missing seconds, emit every model window that
        # was crossed.  Example: a jump from second 9 to 11 must still emit EW10.
        results: List[Dict[str, Any]] = []
        for window in sorted(self.models):
            if previous_second < window <= second:
                results.extend(self._predict_window(session, self.models[window]))
        return results

    def _predict_window(self, session: EventSession, loaded: LoadedWindowModel) -> List[Dict[str, Any]]:
        import torch

        window = loaded.window
        station_ids: List[str] = []
        values_batch: List[np.ndarray] = []
        mask_batch: List[np.ndarray] = []
        skipped: List[Dict[str, Any]] = []

        for station_id in sorted(session.station_values):
            if (station_id, window) in session.emitted:
                continue
            values = np.asarray(session.station_values[station_id][:window], dtype=np.float32)
            valid = np.asarray(session.station_valid[station_id][:window], dtype=np.bool_)
            if len(values) < window:
                continue
            valid_fraction = float(valid.mean())
            if valid_fraction < self.min_valid_fraction or not valid.any():
                skipped.append({
                    "type": "prediction_skipped",
                    "event_id": session.event_id,
                    "station_id": station_id,
                    "window": window,
                    "valid_fraction": valid_fraction,
                    "reason": "insufficient_valid_observations",
                })
                session.emitted.add((station_id, window))
                continue
            station_ids.append(station_id)
            values_batch.append(values / 9.0)
            mask_batch.append(valid)

        results: List[Dict[str, Any]] = []
        model = loaded.model
        model.eval()
        for start in range(0, len(station_ids), self.batch_size):
            end = start + self.batch_size
            x = torch.from_numpy(np.stack(values_batch[start:end])).to(self.device)
            mask = torch.from_numpy(np.stack(mask_batch[start:end])).to(self.device)
            with torch.no_grad():
                out = model(x, mask)
                class_prob = torch.softmax(out["class_logits"], dim=-1)
                confidence, pred_class = class_prob.max(dim=-1)
                alert_prob = torch.sigmoid(out["alert_logit"])

            for j, station_id in enumerate(station_ids[start:end]):
                idx = start + j
                valid_values = np.asarray(session.station_values[station_id][:window], dtype=np.float32)[
                    np.asarray(session.station_valid[station_id][:window], dtype=np.bool_)
                ]
                current_max = int(valid_values.max()) if valid_values.size else 0
                pred_idx = int(pred_class[j].item())
                p_alert = float(alert_prob[j].item())
                results.append({
                    "type": "prediction",
                    "event_id": session.event_id,
                    "origin_time": session.origin_time,
                    "elapsed_second": session.current_second,
                    "station_id": station_id,
                    "window": window,
                    "current_max": current_max,
                    "pred_class": pred_idx,
                    "pred_label": CWA_CLASSES[pred_idx],
                    "class_confidence": float(confidence[j].item()),
                    "expected_class": float(out["expected_class"][j].item()),
                    "alert_prob": p_alert,
                    "alert_threshold": loaded.threshold,
                    "alert": bool(p_alert >= loaded.threshold),
                    "valid_fraction": float(mask_batch[idx].mean()),
                    "model_format_version": loaded.payload.get("format_version", 2),
                })
                session.emitted.add((station_id, window))
        return skipped + results


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def emit(obj: Mapping[str, Any], output) -> None:
    output.write(json.dumps(obj, ensure_ascii=False) + "\n")
    output.flush()


def process_message(engine: StreamingEngine, message: Mapping[str, Any]) -> List[Dict[str, Any]]:
    msg_type = message.get("type")
    if msg_type == "start_event":
        return [engine.start_event(str(message["event_id"]), message.get("origin_time"))]
    if msg_type == "end_event":
        return [engine.end_event(str(message["event_id"]))]
    if msg_type == "tick":
        observations = message.get("observations") or {}
        if not isinstance(observations, dict):
            raise ValueError("tick.observations must be an object mapping station_id to intensity")
        return engine.tick(str(message["event_id"]), int(message["second"]), observations)
    raise ValueError(f"unsupported message type: {msg_type}")


def command_serve(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    engine = StreamingEngine(
        Path(args.model_root),
        windows=args.windows,
        device=device,
        min_valid_fraction=args.min_valid_fraction,
        batch_size=args.batch_size,
        allow_gaps=args.allow_gaps,
    )
    input_handle = sys.stdin if args.input == "-" else open(args.input, "r", encoding="utf-8")
    output_handle = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        for line_no, line in enumerate(input_handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
                for obj in process_message(engine, message):
                    emit(obj, output_handle)
            except Exception as exc:
                emit({"type": "error", "line": line_no, "message": str(exc)}, output_handle)
                if args.fail_fast:
                    raise
    finally:
        if input_handle is not sys.stdin:
            input_handle.close()
        if output_handle is not sys.stdout:
            output_handle.close()


def command_replay(args: argparse.Namespace) -> None:
    with open(args.event_json, "r", encoding="utf-8") as f:
        event = json.load(f)
    intensity = event.get("intensity") or {}
    if not isinstance(intensity, dict) or not intensity:
        raise ValueError("event JSON has no intensity mapping")
    origin_time = (event.get("eq_info") or {}).get("origin_time")
    event_id = args.event_id or str((event.get("eq_info") or {}).get("number") or Path(args.event_json).stem)

    device = resolve_device(args.device)
    engine = StreamingEngine(
        Path(args.model_root),
        windows=args.windows,
        device=device,
        min_valid_fraction=args.min_valid_fraction,
        batch_size=args.batch_size,
        allow_gaps=False,
    )
    output_handle = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        emit(engine.start_event(event_id, origin_time), output_handle)
        max_second = max(args.windows)
        for second in range(1, max_second + 1):
            observations = {
                station_id: (series[second - 1] if isinstance(series, list) and len(series) >= second else None)
                for station_id, series in intensity.items()
            }
            for obj in engine.tick(event_id, second, observations):
                emit(obj, output_handle)
        emit(engine.end_event(event_id), output_handle)
    finally:
        if output_handle is not sys.stdout:
            output_handle.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SSIF v3 event-aligned streaming inference")
    sub = p.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Read event/tick JSONL and emit prediction JSONL")
    serve.add_argument("--model-root", required=True)
    serve.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    serve.add_argument("--input", default="-", help="JSONL input path, or - for stdin")
    serve.add_argument("--output", default="-", help="JSONL output path, or - for stdout")
    serve.add_argument("--device", default="auto")
    serve.add_argument("--batch-size", type=int, default=256)
    serve.add_argument("--min-valid-fraction", type=float, default=0.80)
    serve.add_argument("--allow-gaps", action="store_true")
    serve.add_argument("--fail-fast", action="store_true")
    serve.set_defaults(func=command_serve)

    replay = sub.add_parser("replay", help="Replay one CWA event JSON through the streaming engine")
    replay.add_argument("--model-root", required=True)
    replay.add_argument("--event-json", required=True)
    replay.add_argument("--event-id", default=None)
    replay.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    replay.add_argument("--output", default="-")
    replay.add_argument("--device", default="auto")
    replay.add_argument("--batch-size", type=int, default=256)
    replay.add_argument("--min-valid-fraction", type=float, default=0.80)
    replay.set_defaults(func=command_replay)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if any(w < 10 for w in args.windows):
        raise ValueError("EW05 is intentionally unsupported in SSIF v3")
    args.func(args)


if __name__ == "__main__":
    main()
