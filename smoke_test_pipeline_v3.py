# -*- coding: utf-8 -*-
"""End-to-end smoke test for audit/split, training, checkpoint and gap streaming."""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def make_event(seed: int, event_index: int, n_stations: int = 5) -> dict:
    rng = random.Random(seed)
    intensity = {}
    epi = {}
    stids = {}
    event_positive = event_index % 3 != 0
    for s in range(n_stations):
        station = f"S{s:03d}"
        peak = (4 + (event_index + s) % 4) if event_positive and s < 3 else (event_index + s) % 4
        peak_at = 12 + (event_index * 3 + s * 7) % 70
        series = []
        for t in range(120):
            value = 0
            if abs(t - peak_at) <= 3:
                value = max(0, peak - abs(t - peak_at))
            if rng.random() < 0.01:
                series.append(-99)
            else:
                series.append(value)
        intensity[station] = series
        epi[station] = 10.0 + event_index + s
        stids[station] = {"lat": 22.0 + 0.05 * event_index, "lon": 120.0 + 0.03 * event_index}
    return {
        "eq_info": {
            "number": f"E{event_index:04d}",
            "origin_time": f"{2010 + event_index % 15:04d}-01-01T00:00:00Z",
            "longitude": 120.0 + (event_index % 8) * 0.3,
            "latitude": 22.0 + (event_index % 10) * 0.3,
            "depth": 5.0 + (event_index % 8) * 12.0,
            "magnitude": 3.5 + (event_index % 7) * 0.5,
        },
        "times": list(range(120)),
        "stids": stids,
        "intensity": intensity,
        "epicenter_distance": epi,
    }


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        data = work / "data"
        prepared = work / "prepared"
        trained = work / "trained"
        data.mkdir()
        for i in range(28):
            (data / f"event_{i:03d}.json").write_text(
                json.dumps(make_event(1000 + i, i)), encoding="utf-8"
            )

        run([
            sys.executable, "prepare_ssif_dataset.py", "audit-split",
            "--data-dir", str(data), "--output-dir", str(prepared),
            "--split-candidates", "300", "--seed", "7",
        ], root)
        manifest = json.loads((prepared / "split_manifest.json").read_text(encoding="utf-8"))
        splits = manifest["splits"]
        all_ids = [eid for values in splits.values() for eid in values]
        assert len(all_ids) == len(set(all_ids)) == 28
        assert all(splits[name] for name in ("train", "validation", "calibration", "test"))
        assert manifest["n_common_cohort_records"] > 0

        run([
            sys.executable, "train_ssif_v3.py", "train-all",
            "--data-dir", str(data), "--output-dir", str(trained),
            "--split-manifest", str(prepared / "split_manifest.json"),
            "--windows", "10", "--epochs", "1", "--patience", "0",
            "--batch-size", "16", "--eval-batch-size", "32",
            "--hidden-size", "32", "--conv1", "32", "--conv2", "32",
            "--num-layers", "1", "--num-heads", "4", "--ff-mult", "2",
            "--workers", "0", "--device", "cpu", "--seed", "7",
        ], root)
        assert (trained / "EW10" / "best.pt").exists()
        metrics = json.loads((trained / "EW10" / "metrics.json").read_text(encoding="utf-8"))
        assert set(metrics) == {"validation", "calibration", "test"}

        # Verify --allow-gaps does not skip EW10 when input jumps 9 -> 11.
        stream_input = work / "stream.jsonl"
        rows = [{"type": "start_event", "event_id": "LIVE", "origin_time": "2026-01-01T00:00:00Z"}]
        for second in range(1, 10):
            rows.append({"type": "tick", "event_id": "LIVE", "second": second,
                         "observations": {"S000": min(3, second // 3)}})
        rows.append({"type": "tick", "event_id": "LIVE", "second": 11,
                     "observations": {"S000": 4}})
        stream_input.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        stream_output = work / "stream_out.jsonl"
        run([
            sys.executable, "stream_ssif_v3.py", "serve",
            "--model-root", str(trained), "--windows", "10",
            "--input", str(stream_input), "--output", str(stream_output),
            "--device", "cpu", "--allow-gaps", "--min-valid-fraction", "0.8",
        ], root)
        outputs = [json.loads(line) for line in stream_output.read_text(encoding="utf-8").splitlines()]
        predictions = [x for x in outputs if x.get("type") == "prediction" and x.get("window") == 10]
        assert predictions, outputs
        print("PASS: SSIF v3 end-to-end smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
