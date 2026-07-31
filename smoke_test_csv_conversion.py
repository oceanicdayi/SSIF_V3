# -*- coding: utf-8 -*-
"""End-to-end smoke test for the combined_data.csv event-row converter."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def event_row(index: int) -> dict[str, str]:
    origin = f"2024-04-0{index}T00:00:19"
    sequence_a = [1] * 120
    sequence_b = [2] * 120
    sequence_b[15] = min(9, 4 + index)
    if index == 2:
        sequence_a[10:13] = [-99, None, 99]
    return {
        "times": repr([f"{origin}_{second:03d}" for second in range(120)]),
        "stids": repr({"A001": {"city": "臺北市"}, "A002": {"city": "花蓮縣"}}),
        "intensity": repr({"A001": sequence_a, "A002": sequence_b}),
        "epicenter_distance": repr({"A001": 100 + index, "A002": 20 + index}),
        "variables": repr(["intensity", "epicenter_distance"]),
        "eq_info": repr({
            "origin_time": origin,
            "longitude": 121.7,
            "latitude": 23.8,
            "depth": 10.0,
            "magnitude": 5.0 + index / 10,
            "isnumber": f"E{index:03d}",
        }),
        "source_file": f"/archive/event_{index}.json",
    }


def run() -> None:
    repo = Path(__file__).resolve().parent
    converter = repo / "csv_to_ssif_json.py"
    with tempfile.TemporaryDirectory(prefix="ssif_csv_smoke_") as temp:
        root = Path(temp)
        csv_path = root / "combined_data.csv"
        output_dir = root / "events"
        pd.DataFrame([event_row(1), event_row(2)]).to_csv(csv_path, index=False)

        inspected = subprocess.run(
            [sys.executable, str(converter), "inspect", "--csv", str(csv_path), "--rows", "2"],
            text=True,
            capture_output=True,
            check=True,
        )
        schema = json.loads(inspected.stdout)
        assert schema["detected"]["layout"] == "event_json"

        converted = subprocess.run(
            [
                sys.executable, str(converter), "convert",
                "--csv", str(csv_path),
                "--output-dir", str(output_dir),
                "--layout", "auto",
                "--horizon", "120",
                "--chunk-size", "1",
                "--overwrite",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        summary = json.loads(converted.stdout)
        assert summary["n_events"] == 2
        assert summary["n_station_records"] == 4

        validated = subprocess.run(
            [
                sys.executable, str(converter), "validate",
                "--data-dir", str(output_dir),
                "--horizon", "120",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        validation = json.loads(validated.stdout)
        assert validation["n_event_json"] == 2
        assert validation["n_station_records"] == 4
        assert validation["n_errors"] == 0

        event_files = sorted(output_dir.glob("event_*.json"))
        assert len(event_files) == 2
        second = json.loads(event_files[1].read_text(encoding="utf-8"))
        assert second["intensity"]["A001"][10:13] == [-99.0, -99.0, -99.0]
        assert len(second["intensity"]["A002"]) == 120

    print("PASS: combined_data.csv event-row conversion smoke test")


if __name__ == "__main__":
    run()
