# -*- coding: utf-8 -*-
"""End-to-end smoke test for combined_csv_to_ssif_json.py."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def make_row(index: int) -> dict[str, str]:
    origin = f"2024-04-0{index}T00:00:19"
    seq_a = [1] * 120
    seq_b = [2] * 119 + [4]
    return {
        "times": repr([f"{origin}_{second:03d}" for second in range(120)]),
        "stids": repr({
            "A001": {"city": "臺北市", "elev": float("nan")},
            "A002": {"city": "花蓮縣", "elev": 3.0},
        }),
        "intensity": repr({"A001": seq_a, "A002": seq_b}),
        "epicenter_distance": repr({"A001": 100.0 + index, "A002": 20.0 + index}),
        "variables": repr(["intensity", "epicenter_distance"]),
        "eq_info": repr({
            "origin_time": origin,
            "longitude": 121.7,
            "latitude": 23.8,
            "depth": 10.0,
            "magnitude": 5.0 + index / 10,
        }),
        "source_file": f"/archive/event_{index}.json",
    }


def run_command(script: Path, arguments: list[str]) -> dict:
    completed = subprocess.run(
        [sys.executable, str(script), *arguments],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    repo = Path(__file__).resolve().parent
    script = repo / "combined_csv_to_ssif_json.py"

    with tempfile.TemporaryDirectory(prefix="ssif_combined_csv_") as temp_dir:
        root = Path(temp_dir)
        csv_path = root / "combined_data.csv"
        output_dir = root / "events"

        rows = [make_row(1), make_row(2), make_row(3)]
        duplicate = rows[1].copy()
        duplicate["source_file"] = "/archive/copied/event_2.json"
        rows.append(duplicate)  # same scientific content, different archive path
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        inspected = run_command(script, [
            "inspect", "--csv", str(csv_path), "--rows", "2",
        ])
        assert inspected["layout"] == "event_json"
        assert len(inspected["sample_rows"]) == 2

        scanned = run_command(script, ["scan", "--csv", str(csv_path)])
        assert scanned["counters"]["rows_read"] == 4
        assert scanned["counters"]["exact_duplicate_rows"] == 1
        assert scanned["counters"].get("row_errors", 0) == 0

        converted = run_command(script, [
            "convert",
            "--csv", str(csv_path),
            "--output-dir", str(output_dir),
            "--overwrite",
            "--duplicate-policy", "skip-identical",
        ])
        assert converted["n_events"] == 3
        assert converted["counters"]["rows_skipped_duplicate"] == 1

        validated = run_command(script, [
            "validate", "--data-dir", str(output_dir),
        ])
        assert validated["counters"]["event_json"] == 3
        assert validated["counters"].get("errors", 0) == 0

        first_event = json.loads(sorted(output_dir.glob("event_*.json"))[0].read_text(encoding="utf-8"))
        assert len(first_event["intensity"]["A001"]) == 120
        assert first_event["stids"]["A001"]["elev"] is None

        # Nested batch folders (第一批/第二批/...) must also be discovered.
        nested_root = root / "nested_batches"
        batch_a = nested_root / "第一批"
        batch_b = nested_root / "第二批"
        batch_a.mkdir(parents=True)
        batch_b.mkdir(parents=True)
        for index, source in enumerate(sorted(output_dir.glob("event_*.json")), start=1):
            target_dir = batch_a if index % 2 else batch_b
            target = target_dir / f"2024040{index}_000019.json"
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        (nested_root / "conversion_summary.json").write_text("{}", encoding="utf-8")

        nested_validated = run_command(script, [
            "validate", "--data-dir", str(nested_root),
        ])
        assert nested_validated["counters"]["event_json"] == 3
        assert nested_validated["counters"].get("errors", 0) == 0

    print("PASS: combined CSV streaming conversion smoke test")


if __name__ == "__main__":
    main()
