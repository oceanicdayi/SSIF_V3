# -*- coding: utf-8 -*-
"""Static validation for the dedicated SSIF v3 training Colab notebook."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "notebooks" / "SSIF_V3_Model_Training_ZH_TW.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    cells = notebook.get("cells")
    assert isinstance(cells, list) and cells

    code_sources: list[str] = []
    for index, cell in enumerate(cells):
        assert cell.get("cell_type") in {"markdown", "code", "raw"}
        source = cell.get("source", [])
        assert isinstance(source, list)
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == [], f"cell {index} contains saved output"
        source_text = "".join(source)
        compile(source_text, f"{NOTEBOOK.name}:cell-{index}", "exec")
        code_sources.append(source_text)

    combined = "\n".join(code_sources)
    required_fragments = [
        "os.chdir('/content')",
        "prepare_ssif_dataset.py",
        "train_ssif_v3.py",
        "load_station_records",
        "TRAIN_DATA.rglob('*.json')",
        "datetime.now(timezone.utc)",
        "RUN_QUICK_TRAIN = True",
        "RUN_FULL_TRAIN = False",
        "RUN_EXTERNAL_EVALUATION = False",
        "split_manifest.json",
        "validation",
        "calibration",
        "data_fingerprint_sha256",
        "checkpoint_audit",
        "--cohort",
        "common",
        "--window-seed-mode",
        "same",
        "--min-precision",
        "best.pt",
        "summary.json",
        "run_inventory.json",
    ]
    for fragment in required_fragments:
        assert fragment in combined, f"missing training safeguard: {fragment}"

    assert "WINDOWS = [10,15,20,25,30,35,40]" in combined
    assert "LABEL_HORIZON = 120" in combined
    assert "REBUILD_SPLIT = False" in combined
    assert "OVERWRITE_FULL_MODEL = False" in combined
    assert "datetime.utcnow()" not in combined
    assert "validation['counters'].get('event_json'" not in combined
    assert "combined_csv_to_ssif_json.py','validate'" not in combined

    print(
        f"PASS: {NOTEBOOK.name} contains {len(cells)} cells; "
        "all code cells compile and training-loader safeguards are present"
    )


if __name__ == "__main__":
    main()
