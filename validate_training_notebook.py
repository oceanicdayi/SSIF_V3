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
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "markdown"
    )
    required_fragments = [
        "os.chdir('/content')",
        "prepare_ssif_dataset.py",
        "train_ssif_v3.py",
        "load_station_records",
        "STAGED_ARCHIVE",
        "list_top_level_json",
        "datetime.now(timezone.utc)",
        "PARALLEL_EW_JOBS = 1",
        "prepare_formal_training",
        "run_formal_window",
        "finalize_formal_training",
        "inspect_window_completion",
        "FORCE_RETRAIN_WINDOWS = set()",
        "run_formal_window(10)",
        "run_formal_window(15)",
        "run_formal_window(20)",
        "run_formal_window(25)",
        "run_formal_window(30)",
        "run_formal_window(35)",
        "run_formal_window(40)",
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
        "per_window_cells",
        "skip_completed_windows",
        "不重複訓練",
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
    assert "## 8. 正式訓練 EW10–EW40" in markdown
    assert "### 8.1 正式訓練 EW10" in markdown
    assert "### 8.8 彙整正式訓練結果" in markdown
    for title in (
        "### 8.2 正式訓練 EW15",
        "### 8.3 正式訓練 EW20",
        "### 8.4 正式訓練 EW25",
        "### 8.5 正式訓練 EW30",
        "### 8.6 正式訓練 EW35",
        "### 8.7 正式訓練 EW40",
    ):
        assert title in markdown, f"missing markdown section: {title}"

    print(
        f"PASS: {NOTEBOOK.name} contains {len(cells)} cells; "
        "all code cells compile and per-window formal-training cells are present"
    )


if __name__ == "__main__":
    main()
