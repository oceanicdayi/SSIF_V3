# -*- coding: utf-8 -*-
"""Static validation for the SSIF_V3 Colab tutorial notebook."""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "notebooks" / "SSIF_V3_Colab_Tutorial_ZH_TW.ipynb"


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

        outputs = cell.get("outputs", [])
        assert outputs == [], f"cell {index} contains saved outputs or stale errors"
        assert cell.get("execution_count") is None

        source_text = "".join(source)
        compile(source_text, f"{NOTEBOOK.name}:cell-{index}", "exec")
        code_sources.append(source_text)

    combined = "\n".join(code_sources)
    assert "os.chdir('/content')" in combined, (
        "repository sync must leave /content/SSIF_V3 before cleanup"
    )
    assert "shutil.rmtree(REPO_ROOT" in combined
    assert "git', 'clone', '--depth', '1'" in combined
    assert "rm', '-rf', str(REPO_ROOT)" not in combined, (
        "do not delete a possible current working directory through rm -rf"
    )
    assert "combined_csv_to_ssif_json.py" in combined
    assert "smoke_test_combined_csv_conversion.py" in combined
    assert "RUN_FULL_SCAN = False" in combined
    assert "RUN_FULL_CONVERSION = False" in combined

    print(
        f"PASS: {NOTEBOOK.name} contains {len(cells)} cells; "
        f"all code cells compile and no saved errors are present"
    )


if __name__ == "__main__":
    main()
