from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    notebook_path = Path("colab_train_3dgs.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    markers = ("FRAMEWORK_MODE", "TRAIN_SCRIPT", "GS_ROOT", "CUDA_EXTENSIONS", "normalize_extension", "Running OSN-GS")
    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        hits = [marker for marker in markers if marker in source]
        if not hits:
            continue
        first_line = next((line for line in source.splitlines() if line.strip()), "")
        print(f"CELL {index} {cell.get('cell_type')} hits={hits}")
        print(first_line[:160])
        print(source[:2500])
        print("---")


if __name__ == "__main__":
    main()
