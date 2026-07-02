import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PATH = r"C:\Users\dna10\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

notebook_path = PROJECT_ROOT / "colab_train_3dgs.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
metadata = notebook.setdefault("metadata", {})
metadata["kernelspec"] = {
    "display_name": "Python 3.12.13",
    "language": "python",
    "name": "python3",
}
metadata["language_info"] = {
    "name": "python",
    "version": "3.12.13",
}
notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

vscode_dir = PROJECT_ROOT / ".vscode"
vscode_dir.mkdir(exist_ok=True)
(vscode_dir / "settings.json").write_text(
    json.dumps({"python.defaultInterpreterPath": PYTHON_PATH}, indent=2) + "\n",
    encoding="utf-8",
)

print("updated")
