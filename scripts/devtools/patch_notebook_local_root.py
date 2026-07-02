import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "colab_train_3dgs.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Expected snippet not found:\n{old}")
    return source.replace(old, new, 1)


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))

    if "USE_DRIVE = False" in source and "FRAMEWORK_MODE = 'osn_gs'" in source:
        source = replace_once(
            source,
            "USE_DRIVE = False\n\nif USE_DRIVE:\n    from google.colab import drive\n    drive.mount('/content/drive')\n",
            "USE_DRIVE = False\n\ntry:\n    import google.colab  # type: ignore\n    IS_COLAB = True\nexcept ImportError:\n    IS_COLAB = False\n\nNOTEBOOK_ROOT = Path.cwd().resolve()\n\nif USE_DRIVE:\n    if not IS_COLAB:\n        raise RuntimeError('Google Drive mount is only available in Colab.')\n    from google.colab import drive\n    drive.mount('/content/drive')\n",
        )

        source = replace_once(
            source,
            "PROJECT_ZIP = None  # Optional: Path('/content/OSN-GS.zip')\nPROJECT_EXTRACT_ROOT = Path('/content/project_src')\n",
            "PROJECT_ZIP = None  # Optional: Path('/content/OSN-GS.zip')\nPROJECT_EXTRACT_ROOT = Path('/content/project_src') if IS_COLAB else NOTEBOOK_ROOT / '_project_src'\n",
        )

        source = replace_once(
            source,
            "if GS_ROOT is None or not GS_ROOT.exists():\n    if PROJECT_ZIP is not None:\n        GS_ROOT = extract_project_zip(PROJECT_ZIP)\n    else:\n        candidates = find_project_roots([Path('/content'), Path('/content/drive/MyDrive')])\n        # Prefer a folder named OSN-GS when the notebook is in OSN-GS mode.\n        if FRAMEWORK_MODE == 'osn_gs':\n            candidates = sorted(candidates, key=lambda path: 0 if path.name == 'OSN-GS' else 1)\n        if candidates:\n            GS_ROOT = candidates[0]\n        else:\n            from google.colab import files\n            print(f'Upload a project zip now. It should contain {TRAIN_SCRIPT} somewhere inside.')\n            uploaded = files.upload()\n            zip_names = [name for name in uploaded.keys() if name.lower().endswith('.zip')]\n            if not zip_names:\n                raise FileNotFoundError('No project .zip file was uploaded.')\n            GS_ROOT = extract_project_zip(Path('/content') / zip_names[0])\n",
            "if GS_ROOT is None or not GS_ROOT.exists():\n    if not IS_COLAB:\n        GS_ROOT = NOTEBOOK_ROOT\n    elif PROJECT_ZIP is not None:\n        GS_ROOT = extract_project_zip(PROJECT_ZIP)\n    else:\n        candidates = find_project_roots([Path('/content'), Path('/content/drive/MyDrive')])\n        # Prefer a folder named OSN-GS when the notebook is in OSN-GS mode.\n        if FRAMEWORK_MODE == 'osn_gs':\n            candidates = sorted(candidates, key=lambda path: 0 if path.name == 'OSN-GS' else 1)\n        if candidates:\n            GS_ROOT = candidates[0]\n        else:\n            from google.colab import files\n            print(f'Upload a project zip now. It should contain {TRAIN_SCRIPT} somewhere inside.')\n            uploaded = files.upload()\n            zip_names = [name for name in uploaded.keys() if name.lower().endswith('.zip')]\n            if not zip_names:\n                raise FileNotFoundError('No project .zip file was uploaded.')\n            GS_ROOT = extract_project_zip(Path('/content') / zip_names[0])\n",
        )

        source = replace_once(
            source,
            "print('FRAMEWORK_MODE:', FRAMEWORK_MODE)\nprint('GS_ROOT:', GS_ROOT)\nprint('train script exists:', (GS_ROOT / TRAIN_SCRIPT).exists())\nprint('submodules exists:', (GS_ROOT / 'submodules').exists())\n",
            "print('IS_COLAB:', IS_COLAB)\nprint('NOTEBOOK_ROOT:', NOTEBOOK_ROOT)\nprint('FRAMEWORK_MODE:', FRAMEWORK_MODE)\nprint('GS_ROOT:', GS_ROOT)\nprint('train script exists:', (GS_ROOT / TRAIN_SCRIPT).exists())\nprint('submodules exists:', (GS_ROOT / 'submodules').exists())\n",
        )

        cell["source"] = source.splitlines(keepends=True)
        break

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("patched")
