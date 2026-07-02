import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


data_init = """from osn_gs.data.colmap_scene import load_colmap_scene
from osn_gs.data.cameras import Camera, identity_camera
from osn_gs.data.scene_loader import ImageBatch, Scene
from osn_gs.data.torch_scene import TorchImageBatch, TorchScene

__all__ = [
    "Camera",
    "ImageBatch",
    "Scene",
    "TorchImageBatch",
    "TorchScene",
    "identity_camera",
    "load_colmap_scene",
]
"""

scene_loader = """from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osn_gs.data.cameras import Camera
from osn_gs.gaussian.certain_gaussians import CertainGaussianSet


@dataclass
class ImageBatch:
    cameras: list[Camera]
    images: np.ndarray


@dataclass
class Scene:
    initial_gaussians: CertainGaussianSet
    cameras: list[Camera]
    images: np.ndarray

    def sample_views(self, count: int = 1) -> ImageBatch:
        count = max(1, min(count, len(self.cameras)))
        return ImageBatch(cameras=self.cameras[:count], images=self.images[:count])
"""

write(PROJECT_ROOT / "osn_gs" / "data" / "__init__.py", data_init)
write(PROJECT_ROOT / "osn_gs" / "data" / "scene_loader.py", scene_loader)

notebook_path = PROJECT_ROOT / "colab_train_3dgs.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))

    if "if FRAMEWORK_MODE == 'osn_gs' and OSN_USE_SYNTHETIC:" in source and "OSN_SCENE_NPZ" in source:
        source = (
            "required_paths = [\n"
            "    DATA_ROOT / 'images',\n"
            "    DATA_ROOT / 'sparse',\n"
            "]\n"
            "missing = [str(path) for path in required_paths if not path.exists()]\n"
            "if missing:\n"
            "    print('Expected DATA_ROOT layout:')\n"
            "    print('  DATA_ROOT/images')\n"
            "    print('  DATA_ROOT/sparse')\n"
            "    print('\\nFix one of these:')\n"
            "    print('  1. Put your dataset in NOTEBOOK_ROOT/DATASET.')\n"
            "    print('  2. Set DATA_ROOT to the actual dataset folder.')\n"
            "    print('  3. If running in Colab, upload a dataset zip or mount Drive.')\n"
            "    raise FileNotFoundError('Dataset is not ready. Missing: ' + ', '.join(missing))\n"
            "\n"
            "MODEL_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "print('Dataset looks ready.')\n"
        )
        cell["source"] = source.splitlines(keepends=True)

notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("cleaned")
