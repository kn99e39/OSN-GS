import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


train_py = """from __future__ import annotations

\"\"\"Notebook-compatible OSN-GS training entrypoint.

The renderer notebook discovers a project by searching for `train.py`.
This wrapper keeps that workflow intact while delegating the real work to
`TorchOSNGSTrainer`.
\"\"\"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from osn_gs.core.torch_pipeline import TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.colmap_scene import load_colmap_scene
from osn_gs.interop.colab_args import build_osn_gs_train_parser, output_dir_from_args, save_interval_from_args
from osn_gs.render.cuda_rasterizer_adapter import RasterizerPipelineOptions
from osn_gs.utils.torch_ops import default_device


def main() -> None:
    args = build_osn_gs_train_parser().parse_args()
    device = args.device or default_device(prefer_cuda=True)
    output_dir = output_dir_from_args(args)
    save_interval = save_interval_from_args(args)

    pipeline_config = TorchPipelineConfig(
        base_curve_count=args.base_curve_count,
        uncertain_samples_u=args.uncertain_samples_u,
        uncertain_samples_v=args.uncertain_samples_v,
    )
    training_config = TorchTrainingConfig(
        iterations=args.iterations,
        surface_rebuild_interval=args.surface_rebuild_interval,
        density_control_interval=args.density_control_interval,
        save_interval=save_interval,
        prefer_cuda=device == "cuda",
    )
    rasterizer_options = RasterizerPipelineOptions(prefer_cuda_rasterizer=not args.disable_cuda_rasterizer)
    trainer = TorchOSNGSTrainer(
        pipeline_config=pipeline_config,
        training_config=training_config,
        rasterizer_options=rasterizer_options,
        device=device,
    )

    if not args.source_path:
        raise ValueError("OSN-GS requires --source_path/-s pointing to a COLMAP dataset root.")

    scene = load_colmap_scene(
        args.source_path,
        device=device,
        image_dir_name=args.images,
        sparse_dir_name=args.sparse_dir,
        image_downscale=args.image_downscale,
        max_images=args.max_images,
    )

    result = trainer.train(scene, output_dir)
    print(
        "OSN-GS train.py complete: "
        f"iteration={result.state.iteration}, "
        f"loss={result.state.last_loss:.6f}, "
        f"psnr={result.state.last_psnr:.3f}, "
        f"gaussians={len(result.state.model)}, "
        f"uncertain={int(result.state.model.is_uncertain.sum().item())}, "
        f"output={result.output_dir}"
    )


if __name__ == "__main__":
    main()
"""

colab_args = """from __future__ import annotations

\"\"\"Argument helpers for Colab/notebook entrypoints.

The `3DGS_Renderer/colab_train_3dgs.ipynb` notebook was originally written
around Graphdeco-style `train.py` arguments. OSN-GS has its own Torch trainer,
so this module translates the shared subset of notebook arguments into OSN-GS
configuration without forcing the notebook to know every internal class.
\"\"\"

import argparse
from pathlib import Path


def build_osn_gs_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train OSN-GS from notebook-compatible arguments.")

    # Graphdeco-compatible aliases. `source_path` points to a COLMAP/3DGS scene.
    parser.add_argument("-s", "--source_path", default="", help="COLMAP scene root with images/ and sparse/.")
    parser.add_argument("-m", "--model_path", default="outputs/osn_gs", help="Output directory.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--densify_until_iter", type=int, default=0)
    parser.add_argument("--densification_interval", type=int, default=0)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0)

    # Native OSN-GS options.
    parser.add_argument("--images", type=str, default="images", help="Image folder name under --source_path.")
    parser.add_argument("--sparse_dir", type=str, default="sparse/0", help="Sparse COLMAP folder under --source_path.")
    parser.add_argument("--image_downscale", type=int, default=1, help="Integer image downscale for COLMAP loading.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit loaded COLMAP images; 0 means all.")
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument("--base_curve_count", type=int, default=8)
    parser.add_argument("--uncertain_samples_u", type=int, default=16)
    parser.add_argument("--uncertain_samples_v", type=int, default=3)
    parser.add_argument("--surface_rebuild_interval", type=int, default=1000)
    parser.add_argument("--density_control_interval", type=int, default=500)
    parser.add_argument("--disable_cuda_rasterizer", action="store_true")
    return parser


def save_interval_from_args(args: argparse.Namespace) -> int:
    \"\"\"Choose a single periodic save interval from Graphdeco-style save iterations.\"\"\"

    save_iterations = sorted({int(value) for value in args.save_iterations if int(value) > 0})
    if save_iterations:
        return max(1, save_iterations[0])
    return max(1, int(args.iterations))


def output_dir_from_args(args: argparse.Namespace) -> Path:
    return Path(args.model_path)
"""

train_torch = """from __future__ import annotations

\"\"\"OSN-GS Torch training CLI.\"\"\"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osn_gs.core.torch_pipeline import TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.colmap_scene import load_colmap_scene
from osn_gs.render.cuda_rasterizer_adapter import RasterizerPipelineOptions
from osn_gs.utils.torch_ops import default_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the OSN-GS torch framework.")
    parser.add_argument("-s", "--source_path", type=str, default="", help="COLMAP scene root with images/ and sparse/.")
    parser.add_argument("--images", type=str, default="images", help="Image folder name under --source_path.")
    parser.add_argument("--sparse_dir", type=str, default="sparse/0", help="Sparse COLMAP folder under --source_path.")
    parser.add_argument("--image_downscale", type=int, default=1, help="Integer image downscale for COLMAP loading.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit loaded COLMAP images; 0 means all.")
    parser.add_argument("--output", type=str, default="outputs/osn_gs_torch", help="Output directory.")
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--base_curve_count", type=int, default=8)
    parser.add_argument("--uncertain_samples_u", type=int, default=16)
    parser.add_argument("--uncertain_samples_v", type=int, default=3)
    parser.add_argument("--surface_rebuild_interval", type=int, default=1000)
    parser.add_argument("--density_control_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--disable_cuda_rasterizer", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or default_device(prefer_cuda=True)

    pipeline_config = TorchPipelineConfig(
        base_curve_count=args.base_curve_count,
        uncertain_samples_u=args.uncertain_samples_u,
        uncertain_samples_v=args.uncertain_samples_v,
    )
    training_config = TorchTrainingConfig(
        iterations=args.iterations,
        surface_rebuild_interval=args.surface_rebuild_interval,
        density_control_interval=args.density_control_interval,
        save_interval=args.save_interval,
        prefer_cuda=device == "cuda",
    )

    rasterizer_options = RasterizerPipelineOptions(prefer_cuda_rasterizer=not args.disable_cuda_rasterizer)
    trainer = TorchOSNGSTrainer(
        pipeline_config=pipeline_config,
        training_config=training_config,
        rasterizer_options=rasterizer_options,
        device=device,
    )

    if not args.source_path:
        raise ValueError("OSN-GS requires --source_path/-s pointing to a COLMAP dataset root.")

    scene = load_colmap_scene(
        args.source_path,
        device=device,
        image_dir_name=args.images,
        sparse_dir_name=args.sparse_dir,
        image_downscale=args.image_downscale,
        max_images=args.max_images,
    )

    result = trainer.train(scene, args.output)
    print(
        "OSN-GS torch training complete: "
        f"iteration={result.state.iteration}, "
        f"loss={result.state.last_loss:.6f}, "
        f"psnr={result.state.last_psnr:.3f}, "
        f"gaussians={len(result.state.model)}, "
        f"uncertain={int(result.state.model.is_uncertain.sum().item())}, "
        f"output={result.output_dir}"
    )


if __name__ == "__main__":
    main()
"""

torch_scene = """from __future__ import annotations

\"\"\"Torch training scene helpers.\"\"\"

from dataclasses import dataclass
from typing import Any

from osn_gs.render.torch_fallback import TorchCamera
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchImageBatch:
    \"\"\"trainer에 전달되는 camera/image batch.\"\"\"

    cameras: list[TorchCamera]
    # Shape: (B, 3, H, W).
    images: Any


@dataclass
class TorchScene:
    \"\"\"OSN-GS Torch trainer가 요구하는 최소 scene protocol.\"\"\"

    initial_points: Any
    initial_colors: Any
    cameras: list[TorchCamera]
    images: Any
    extent: float = 1.0

    def sample_views(self, iteration: int, batch_size: int = 1) -> TorchImageBatch:
        \"\"\"Sample a deterministic batch of views for training.\"\"\"

        torch = require_torch()
        count = len(self.cameras)
        if count == 0:
            raise ValueError("TorchScene requires at least one camera.")
        indices = [(iteration + offset) % count for offset in range(batch_size)]
        image_indices = torch.as_tensor(indices, dtype=torch.long, device=self.images.device)
        return TorchImageBatch(cameras=[self.cameras[idx] for idx in indices], images=self.images[image_indices])
"""

readme = """# OSN-GS

OSN-GS is an experimental 3D Gaussian Splatting framework that predicts occluded surface structure from observed Gaussian geometry. It fits base curves on observed Gaussian centers, extrapolates occlusion curves, builds a NURBS-like parametric surface, and places uncertain Gaussians on the inferred surface.

## Current Training Path

The active training implementation is the Torch path:

```bash
python scripts/train_osn_gs_torch.py \\
  -s /path/to/scene_root \\
  --device cuda \\
  --iterations 30000 \\
  --output outputs/osn_gs_run
```

The project also exposes a notebook-compatible wrapper:

```bash
python train.py -s /path/to/scene_root -m outputs/osn_gs_run --iterations 30000
```

To train from a COLMAP/Graphdeco-style dataset:

```bash
python train.py \\
  -s /path/to/scene_root \\
  -m outputs/osn_gs_colmap \\
  --iterations 30000 \\
  --image_downscale 2
```

If `diff_gaussian_rasterization` is installed, OSN-GS uses it automatically. Otherwise it uses a differentiable fallback renderer that is useful for debugging the OSN-GS pipeline, but not for final 3DGS-quality results.

## Colab Notebook

`../3DGS_Renderer/colab_train_3dgs.ipynb` now has an `osn_gs` framework mode. In the project setup cell, keep:

```python
FRAMEWORK_MODE = 'osn_gs'
```

The notebook will discover an uploaded `OSN-GS` project zip by its top-level `train.py`, discover/upload a COLMAP-style dataset, and pass that dataset to OSN-GS through `train.py -s DATA_ROOT`.

## Inputs

OSN-GS now supports a COLMAP/Graphdeco-style scene directly:

```text
scene_root/
  images/
  sparse/0/cameras.bin
  sparse/0/images.bin
  sparse/0/points3D.bin
```

Text exports are also accepted:

```text
scene_root/
  images/
  sparse/0/cameras.txt
  sparse/0/images.txt
  sparse/0/points3D.txt
```

## Outputs

Each save directory contains:

- `point_cloud.ply`: trained Gaussian cloud with an `uncertain` vertex property
- `render.ppm`: rendered image preview
- `checkpoint.pt`: Torch checkpoint
- `metrics.txt`: iteration, loss, PSNR, Gaussian counts, and rasterizer backend flag

## Main Modules

- `osn_gs/core/torch_pipeline.py`: observed curve fitting, occlusion curve prediction, surface construction, uncertain Gaussian initialization
- `osn_gs/core/torch_trainer.py`: differentiable training loop and output saving
- `osn_gs/gaussian/torch_model.py`: 3DGS-style Torch Gaussian parameter container
- `osn_gs/gaussian/torch_density_control.py`: uncertain pruning and promotion policy
- `osn_gs/surface/torch_nurbs.py`: Torch NURBS-like surface representation
- `osn_gs/render/cuda_rasterizer_adapter.py`: CUDA rasterizer bridge with fallback renderer
- `osn_gs/losses/torch_losses.py`: image, surface, uncertainty, and anchor losses

## CUDA Dependencies

For full 3DGS-quality training, install the standard 3DGS CUDA submodules in the Python environment:

- `diff_gaussian_rasterization`
- `simple_knn`

The workspace already contains a reference `gaussian-splatting` checkout, so those submodules can be installed from there on the target Linux/CUDA machine.
"""

write(PROJECT_ROOT / "train.py", train_py)
write(PROJECT_ROOT / "osn_gs" / "interop" / "colab_args.py", colab_args)
write(PROJECT_ROOT / "scripts" / "train_osn_gs_torch.py", train_torch)
write(PROJECT_ROOT / "osn_gs" / "data" / "torch_scene.py", torch_scene)
write(PROJECT_ROOT / "README.md", readme)

notebook_path = PROJECT_ROOT / "colab_train_3dgs.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))

    if "DATASET_DRIVE_URL =" in source and "DATA_ROOT = Path('/content/data/scene')" in source:
        source = source.replace(
            "# Upload, download, or auto-detect a COLMAP-style dataset zip.\n"
            "# OSN-GS can now consume the same DATA_ROOT layout through its `train.py -s` path.\n"
            "# Set OSN_USE_SYNTHETIC = True only when you want to skip real data and run the built-in smoke scene.\n"
            "OSN_USE_SYNTHETIC = False if FRAMEWORK_MODE == 'osn_gs' else False\n"
            "OSN_SCENE_NPZ = None  # Example: Path('/content/data/osn_scene.npz')\n"
            "\n"
            "# Prefer DATASET_DRIVE_URL for large datasets. Leave it empty to use file upload.\n"
            "DATASET_DRIVE_URL = 'https://drive.google.com/file/d/1lIEfFofR2RAA2rh3lNs3cOM2tU21COeY/view?usp=share_link'  # Example: 'https://drive.google.com/file/d/FILE_ID/view?usp=sharing'\n"
            "UPLOAD_DATASET_ZIP = True\n"
            "DATA_UNZIP_DIR = Path('/content/data')\n"
            "DOWNLOADED_DATASET_ZIP = Path('/content/dataset_from_drive.zip')\n"
            "DATA_ROOT = Path('/content/data/scene')\n"
            "MODEL_ROOT = Path('/content/output/osn_gs_scene' if FRAMEWORK_MODE == 'osn_gs' else '/content/output/scene')\n"
            "\n"
            "# Drive example, if you prefer persistent storage:\n"
            "# DATA_ROOT = Path('/content/drive/MyDrive/datasets/my_scene')\n"
            "# MODEL_ROOT = Path('/content/drive/MyDrive/3dgs_outputs/my_scene')\n",
            "# Upload, download, or auto-detect a COLMAP-style dataset zip.\n"
            "# OSN-GS consumes the same DATA_ROOT layout through `train.py -s DATA_ROOT`.\n"
            "LOCAL_DATASET_ROOT = NOTEBOOK_ROOT / 'DATASET'\n"
            "\n"
            "# Prefer DATASET_DRIVE_URL for large datasets. Leave it empty to use file upload.\n"
            "DATASET_DRIVE_URL = 'https://drive.google.com/file/d/1lIEfFofR2RAA2rh3lNs3cOM2tU21COeY/view?usp=share_link'  # Example: 'https://drive.google.com/file/d/FILE_ID/view?usp=sharing'\n"
            "UPLOAD_DATASET_ZIP = IS_COLAB\n"
            "DATA_UNZIP_DIR = Path('/content/data') if IS_COLAB else LOCAL_DATASET_ROOT\n"
            "DOWNLOADED_DATASET_ZIP = Path('/content/dataset_from_drive.zip') if IS_COLAB else LOCAL_DATASET_ROOT / 'dataset_from_drive.zip'\n"
            "DATA_ROOT = Path('/content/data/scene') if IS_COLAB else LOCAL_DATASET_ROOT\n"
            "MODEL_ROOT = Path('/content/output/osn_gs_scene' if FRAMEWORK_MODE == 'osn_gs' else '/content/output/scene') if IS_COLAB else NOTEBOOK_ROOT / 'output' / ('osn_gs_scene' if FRAMEWORK_MODE == 'osn_gs' else 'scene')\n"
            "\n"
            "# Drive example, if you prefer persistent storage:\n"
            "# DATA_ROOT = Path('/content/drive/MyDrive/datasets/my_scene')\n"
            "# MODEL_ROOT = Path('/content/drive/MyDrive/3dgs_outputs/my_scene')\n"
            "# Local example: NOTEBOOK_ROOT / 'DATASET'\n",
        )

        source = source.replace(
            "if FRAMEWORK_MODE == 'osn_gs' and OSN_SCENE_NPZ is not None:\n"
            "    OSN_USE_SYNTHETIC = False\n"
            "    print('OSN-GS mode: using NPZ scene:', OSN_SCENE_NPZ)\n"
            "elif FRAMEWORK_MODE == 'osn_gs' and OSN_USE_SYNTHETIC:\n"
            "    print('OSN-GS mode: using built-in synthetic scene; dataset discovery skipped.')\n"
            "else:\n"
            "    search_roots = [\n"
            "        DATA_ROOT,\n"
            "        GS_ROOT,\n"
            "        DATA_UNZIP_DIR,\n"
            "        Path('/content'),\n"
            "        Path('/content/drive/MyDrive'),\n"
            "    ]\n"
            "\n"
            "    candidates = find_3dgs_datasets(search_roots)\n"
            "    if candidates:\n"
            "        DATA_ROOT = candidates[0]\n"
            "    elif DATASET_DRIVE_URL.strip():\n"
            "        print('Downloading dataset from Google Drive...')\n"
            "        gdown.download(DATASET_DRIVE_URL.strip(), str(DOWNLOADED_DATASET_ZIP), quiet=False, fuzzy=True)\n"
            "        DATA_ROOT = extract_dataset_zip(DOWNLOADED_DATASET_ZIP)\n"
            "    else:\n"
            "        zip_candidates = find_dataset_zips([GS_ROOT, Path('/content'), Path('/content/drive/MyDrive')])\n"
            "        valid_zip = None\n"
            "        for zip_path in zip_candidates:\n"
            "            if zipfile.is_zipfile(zip_path):\n"
            "                valid_zip = zip_path\n"
            "                break\n"
            "            print('Skipping invalid or incomplete zip:', zip_path)\n"
            "\n"
            "        if valid_zip is not None:\n"
            "            DATA_ROOT = extract_dataset_zip(valid_zip)\n"
            "\n"
            "    if UPLOAD_DATASET_ZIP and not has_3dgs_dataset_layout(DATA_ROOT):\n"
            "        from google.colab import files\n"
            "        print('Upload your dataset zip now. It should contain images/ and sparse/ somewhere inside.')\n"
            "        uploaded = files.upload()\n"
            "        zip_names = [name for name in uploaded.keys() if name.lower().endswith('.zip')]\n"
            "        if not zip_names:\n"
            "            raise FileNotFoundError('No .zip file was uploaded.')\n"
            "\n"
            "        uploaded_zip = Path('/content') / zip_names[0]\n"
            "        DATA_ROOT = extract_dataset_zip(uploaded_zip)\n"
            "\n"
            "    if not has_3dgs_dataset_layout(DATA_ROOT):\n"
            "        print('Searched for datasets in:')\n"
            "        for root in search_roots:\n"
            "            print(' ', root)\n"
            "        raise FileNotFoundError('No COLMAP-style dataset found. DATA_ROOT must contain images/ and sparse/.')\n"
            "\n"
            "print('FRAMEWORK_MODE:', FRAMEWORK_MODE)\n"
            "print('DATA_ROOT:', DATA_ROOT)\n"
            "print('MODEL_ROOT:', MODEL_ROOT)\n"
            "print('OSN_USE_SYNTHETIC:', OSN_USE_SYNTHETIC if FRAMEWORK_MODE == 'osn_gs' else '<not used>')\n"
            "print('OSN_SCENE_NPZ:', OSN_SCENE_NPZ if FRAMEWORK_MODE == 'osn_gs' else '<not used>')\n"
            "print('images exists:', (DATA_ROOT / 'images').exists())\n"
            "print('sparse exists:', (DATA_ROOT / 'sparse').exists())\n",
            "search_roots = [DATA_ROOT, GS_ROOT, DATA_UNZIP_DIR]\n"
            "if IS_COLAB:\n"
            "    search_roots += [Path('/content'), Path('/content/drive/MyDrive')]\n"
            "\n"
            "candidates = find_3dgs_datasets(search_roots)\n"
            "if candidates:\n"
            "    if not IS_COLAB:\n"
            "        candidates = sorted(candidates, key=lambda path: 0 if path.resolve() == LOCAL_DATASET_ROOT.resolve() else 1)\n"
            "    DATA_ROOT = candidates[0]\n"
            "elif IS_COLAB and DATASET_DRIVE_URL.strip():\n"
            "    print('Downloading dataset from Google Drive...')\n"
            "    gdown.download(DATASET_DRIVE_URL.strip(), str(DOWNLOADED_DATASET_ZIP), quiet=False, fuzzy=True)\n"
            "    DATA_ROOT = extract_dataset_zip(DOWNLOADED_DATASET_ZIP)\n"
            "else:\n"
            "    zip_search_roots = [GS_ROOT]\n"
            "    if IS_COLAB:\n"
            "        zip_search_roots += [Path('/content'), Path('/content/drive/MyDrive')]\n"
            "    zip_candidates = find_dataset_zips(zip_search_roots)\n"
            "    valid_zip = None\n"
            "    for zip_path in zip_candidates:\n"
            "        if zipfile.is_zipfile(zip_path):\n"
            "            valid_zip = zip_path\n"
            "            break\n"
            "        print('Skipping invalid or incomplete zip:', zip_path)\n"
            "\n"
            "    if valid_zip is not None:\n"
            "        DATA_ROOT = extract_dataset_zip(valid_zip)\n"
            "\n"
            "if UPLOAD_DATASET_ZIP and not has_3dgs_dataset_layout(DATA_ROOT):\n"
            "    from google.colab import files\n"
            "    print('Upload your dataset zip now. It should contain images/ and sparse/ somewhere inside.')\n"
            "    uploaded = files.upload()\n"
            "    zip_names = [name for name in uploaded.keys() if name.lower().endswith('.zip')]\n"
            "    if not zip_names:\n"
            "        raise FileNotFoundError('No .zip file was uploaded.')\n"
            "\n"
            "    uploaded_zip = Path('/content') / zip_names[0]\n"
            "    DATA_ROOT = extract_dataset_zip(uploaded_zip)\n"
            "\n"
            "if not has_3dgs_dataset_layout(DATA_ROOT):\n"
            "    print('Searched for datasets in:')\n"
            "    for root in search_roots:\n"
            "        print(' ', root)\n"
            "    raise FileNotFoundError('No COLMAP-style dataset found. DATA_ROOT must contain images/ and sparse/.')\n"
            "\n"
            "print('FRAMEWORK_MODE:', FRAMEWORK_MODE)\n"
            "print('LOCAL_DATASET_ROOT:', LOCAL_DATASET_ROOT)\n"
            "print('DATA_ROOT:', DATA_ROOT)\n"
            "print('MODEL_ROOT:', MODEL_ROOT)\n"
            "print('images exists:', (DATA_ROOT / 'images').exists())\n"
            "print('sparse exists:', (DATA_ROOT / 'sparse').exists())\n",
        )

        source = source.replace(
            "if FRAMEWORK_MODE == 'osn_gs' and OSN_USE_SYNTHETIC:\n"
            "    MODEL_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "    print('OSN-GS synthetic output folder is ready:', MODEL_ROOT)\n"
            "elif FRAMEWORK_MODE == 'osn_gs' and OSN_SCENE_NPZ is not None:\n"
            "    MODEL_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "    if not Path(OSN_SCENE_NPZ).exists():\n"
            "        raise FileNotFoundError(f'OSN_SCENE_NPZ does not exist: {OSN_SCENE_NPZ}')\n"
            "    print('OSN-GS NPZ output folder is ready:', MODEL_ROOT)\n"
            "else:\n"
            "    required_paths = [\n"
            "        DATA_ROOT / 'images',\n"
            "        DATA_ROOT / 'sparse',\n"
            "    ]\n"
            "    missing = [str(path) for path in required_paths if not path.exists()]\n"
            "    if missing:\n"
            "        print('Expected DATA_ROOT layout:')\n"
            "        print('  DATA_ROOT/images')\n"
            "        print('  DATA_ROOT/sparse')\n"
            "        print('\\nFix one of these:')\n"
            "        print('  1. Set DATA_ROOT to the actual dataset folder.')\n"
            "        print('  2. Set UPLOAD_DATASET_ZIP = True in the previous cell and upload a dataset zip.')\n"
            "        print('  3. Mount Drive and point DATA_ROOT to your Drive dataset path.')\n"
            "        raise FileNotFoundError('Dataset is not ready. Missing: ' + ', '.join(missing))\n"
            "\n"
            "    MODEL_ROOT.mkdir(parents=True, exist_ok=True)\n"
            "    print('Dataset looks ready.')\n",
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
            "print('Dataset looks ready.')\n",
        )

        cell["source"] = source.splitlines(keepends=True)

    if "OSN_POINT_COUNT = 48" in source and "OSN_IMAGE_SIZE = 96" in source:
        source = source.replace(
            "# OSN-GS-specific knobs used only when FRAMEWORK_MODE == 'osn_gs'.\n"
            "OSN_POINT_COUNT = 48\n"
            "OSN_IMAGE_SIZE = 96\n"
            "OSN_BASE_CURVE_COUNT = 8\n",
            "# OSN-GS-specific knobs used only when FRAMEWORK_MODE == 'osn_gs'.\n"
            "OSN_BASE_CURVE_COUNT = 8\n",
        )
        source = source.replace(
            "        '--point_count', str(OSN_POINT_COUNT),\n"
            "        '--image_size', str(OSN_IMAGE_SIZE),\n",
            "",
        )
        source = source.replace(
            "    if OSN_SCENE_NPZ is not None:\n"
            "        cmd += ['--scene_npz', str(OSN_SCENE_NPZ)]\n"
            "    elif not OSN_USE_SYNTHETIC:\n"
            "        cmd += ['-s', str(DATA_ROOT), '--image_downscale', str(OSN_IMAGE_DOWNSCALE), '--max_images', str(OSN_MAX_IMAGES)]\n",
            "    cmd += ['-s', str(DATA_ROOT), '--image_downscale', str(OSN_IMAGE_DOWNSCALE), '--max_images', str(OSN_MAX_IMAGES)]\n",
        )
        source = source.replace(
            "    if OSN_SCENE_NPZ is not None:\n"
            "        print('NPZ scene:', OSN_SCENE_NPZ)\n"
            "    elif OSN_USE_SYNTHETIC:\n"
            "        print('Synthetic scene: built-in OSN-GS smoke scene')\n"
            "    else:\n"
            "        print('COLMAP DATA_ROOT:', DATA_ROOT)\n",
            "    print('COLMAP DATA_ROOT:', DATA_ROOT)\n",
        )
        cell["source"] = source.splitlines(keepends=True)

notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("patched")
