# OSN-GS

OSN-GS is an experimental 3D Gaussian Splatting framework that predicts occluded surface structure from observed Gaussian geometry. It fits base curves on observed Gaussian centers, extrapolates occlusion curves, builds a NURBS-like parametric surface, and places uncertain Gaussians on the inferred surface.

## Current Training Path

The active training implementation is the Torch path:

```bash
python scripts/train_osn_gs_torch.py \
  --device cuda \
  --iterations 30000 \
  --output outputs/osn_gs_run
```

The project also exposes a notebook-compatible wrapper:

```bash
python train.py -m outputs/osn_gs_run --iterations 30000
```

To train from a COLMAP/Graphdeco-style dataset:

```bash
python train.py \
  -s /path/to/scene_root \
  -m outputs/osn_gs_colmap \
  --iterations 30000 \
  --image_downscale 2
```

If `diff_gaussian_rasterization` is installed, OSN-GS uses it automatically. Otherwise it uses a differentiable fallback renderer that is useful for debugging the OSN-GS pipeline, but not for final 3DGS-quality results.

## Colab Notebook

`../3DGS_Renderer/colab_train_3dgs.ipynb` now has an `osn_gs` framework mode. In the project setup cell, keep:

```python
FRAMEWORK_MODE = 'osn_gs'
```

The notebook will discover an uploaded `OSN-GS` project zip by its top-level `train.py`, discover/upload a COLMAP-style dataset, and pass that dataset to OSN-GS through `train.py -s DATA_ROOT`.

For a quick synthetic smoke run, set:

```python
OSN_USE_SYNTHETIC = True
```

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

The script can run a synthetic smoke scene:

```bash
python scripts/train_osn_gs_torch.py --device cuda
```

It can also read an NPZ scene:

```bash
python scripts/train_osn_gs_torch.py \
  --scene_npz data/scene.npz \
  --device cuda \
  --output outputs/custom_scene
```

The NPZ file should contain:

- `points`: `(N, 3)` initial Gaussian centers
- `colors`: `(N, 3)` RGB values in `[0, 1]`
- `images`: `(V, 3, H, W)` or `(V, H, W, 3)` target images
- `extent`: optional scalar scene extent

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
