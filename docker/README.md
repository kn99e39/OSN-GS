# OSN-GS CUDA Docker Environment

This environment isolates OSN-GS from the CentOS 7 host while using its NVIDIA GPUs. It targets the host's NVIDIA 525.60.13 driver and RTX 3090 / RTX 3080 Ti GPUs with CUDA 11.8, PyTorch 2.1.2, Python 3.10, and a prebuilt vendored diff-Gaussian rasterizer.

## Build and validate

Run these commands on the server from the repository root:

```bash
docker compose -f docker/compose.yml build
docker compose -f docker/compose.yml run --rm osn-gs \
  python3.10 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0)); import diff_gaussian_rasterization; print('rasterizer: ok')"
```

The Compose service exposes all GPUs to Docker, but defaults the process to GPU 0 (RTX 3090). Select the 3080 Ti explicitly when needed:

```bash
CUDA_VISIBLE_DEVICES=1 docker compose -f docker/compose.yml run --rm osn-gs bash
```

## Training and synthetic constructor benchmark

```bash
docker compose -f docker/compose.yml run --rm osn-gs \
  python3.10 -m nurbs_constructor_benchmark --device cuda

docker compose -f docker/compose.yml run --rm osn-gs \
  python3.10 train.py -s /workspace/DATASET -m /workspace/outputs/osn_gs_run --device cuda
```

## Jupyter through SSH

Start Jupyter on the server inside `tmux` so it survives SSH disconnects:

```bash
tmux new -s osn-gs-jupyter
docker compose -f docker/compose.yml run --rm --service-ports osn-gs \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

On the local machine, forward the server-only port:

```bash
ssh -N -L 8888:127.0.0.1:8888 nd@100.115.109.101
```

In VS Code, select **Existing Jupyter Server** and paste the token URL printed by Jupyter, replacing its host with `127.0.0.1` if necessary. The port is bound to the server's loopback interface, so it is not publicly exposed.

## Notes

- The project source is bind-mounted. Rebuild the image after dependency or CUDA-extension source changes.
- Do not install packages into the CentOS 7 host Python; use `python3.10` inside the service.
- CUDA extension artifacts are stored in the named `torch_extensions` volume. Remove that volume only when deliberately forcing a JIT rebuild.
