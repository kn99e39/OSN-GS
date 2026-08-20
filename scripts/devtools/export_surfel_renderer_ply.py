"""arch/2dgs-coverage-first-surface -- WebRenderer-compatible PLY export for a 2DGS surfel checkpoint.

WebRenderer (`WebRenderer/Gaussian_Interpreter.js`) has no concept of a 2D
surfel primitive: it always reads three scale properties and falls back to
`exp(0) = 1.0` for any missing one (`raw.scale_2 ?? 0`). A 2DGS surfel's real
tangent scales are typically ~0.01-0.05, so simply omitting `scale_2` (as
`osn_gs/gaussian/torch_surfel_model.py::TorchGaussianSurfelModel.save_ply`
does, correctly, for the checkpoint's own record) would make WebRenderer
render every surfel as a huge near-unit-radius blob -- not a meaningful
"scene preview".

This script is a VISUALIZATION-ONLY export, never the checkpoint's own
representation: it fills `scale_2` with a small fraction of the surfel's own
tangent extent (`normal_thickness_fraction * min(s_u, s_v)`, default 1%) so
WebRenderer draws a thin near-flat disk instead of a unit-radius sphere. The
trained checkpoint (`scale_dim == 2`, no third scale tensor) is never
modified; this only ever writes a separate PLY file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEVTOOLS_DIR.parent.parent
for path in (str(DEVTOOLS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from osn_gs.gaussian.torch_primitive_evidence_adapter import (
    PRIMITIVE_SURFEL_2D,
    checkpoint_primitive,
    load_primitive_model,
)

_ITERATION_DIR = "iteration_0000001"

_PLY_HEADER = (
    "ply\nformat binary_little_endian 1.0\n"
    "element vertex {count}\n"
    "property float x\nproperty float y\nproperty float z\n"
    "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    "property float opacity\n"
    "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
    "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
    "end_header\n"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Output directory (gets iteration_0000001/point_cloud.ply).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--normal-thickness-fraction", type=float, default=0.01,
        help="scale_2 (linear) = this fraction * min(s_u, s_v) per surfel. Visualization-only, never part of the trained model.",
    )
    arguments = parser.parse_args()

    print(f"loading checkpoint {arguments.checkpoint}", flush=True)
    model, payload = load_primitive_model(arguments.checkpoint, device=arguments.device)
    primitive = checkpoint_primitive(payload)
    if primitive != PRIMITIVE_SURFEL_2D or int(getattr(model, "scale_dim", 3)) != 2:
        raise ValueError(
            f"{arguments.checkpoint} is not a 2DGS surfel checkpoint (primitive={primitive!r}, "
            f"scale_dim={getattr(model, 'scale_dim', None)!r}); this script is surfel-specific."
        )

    with torch.no_grad():
        uncertain_mask = model.is_uncertain.reshape(-1).to(torch.bool)
        visible_selector = torch.nonzero(~uncertain_mask, as_tuple=False).reshape(-1)
        count = int(visible_selector.shape[0])
        print(f"surfels: total={len(model)} visible={count} iteration={payload.get('iteration')}", flush=True)

        xyz = model.get_xyz.detach()[visible_selector]
        f_dc = model._features_dc.detach()[visible_selector][:, 0, :]
        opacity_logit = model._opacity.detach().reshape(-1)[visible_selector]
        rotation = model.get_rotation.detach()[visible_selector]
        linear_scaling = model.get_scaling.detach()[visible_selector]  # (N, 2) = [s_u, s_v]

        minor_tangent = linear_scaling.min(dim=1).values
        normal_thickness = (float(arguments.normal_thickness_fraction) * minor_tangent).clamp_min(1e-8)
        log_scale_uv = torch.log(linear_scaling)
        log_scale_normal = torch.log(normal_thickness).unsqueeze(-1)
        log_scale_full = torch.cat([log_scale_uv, log_scale_normal], dim=-1)  # (N, 3)

        columns = np.concatenate(
            [
                xyz.cpu().numpy().astype(np.float32),
                f_dc.cpu().numpy().astype(np.float32),
                opacity_logit.cpu().numpy().astype(np.float32).reshape(-1, 1),
                log_scale_full.cpu().numpy().astype(np.float32),
                rotation.cpu().numpy().astype(np.float32),
            ],
            axis=1,
        )

    out_path = arguments.out / _ITERATION_DIR / "point_cloud.ply"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        handle.write(_PLY_HEADER.format(count=count).encode("ascii"))
        handle.write(np.ascontiguousarray(columns, dtype="<f4").tobytes())

    print(f"wrote {out_path} ({count} surfels)", flush=True)
    print(
        f"scale_2 fill: {arguments.normal_thickness_fraction:.4f} * min(s_u, s_v) per surfel "
        "-- VISUALIZATION ONLY, not part of the trained checkpoint (scale_dim stays 2 in the model itself).",
        flush=True,
    )


if __name__ == "__main__":
    main()
