"""Offline replay + stage-by-stage diagnostic for the long-horizon Visible
Surface Constructor reliability collapse (worklog 135).

Loads a saved OSN-GS checkpoint (format_version=2) directly -- no training,
no trainer, no renderer -- and re-derives exactly the same tensors
``reconstruct_visible_after_adc`` (osn_gs/core/torch_pipeline.py) would feed
into the canonical constructor, instrumenting every stage between "raw
checkpoint tensors" and "eigendecomposition input" for non-finite values.

Usage:
    python -m scripts.devtools.diagnose_long_horizon_reliability_collapse \
        --checkpoint output/osn_gs_scene/3000/checkpoint.pt --cap 2048
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.surface.torch_gaussian_covariance_frame import (
    GaussianCovarianceFrame,
    covariance_from_scale_rotation,
    extract_covariance_frame,
)

# cuSOLVER's batched syevj/syevd has an empirically-confirmed hard batch-size
# ceiling around 2,064,888 (measured on this machine's GPU/driver/cusolver
# combination -- not a documented constant, so a conservative chunk size is
# used rather than hugging the exact boundary). Diagnostic-only helper that
# previews the eventual production chunking fix; NOT wired into any
# production path by this script.
_EIGH_SAFE_CHUNK = 1_000_000


def _chunked_extract_covariance_frame(covariance: torch.Tensor) -> GaussianCovarianceFrame:
    count = int(covariance.shape[0])
    if count <= _EIGH_SAFE_CHUNK:
        return extract_covariance_frame(covariance)
    parts = [
        extract_covariance_frame(covariance[start:start + _EIGH_SAFE_CHUNK])
        for start in range(0, count, _EIGH_SAFE_CHUNK)
    ]
    fields = {}
    for name in GaussianCovarianceFrame.__dataclass_fields__:
        values = [getattr(part, name) for part in parts]
        if name == "shape_class":
            fields[name] = tuple(item for part_values in values for item in part_values)
        else:
            fields[name] = torch.cat(values, dim=0)
    return GaussianCovarianceFrame(**fields)
from osn_gs.surface.torch_gaussian_structural_reliability import (
    evaluate_intrinsic_reliability,
    INTRINSIC_RELIABLE,
    INTRINSIC_AMBIGUOUS,
    INTRINSIC_REJECTED,
)


def _stats(name: str, tensor: torch.Tensor) -> dict:
    flat = tensor.detach()
    finite_mask = torch.isfinite(flat)
    finite = flat[finite_mask]
    out = {
        "stage": name,
        "count": int(flat.numel()),
        "finite_count": int(finite_mask.sum().item()),
        "nan_count": int(torch.isnan(flat).sum().item()),
        "pos_inf_count": int(torch.isposinf(flat).sum().item()),
        "neg_inf_count": int(torch.isneginf(flat).sum().item()),
    }
    if finite.numel() > 0:
        out["min"] = float(finite.min().item())
        out["max"] = float(finite.max().item())
        out["mean"] = float(finite.mean().item())
    else:
        out["min"] = out["max"] = out["mean"] = None
    return out


def _row_stats(name: str, tensor: torch.Tensor) -> dict:
    """Per-Gaussian-row finiteness (any non-finite component in the row)."""
    flat = tensor.detach().reshape(tensor.shape[0], -1)
    row_finite = torch.isfinite(flat).all(dim=1)
    bad_rows = torch.nonzero(~row_finite, as_tuple=False).reshape(-1)
    return {
        "stage": name,
        "row_count": int(flat.shape[0]),
        "finite_row_count": int(row_finite.sum().item()),
        "nonfinite_row_count": int(bad_rows.numel()),
        "first_offending_row": int(bad_rows[0].item()) if bad_rows.numel() > 0 else None,
    }


def _git_fingerprint() -> dict:
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[2]).decode().strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"<error: {exc}>"

    return {
        "commit": _run("rev-parse", "HEAD"),
        "dirty_file_count": len(_run("status", "--porcelain").splitlines()),
    }


def diagnose(checkpoint_path: Path, cap: int, device: str) -> dict:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw = payload["model_raw"]
    report: dict = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_fingerprint": {
            "iteration": int(payload["iteration"]),
            "format_version": int(payload.get("format_version", 0)),
            "sha256": _sha256(checkpoint_path),
        },
        "git": _git_fingerprint(),
        "cap": int(cap),
        "stages": [],
    }

    model = TorchGaussianModel(sh_degree=3, device=device)
    model.replace_tensors(
        xyz=raw["xyz"], features_dc=raw["features_dc"], features_rest=raw["features_rest"],
        opacity=raw["opacity"], scaling=raw["scaling"], rotation=raw["rotation"],
        uncertain_confidence=raw["uncertain_confidence"], uncertain_mask=raw["is_uncertain"],
        surface_uv=raw["surface_uv"], cluster_ids=raw["cluster_ids"],
        surface_owner_kind=raw.get("surface_owner_kind"),
        surface_owner_id=raw.get("surface_owner_id"),
        stable_gaussian_ids=raw.get("stable_gaussian_ids"),
    )
    report["source_gaussian_count"] = len(model)

    # --- Stage N1: raw model state as stored in the checkpoint ---
    report["stages"].append({"tag": "N1_raw_position", **_row_stats("N1_raw_position", model._xyz)})
    report["stages"].append({"tag": "N1_raw_log_scale", **_row_stats("N1_raw_log_scale", model._scaling)})
    report["stages"].append({"tag": "N1_raw_quaternion", **_row_stats("N1_raw_quaternion", model._rotation)})
    report["stages"].append({"tag": "N1_raw_opacity_logit", **_row_stats("N1_raw_opacity_logit", model._opacity)})

    # --- Stage N2: scale activation (exp) ---
    activated_scale = model.get_scaling.detach()
    report["stages"].append({"tag": "N2_activated_scale", **_row_stats("N2_activated_scale", activated_scale)})
    report["stages"].append({"tag": "N2_activated_scale_values", **_stats("N2_activated_scale_values", activated_scale)})

    # --- Stage N3: quaternion normalization ---
    raw_quat_norm = model._rotation.detach().norm(dim=-1)
    zero_quat_count = int((raw_quat_norm < 1e-12).sum().item())
    normalized_rotation = model.get_rotation.detach()
    report["stages"].append({"tag": "N3_raw_quaternion_norm", **_stats("N3_raw_quaternion_norm", raw_quat_norm)})
    report["zero_or_near_zero_quaternion_count"] = zero_quat_count
    report["stages"].append({"tag": "N3_normalized_quaternion", **_row_stats("N3_normalized_quaternion", normalized_rotation)})

    # --- Stage N4: rotation matrix + covariance assembly ---
    covariance = covariance_from_scale_rotation(activated_scale, normalized_rotation)
    report["stages"].append({"tag": "N4_covariance", **_row_stats("N4_covariance", covariance)})

    symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
    report["stages"].append({"tag": "N4_symmetrized_covariance", **_row_stats("N4_symmetrized_covariance", symmetric)})

    finite_row_mask = torch.isfinite(symmetric.reshape(symmetric.shape[0], -1)).all(dim=1)
    bad_indices = torch.nonzero(~finite_row_mask, as_tuple=False).reshape(-1)
    report["nonfinite_covariance_row_count"] = int(bad_indices.numel())
    if bad_indices.numel() > 0:
        first_bad = int(bad_indices[0].item())
        report["first_offending_gaussian"] = {
            "index_in_model": first_bad,
            "stable_id": int(model.stable_gaussian_ids[first_bad].item()),
            "is_uncertain": bool(model.is_uncertain[first_bad].item()),
            "surface_owner_kind": int(model.surface_owner_kind[first_bad].item()),
            "raw_position": model._xyz[first_bad].detach().cpu().tolist(),
            "raw_log_scale": model._scaling[first_bad].detach().cpu().tolist(),
            "activated_scale": activated_scale[first_bad].detach().cpu().tolist(),
            "raw_quaternion": model._rotation[first_bad].detach().cpu().tolist(),
            "raw_quaternion_norm": float(raw_quat_norm[first_bad].item()),
            "normalized_quaternion": normalized_rotation[first_bad].detach().cpu().tolist(),
            "raw_opacity_logit": float(model._opacity[first_bad, 0].item()),
        }

    # --- Stage N5: eigendecomposition (only attempt on the FINITE subset,
    # to avoid crashing the whole diagnostic script; the whole point is to
    # observe this stage without letting the known-crashing call kill us). ---
    eigh_crashed = False
    eigh_error = None
    if int(finite_row_mask.sum().item()) > 0:
        try:
            _ = torch.linalg.eigh(symmetric[finite_row_mask])
        except Exception as exc:  # noqa: BLE001 - diagnostic capture
            eigh_crashed = True
            eigh_error = f"{type(exc).__name__}: {exc}"
    report["eigh_on_finite_subset_crashed"] = eigh_crashed
    report["eigh_on_finite_subset_error"] = eigh_error

    # Also attempt eigh on the FULL (possibly non-finite) tensor, matching
    # production exactly, to see whether cuSOLVER crashes outright or
    # returns NaN silently (backend/version dependent).
    eigh_full_crashed = False
    eigh_full_error = None
    try:
        _ = torch.linalg.eigh(symmetric)
    except Exception as exc:  # noqa: BLE001
        eigh_full_crashed = True
        eigh_full_error = f"{type(exc).__name__}: {exc}"
    report["eigh_on_full_tensor_crashed"] = eigh_full_crashed
    report["eigh_on_full_tensor_error"] = eigh_full_error

    # --- Reliability waterfall on the FINITE subset only (eligibility mask
    # mirrors reconstruct_visible_after_adc: not uncertain, not occluded-owned) ---
    eligible_mask = (~model.is_uncertain) & (model.surface_owner_kind != 2)
    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).reshape(-1)
    report["eligible_count"] = int(eligible_indices.numel())
    finite_eligible = finite_row_mask[eligible_indices]
    report["eligible_finite_count"] = int(finite_eligible.sum().item())
    report["eligible_nonfinite_count"] = int((~finite_eligible).sum().item())

    finite_eligible_indices = eligible_indices[finite_eligible]
    if int(finite_eligible_indices.numel()) >= 4:
        frame = _chunked_extract_covariance_frame(covariance[finite_eligible_indices])
        intrinsic = evaluate_intrinsic_reliability(frame)
        counts = {INTRINSIC_RELIABLE: 0, INTRINSIC_AMBIGUOUS: 0, INTRINSIC_REJECTED: 0}
        for cls in intrinsic.intrinsic_class:
            counts[cls] += 1
        report["intrinsic_reliability_on_finite_eligible"] = counts
        reason_counts: dict[str, int] = {}
        for reasons in intrinsic.reasons:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        report["intrinsic_rejection_reason_counts"] = reason_counts
    else:
        report["intrinsic_reliability_on_finite_eligible"] = None

    return report


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = diagnose(args.checkpoint, args.cap, args.device)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
