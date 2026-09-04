from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from devtools.demo.worklog_155_intrinsic_normal_gaussian_region_viability_audit import (
    ITERATION_DIR,
    REVIEW_CAMERAS,
    _contract_reconciliation,
    _mapping_digest,
    _partition_status,
    _point_in_polygon,
    _summary,
    _write_visualization_readmes,
)


def test_active_partition_contract_is_intrinsic_and_not_covariance_derived() -> None:
    contract = _contract_reconciliation()
    assert contract["contract_status"] == "INTENT_ALIGNED_NO_CONTRACT_DRIFT"
    assert contract["covariance_construction_called"] is False
    assert contract["eigendecomposition_called"] is False
    assert contract["covariance_minor_axis_normal"] is False
    assert contract["covariance_normal_override_of_t_w"] is False
    assert contract["lambda2_lambda3_axis_separability_membership_rule"] is False


def test_existing_partition_roles_map_to_wl154_membership_without_reclassification() -> None:
    partition = SimpleNamespace(
        partition_role=torch.tensor([0, 1, 1, 2, 0], dtype=torch.int8),
        ambiguous_multi_region=torch.tensor([False, False, True, False, False]),
    )
    status, accepted, counts = _partition_status(partition)
    assert status.tolist() == [0, 1, 2, 4, 0]
    assert accepted.tolist() == [True, True, False, False, True]
    assert counts == {"core": 2, "attached": 1, "ambiguous": 1, "rejected": 0, "unassigned": 1}


def test_mapping_digest_is_stable_for_same_stable_id_mapping() -> None:
    orientation = SimpleNamespace(gaussian_ids=torch.tensor([30, 10, 20], dtype=torch.int64))
    partition = SimpleNamespace(subset_ids=torch.tensor([4, 2, 3], dtype=torch.int64))
    status = torch.tensor([0, 4, 1], dtype=torch.int64)
    assert _mapping_digest(orientation, partition, status) == _mapping_digest(orientation, partition, status)


def test_review_polygon_and_summary_are_diagnostic_helpers_only() -> None:
    x = torch.tensor([1.0, 3.0, 1.0]).numpy()
    y = torch.tensor([1.0, 1.0, 3.0]).numpy()
    assert _point_in_polygon(x, y, ((0, 0), (2, 0), (2, 2), (0, 2))).tolist() == [True, False, False]
    summary = _summary([1, 2, 3, 4, 5])
    assert summary["count"] == 5
    assert {"min", "median", "p75", "p90", "p95", "p99", "max"}.issubset(summary)


def test_visualization_readmes_cover_nested_review_artifacts() -> None:
    review = {"camera_metadata": {name: {"resolution": [648, 420]} for name in REVIEW_CAMERAS}}
    pair = {"row_count": 5, "state_counts": {"OBSERVED": 3, "OCCLUDED": 2, "UNRESOLVED": 0}}
    tmp_root = Path(tempfile.mkdtemp(prefix="w155_readme_test_"))
    try:
        _write_visualization_readmes(tmp_root, review, pair)
        directories = [tmp_root] + [path for path in tmp_root.rglob("*") if path.is_dir()]
        assert directories
        assert all((path / "README.md").is_file() for path in directories)
        assert "intrinsic `t_w`" in (tmp_root / "review_views" / "B_intrinsic_tw_normal" / "README.md").read_text(encoding="utf-8")
        assert (tmp_root / "review_views" / "cameras" / REVIEW_CAMERAS[0] / "F_original_plus_region_overlay" / "README.md").is_file()
        assert (tmp_root / "mandatory_gaussian_visualization_pair" / "Original Scene" / ITERATION_DIR / "README.md").is_file()
    finally:
        shutil.rmtree(tmp_root)
