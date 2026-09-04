"""Focused tests for Worklog 161's stop-safe spatial-domain audit."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "devtools/demo/worklog_161_global_persistent_occlusion_spatial_domain_audit.py"
SPEC = importlib.util.spec_from_file_location("worklog_161_spatial_domain_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_historical_semantics_a_preserves_missing_evidence_as_unresolved() -> None:
    historical = np.asarray([MODULE.STATE_UNRESOLVED, MODULE.STATE_OCCLUDED], dtype=np.int8)
    geometric = np.asarray([True, True], dtype=bool)
    evidence = np.asarray([False, True], dtype=bool)
    result = MODULE.relabel_semantics_a_to_b(
        historical_states=torch_tensor(historical),
        geometrically_relevant=torch_tensor(geometric),
        evidence_available=torch_tensor(evidence),
    )
    assert int(result[0]) == MODULE.STATE_NON_RELEVANT
    assert int(result[1]) == MODULE.STATE_OCCLUDED
    assert MODULE.w160.aggregate_persistent_states(historical[None, :])[0] == MODULE.STATE_UNRESOLVED
    assert MODULE.w160.aggregate_persistent_states(result.numpy()[None, :])[0] == MODULE.STATE_OCCLUDED


def torch_tensor(value: np.ndarray):
    import torch

    return torch.as_tensor(value)


def test_synthetic_contracts_pass_and_show_semantics_impact() -> None:
    result = MODULE.synthetic_contracts()
    assert result["all_pass"] is True
    missing = next(case for case in result["cases"] if case["name"].startswith("D_"))
    assert missing["actual_historical"] == "UNRESOLVED"
    assert missing["hypothetical_semantics_b"] == "OCCLUDED"


def test_spatial_domain_audit_is_fail_closed_without_new_domain() -> None:
    result = MODULE.spatial_domain_audit()
    assert result["canonical_pre_latent_spatial_query_domain_exists"] is False
    assert result["canonical_domain_candidate"] is None
    assert result["forbidden_substitutes_not_used"]
    assert all(candidate["eligible_as_occlusion_domain"] is False for candidate in result["candidates"])


def test_global_accumulator_keeps_observed_and_unresolved_semantics() -> None:
    import torch

    observed = torch.tensor([True, False, False, False])
    relevant = torch.tensor([True, True, False, True])
    unresolved = torch.tensor([False, False, False, True])
    result = MODULE._global_from_accumulators(observed, relevant, unresolved).numpy()
    assert result.tolist() == [MODULE.STATE_OBSERVED, MODULE.STATE_OCCLUDED, MODULE.STATE_UNRESOLVED, MODULE.STATE_UNRESOLVED]
