# Worklog 57: Occluded Candidate → Safe Uncertain Proposal Production Integration

## 결과

Worklog 56의 eligible-boundary bridge를 입력으로 사용해, 기존 Phase F constrained occluded-NURBS fitting → Phase F.1 sampled safety gate → Phase G uncertain Gaussian proposal을 하나의 production orchestration으로 연결했다. Visible construction과 Phase D/E의 내부 판정은 변경하거나 재감사하지 않았다.

- `run_safe_uncertain_proposals_from_gaussians(...)`: raw Gaussian evidence에서 Worklog 56 bridge를 거쳐 safe uncertain proposal까지 실행하는 단일 진입점.
- `build_safe_uncertain_proposals_from_bridge(...)`: 이미 생성된 bridge result에서 candidate별로 같은 Phase F/F.1/G 경로를 실행하는 entry point.
- 모든 `OccludedRegionCandidate`는 정확히 하나의 `SafeUncertainProposalAttempt`로 성공 또는 typed rejection이 기록된다.
- candidate ID와 supporting domain/boundary/patch ID는 chart, safety, proposal까지 보존된다.
- model append, appearance 설정, opacity 설정은 수행하지 않는다. proposal은 `appearance_state="unset"`, `opacity_state="unset"`, `append_state="not_appended"` 상태를 유지한다.

## Candidate-ready fail-closed 계약

Phase E의 기존 입력 정책은 `valid=pair`, `degenerate=pair-but-record-provenance`, `rejected=exclude`다. 이는 geometric candidate의 provenance 보존 계약이며, Phase F/F.1/G 승인 조건은 아니다.

Phase F constrained fitting은 다음을 모두 만족할 때만 실행한다.

1. `candidate.state == "candidate"`
2. supporting domain/boundary/patch ID가 pairwise cardinality와 provenance를 만족
3. 두 `ContinuationDomain.state == "valid"`
4. domain → boundary → patch registry chain이 모두 해소

따라서 `state != rejected`를 승인 조건으로 사용하지 않았다. `degenerate` domain, unsupported/rejected candidate, provenance 불일치, fitting 실패, non-validated chart, unsafe/ambiguous safety 결과, non-eligible proposal은 모두 proposal 0개인 typed rejection으로 남긴다.

## Fixture 결과

- candidate-ready planar bridge는 실제 Phase F fit, Phase F.1 safety, Phase G sampling을 통과해 non-empty uncertain proposal batch를 생성했고 source provenance가 유지됐다.
- Box(cap 64) 7개와 Thin-slab(cap 64) 3개 candidate는 실제 orchestration까지 실행했다. 다만 모두 supporting domain의 `degenerate` 상태 또는 rejected candidate 상태여서 fail-closed typed rejection으로 끝났고 proposal은 0개다. 상태 승격이나 synthetic fallback은 만들지 않았다.
- Sphere는 candidate/proposal 모두 0을 유지했다. raw Gaussian 단일 entry point에서도 동일하게 0이다.
- Worklog 56의 real 3k/10k candidate 0 및 real 5k candidate 0(AABB 비접촉)도 후보가 없을 때 proposal을 만들지 않으므로 유지된다.

## 검증

```text
python -m pytest -q tests/test_safe_uncertain_proposal_production.py \
    tests/test_eligible_boundary_continuation_bridge.py tests/test_occluded_chart.py \
    tests/test_occluded_chart_hardening.py tests/test_uncertain_gaussian_proposal.py
71 passed, 4 subtests passed

python -m pytest -q
744 passed, 1 skipped, 1 warning, 12 subtests passed in 231.78s
```

## 남은 위험

Box/Thin-slab candidate가 Phase F/G candidate-ready domain을 갖지 않는 문제는 우회하지 않았다. `degenerate` continuation domain을 Phase F/F.1/G 입력으로 허용하려면 Phase D/F 계약 변경의 별도 승인이 필요하다. 현재 production 경로는 fail-closed 상태를 유지한다.