# Candidate-local continuation certificate 및 physical termination 추출 감사

## 상태

**부분 완료 / production certificate 채택 보류.** 이 기록은 Worklog 40 이후 현재 작업 트리의 candidate-local continuation 관련 변경과 실제 3k/5k/10k replay를 재검증한 결과다. 전역 region-pair verdict의 비국소성은 확인됐지만, 이를 안전하게 대체하는 candidate-local certificate는 아직 production에 연결되지 않았다.

## 수행 내용

- 첨부된 작업 요구와 Worklog 40, 현재 구현을 대조했다.
- `smooth_cross_region_continuation` typed provenance와 `boundary_smooth_cross_region_candidate_count` 진단 항목이 현재 작업 트리에 존재함을 확인했다. 이는 `reliability_frontier`의 의미를 조용히 바꾸지 않고, smooth cross-region continuation을 별도 reason으로 보존한다.
- 현재 `construct_visible_nurbs_from_gaussians`는 support-termination 반경으로 raw Gaussian footprint 대신 resolved representative graph `candidate_scale`을 전달한다. representative 간격이 footprint보다 훨씬 큰 실제 snapshot에서 accepted neighbor가 반경 밖으로 탈락하는 전달 결함을 피하기 위한 보정이다. candidate/residual scale의 분리는 유지된다.
- `trace_physical_termination_gates.py`로 3k/5k/10k 실제 checkpoint의 node-level first-failure 및 evidence 기반 R1/R2 proxy를 실행했다.
- `trace_real_snapshot_boundary_waterfall.py`로 같은 checkpoint에서 compatibility/order 단계를 재실행했다.

## Worklog 40 global verdict 감사

현재 `extract_support_termination_candidates()`의 representative-only 경로는 outward arc 안의 다른 region node를 찾은 뒤, 해당 region pair의 전역 `smooth_continuation` verdict만으로 후보를 `smooth_cross_region_continuation`으로 재분류한다. 따라서 relation edge가 후보 근처에 존재하는지, outward arc와 겹치는지, pair 안에 crease/parallel evidence가 공간적으로 혼재하는지를 production 판정이 검사하지 않는다.

`PAIR_SPATIALLY_MIXED` 등 mixed-relation 상태 상수와 relation source 수집 함수는 현재 파일에 정의되어 있으나, production verdict 또는 candidate-local 승인 조건에는 아직 연결되지 않았다. 따라서 region pair 어딘가의 same-surface evidence가 멀리 있는 candidate에도 적용될 수 있고, global aggregation이 false suppression을 만들지 않는다는 회귀 증거도 아직 없다.

## Candidate-local certificate 평가

현재 소스 주석에는 support-radius direct locality와 local crease/parallel veto를 시험했을 때 sphere seam physical candidate가 각각 11개 및 8개로 되살아났다는 측정이 남아 있다. 이 변형은 bounded-kNN relation evidence가 seam을 균등하게 덮지 못한다는 문제 때문에 채택되지 않았다.

이번 검증에서는 이 수치를 독립 재현하거나 새 threshold를 도입하지 않았다. 따라서 다음 candidate-local certificate는 **미구현**이다.

```text
candidate outward support
  + bounded local same-surface path/evidence
  + local crease/parallel/competing contradiction 부재
  + tangent/normal/residual continuity
  + region-pair verdict는 prior만 사용
```

이를 구현하기 전에는 mixed smooth/crease, smooth/gap, localized crease, touching-point fixture가 필요하다. 전역 verdict를 단순히 제거하거나 direct relation만 요구하면 sphere의 bounded-graph coverage 부족으로 회귀한다.

## Real physical-termination waterfall

`trace_physical_termination_gates.py --cap 2048` 실행 결과다. R1은 region 내부 accepted-neighbor angular gap조차 없는 경우, R2는 perimeter proxy는 있으나 physical candidate가 세 개 미만인 경우다. 이는 ground truth가 아닌 evidence-based diagnostic 분류다.

| checkpoint | generated physical candidate | R1 | R2 | R3-or-later |
| --- | ---: | ---: | ---: | ---: |
| `3000` | 136 | 2 | 138 | 17 |
| `5000` | 167 | 3 | 124 | 21 |
| `10000` | 106 | 3 | 130 | 8 |

R1은 2~3 region에 그치고 R2가 124~138 region으로 지배적이다. 즉 Worklog 40의 `R1/R2 candidate-starved` 합산 결론은 현재 경로에서 **perimeter proxy 도달 뒤 candidate extraction이 누락되는 R2**로 더 좁혀진다.

node-level first failure에는 `not_region_member`, `no_neighbor_support`, `insufficient_termination_evidence`, `histogram_veto_no_missing_sector_run`, `ambiguous_continuation`, `reliability_frontier`, `unresolved_sampling_gap`, `generated_physical_candidate`가 기록된다. full all-failure vector와 signed margin, outward-arc relation 증거는 아직 이 script에 완결되지 않았으므로 다음 batch의 명시적 작업이다.

## Compatibility/order 재확인 및 진단 불일치

`trace_real_snapshot_boundary_waterfall.py`는 physical candidate를 각각 153/181/121개로 보고했고, closed component는 세 checkpoint 모두 0개였다. compatibility 부족(R3)은 3k/5k/10k에서 각각 19/24/9 region으로 보고했다.

이는 위 physical gate trace의 generated count 136/167/106과 일치하지 않는다. 두 도구가 normalization 전후 및 source-id 단위를 다르게 집계하는지 확인이 필요하다. 이 불일치를 해소하기 전에는 candidate extraction repair의 before/after 수치나 Box face 4의 원인을 확정하지 않는다.

## Box face 4 및 angular exposure

Box face 4의 missing corner 2개와 interior false candidate 4개에 대한 member-level trace는 아직 없다. fixed-sector histogram은 sphere false candidate를 크게 줄이는 load-bearing guard라는 Worklog 39의 결론을 유지한다. bin 수, smearing coefficient, candidate threshold를 fixture에 맞추어 변경하지 않았다.

## 검증

- Focused regression: `35 passed in 20.24s`.
- Repository-wide pytest: `707 passed, 1 skipped, 1 warning, 8 subtests passed in 183.97s`.

## 결론과 다음 병목

Worklog 40의 region-pair global continuation verdict가 공간적으로 혼재한 관계에서 안전하다고 결론낼 수 없다. typed provenance와 scale 전달 보정은 유효하지만, candidate-local certificate의 production 채택은 보류한다.

다음 batch는 seed admission, Hungarian solver, NURBS fitting을 변경하지 않고 다음 순서를 따른다.

1. 두 waterfall의 candidate identity/normalization 단위를 stable ID 기준으로 통일한다.
2. mixed-relation adversarial fixture로 global leakage를 재현하고, local certificate의 필요 조건을 검증한다.
3. Box face 4의 각 member에 대해 corner/edge/interior diagnostic trace를 완성한다.
4. 그 결과가 명백한 전달·normalization·neighbor-set 결함을 보일 때만 narrow repair를 적용한다.
