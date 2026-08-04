# Worklog 51: Full-cloud Boundary Support Materialization 감사

## 배경

Worklog 50의 multi-scale fail-closed 정책은 그대로 두었다. 이번 작업의 목표는 3k/10k 주요 region의 raw full cloud(representative 축약 이전)에 region-local physical boundary chain이 실제로 존재하는지 확인하고, 존재하면 representative 축약과 분리된 boundary-anchor sidecar를 구현하는 것이었다.

## 감사 방법

`scripts/devtools/trace_raw_full_cloud_boundary_evidence.py`(신규)로, representative-level open-chain의 두 endpoint 사이 직선 경로를 5개 지점(t=0.15~0.85)으로 샘플링해 각 지점에서:

1. 가장 가까운 RAW full-cloud Gaussian을 anchor로 잡고,
2. 그 Gaussian 자신의 frame(정확히는 정확한 자기 normal/scale)을 기준으로,
3. 같은 region에 속한(representative 배정 기준 `nearest_representative_index`) 다른 RAW Gaussian들만으로 **production과 동일한** same-mode + largest-circular-gap 알고리즘(`_largest_circular_gap_from_bins`)을 그대로 재사용해 실제 각도 gap을 측정했다.

representative 축약을 전혀 거치지 않은, "만약 이 지점에 representative가 있었다면 무엇을 봤을까"에 대한 ground-truth 측정이다.

## 결과: 5개 gap 사례

| region | endpoint 쌍 | 거리 | raw 결과 요약 |
|---|---|---:|---|
| 3k-60 | 421378 → 949538 | 2.40 | 5개 지점 전부 anchor 발견, gap 2~28도(전부 임계값 24도 이하 또는 근접) — **일관되게 no_gap, 진짜 interior** |
| 3k-60 | 949538 → 1268028 | 1.96 | t=0.15만 102도(진짜 gap), t=0.35/0.5는 **raw anchor 자체가 없음**(순수 관측 부재), t=0.65/0.85는 22/34도(경계) — **패치성, 일관된 chain 아님** |
| 3k-52 | 666904 → 1086120 | 0.13 | 5개 지점 전부 56~158도(전부 임계값 초과, well-supported 33~70개) — **일관된 edge 신호** |
| 3k-56 | 1110285 → 278207 | 0.12 | 5개 지점 전부 82~226도(전부 임계값 초과, 39~47개) — **일관된 edge 신호** |
| 3k-77 | 672047 → 1673117 | 9.79 | 5개 지점 전부 raw anchor 없음 — **완전히 관측 부재, 애초에 인접 boundary가 아님** |

## 판정

- **77 (거리 9.79)**: raw cloud에도 데이터가 전혀 없다. 두 endpoint는 애초에 서로 인접한 boundary 구간이 아니라 완전히 분리된 두 fragment다 — "no chain" 케이스, 억지로 만들지 않는다.
- **60의 두 번째 gap (거리 1.96)**: raw cloud가 patchy하다 — 일부 지점은 진짜 gap을, 일부는 관측 자체가 없음을 보인다. 연속된 chain으로 볼 근거가 없다 — "no chain" 케이스.
- **52, 56 (거리 0.12~0.13, 매우 짧은 gap)**: raw cloud가 **경로 전체에서 일관되게 진짜 gap(56~226도)**을 보인다. 그런데 원래 trace(`trace_real_physical_candidate_chains.py`)를 다시 확인하면, 이 두 endpoint는 **representative 레벨에서도 이미 서로를 `first_gate: compatible`로 인식하고 있었다** — 즉 representative 자체는 이미 그 자리에 있고, 기하 호환성 게이트도 이미 통과했는데, 최종 Hungarian one-in-one-out matching이 이 edge를 선택하지 않아서 열린 채로 남은 것이다. **이건 representative 축약에서 evidence가 소실된 사례가 아니다** — raw와 representative 양쪽 다 evidence가 있고, 문제는 그 다음 matching/ordering 단계의 경쟁(다른 후보와의 경쟁에서 밀림)이다.

## 결론: boundary-anchor sidecar를 구현하지 않는다

이번 task의 boundary-anchor는 "raw full cloud에는 chain이 있으나 representative 축약에서 소실"된 경우를 위한 것이다. 5개 사례 중 어느 것도 이 조건을 정확히 만족하지 않았다:

- 3개는 raw cloud 자체에 coherent chain이 없다(순수 관측 부재 또는 patchy) — worklog의 지시대로 "raw full cloud에도 chain이 없다면 ... unsupported로 남긴다"에 해당한다.
- 2개(52, 56)는 raw와 representative 양쪽에 evidence가 이미 존재하고 서로 compatible로 인식되지만, **Hungarian matching 경쟁에서 탈락**한 사례다. 이건 진짜 결함이지만, 이번 작업이 명시적으로 금지한 "Hungarian solver 변경" 없이는 고칠 수 없는 위치의 결함이고, boundary-anchor sidecar(축약 손실을 복구하는 메커니즘)로 고칠 수 있는 문제가 아니다 — sidecar를 붙여도 representative가 이미 거기 있으므로 아무것도 바뀌지 않는다.

따라서 이번 라운드는 가설을 기각한다: **주요 region에서 raw full cloud가 representative 축약으로 잃어버린 coherent physical boundary chain을 갖고 있다는 근거를 찾지 못했다.** production 코드는 변경하지 않았다.

## Real 3k/5k/10k before/after

프로덕션 코드를 변경하지 않았으므로 before/after는 정의상 동일하다 — worklog 50의 최종 상태(physical 154/185/125, closed/materialized 0/0, 2/2, 0/0)가 그대로 유지된다.

## Negative-control 및 검증

프로덕션 코드 변경이 없으므로 Box 6/Cylinder 3/Sphere 0/Thin slab 분리는 자동으로 worklog 50과 동일하게 유지된다. 그럼에도 완료 조건에 따라 focused 및 full pytest를 재실행해 확인했다.

```text
python -m pytest -q tests/test_full_cloud_continuation_shell.py tests/test_visible_surface_construction.py \
    tests/test_directed_boundary_ordering.py tests/test_boundary_topology_safety.py
58 passed in 21.85s

python -m pytest -q
720 passed, 1 skipped, 1 warning, 8 subtests passed
```

## 남은 단일 병목 (그리고 새로 확정된 별도 결함 위치)

worklog 47~50과 동일하게, 남은 병목은 큰 region perimeter의 real observed termination evidence 밀도다. 이번 감사로 **추가로 확정한 사실**: region 52/56처럼 짧은 gap 중 일부는 evidence 문제가 아니라 directed ordering/Hungarian matching 단계의 후보 경쟁 문제다 — 이건 이번 작업 범위 밖(Hungarian solver 변경 금지)이므로 고치지 않았지만, 다음 라운드가 ordering/matching 쪽을 다룬다면 이 두 구체적 사례(region 52: 666904↔1086120, region 56: 1110285↔278207)가 재현 가능한 시작점이다.
