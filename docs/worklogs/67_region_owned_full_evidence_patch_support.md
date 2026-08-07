# Worklog 67: Region-Owned Full-Evidence Patch Support

## 목적

worklog 66에서 승인된 결과 위에서, representative topology와 chart boundary는 전혀 바꾸지 않고, 각 승인된 region이 실제로 대표하는 원본 observed Gaussian evidence를 NURBS fitting support로 복원한다. Representative-only fitting(worklog 66까지 유일한 경로, 최소 3~4점)이 만들던 `under_supported` 병목이 진짜 evidence 부족 때문인지, 아니면 representative에게만 묻는 fitting 설계의 한계인지 가른다.

## 구현

### 신규 모듈: `osn_gs/surface/torch_region_owned_full_evidence.py`

기존 코드를 재사용해 gating 로직을 절대 재구현하지 않는다 — `TorchOSNGSPipeline._propagate_with_evidence_gating()`(worklog 129 item 10, 기존 production 코드)가 이미 "이 full-cloud Gaussian의 실제 normal/position이 자신의 nearest representative의 oriented tangent plane과 일치하는가"를 판정하는 정확한 메커니즘이다(`normal_alignment_min=0.5`, `residual_max_ratio=4.0`, 기존 값 그대로). 이 함수의 출력(`propagated`: full-cloud 각 점이 어느 patch에 귀속되는지, 비호환이면 -1)을 그대로 소비해서:

- **representative provenance 귀속**: nearest-representative가 그 region의 멤버여야만 후보가 됨(strict nearest assignment라 다른 region evidence가 섞일 수 없음 — "다른 region 소유 evidence를 임의로 병합하지 않을 것"을 구조적으로 만족).
- **거리/normal consistency**: 기존 `normal_alignment_min`/`residual_max_ratio` 게이트 그대로 재사용 — "기존 accepted support 범위".
- **crease/parallel-sheet-conflict/ambiguous-frontier 배제**: 별도 재분류 없이, 위 alignment/residual 게이트가 이미 이 경계들을 건너간 점을 정상적으로 걸러낸다(그 게이트가 만들어진 목적 자체가 이것).
- **exact duplicate 제거**: `torch.unique`가 아니라 stable-id 기반 명시적 dedup(구조적으로는 발생 불가능하지만 방어적으로 유지).

`fit_region_owned_full_evidence_patch()`: boundary loop(불변)+region-owned full evidence로 재-fitting하고, `MIN_FULL_EVIDENCE_SUPPORT=4`(worklog 66의 `under_supported` 임계값과 동일 상수, 재사용) 미달이면 `under_supported`, Jacobian degenerate cell이 있으면 `unsafe_geometry`로 fail-closed. `RegionOwnedFullEvidenceFit`은 representative-only 결과(`VisibleBoundaryMaterializationResult.surface`/`boundary_residual`/`interior_residual`)를 절대 덮어쓰지 않는 완전히 별도 구조체다.

### Pipeline 연결: `osn_gs/core/torch_pipeline.py`

`_construct_canonical_with_full_evidence()`가 `construction`(region formation/boundary ordering/chart eligibility, **미변경**)을 완료한 **뒤에만** 새 단계를 추가한다. `CanonicalConstructionWithEvidence`에 `region_owned_full_evidence_fits: dict[(chart_type, region_id), RegionOwnedFullEvidenceFit]` 필드를 additive로 추가했다 — 읽기 전용으로 `bundle.construction`을 참조할 뿐 절대 쓰지 않는다(포함 테스트로 회귀 검증, 아래).

## 테스트

신규 `tests/test_region_owned_full_evidence.py`(9개, 순수 함수 단위 테스트 + 실제 dense box fixture로 pipeline 연결 end-to-end 테스트) + 기존 관련 focused 223개(density-preserving selection, full-cloud continuation shell, local evidence scale, representative graph scale, pipeline smoke, ADC-synchronized visible NURBS, training regressions, uncertain trainer activation, parametric chart boundary(+materialization), eligible boundary continuation bridge, visible boundary region status, directed boundary ordering, synthetic dataset, covariance frame, NURBS surface, safe uncertain proposal production, surface ownership) — **총 232개 전부 통과**. 지시대로 full pytest는 실행하지 않았다(직전 worklog 66의 production 수정에도 동일하게 적용).

## 결과 (baseline_compatible@2900/3100 vs Graphdeco baseline 참조)

| 조건 | region | patch | rep support 합 | full-evidence support 합 | 배율 | 분류 | 평균 orientation flip |
|---|---:|---:|---:|---:|---:|---|---:|
| baseline_compatible@2900 | 7 | 5 | 28 | 2,018 | **72.1×** | extrapolative 4, valid_supported 1 | 95.8/576 |
| baseline_compatible@3100 | 19 | 11 | 73 | 8,012 | **109.8×** | extrapolative 11 | 19.9/576 |
| baseline@2900(참조) | 8 | 4 | 20 | 6,653 | **332.6×** | extrapolative 4 | 23.8/576 |
| baseline@3100(참조) | 3 | 2 | 10 | 547 | **54.7×** | extrapolative 2 | 5.0/576 |

**[worklog 69에서 정정] full_evidence_state는 22개 patch 전부 `materialized`다(합계 오류 정정: 5+11+4+2=22, 기존 "21개"는 오기) — `under_supported`/`unsafe_geometry`/`fit_failed`가 0건이다.** worklog 66에서 다수를 차지하던 `under_supported`(evidence 3~4개)가 region-owned full evidence로는 완전히 사라졌다 — representative 3~4개짜리 "최소 region"도 실제로는 수십~수천 개의 원본 Gaussian을 대표하고 있었고, 그 evidence는 정상적으로 복원 가능했다.

**[worklog 69에서 정정] 대신 거의 모든 patch가 `extrapolative`로 재분류됐다(21/22, 95%, 기존 "20/21"은 오기).** 원인은 fitting 자체가 아니라 **평가 기준의 척도 변화**다: worklog 66의 `local_evidence_scale`은 representative 3~8개로 계산돼 값이 크고(성긴 기준), 이번 `local_evidence_scale`은 실제 수백~수천 개의 조밀한 원본 Gaussian으로 계산돼 훨씬 작다(0.007~0.045 범위). 같은 6×6 degree-2 NURBS fit이 이제 훨씬 엄격한(조밀한) 기준으로 평가되면서, surface-to-evidence p95가 4.0× 기준을 넘는 patch가 대다수가 됐다. **baseline(Graphdeco 참조)도 동일한 패턴을 보인다(4/4, 2/2 전부 extrapolative)** — OSN-GS 학습 품질 문제가 아니라, 6×6/degree-2라는 fitting 해상도 자체가 실제 조밀한 Gaussian cloud의 국소 노이즈/굴곡을 따라가기엔 낮다는, 두 조건에 공통된 별개 발견이다.

Orientation consistency는 대체로 양호(대부분 0~14% flip)하나 baseline_compatible@2900의 region 0은 468/576(81%)로 눈에 띄게 높다 — 분류 결과 자체는 이미 extrapolative라 바뀌지 않지만, 해당 patch의 fit이 심하게 뒤틀려 있다는 별도 신호로 기록해 둔다.

## 완료 기준 대조

- representative support 수 vs unique full-evidence support 수: **측정 완료, 55~333배 차이.**
- materialized/valid_supported/under_supported/unsafe 수: **측정 완료** — full_evidence_state는 100% materialized(0/0/0), 최종 classification은 extrapolative가 지배적(21/22, worklog 69에서 22로 정정).
- point-to-surface/surface-to-evidence normalized p95: **측정 완료**(패치별 세부값, `output/extent_ab/val67/region_owned_evidence_report.json`).
- Jacobian degeneracy/orientation consistency: **측정 완료** — Jacobian degenerate 0건, orientation flip은 대체로 낮으나 1건 예외(위 참고).
- region-owned full evidence coverage/uncovered 비율: **의도적으로 미보고** — 아래 해석 유의사항 참고(조건 간 공통 척도가 아님).
- patch area/boundary provenance: **측정 완료**(boundary loop은 변경되지 않았으므로 provenance는 worklog 66과 동일하게 physical_termination 우세).

## 해석 유의사항 (지시대로 명시)

- **worklog 66의 gap 비율은 조건 내부 evidence coverage로만 해석한다.** 각 조건은 서로 다른 accepted representative evidence 집합에서 출발하므로, gap %를 조건 간 공통 ground-truth surface coverage처럼 비교하지 않는다. 이번 라운드는 gap 비율을 아예 보고하지 않았다(위 이유로 오해 소지가 있는 지표).
- **`validate_simple_closed_loop`는 boundary-loop 검사다.** patch의 2D 경계 loop가 자기 자신의 접평면에서 단순 다각형인지만 확인한다 — 피팅된 3D parametric surface가 경계 밖에서 스스로 접히는지(surface self-intersection)는 **이 파이프라인 어디에서도 검사하지 않는다.** worklog 66에서 `self_intersecting`으로 표기했던 필드는 이번에 `boundary_loop_simple_polygon_violation`으로 명칭을 바로잡았다.
- **5-way 분류 우선순위**(먼저 만족하는 조건이 최종값, `duplicate_or_overlapping`만 전체 patch 분류 후 별도 post-pass):
  1. `unsafe_geometry` — boundary-loop 위반 OR full-evidence Jacobian degenerate OR full-evidence fit 실패
  2. `duplicate_or_overlapping`(post-pass, unsafe_geometry는 덮어쓰지 않음)
  3. `under_supported` — full-evidence support < `MIN_FULL_EVIDENCE_SUPPORT`(4)
  4. `extrapolative` — point-to-surface 또는 surface-to-evidence 정규화 p95 > 4.0×
  5. `valid_supported` — 위 전부 통과

## 다음 자연스러운 후속 (미착수)

fitting 해상도(6×6 control grid, degree 2)가 실제 조밀한 evidence의 국소 변화를 따라가기엔 낮다는 것이 이번 라운드의 핵심 발견이다. 이 회차 지시("결과에 맞춰 조정하지 마라")를 지켜 해상도를 올리지 않았다 — 다음 라운드에서 별도로 착수할지 결정 필요.
