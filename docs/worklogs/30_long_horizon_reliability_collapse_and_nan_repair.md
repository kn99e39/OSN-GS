# Worklog 30 — Visible Surface Constructor Long-Horizon Reliability Collapse 및 NaN 복구

## 최종 질문에 대한 결론 먼저

> 장시간 학습에서 Visible Surface Constructor가 reliable region 형성 전에 붕괴하는 정확한 원인은 무엇이며, 동일 constructor semantics를 유지하면서 이를 실제 1.6M~3M Gaussian 입력에서 복구했는가?

두 개의 독립적이고 실증된 결함을 찾아 constructor 내부에서 좁게 수정했다.

- **10k iteration 크래시의 진짜 원인은 NaN이 아니었다.** cuSOLVER 배치 eigensolver(`cusolverDnXsyevBatched`)가 이 GPU/드라이버 조합에서 배치 크기 약 2,064,888개를 넘으면 완전히 finite한 입력에서도 하드 크래시한다는 것을 순수 synthetic identity-matrix 배치로 직접 재현했다. `extract_covariance_frame`이 이 한계 없이 전체 관측 클라우드를 한 번에 배치 처리하고 있었다. Chunking으로 수정 → 10k에서 크래시 완전히 제거, 결과는 수학적으로 완전히 동일.
- **3k/5k(그리고 수정 후 10k)의 `reliable_count≈0`은 intrinsic covariance 붕괴가 아니라 contextual evidence 오염이었다.** 대표점(representative)의 97~98%가 intrinsic으로는 reliable했지만, `assign_nearest_representative`가 대표점 2048개에 대해 전체 관측 클라우드(1.6M~3M)를 아무 반경 제한 없이 global nearest-Voronoi로 배정하면서 대표점 하나가 최대 23,881개의, 공간적으로 서로 무관한 Gaussian을 "이웃 증거"로 흡수하고 있었다. Local-radius 경계(대표점 자신의 intrinsic tangent scale × 6)를 도입해 evidence aggregation을 실제 로컬 이웃으로 제한 → `tangent_residual_mean` 중앙값이 4.6~5.7에서 1.6~1.7로 즉시 개선, `reliable_count`가 4→7(3k), 2→7(5k), 0→9(10k)로 개선. **다만 threshold(0.35) 대비 여전히 ~5배 높은 상태로 남아, 완전히 "건강한" 수준은 아니다 — 이는 정직하게 disclosed된 별도의 남은 병목이다(아래 §15).**

Raw learned Gaussian state(position/scale/quaternion/covariance) 자체는 3k/5k/10k 세 스냅샷 전부에서 **100% finite**였다 — C1(raw state invalid)은 명확히 기각됐다.

## 1. 스냅샷과 fingerprint

| iteration | Gaussian count | SHA-256 |
|---|---|---|
| 3000 | 1,603,415 | `de6d93e2b0367c7e64de04bc86b04ea30a6f2133805f878ed9e354596cd242e8` |
| 5000 | 1,863,580 | `462335c31bb52fd6c3a0124dad2f17b412b0ee660632ae61c75122ebff62ad2b` |
| 10000 | 3,038,441 | `e31af8b51c4a5115d206bf8d0fbb7225219c88584f3294a7f66c084c6a9e77c3` |

Git commit(작업 시작 시점): `d359c5ebf23f15d99e60267f3978021e8a86902d`. GPU: NVIDIA GeForce RTX 5080, CUDA 13.0, PyTorch 2.12.1+cu130.

## 2. Offline reproduction

새 스크립트 `scripts/devtools/replay_long_horizon_snapshot.py`와 `scripts/devtools/diagnose_long_horizon_reliability_collapse.py`가 checkpoint(`output/osn_gs_scene/{it}/checkpoint.pt`, `format_version=2`)의 raw model tensor를 직접 `TorchGaussianModel.replace_tensors(...)`로 로드하고, `reconstruct_visible_after_adc`가 실제로 호출하는 것과 정확히 동일한 코드 경로(`covariance_from_scale_rotation(model.get_scaling, model.get_rotation)` → `TorchOSNGSPipeline._construct_canonical_with_full_evidence`)를 재현한다. Training을 재실행하지 않았다.

수정 전 재현 결과가 실제 학습에서 기록된 `nurbs_surface.json`과 정확히 일치했다:

```text
3k → reliable_count=0, no_admissible_region, boundary_failure_stage=A_candidate_generation_failed
5k → reliable_count=1, no_admissible_region, boundary_failure_stage=A_candidate_generation_failed
10k → CUSOLVER_STATUS_INVALID_VALUE 크래시 (실제 학습 로그와 동일 메시지)
```

## 3. NaN 최초 발생 위치 — 사실은 어디에도 없었다

`diagnose_long_horizon_reliability_collapse.py`가 raw position → raw log-scale → raw quaternion → activated scale(exp) → normalized quaternion → covariance → symmetrized covariance까지 모든 단계에서 finite/NaN/Inf 카운트를 기록했다. **세 스냅샷 전부, 모든 단계에서 non-finite 값이 0개였다.**

```text
3000: N1~N4 전 단계 nonfinite_row_count = 0 (1,603,415/1,603,415 finite)
5000: N1~N4 전 단계 nonfinite_row_count = 0 (1,863,580/1,863,580 finite)
10000: N1~N4 전 단계 nonfinite_row_count = 0 (3,038,441/3,038,441 finite)
```

그런데도 10k에서 `torch.linalg.eigh(symmetric)`을 직접 호출하면 완전히 finite한 입력에서 동일한 `CUSOLVER_STATUS_INVALID_VALUE` 예외가 재현됐다. 이를 완전히 분리해 확인하기 위해 실제 checkpoint 데이터와 무관한 순수 synthetic 배치(모두 단위행렬)로 이진 탐색했다:

```text
2,064,887개 배치 → 성공
2,064,888개 배치 → CUSOLVER_STATUS_INVALID_VALUE
```

**이 예외 메시지의 "may appear if the input matrix contains NaN" 힌트는 이 케이스에서는 완전히 오도됐다.** 실제 원인은 `cusolverDnXsyevBatched`의 undocumented 배치 크기 상한이다(N5 확정, 단 원인은 "NaN 데이터"가 아니라 "배치 크기"). 이 상한은 GPU/드라이버/cuSOLVER 버전에 따라 달라질 수 있는 값이라 정확한 경계값에 의존하지 않고 훨씬 보수적인 청크 크기(1,000,000)를 채택했다.

부가 확인: 작은 배치(100개)에 실제 NaN 하나를 주입해도 cuSOLVER는 크래시하지 않고 해당 행만 NaN eigenvalue를 반환했다(`covariance_conditioning_score`가 기존에 이미 이를 `intrinsic_rejected`로 정확히 분류). 즉 진짜 NaN 데이터가 있었다면 애초에 크래시가 아니라 정상적인 reject 분류로 처리됐을 것이다 — crash는 순전히 batch-size 문제였다.

## 4. NaN/배치-한계 복구 (R1)

`osn_gs/surface/torch_gaussian_covariance_frame.py`에 `_batched_eigh()` 헬퍼를 추가하고 `extract_covariance_frame`의 `torch.linalg.eigh` 호출을 이걸로 교체했다. eigh는 독립된 (3,3) 블록 배치 연산이라 청크로 나눠 이어붙여도 수학적으로 완전히 동일하다 — frame semantics 변경 없음, eigenframe 대체 경로도 도입하지 않았다(요청받은 "learned scale/quaternion direct-frame 경로" 대안은 필요 없었다: 문제가 애초에 covariance 품질이 아니라 배치 크기였으므로).

검증:
- 3,038,441개 단위행렬 배치를 `extract_covariance_frame`에 직접 통과 → 크래시 없이 성공.
- `tests/test_long_horizon_reliability_collapse_repair.py::ChunkedEighEquivalenceTest`: 5000개 무작위 covariance에서 청크(강제로 777개씩) 결과와 비청크 결과가 `torch.testing.assert_close`로 완전히 일치(`eigenvalues`, `normal_candidate`, `tangent_major_scale`, `shape_class` 전부). 한계 미만(50개)에서는 기존 단일 호출과 동일함도 별도로 검증.

## 5. 3k/5k reliability rejection waterfall

수정 전, 실제 production 경로(`_construct_canonical_with_full_evidence`, cap=2048)를 그대로 실행해 각 단계를 분해했다.

| iteration | 대표점 intrinsic reliable | 대표점 intrinsic ambiguous | **최종 reliable** | **최종 ambiguous** |
|---|---|---|---|---|
| 3000 | 2004 / 2048 (97.9%) | 44 | **4** | 2044 |
| 5000 | 1958 / 2048 (95.6%) | 90 | **2** | 2046 |
| 10000 | 1903 / 2048 (92.9%) | 144 | **0** | 2047 |

질문 A/B/C에 대한 답:

```text
A. intrinsic reliable representative 자체가 거의 없는가? -> 아니다. 92.9~97.9%가 intrinsic reliable.
B. intrinsic reliable은 충분하지만 contextual mixed/insufficient로 전부 내려가는가? -> 그렇다. 명확하게.
C. representative selection 단계에서 structurally valid mode가 선택되지 않는가? -> 아니다. selection_mode=weighted_farthest_point가 모드를 잘 커버했다(occupied_cell_count 1476~1667, multi_mode_cell_count 1399~1615 — 대부분의 cell에서 여러 mode가 정상적으로 보존됨).
```

`evaluate_contextual_consistency_from_full_evidence`의 게이트 `consistent_max_mutual_tangent_residual=0.35`를 실제로 결정하는 evidence 값:

| iteration | tangent_residual_mean 중앙값 | p90 | 대표점당 source count 평균 | 최대 |
|---|---|---|---|---|
| 3000 | 4.59 (임계값의 13배) | 11.36 | 728 | 7,714 |
| 5000 | 5.06 (14배) | 14.09 | 832 | 11,999 |
| 10000 | 5.69 (16배) | 18.11 | 1,339 | 23,881 |

Gaussian 수가 늘수록 대표점당 source count와 tangent_residual이 단조 증가 — density-scaling failure 가설(C6)과 정확히 일치.

## 6. 근본 원인 — unbounded global Voronoi 배정

`torch_full_neighborhood_evidence.py::assign_nearest_representative`는 `torch.cdist(...).min(dim=1)`로 전체 관측 클라우드를 대표점 2048개에 대해 **반경 제한 없이** 최근접 배정한다. cap이 고정된 채 Gaussian 수가 늘면 각 대표점의 "Voronoi cell"이 물리적으로 훨씬 넓은 영역(그리고 완전히 다른 곡률/구조)을 흡수하게 되고, `tangent_residual_mean`(대표점 자신의 tangent scale로 정규화된 offset)이 구조적으로 폭증한다. `full_evidence_saturating_support_count=24`라는 기존 threshold 값 자체가 "지역적 이웃 수십 개" 규모를 가정하고 있었다는 것도 방증이다 — 실측 대표점당 source count(728~23,881)는 이 threshold가 가정한 규모를 30~1000배 초과한다.

## 7. 139k 스냅샷과의 비교 — 데이터 없음, 정직하게 disclosed

Worklog 129가 언급한 ~139k-Gaussian 실제 DATASET 스냅샷이나 worklog 133의 replay artifact(`C:\tmp\osn_gs_mode_aware_selection_replay_v3.pt`)를 파일시스템에서 찾을 수 없었다(`C:\tmp` 하위에 없음). 이번 세션에서 직접 재현 가능한 재현 데이터가 없어 byte-identical 비교는 하지 못했다. 대신 §5의 실측 대표점당 source count(3k: 728, 5k: 832, 10k: 1339)와 worklog 129 자체가 기록한 ~139k/2048 ≈ 68이라는 비율을 대조하면, 이번 3k~10k 스냅샷은 그 10~20배에 달하는 대표점당 source 밀도를 갖고 있다 — Gaussian 수 증가 + 고정 cap → 대표점당 source population 증가 → 서로 다른 local surface mode/evidence 혼합이라는 가설과 일치하는 간접 증거다. Byte-identical 재현이 필요하면 별도 세션에서 replay artifact를 재수집해야 한다(scope 밖으로 disclosed).

## 8. Diagnostic-only representative cap sweep (수정 후)

Production default(2048)는 바꾸지 않았다. Region/boundary/NURBS fitting까지는 실행하지 않고 reliability 단계까지만 sweep했다(8192는 GPU 메모리/시간상 3000/10000 두 스냅샷만).

| iteration | cap | 최종 reliable | tangent_residual_mean 중앙값 | construction_seconds |
|---|---|---|---|---|
| 3000 | 2048 | 7 | 1.72 | 20.0s |
| 3000 | 4096 | 11 | 1.67 | 24.2s |
| 3000 | 8192 | 14 | 1.57 | 32.7s |
| 10000 | 2048 | 9 | 1.68 | 35.1s |
| 10000 | 4096 | 26 | 1.62 | 39.5s |

해석: cap을 늘리면 reliable_count가 절대 개수로는 완만히 증가하지만(대표점 수 자체가 늘므로 당연함), **tangent_residual_mean은 거의 변화가 없다(1.57~1.72 범위 고정)**. 이는 local-radius 수정 이후에는 cap 크기가 더 이상 핵심 병목이 아님을 의미한다 — C4(fixed-cap mode mixing)는 local-radius 수정으로 이미 대부분 해소됐고, cap을 더 늘리는 것은 이 시점에서 근본 해결책이 아니다(§15 참고). Production cap은 변경하지 않았다.

## 9. Representative selection coverage audit

`selection_diagnostics`(수정 전/후 동일값 — selection 자체는 건드리지 않음)에서 3000/5000/10000 전부 `selection_mode=weighted_farthest_point`, `occupied_cell_count` 1476~1667, `multi_mode_cell_count` 1399~1615(전체 occupied cell의 90%+가 multi-mode)로, mode-aware 후보 생성과 FPS 선택이 정상적으로 다양한 mode를 커버하고 있음을 확인했다. 이번 세션에서 FPS 정책은 변경하지 않았다(요청받은 대로).

## 10. 적용한 narrow repair

**R1 — `osn_gs/surface/torch_gaussian_covariance_frame.py`**: `_EIGH_MAX_BATCH_SIZE = 1_000_000`, `_batched_eigh()` 헬퍼 추가, `extract_covariance_frame`이 이를 사용하도록 교체. 순수 배치 분할이라 frame semantics 변경 없음.

**R5 — `osn_gs/surface/torch_full_neighborhood_evidence.py`**: `FullNeighborhoodEvidenceConfig`에 `local_radius_tangent_scale_multiplier=6.0`(worklog130의 continuation shell이 이미 쓰는 `6 × tangent_major_scale` 관례를 그대로 재사용 — 새 상수 발명 아님), `local_radius_min_absolute=1e-6` 추가. `compute_full_neighborhood_evidence`가 `assign_nearest_representative`의 결과(`nearest`/`spacing`)는 그대로 두되(다른 호출자 — continuation shell, propagation — 에 영향 없음), 자기 자신의 evidence aggregation(`support_count`와 그로부터 파생되는 모든 값)만 local radius 이내 member로 제한하도록 수정. `out_of_local_radius_count` 필드를 새로 추가해 제외된 member 수를 투명하게 기록(silent filtering 아님).

**진단 전용 추가 — `osn_gs/surface/torch_visible_surface_construction.py`**: `diagnostic_summary`에 `reliability_failure_stage`(`intrinsic_reliability_collapse` / `contextual_reliability_collapse` / `partial_contextual_reliability_collapse` / `not_failed`)와 `intrinsic_reliable_count`/`intrinsic_ambiguous_count`/`intrinsic_rejected_count`를 추가 — 기존 `construction_state`/`_state()`/public API는 전혀 건드리지 않은 순수 additive 필드.

**Reliability threshold(0.35 등)는 전혀 손대지 않았다.** Representative cap도 production default(2048)를 그대로 유지했다.

## 11. Repair 후 동일 snapshot replay

| iteration | reliable_count (전→후) | construction_state (후) | 크래시 |
|---|---|---|---|
| 3000 | 4 → **7** | no_admissible_region | 없음(원래도 없었음) |
| 5000 | 2 → **7** | no_admissible_region | 없음(원래도 없었음) |
| 10000 | 크래시(0) → **9** | no_admissible_region | **제거됨** |

10k는 크래시가 완전히 사라지고 `no_admissible_region`으로 정상 완료됐다(0/2048 reliable, 1 rejected). 3k/10k reliable_count가 개선됐지만 **여전히 0.3~0.4% 수준으로 절대적으로 낮다** — "완료"로 간주하지 않는다. `tangent_residual_mean`이 threshold 대비 여전히 ~5배 높은 채로 남아 있다(§13에서 원인 분석).

## 12. Runtime / memory

Chunked eigh는 3,038,441개 배치에서도 크래시 없이 동작했고, `_construct_canonical_with_full_evidence` 전체 runtime은 3k 20.0s → 5k 22.7s → 10k 35.1s(전부 fix 적용 후, cap=2048)로 worklog 131/132가 이미 달성한 GPU synchronization 최적화 범위 내에 있다. Local-radius 마스킹은 기존 `index_add_` scatter 연산에 elementwise 곱 하나를 추가하는 정도라 별도의 O(N) 이상 비용을 추가하지 않았다. Detailed audit(diagnose 스크립트)은 별도 offline 스크립트로만 존재하고 production hot path에는 포함되지 않는다.

## 13. C1~C7 최종 판정

```text
C1. Raw learned Gaussian state invalid -> 기각. 3k/5k/10k 전부 100% finite 확인.
C2. Constructor covariance/frame derivation bug -> 확정, 수정함(R1). 단 "covariance 품질" 버그가 아니라 "배치 크기 한계 미고려" 버그였다.
C3. Intrinsic covariance quality collapse -> 기각. intrinsic reliable 92.9~97.9%로 항상 건강했다.
C4. Fixed representative cap에 의한 contextual mode mixing -> 부분 기여, R5로 대부분 해소. cap sweep(§8)에서 residual이 cap에 거의 반응하지 않는 것으로 확인.
C5. Representative selection spatial/structural coverage loss -> 기각. selection diagnostics(§9)에서 다양한 mode가 정상적으로 대표점에 반영되고 있음을 확인.
C6. Full-neighborhood reliability definition의 density-scaling failure -> 확정, 이번 작업의 핵심 root cause. R5로 부분 수정(tangent_residual 중앙값 13~16배->약 5배 초과로 개선, 완전 해소는 아님).
C7. 복합 원인 -> 해당. 크래시는 C2 단독, reliable_count=0 붕괴는 C6(주) + C4(부) 복합.
```

## 14. Focused / full pytest

- `tests/test_long_horizon_reliability_collapse_repair.py`(신규, 5 tests): chunked-vs-unchunked eigh 완전 일치, 한계 미만 시 기존 동작 유지, local-radius가 실제로 먼 cluster를 배제하는지, 전부 로컬일 때 아무것도 배제하지 않는지(no-op 회귀 가드), config 필드 존재 확인.
- `tests/test_visible_surface_construction.py`(1 test 추가): `reliability_failure_stage`가 실제 intrinsic/final reliable count와 논리적으로 모순되지 않는지.
- 관련 기존 suite 재확인: `test_density_preserving_representative_selection.py` + `test_full_cloud_continuation_shell.py` + `test_adc_synchronized_visible_nurbs.py` + `test_surface_ownership.py` = 66/66 pass(회귀 없음).
- **Repository-wide `pytest`: 604 passed, 1 skipped, 0 failed, 8 subtests passed** (150.68s). 세션 시작 시점 메모리에 기록됐던 "2개 unrelated topology 실패"는 이 시점에는 이미 사라져 있었다(다른 동시 세션에 의한 것으로 보이며, 이번 작업과 무관 — 조사하지 않았다).

## 15. 다음 Visible Surface Constructor 병목

R5 적용 후에도 `tangent_residual_mean` 중앙값(1.6~1.7)이 여전히 `consistent_max_mutual_tangent_residual=0.35`의 약 5배다. Cap sweep(§8)에서 이 값이 cap 크기에 거의 반응하지 않는 것으로 볼 때, 남은 원인은 cap도 locality도 아니라 **정규화 분모의 선택**으로 보인다: 현재 `tangent_residual`은 각 대표점 "자기 자신"의 학습된 `tangent_major_scale`(개별 Gaussian 하나의 covariance 크기)로 정규화되는데, 실제 baseline 3DGS 학습 Gaussian은 개별 크기가 로컬 point spacing보다 훨씬 작은 경우가 흔하다(worklog 134에서 다룬 baseline anisotropy 통계와 일관). 즉 "이 대표점 근처의 실제 표면 규모"를 개별 Gaussian 하나의 자기 자신 covariance로 추정하는 것 자체가 real 학습 데이터에서는 지나치게 작은 분모가 되어 residual을 구조적으로 부풀릴 수 있다는 가설이다 — **검증되지 않았고, 이번 세션에서 추측성으로 수정하지 않았다.** 다음 라운드에서 별도로 조사할 병목으로 남긴다(threshold 완화나 scale 정규화 방식 변경은 여러 정책을 동시에 추측 수정하지 말라는 이번 작업 지침에 따라 보류).

## 16. 남겨진 non-goal (이번 작업 범위 밖, 확인만 하고 손대지 않음)

open boundary NURBS fitting, multi-loop ownership, annulus materialization, derived seam 생성, sphere chart atlas, boundary linking 정책, representative cap의 production 기본값 변경, FPS/selection 정책 변경, reliability threshold 완화, scene-specific tuning, ADC cadence, Uncertain/Occluded pipeline, worklog 134(synthetic surface-aligned covariance) 확장.
