# OSN-GS 문서 안내

## 현재 기준 문서

- [Urgent Work Master](Urgent_Work/OSN_GS_Urgent_Work_Master.md): 현재 방향, 활성 작업, 승인 경계의 단일 기준
- [작업로그 보존 정책](worklogs/README.md): 유지 중인 최소 검증 기록 목록
- [Architecture](architecture.md): 프레임워크 수준의 설계 결정
- [NURBS Construction](nurbs_construction.md): NURBS 중간 표현과 구현 계약

## 현재 상태

현재는 두 범위를 병행한다.

- 모든 topology에 공통인 isolated Boundary-first visible-surface hardening. 기본 dispatcher나 production path에는 아직 연결하지 않는다.
- Uncertain Gaussian의 proposal/append/ownership model foundation. optimizer, trainer, renderer, checkpoint 및 global selection 통합은 범위 밖이다.
- NURBS Construction benchmark는 depth-bearing 3D shell과 baseline-like flattened covariance를 기본 입력으로 사용한다. [Worklog 111](worklogs/111_nurbs_construction_synthetic_3d_gaussian_dataset.md)을 따른다.
- covariance-guided pairwise affinity 위에 consensus-aware surface-region candidate foundation을 추가했다. 이는 ordered boundary/builder/production path와 연결되지 않은 격리 진단 단계다. [Worklog 116](worklogs/116_consensus_aware_surface_region_formation_foundation.md)를 따른다.
- trimmed-component Jacobian test-health 부채를 해소했고, 최신 전체 pytest는 `568 passed, 1 skipped, 1 warning, 8 subtests passed`다. [Worklog 114](worklogs/114_trimmed_component_jacobian_test_health.md)와 [Worklog 116](worklogs/116_consensus_aware_surface_region_formation_foundation.md)을 따른다.

이전 실험·폐기 방향의 상세 작업로그는 작업 트리에서 제거했다. 필요한 경우 Git 이력으로 조회하며, 현재 결정을 위해 과거 로그를 canonical source로 사용하지 않는다.
## 2026-07-29 Boundary-first NURBS materialization 상태

- consensus-aware region formation의 real trained Gaussian 결과는 full sheet segmentation이 아니라 reliable-core-only core-island 추출로 해석한다. Worklog 117을 따른다.
- world-space half-edge와 ordered graph의 admissible closed outer loop는 canonical evaluable visible NURBS materialization adapter로 전달한다. dispatcher, builder, production path에는 연결하지 않는다. Worklog 120을 따른다.
- 최신 전체 pytest: 577 passed, 1 skipped, 1 warning, 8 subtests passed.
- Worklog 123: [Canonical Tangent Frame Invariance Repair](worklogs/123_canonical_tangent_frame_invariance_repair.md) — Gaussian-only smooth_curved_sheet의 rotation/scale/shuffle/sign-equivalence 안정성 수정 및 검증.

## 2026-07-30 canonical visible NURBS 학습 통합

- `train.py`의 visible NURBS 초기화·주기적 재구축·file/stream payload는 이제 `construct_visible_nurbs_from_gaussians` 하나만 사용한다. `legacy`, `voxel_patch_stage1`, IDW/local split fallback과 해당 CLI는 제거했다.
- 대규모 점군은 deterministic voxel-center 표본(`canonical_construction_max_points`, 기본 2048)에서 canonical topology를 구축하고 ownership/UV/covariance frame을 전체 Gaussian으로 전파한다.
- 지원되는 curved sheet의 실제 trainer 1 iteration은 통과한다. 현재 canonical 범위 밖인 로컬 `DATASET` 복합 장면은 `review_required`로 fail-closed하며 fallback을 사용하지 않는다.
- 상세 근거와 남은 production blocker는 [Worklog 124](worklogs/124_canonical_visible_nurbs_training_integration.md)를 따른다.
- Worklog 124 최종 repository-wide pytest: `570 passed, 1 skipped, 2 warnings, 8 subtests passed in 153.38s`.
## 2026-07-30 ADC 동기화 canonical visible NURBS 실험

- 기본 `initialize` 스케줄은 그대로 유지한다. 실험 플래그 `--visible_nurbs_update_schedule adc_post_commit`은 초기 NURBS 없이 Gaussian 학습을 시작하고 구조적 ADC와 Gaussian optimizer commit 뒤에만 detached canonical 재구축한다.
- 실패/review/0 surface는 stale patch·optimizer·visible binding을 제거한다. clone/split/prune/checkpoint를 통과하는 stable Gaussian ID와 event별 sample/full/opacity coverage·fingerprint·runtime JSONL 진단을 추가했다.
- controlled multi-ADC 대조 실험은 Gaussian trainable tensor bitwise equality를 포함해 통과했다. 실제 `DATASET` 6-iteration/3-event CUDA 실험도 실행했다. 모든 bounded canonical sample은 `no_admissible_region`으로 fail-closed했고 stale NURBS 없이 Gaussian ADC는 계속됐다. cap sensitivity(512/1024/2048)도 같은 결론이었다.
- 최신 repository-wide pytest: `578 passed, 1 skipped, 1 warning, 8 subtests passed in 132.85s`. 실데이터 CUDA ADC는 run-to-run bitwise 재현적이지 않아 model equality는 controlled CPU 대조로만 보장한다. 최종 판정은 `PARTIALLY_SUPPORTED`다.
- 상세 구현·검증·남은 위험은 [Worklog 126](worklogs/126_adc_synchronized_canonical_visible_nurbs_experiment.md)을 따른다. 노트북 Train 셀은 [Worklog 127](worklogs/127_notebook_canonical_adc_nurbs_schedule.md)처럼 현재 canonical 경로와 `adc_post_commit` 스케줄을 기본으로 전달한다.

## 2026-07-30 WebRenderer Gaussian 진단

- `WebRenderer`는 PLY와 training stream 양쪽에서 certain/uncertain reliability, confidence, canonical surface ownership, NURBS patch ID를 시각화한다.
- field 없는 기존 Graphdeco PLY는 계속 로드되며 diagnostic mode에서는 중립 회색으로 나타난다. 상세 계약과 검증 한계는 [Worklog 128](worklogs/128_webrenderer_gaussian_diagnostics.md)을 따른다.
