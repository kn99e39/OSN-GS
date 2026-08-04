# Worklog 18: canonical visible NURBS 학습 경로 통합

## 수행 작업

- `train.py`와 `TorchOSNGSPipeline.initialize()`의 유일한 visible NURBS 구축 경로를 `construct_visible_nurbs_from_gaussians`로 통합했다.
- `legacy`, `voxel_patch_stage1`, IDW seed fit, local voxel split/refit 구현과 선택 CLI, Stage 1 전용 테스트·ablation 실행기를 제거했다.
- canonical 구축 결과의 region membership을 `cluster_ids`로, NURBS foot-point를 `surface_uv`로 연결하고, 재구축 시 전체 patch registry와 surface optimizer를 함께 교체한다.
- 대규모 입력에서는 voxel 중심 최근접 stable point를 최대 `canonical_construction_max_points`개 선택한다. local-PCA covariance와 O(N^2) topology는 이 표본에서만 계산하고 frame, scale, patch membership, UV를 전체 Gaussian으로 전파한다.
- NURBS file/stream payload source를 canonical construction으로 통일하고 retired voxel/Stage 1 provenance를 제거했다.

## 결과

- 기본 CLI에는 `canonical_covariance_knn`, `canonical_construction_max_points`, covariance scale, projection iteration만 남았으며 constructor selector와 fallback 옵션은 없다.
- canonical curved-sheet 기반 실제 `TorchOSNGSTrainer` 1 iteration 회귀가 통과한다.
- 441개 평면 Gaussian을 81개 canonical 표본으로 구축한 뒤 441개 전체에 membership과 UV가 전파되는 회귀를 추가했다.
- canonical invariance, trainer, NURBS, ownership 대상 테스트는 green이다.

## 평가

- 지원되는 단일 visible sheet 입력에서는 학습 시작부터 저장·streaming·주기적 재구축까지 동일 canonical 파이프라인을 사용한다.
- materialized surface가 없으면 이전 patch를 유지하거나 직사각형/voxel NURBS를 합성하지 않고 명시적으로 실패한다.

## 남은 위험

- 현재 canonical 알고리즘은 arbitrary trained scene/full scene coverage를 아직 보장하지 않는다. 로컬 `DATASET`의 실제 `train.py --iterations 0` 스모크는 canonical 단계까지 진입했지만 `review_required` (`regions=11`, `components=5`, materialized surface 0)로 fail-closed했다. 이는 legacy fallback으로 우회하지 않았다는 증거이면서, 해당 복합 장면을 학습하려면 canonical region/boundary materialization 범위를 별도 확장해야 한다는 production blocker다.
- `canonical_construction_max_points`는 topology 비용을 제한하지만, 장면별 표본 예산과 voxel 대표성의 품질 평가는 추가로 필요하다.
## 검증

- 대상 canonical/trainer/ownership/invariance 회귀: 통과.
- `train.py --help`, `scripts/train_osn_gs_torch.py --help`, `python -m nurbs_constructor_benchmark.runner --help`: 통과하며 production selector 기본은 canonical이다.
- repository-wide pytest: `570 passed, 1 skipped, 2 warnings, 8 subtests passed in 153.38s`.
- 최종 로컬 `DATASET` actual `train.py --iterations 0` 스모크: canonical KNN/topology 단계에 진입한 뒤 예상대로 `review_required` (`regions=11`, `components=5`)로 fail-closed. legacy/voxel fallback 호출 없음.