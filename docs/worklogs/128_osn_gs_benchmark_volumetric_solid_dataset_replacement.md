# Worklog 128 — osn-gs benchmark Volumetric Solid Dataset Replacement

(작업 시작 전 `ls docs/worklogs/`로 확인한 결과 127까지 사용 중이어서 **128**로 지정.)

## 0. 배경과 범위

Worklog 125는 `nurbs_constructor_benchmark/gaussian_reliability_scenes.py`(worklog 111-124의 격리된 covariance-guided 실험 파이프라인 전용 fixture)를 box/cylinder/sphere로 교체했다. 그러나 사용자가 지적한 대로, **실제 `osn-gs benchmark` CLI가 로드하는 synthetic dataset은 별도의 모듈**(`nurbs_constructor_benchmark/scenes.py` + `ground_truth.py` + `metrics.py`)이며, 이 dataset은 여전히 `z = f(x, y)` 단일 height-field 가정 위에 만들어져 있었다(`SCENE_NAMES = ("saddle_shell", "spherical_cap", "folded_roof", "wave_annulus")`).

사용자 지시: "synthetic dataset은 오직 osn-gs benchmark에서 사용할 목적으로 만들어지는 것"이므로 이 dataset을 **전량 폐기하고 실제 3D volumetric solid로 대치**하되, ground-truth/metrics(hole count, topology, support-domain 등)까지 **3D-native로 완전히 재설계**한다(옵션 1, 사용자가 명시적으로 선택).

## 1. 핵심 아키텍처 문제와 해법

`ground_truth.py`/`metrics.py`의 hole/topology/support-domain 계산 전부가 `[-1,1]^2` XY 평면 래스터 기반이었다 — box의 옆면처럼 위에서 보면 선으로 뭉개지는 면은 이 모델로 표현 불가능. 해법: `GroundTruthFace` 신규 엔티티(자체 local `[-1,1]^2` 파라미터 도메인 + `to_world`/`to_local`(analytic 역함수)/`normal_fn`/`support_predicate`)를 도입하고, `SyntheticGaussianScene`에 `faces: tuple[GroundTruthFace,...] | None` 필드를 추가했다. `faces`가 설정된 scene(신규 box/cylinder/sphere)은 face-aware 경로를, `None`인 기존 height-field scene(legacy 15종 + 이전 4종)은 기존 경로를 그대로 사용하도록 모든 함수를 분기시켰다 — 기존 legacy 테스트(약 10개 파일, `test_gt_nurbs.py` 등)는 단 한 줄도 건드리지 않고 그대로 통과한다.

## 2. `scenes.py`

- `_box_faces`/`_cylinder_faces`/`_sphere_faces`: 각 면의 정확한 analytic `to_world`/`to_local`/`normal_fn`.
- `_box_oracle`/`_cylinder_oracle`/`_sphere_oracle`: **임의의 3D 쿼리점**(표면 위가 아니어도 됨)에 대해 정확한 signed-distance + normal을 반환하는 closed-form SDF. `runner.py`의 `score_state`가 재구성된 patch 샘플을 채점하려면 이 계약이 필수임을 확인 후 구현.
- `_box_patch_label`/`_cylinder_patch_label`/`_sphere_patch_label`: nearest-face 기반 정확한 topology label (`gt_patch_count`=6/3/1).
- `SCENE_NAMES = ("box", "cylinder", "sphere")`로 교체. 이전 `saddle_shell`/`spherical_cap`/`folded_roof`/`wave_annulus`(height-field 기반 "depth-bearing" 시도)는 코드까지 완전히 삭제했다 — 유일한 소비처(`test_synthetic_gaussian_dataset.py`)도 함께 갱신. 나머지 legacy 15종은 `LEGACY_SCENE_NAMES`로 그대로 유지(다른 KDE/raster 기반 회귀 스위트 전용, `osn-gs benchmark`와 무관).

### 발견하고 수정한 버그 (모두 실제 버그, 이번에 처음 노출됨)

1. **`_tangent_frame_quaternion`의 quaternion 추출 공식이 180도급 회전에서 붕괴**: trace 기반 단일 분기 공식은 trace≈-1일 때 qw와 분자가 동시에 0에 가까워져 예를 들어 normal=(0,0,-1)이 identity quaternion으로 잘못 디코딩됨(box의 모든 축정렬 normal이 정확히 이 경우에 해당). Height-field scene은 normal이 항상 +z 근방의 연속 섭동이라 이 경우를 만난 적이 없었다. Shepperd's method(대각원소 최대값 기준 4-분기 robust 추출)로 교체, 2000개 랜덤 방향 + 6개 축정렬 방향 모두 alignment=1.0 확인.
2. **Box/cylinder면의 random uniform 샘플링이 밀도 분산을 유발**: `torch.rand` 샘플링은 국소 밀도 분산이 커서 (worklog 115와 동일한 종류의 문제) 600점짜리 box 하나가 pairwise same_surface 그래프에서 25개 region으로 허위 분절됨. Grid 기반(+jitter) 샘플링(`_grid_uv`)으로 교체 후 정확히 6개 region으로 정정.
3. **Cylinder 옆면의 원주/높이 물리적 종횡비 불일치**: 정사각 grid를 그대로 쓰면 원주(4.4) 대 높이(2.0) 비율만큼 이방성 밀도가 생겨 여전히 분절됨 — `aspect_ratio` 보정 추가.
4. **Cylinder 옆면 seam 중복점**: local u=-1과 u=+1이 정확히 같은 물리적 각도(0=2π)로 매핑되어 seam 전체가 중복 샘플링되고, 그 중복쌍들이 본체와 단절된 수십 개의 2점짜리 고립 region으로 나타남. Periodic 축의 끝점 하나를 제거(`periodic_u=True`)해 해결. 수정 후 pairwise region은 정확히 3개(옆면+캡2개)로 안정화됨.

## 3. `ground_truth.py`

- `gt_solid_surface_points`: 각 face의 local grid를 `support_predicate`로 거른 뒤 `to_world`로 lift, 전체 face를 합집합 — accuracy/chamfer/completeness metric의 dense GT 표본.
- `observed_gt_surface_points`: volumetric scene일 때 XY만 비교하던 기존 로직(다른 방향을 향한 면에는 의미 없음)을 전체 3D 거리 비교로 교체.
- `gt_nurbs_charts`: face마다 하나의 chart. `circle` support(원기둥 캡)는 trim mask 없이 polar(각도, 반지름) 파라미터화로 boundary-conformal하게 처리(기존 annulus/crescent와 동일한 설계 원칙 유지).
- `TorchNURBSSurface`로 실제 평가해 확인: box 6개 chart 모두 residual < 1.2e-7(정확), cylinder/sphere의 곡면 degree-1 rect chart는 약간의 보간 sag(0.03~0.04)가 있으나 — 이는 GT NURBS 오버레이 **시각화 전용**이며 실제 채점(accuracy/chamfer)은 `gt_surface_points`의 analytic 표본을 직접 사용하므로 채점 정확도에 영향 없음. 기존 "sine" scene도 동일한 종류의 sag(0.006)를 이미 갖고 있었음을 확인(신규 회귀 아님).

## 4. `metrics.py`

- `_assign_patch_to_face`: 재구성된 patch를 `gt_patch_label`의 다수결로 가장 가까운 GT face에 배정.
- `_patch_face_mask`: 기존 `_patch_xy_mask`와 동일한 밀도 적응형 UV 오버샘플링이나, world 좌표를 그대로 쓰지 않고 배정된 face의 `to_local` 역함수로 변환한 뒤 래스터화.
- `_combined_face_raster`: 각 face의 local raster를 gap(1 face-width)을 두고 옆으로 이어붙여, 기존 `_components`/`_holes`/`_boundary_distances`/`_enclosed_hole_mask` 등 모든 2D 헬퍼를 **코드 변경 없이 그대로** 재사용하면서도 서로 다른 face가 flood-fill로 잘못 이어지지 않도록 함.
- `support_domain_metrics`/`patch_union_metrics`: `scene.faces is not None`일 때 위 face-aware 경로로 분기, 그렇지 않으면 기존 코드 100% 그대로. 반환 dict의 키는 전부 동일(신규 `support_face_count`/`union_face_count`만 추가)이라 `runner.py`의 소비 코드는 수정 불필요.

## 5. 실제 파이프라인 검증 (중요한 발견, 정직하게 공개)

`osn-gs benchmark`의 **기본 경로(`--constructor canonical`, worklog 124/126이 연결한 covariance-guided 파이프라인)** 로 box/cylinder/sphere를 600점(CLI 기본값)으로 실행하면 **materialize 실패로 하드 에러**가 발생한다:

```
RuntimeError: Canonical visible-surface construction produced no materialized NURBS
(state='review_required', regions=6, components=46). No legacy or voxel fallback is available.
```

원인 조사 결과 이는 **본 데이터셋의 결함이 아니다** — pairwise same_surface 그래프는 정확히 6/3/1개 region으로 완벽하게 분리되고(§2의 버그 수정 후 확인), covariance/오라클/라벨도 모두 정확하다. 문제는 그 이후 단계인 world-space boundary half-edge/ordered-loop/NURBS materialization(worklog 118-123, 다른 세션이 구축한 별도 시스템)이 **닫힌 다면(multi-face) solid**를 아직 검증해본 적이 없다는 데 있다 — 그 세션들의 자체 기록(worklog 121)도 "clean plane과 curved sheet만" 확인했다고 명시하고 있다. 이를 고치는 것은 이번 "데이터셋 교체" 범위를 크게 벗어나는 별도의 대형 작업이다.

**대안 경로(`--constructor boundary_first`, KDE/raster 기반 기존 비교기)는 세 scene 모두 정상 동작함을 확인**:

| scene | accuracy_rms | chamfer_rms | gt_patch_count | gen_patch_count | support_face_count |
|---|---|---|---|---|---|
| box | 0.0209 | 0.1204 | 6 | 24 | 6 |
| cylinder | 0.0206 | 0.0601 | 3 | 11 | 3 |
| sphere | 0.0124 | 0.0364 | 1 | 8 | 1 |

이는 본 데이터셋+ground-truth+metrics 재설계가 **실제 재구성 파이프라인을 통해 end-to-end로 올바르게 작동**함을 확인시켜준다(face-aware topology/support 지표 포함). `--constructor canonical`이 기본값이므로, 이 갭을 사용자에게 명확히 알린다: **지금 `osn-gs benchmark`를 인자 없이 실행하면 하드 에러가 난다.** `--constructor boundary_first`를 쓰거나, canonical 경로의 닫힌 다면체 지원이 보강될 때까지 기다려야 한다.

## 6. 검증

- 신규 `_grid_uv`/`_box_oracle`/`_cylinder_oracle`/`_sphere_oracle`/`_box_patch_label`/`_cylinder_patch_label` 등을 직접 검증: 표면 위 점 residual < 1.2e-7, 오프셋 점(0.05 이동) residual ≈ 0.05, normal alignment = 1.0(랜덤 2000개 포함), pairwise region 6/3/1 정확히 분리.
- `tests/test_synthetic_gaussian_dataset.py` 재작성: `SCENE_NAMES == ("box","cylinder","sphere")`, 각 scene의 오라클 residual < 1e-4, `faces`와 `gt_patch_count` 일치 확인 추가.
- `tests/test_gt_nurbs.py`(legacy ground_truth.py 최대 소비처, 11+5개 legacy scene 이름을 하드코딩) 및 나머지 legacy 소비 테스트(`test_boundary_central_cap.py`, `test_boundary_component_recovery.py`, `test_boundary_first_support_pipeline.py`, `test_boundary_surface_quality.py`, `test_component_boundary.py`, `test_gaussian_support_continuity.py`, `test_patch_boundary.py`, `test_surface_candidate_graph.py`, `test_surface_decomposition.py`, devtools 스크립트 4개)는 **한 줄도 수정하지 않고 그대로 통과**.
- 전체 저장소 `pytest`: **578 passed, 1 skipped, 0 failed**.

## 7. 명시적으로 하지 않은 것

- `--constructor canonical`(기본값) 경로의 닫힌 다면체 boundary/materialization 강건화 — §5에서 발견한 진짜 문제지만 별도의 대형 작업이며 오늘 범위 밖. worklog 118-123을 만든 다른 세션의 소관.
- Worklog 125의 격리된 실험 fixture(`gaussian_reliability_scenes.py`)는 이미 지난 라운드에서 처리 완료, 이번 라운드는 건드리지 않음.
- Default dispatcher/trainer 변경 없음(애초에 이번 작업은 synthetic benchmark dataset 교체이지 production pipeline 변경이 아님).
- 복합/불규칙 solid는 만들지 않음(box/cylinder/sphere 기본 라이브러리까지만, 사용자가 이전에 선택한 범위).

## 8. 이번 작업의 의의

`osn-gs benchmark`가 실제로 로드하는 synthetic dataset이 처음으로 **진짜 3차원 부피를 가지는 solid**(6면 box, 곡면+평면 혼합 cylinder, 완전 폐곡면 sphere)가 되었고, 그 solid의 ground-truth(정확한 signed-distance 오라클, face별 topology label, face-aware hole/support/topology metric)까지 XY 평면 투영이 아니라 각 면 자신의 좌표계에서 올바르게 계산되도록 재설계했다. 이 과정에서 실제로 존재하던 4가지 결함(quaternion 추출 공식, 샘플링 밀도 분산, 원기둥 종횡비, seam 중복점)을 발견해 고쳤는데, 이들은 전부 "밀도가 균일하고 방향이 하나뿐인 평면"이라는 기존 가정 때문에 지금까지 한 번도 노출된 적이 없던 버그였다. 그리고 이 재구축 과정에서 `--constructor canonical`(현재 기본값이자 최신 covariance-guided 파이프라인)이 아직 닫힌 다면체 solid를 끝까지 처리하지 못한다는, 데이터셋만으로는 가려져 있던 진짜 파이프라인 한계를 발견해 정직하게 기록했다 — 이는 이번 작업이 아니었다면 계속 평면·height-field 데이터로만 테스트되어 드러나지 않았을 문제다. 결과적으로 `osn-gs benchmark`는 이제 실제 3D 물체를 재구성하는 시나리오에 훨씬 가까운 입력과 지표로 평가되며, 어느 부분(데이터셋/ground-truth는 완성, canonical materialization은 아직)이 다음 단계에서 다뤄야 할 진짜 병목인지가 명확해졌다.
