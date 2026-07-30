# Worklog 125 — Volumetric Gaussian Reliability Scene Dataset

(사용자 요청에 별도 번호 지정은 없었으나, 작업 시작 직전 `ls docs/worklogs/`로 재확인한 결과 동시 진행 중이던 다른 세션이 이미 `124_canonical_visible_nurbs_training_integration.md`를 선점하여 **125**로 번호를 지정함.)

## 0. 배경과 범위

`nurbs_constructor_benchmark/gaussian_reliability_scenes.py`와 `surface_region_adversarial_scenes.py`는 Worklog 111~123 전체(covariance frame → structural reliability → manifold affinity → surface-region formation → world-space boundary half-edge/ordering → visible NURBS materialization)의 유일한 synthetic 입력이었다. 그러나 기존 fixture는 전부 **평면 또는 완만하게 굽은 2차원 sine-height-field**(무한 평면의 국소 패치, 또는 사인파 곡면)였고, 실제 부피를 가지는 3차원 solid는 하나도 없었다.

이번 라운드는 사용자 지시에 따라 **기존 synthetic dataset을 전량 폐기**하고, box(직육면체) / cylinder(원기둥) / sphere(구)에 실제로 Splat된 Gaussian dataset으로 교체했다. 범위는 새 dataset 구축과 이를 사용하는 **전체 다운스트림 테스트(15개 파일) 재작성**까지 포함한다. 이번 라운드에서도 covariance는 여전히 pipeline 전체의 1차 증거로 유지·강조했다(사용자 추가 요청).

## 1. 새 volumetric primitive

`nurbs_constructor_benchmark/gaussian_reliability_scenes.py`를 전면 재작성:

- **`_box_faces`**: 축정렬 box의 6개 면을 각각 `_flat_grid`(기존 helper, 그대로 유지)로 샘플링해 결합. 인접한 두 면은 실제 90도 edge에서 만나고, 코너에서는 3개 면이 동시에 만난다 — 기존 "두 평면이 이루는 crease 하나"보다 훨씬 풍부한 다면체 구조.
- **`_cylinder_surface`**: 옆면은 원주 방향으로 연속 곡률(anisotropic curvature), 축 방향으로는 평평 — 상/하단 캡(flat, ±z normal)과는 원형 crease로 분리.
- **`_sphere_surface`**: Fibonacci-sphere 샘플링으로 극점 특이점 없이 거의 균일한 밀도의 완전히 닫힌(경계 없는) 곡면.
- **`_spherical_patch`**: 곡률 스윕 전용 — 반지름에 따라 arc-length 밀도가 바뀌지 않도록 정규화한 국소 patch(닫힌 구 전체가 아님). 초기 버전은 고정 `count=81`로 전체 닫힌 구를 스윕했다가 반지름이 커질수록 표면적이 커져 밀도가 붕괴, region 수가 0이 되는 버그를 발견하고 국소 patch 방식으로 교체했다.

각 primitive는 기존 `GaussianReliabilityScene(name, positions, covariances, description, group_labels)` 계약을 그대로 유지해, 다운스트림 15개 테스트 파일이 대부분 **scene 이름 교체만으로** 마이그레이션 가능하도록 설계했다.

## 2. 기존 scene → 새 scene 매핑

| 기존 | 신규 | 비고 |
| --- | --- | --- |
| `plane` | `box_face` | box의 한 면(단일 평면 patch) |
| `two_perpendicular_surfaces` | `box` | 6면 전체, 12개 edge, 8개 corner |
| `close_parallel_sheets` | `thin_slab` | 얇은 box의 상/하단 면, normal이 서로 반대(+z/-z) — 기존엔 둘 다 +z였던 것을 실제 물리적 앞/뒷면처럼 수정 |
| `smooth_curved_sheet` | `make_curvature_sweep_scene(radius)` | sine-height-field → 실제 구면 patch, 반지름↔곡률 |
| `isolated_floater` | `box_isolated_floater` | |
| `isotropic_blob` | `box_isotropic_contamination` | |
| `oversized_bridge` / `anisotropic_planar_bridge` | `box_with_bridge` / 동일 함수명 유지 | box 내부에 삽입 |
| (신규) | `cylinder`, `sphere` | 곡면+평면 혼합 / 완전 폐곡면 |
| `make_gap_sweep_scene` | 그대로 유지, 내부 구현만 thin-slab 방식으로 교체 | |
| `make_shape_ratio_sweep_scene` | **변경 없음** | 순수 per-Gaussian covariance-shape 테스트로, surface topology와 무관하므로 유지 |

`surface_region_adversarial_scenes.py`에 신규 **`make_cylinder_phase_alias_scene()`** 추가: cylinder 옆면은 원주 방향으로 주기적이라 반대편 지점의 normal이 정확히 반평행(anti-parallel)이며, `abs(dot(n_i,n_j))`를 쓰는 orientation-insensitive 비교에서는 완전히 정렬(1.0)로 읽힌다 — 기존의 인위적인 sine-sheet "long shortcut" fixture보다 훨씬 사실적인 phase-alias 스트레스 테스트다.

## 3. 발견하고 수정한 버그 (fixture 자체, 알고리즘 아님)

1. **곡률 스윕 밀도 붕괴**: 고정 `count=81`로 닫힌 구 전체를 반지름만 키워가며 스윕했더니, 반지름이 커질수록 표면적 대비 밀도가 떨어져 candidate 자체가 형성되지 않아 region 0개가 나옴. 국소 patch(고정 arc-length extent) 방식으로 교체해 해결.
2. **작은 반지름에서의 자기 교차**: `patch_half_extent=0.5`에 반지름이 그보다 작으면 patch가 반구 이상을 감아 자기 자신과 겹치는 현상 발견 — 실제로 물리적으로도 타당한 현상(작은 공에 큰 평면을 감으면 겹침)이므로 버그로 취급하지 않고, 스윕의 유효 반지름 범위(≥0.5)를 문서화.
3. **곡면 patch의 회전 불변성 경계**: `make_curvature_sweep_scene`의 기본 `count_per_axis=9`에서는 특정 회전/스케일 조합 하나가 `constructed`→`review_required`로 flip하는 경우를 발견. `count_per_axis=11`(밀도 상향)로 기본값을 올려 6종 강체 변환(3회전+2스케일+순서 반전) 전부에서 안정화됨을 확인 후 채택.

## 4. 검증

**신규 primitive 단독 검증** (7개 기본 scene + gap/curvature/density/noise/contamination sweep 전체): 기존과 동일한 정성적 결과 재현 — box 6면 각각 독립 region, cylinder 옆면 1개+캡 2개, sphere 1개(단, 아래 §5 참고), thin_slab 2개, contamination/bridge/floater 전부 기존과 동일한 배제 패턴.

**Sphere의 region-formation 결과 (정직하게 공개하는 한계)**: pairwise `same_surface` connected-component 기준으로는 sphere가 1개 region으로 완전히 연결되지만, Worklog 116의 consensus-aware `form_surface_regions`를 적용하면 4개 region(57/45/5/3)으로 다소 보수적으로 분절된다. Crease 관계는 0건, `boundary_conflict_edge_ids`도 0건으로 확인 — **허위 crease나 허위 경계를 만든 것은 아니며**, 단지 완전히 닫힌 균일 곡률 표면이라는 이번에 처음 등장한 새 테스트 케이스에 대해 consensus/bridge-veto 메커니즘이 병합을 보수적으로 보류하는 것으로 판단된다. 이 라운드의 범위는 dataset 교체이지 region-formation 알고리즘 재조정이 아니므로, 원인 조사와 개선은 다음 라운드로 이월한다.

**다운스트림 테스트 15개 파일 전체 재작성**: `test_gaussian_covariance_frame.py`(변경 없음, scene 미사용), `test_gaussian_structural_reliability.py`, `test_gaussian_manifold_affinity.py`, `test_gaussian_reliability_affinity_robustness.py`, `test_gaussian_surface_region_formation.py`, `test_surface_region_validation.py`, `test_surface_region_adversarial_validation.py`(변경 없음, 이미 통과), `test_surface_region_invariance.py`, `test_surface_region_phase_alias.py`(cylinder 기반으로 교체), `test_world_space_boundary_halfedges.py`, `test_ordered_world_boundary_graph.py`(변경 없음), `test_visible_boundary_materialization_adapter.py`(변경 없음), `test_visible_surface_construction.py`, `test_visible_surface_construction_invariance.py`, `test_synthetic_gaussian_dataset.py`.

**전체 저장소 `pytest`**: **570 passed, 1 skipped, 0 failed** (기존 트래킹 대비 회귀 없음; 별도 세션이 진행 중인 worklog 124 관련 파일도 그대로 green).

## 5. 명시적으로 하지 않은 것

- Region-formation 알고리즘 자체의 재조정(위 sphere 분절 현상은 발견·공개만 하고 수정하지 않음).
- `nurbs_constructor_benchmark/scenes.py`(별도 세션이 다루는 saddle_shell/spherical_cap/folded_roof/wave_annulus 포인트클라우드 벤치마크)는 건드리지 않음 — 이번 라운드는 covariance-bearing `gaussian_reliability_scenes.py` 계열에 한정.
- Default dispatcher, trainer, production pipeline, ownership/checkpoint 비접촉 (기존 규율 그대로 유지 — 이번 라운드는 애초에 프로덕션과 무관한 테스트 fixture 교체 작업).
- 복합/불규칙 solid(사용자가 제시한 3안)는 만들지 않음 — box/cylinder/sphere 기본 라이브러리(2안)까지만 구현.

## 6. 이번 작업의 의의

이번 교체로 covariance-guided reliability/affinity/region-formation/boundary/NURBS 파이프라인 전체가 **처음으로 실제 3차원 부피를 가지는 solid** — 6면 box(다중 edge/corner), 곡률+평면이 섞인 cylinder, 경계 없는 완전 곡면 sphere — 위에서 검증되었다. 이전까지는 무한 평면 국소 patch나 사인파 곡면 같은 사실상 2차원적인 fixture만으로 "표면 하나 vs 표면 둘"을 구분했지만, 이제는 하나의 물체가 스스로 가지는 **여러 개의 서로 다른 face, 여러 개의 real edge/corner, 진짜 폐곡면**을 상대로 같은 판정 로직이 여전히 올바르게 동작하는지 확인할 수 있게 됐다. 특히 box는 기존에 없던 "3개 면이 만나는 corner"를, cylinder는 "평면과 곡면이 공존하는 solid"를, sphere는 "경계가 아예 없는 표면"을 각각 처음으로 테스트 매트릭스에 추가했고, sphere에서 발견된 보수적 분절 현상은 이 dataset 교체가 아니었다면 드러나지 않았을 실제 한계다. 즉 이번 작업은 단순한 fixture 이름 교체가 아니라, 파이프라인이 실제 물체에 적용됐을 때 마주칠 법한 위상적 상황(다중 면, 다중 edge, 폐곡면)을 처음으로 정직하게 시험대에 올린 것이다 — 앞으로 이 파이프라인을 실제 학습된 3DGS 장면(사물, 방, 캐릭터 등 전부 부피를 가진 solid)에 적용하기 전 반드시 거쳐야 했을 검증을, 더 늦기 전에 지금 통과시킨 셈이다.
