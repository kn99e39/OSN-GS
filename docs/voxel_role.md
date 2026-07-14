# Voxel Bootstrap이 NURBS 형성에서 하는 역할

Voxel은 NURBS 생성 파이프라인의 부가적인 전처리가 아니라, **"몇 개의 NURBS patch가 필요한가"와 "각 patch가 어떤 데이터를 fitting해야 하는가"를 결정하는 단계**다. 아래 단계(PCA parameterize → IDW seed → LSQ fit)는 voxel이 넘겨준 그룹을 그대로 처리할 뿐이다. 상세 알고리즘은 `docs/nurbs_construction.md`를 참고하고, 이 문서는 voxel이 **다운스트림 각 단계에 정확히 어떤 영향을 주는지**만 정리한다.

관련 발견: `nurbs_constructor_benchmark`의 `crease` scene에서 실제로 나타난 aspect-ratio 버그(가늘고 긴 patch가 정사각형에 가까운 grid로 fitting되는 문제)는 voxel이 만든 patch topology 정보가 다운스트림으로 온전히 전달되지 않아서 생긴다 — 자세한 내용과 수정 계획은 `TODO.md`를 참고.

## 1. Patch topology(NURBS 개수와 경계) 자체를 voxel이 결정한다

`build_torch_voxel_surface_regions`(`osn_gs/surface/torch_voxel_regions.py:31`)가:

1. Gaussian point cloud를 밀도 적응형 voxel cell로 나누고,
2. 각 cell의 normal을 PCA로 추정하고,
3. 인접한 cell 쌍의 normal 각도차가 `voxel_boundary_angle_degrees`(기본 35°)를 넘으면 그 edge를 자르고,
4. 남은 인접 그래프에서 connected component를 구해 `region_patch_ids`를 부여한다.

**하나의 NURBS patch = normal이 부드럽게 이어지는 voxel region들의 연결 성분**이라는 정의 자체가 여기서 나온다. `crease` scene이 4개 patch로 갈라지는 것도, 그중 일부가 가늘고 긴 리본 모양이 되는 것도 전부 이 단계에서 이미 결정된다. Surface fitting은 이 topology를 바꾸지 못한다 — 주어진 그룹을 fitting할 뿐이다.

## 2. Fitting 입력 자체가 voxel region centroid다

`_curve_placement_points`(`osn_gs/core/torch_pipeline.py:521`)는 개별 Gaussian이 아니라 **voxel region의 밀도 가중 평균 위치**를 base curve/surface fitting의 실제 입력(`curve_points`)으로 쓴다. 이는:

- 수만~수십만 개의 raw point를 수십~수백 개의 안정된 centroid로 다운샘플링하고,
- voxel 평균화에 따른 smoothing 효과를 내고,
- `voxel_grid_resolution`과 `adaptive_voxel_density`(subdivision)가 fitting이 실제로 "보는" 데이터의 밀도/분포를 결정하게 만든다.

Voxel을 끄면(`use_voxel_surface_regions=False`) raw Gaussian point가 그대로 쓰인다 (`nurbs_constructor_benchmark --disable-voxel`로 비교 가능).

## 3. Control-point 예산 배분이 voxel 밀도 기반이다

`_fit_surface_patches`(`osn_gs/core/torch_pipeline.py:341`)의 patch별 control point 개수는:

```text
score[patch] = sqrt(sum(voxel region_density in patch)) * (1 + mean(boundary_mask in patch))
```

로 정해진다. Voxel 밀도 정보가 없으면 이 배분 자체가 불가능하고, 전체 patch에 균등 분배할 수밖에 없다.

## 4. LSQ fitting의 point weight도 voxel 밀도다

`_fit_visible_patch`에 전달되는 `point_weights`가 `regions.region_density`다 (`_fit_surface_patches`의 `weight_groups`). 밀도가 높은 voxel region이 정규화 최소제곱(LSQ) fitting에서 더 큰 영향력을 가진다 — `docs/nurbs_construction.md` 5.3절의 `_solve_control_grid_lsq` 정규방정식에서 `weight_k` 항이 바로 이것이다.

## 5. Adaptive voxel density가 비균일 데이터의 fitting 품질을 좌우한다

`adaptive_voxel_density=True`일 때, coarse voxel의 밀도가 `voxel_density_quantile`(기본 0.75) 이상이면 해당 영역만 `voxel_max_subdivision_depth`만큼 더 세분화된다. `nurbs_constructor_benchmark`의 `density_gradient` scene(밀집 중심부 + 희박 주변부)이 정확히 이 경로를 검증하도록 추가됐다: 균일 grid로는 밀집 영역이 과소해상도, 희박 영역이 과대해상도가 되어 `chart_rms`가 다른 scene보다 확연히 나쁘고(0.071 vs sine의 0.017), `--adaptive-voxel`을 켜면 개선된다(0.067).

## 6. 학습 중 국소 재분할(maintenance)도 voxel을 재사용한다

`_split_failed_patch`(`osn_gs/core/torch_pipeline.py:265`)는 지속적으로 fitting 품질이 나쁜 patch를 감지하면, **그 patch에 속한 Gaussian만** (opacity·covariance volume 기반 밀도 가중으로) 다시 voxel화해서 connected component를 구하고, 지배적 component를 제외한 나머지를 새 독립 patch로 떼어낸다. 전역 voxel topology를 다시 만들지 않고, voxel 메커니즘 자체를 국소적으로 재사용하는 것이 이 설계의 핵심이다.

## 요약

| Voxel이 제공하는 것 | 안 쓰면 무엇이 사라지는가 |
|---|---|
| Patch 개수/경계 (connected component) | 전체 scene이 patch 1개로 처리됨 |
| Fitting 입력(curve_points = centroid) | Raw Gaussian point 전체를 그대로 fitting (더 느리고, smoothing 없음) |
| Control-point 예산 배분 근거(밀도) | 모든 patch에 균등 분배 |
| LSQ point weight | 모든 point가 동일 가중치로 fitting |
| 비균일 밀도 대응(adaptive subdivision) | 밀집/희박 영역이 같은 해상도로 처리되어 fitting 품질 저하 |
| 국소 재분할의 근거 데이터 | 지속 실패 patch를 세분화할 방법이 없음 |

**voxel이 하지 않는 것**: 실제 NURBS control point의 3D 좌표 값 자체를 계산하지 않는다(그건 IDW/LSQ의 역할). Voxel은 "무엇을 어떻게 나눠서 fitting할지"를 정하고, 실제 curve/surface 수학은 `osn_gs/surface/torch_nurbs.py`가 담당한다.
