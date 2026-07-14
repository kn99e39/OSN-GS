# TODO: NURBS patch aspect-ratio mismatch (elongated/sliver patches)

상태: **원인 확인 완료, 수정 미착수.** 다른 세션/컨텍스트에서 이어서 작업할 수 있도록 재현 방법과 근본 원인, 수정 방향을 전부 이 문서에 남겨둔다. 배경 지식은 `docs/nurbs_construction.md`(전체 파이프라인)와 `docs/voxel_role.md`(voxel의 역할)를 먼저 읽을 것.

## 증상

`3DGS_Renderer`/`WebRenderer`에서 `nurbs_constructor_benchmark`가 만든 `NURBS_output/crease/nurbs_surface.json`을 로드하면, NURBS iso-parametric wireframe 격자 상당수가 화면의 한 점 근처로 수렴하는 부채꼴(fan) 형태로 보인다. 처음엔 카메라 원근 효과로 의심했으나, 실제로는 **fitting 결과 자체가 비정상적으로 가늘고 긴 형태**임을 수치로 확인했다.

## 재현 방법

```bash
python -m nurbs_constructor_benchmark --scenes crease
```

`nurbs_constructor_benchmark/results/NURBS_output/crease/nurbs_surface.json`을 3DGS_Renderer/WebRenderer(`RENDERER_INPUT_FORMAT.md` 참고)에 로드해서 iso-parametric wireframe을 보면 재현된다. 코드로는 아래 스크립트로 문제를 직접 확인할 수 있다:

```python
import torch
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from nurbs_constructor_benchmark.scenes import make_scene

scene = make_scene('crease', 600, 0, 0.0)
config = TorchPipelineConfig(base_curve_count=4, visible_surface_resolution_u=8, visible_surface_resolution_v=4,
                              surface_fit_mode='lsq', use_voxel_surface_regions=True, voxel_grid_resolution=6,
                              adaptive_voxel_density=False, voxel_max_subdivision_depth=0, max_surface_control_points=4096)
pipeline = TorchOSNGSPipeline(config, device='cpu')
state = pipeline.initialize(scene.points, scene.colors)

for patch_id in range(len(state.surface_patches)):
    mask = state.model.cluster_ids == patch_id
    pts = state.model.get_xyz[mask]
    centered = pts - pts.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh.T
    extent = coords.max(dim=0).values - coords.min(dim=0).values
    shape = state.surface_patches[patch_id].control_grid.shape
    print(f"patch {patch_id}: true_aspect={extent[0]/extent[1]:.2f} grid_aspect={shape[0]/shape[1]:.2f}")
```

측정 결과 (2026-07-14, seed=0, points=600):

```text
patch 0: n=104  실제 PCA extent = 1.93 x 0.36  (진짜 aspect 5.31:1)  grid = 7x3 (aspect 2.33:1)
patch 1: n=105  실제 aspect 5.36:1                                  grid = 7x3 (aspect 2.33:1)
patch 2: n=298  실제 aspect 1.91:1                                  grid = 8x4 (aspect 2.00:1)  <- 거의 일치, 정상
patch 3: n=93   실제 aspect 5.54:1                                  grid = 7x3 (aspect 2.33:1)
```

patch 0/1/3(4개 중 3개, `crease` scene에서 능선을 따라 갈라진 얇은 조각들)은 실제로 5.3배 이상 길쭉한데 2.33:1짜리 grid로 fitting되고 있다. patch 2(가장 크고 정사각형에 가까운 덩어리)만 우연히 실제 형태와 grid 비율이 맞아서 정상으로 보인다.

## 근본 원인

### 원인 A — `pca_parameterize_points`가 두 축을 독립적으로 `[0,1]`로 정규화한다

`osn_gs/surface/torch_nurbs.py:306`:

```python
def pca_parameterize_points(points):
    ...
    coord_min = coords.min(dim=0).values
    span = torch.clamp(coords.max(dim=0).values - coord_min, min=1e-6)
    return torch.clamp((coords - coord_min) / span, 0.0, 1.0)
```

PC1(axis 0)이 실제로 1.93 단위를 커버하든, PC2(axis 1)가 0.36 단위만 커버하든 상관없이 **둘 다 그냥 `[0,1]`로 늘어난다.** 이 시점에 이미 "이 patch가 실제로는 5배 이상 길쭉하다"는 정보가 UV 파라미터화에서 사라진다.

### 원인 B — `_fit_surface_patches`가 모든 patch에 동일한 고정 aspect ratio를 쓴다

`osn_gs/core/torch_pipeline.py:341`, 특히 388행 부근:

```python
aspect = float(base_u) / float(base_v)   # base_u/base_v = 전역 config (기본 8:4 = 2:1), 모든 patch 공통
resolution_u = max(2, min(base_u, int(round((target * aspect) ** 0.5))))
resolution_v = max(2, min(base_v, int(round(target / resolution_u))))
```

Control-point 예산(`target`)은 patch별 밀도로 배분되지만(`docs/nurbs_construction.md` Stage C), **그 예산을 U/V로 쪼개는 aspect ratio는 항상 전역 설정값(기본 2:1)** 이다. patch의 실제 PCA extent 비율은 전혀 반영되지 않는다.

### 두 원인의 결합 효과

원인 A 때문에 짧은 축(v, 실제 0.36 단위)에 배정된 control point들이 [0,1] 파라미터 공간 전체에 퍼진 것처럼 보이지만 실제 3D 공간에서는 서로 매우 가깝게 뭉쳐 있고, 원인 B 때문에 애초에 짧은 축에 배정되는 control point 개수(3개)도 실제 형태를 고려하지 않은 값이다. 결과적으로 렌더러가 그리는 "v=const" iso-line 9개(각각 짧은 축을 가로지르는 긴 곡선, `WebRenderer/util/NurbsGeometry.js`의 `uGridColor` 라인)가 실제 3D에서 거의 겹쳐 있는 리본이 되고, 카메라가 리본의 긴 축을 따라 보면 화면상 한 점으로 수렴하는 것처럼 보인다.

## 수정 방향 (택1 또는 병행)

### 방향 1 — patch별 실제 aspect ratio를 반영해 `resolution_u`/`resolution_v` 재계산

`_fit_surface_patches`(`osn_gs/core/torch_pipeline.py:341`)에서 각 patch group의 point에 대해 PCA extent(현재 `pca_parameterize_points` 내부에서만 계산되고 버려지는 `coords.max-coords.min`)를 미리 구해서, 388행의 `aspect = base_u/base_v` 대신 `aspect = 실제_extent[0]/실제_extent[1]`을 쓰도록 바꾼다.

- 장점: 국소적인 수정, 다른 로직(LSQ, foot-point projection) 무변경.
- 주의: `target`(control point 예산)은 그대로 두되 U/V 분배만 바뀌므로, 매우 극단적인 aspect(예: 20:1)에서는 `resolution_v`가 2 밑으로 내려가지 않도록 기존 `max(2, ...)` clamp를 유지해야 함. 또한 `_split_failed_patch`(265행)도 동일한 aspect 계산 로직을 복붙하고 있으므로 같이 고쳐야 함(공통 헬퍼로 뽑는 게 좋음).

### 방향 2 — `pca_parameterize_points`가 실제 aspect ratio를 보존하도록 정규화 방식 변경

두 축을 독립적으로 `[0,1]`로 정규화하지 않고, **더 긴 축을 기준으로 정규화**하고 짧은 축은 그 비율만큼만 `[0, short/long]`으로 채우는 방식(non-square parameter domain)으로 바꾸는 안. 이러면 UV 공간 자체가 실제 형태를 반영하게 되어 원인 A가 근본적으로 해소된다.

- 장점: 더 근본적인 수정, `fit_torch_visible_surface`(IDW seed)와 `fit_torch_visible_surface_lsq` 양쪽에 자동으로 적용됨.
- 주의: `TorchNURBSSurface.evaluate()`가 기대하는 `uv ∈ [0,1]^2` 가정과 어긋나지 않는지 확인 필요 (`_bspline_basis_pair`가 clamp(uv, 0, 1)을 하므로, 짧은 축의 유효 범위가 `[0, short/long]`으로 줄어들면 나머지 `(short/long, 1]` 구간은 절대 안 쓰이는 낭비 영역이 됨 — knot vector 자체를 그 범위에 맞게 다시 잡아야 진짜로 의미가 있음). **방향 1보다 훨씬 침습적이라, 먼저 방향 1로 실용적 수정을 하고 방향 2는 별도 실험으로 남기는 것을 권장.**

### 부가 고려사항

- `project_points_to_patches`(`osn_gs/core/torch_pipeline.py:447`)의 foot-point projection(Gauss-Newton)은 U/V 해상도와 무관하게 동작하므로 이 수정과 별개로 안전함 — 다만 patch 모양이 바뀌면 fitting 품질(따라서 foot-point 수렴 속도)이 개선될 것으로 기대.
- `WebRenderer/util/NurbsGeometry.js`의 `buildGeometry`가 `surface.patches[]`를 아직 안 읽고 top-level(`patches[0]`)만 그리는 문제는 **별개의 이슈**다 (렌더러 쪽 코드, 이 리포 범위 밖). 이 TODO의 fitting 수정과는 독립적으로 처리 필요.

## 검증 계획

1. 위 "재현 방법" 스크립트로 수정 전/후 `true_aspect` vs `grid_aspect` 비율이 근접하는지 확인 (목표: 비율 오차 20% 이내).
2. `nurbs_constructor_benchmark --scenes crease`를 돌려 `surface_chart_rms`/`normal_p95_degrees`가 수정 전보다 개선되는지 확인 (현재 baseline: `crease: patches=4 controls=95 fit_rms=0.037949 chart_rms=0.006273 normal_mean=1.68deg` — `nurbs_constructor_benchmark/results/report.json` 참고).
3. `nurbs_constructor_benchmark`의 나머지 3개 scene(`plane`, `sine`, `density_gradient`, 전부 정사각형에 가까운 단일 patch)이 회귀 없는지 확인 — 이 scene들은 aspect 문제가 원래 없었으므로 수치가 거의 그대로여야 함.
4. `tests/` 전체(`python -m unittest discover -s tests`, 현재 26개 통과)가 여전히 통과하는지 확인.
5. (가능하면) `NURBS_output/crease/`를 다시 export해서 실제 렌더러에서 시각적으로 부채꼴이 사라졌는지 확인.

## 참고 자료

- `docs/nurbs_construction.md` — 전체 Gaussian → NURBS 파이프라인 상세 (Stage A~F, 수식, 코드 참조 지도)
- `docs/voxel_role.md` — voxel이 patch topology/fitting 입력/예산 배분에 미치는 역할
- `nurbs_constructor_benchmark/README.md` — 벤치마크 사용법, scene 설명
- `RENDERER_INPUT_FORMAT.md` — 렌더러가 기대하는 `nurbs_surface.json` 스키마
