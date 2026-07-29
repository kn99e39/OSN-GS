# NURBS Construction Synthetic Benchmark

이 benchmark는 OSN-GS visible-surface constructor를 실제 3D Gaussian set 형태로 검증한다. 기본 dataset은 점 중심만 흩뿌린 평면 샘플이 아니라, depth extent·법선 회전·표면 접선 방향 covariance를 함께 갖는 관측 Gaussian set이다.

## 기본 3D dataset

`python -m nurbs_constructor_benchmark`는 다음 네 장면을 실행한다.

- `saddle_shell`: 두 주곡률과 큰 depth 변화를 갖는 saddle shell
- `spherical_cap`: lateral extent에 비견되는 depth를 갖는 spherical cap
- `folded_roof`: 두 곡률 방향이 지속되는 rounded folded shell
- `wave_annulus`: 곡면 위 observed inner boundary를 갖는 annular shell

이전 `plane`, `sine`, `crease` 등은 focused compatibility test에서만 이름으로 호출할 수 있는 legacy scene이다. 기본 benchmark population에는 포함하지 않는다.

## Gaussian covariance 계약

각 scene은 `points`, `colors`뿐 아니라 다음을 제공한다.

- `covariance_scales`: linear `(N, 3)` scale
- `covariance_rotations`: normalized wxyz `(N, 4)` quaternion
- `covariance_normals`: local surface normal

covariance는 `output/graphdeco_ab_3k/point_cloud/iteration_3000/point_cloud.ply`의 baseline 3DGS 분포를 표본 분석해 구성한다. baseline의 Gaussian별 major/minor anisotropy는 median `5.44`, p25/p75 `3.14/10.09`, 2x 이상 비율 `90.9%`였다. synthetic set은 이 분포를 국소 nearest-neighbor spacing에 맞춰 scale하고, local z covariance axis를 analytic surface normal에 정렬한다. 따라서 절대 길이는 scene point density를 따르지만 "표면에 납작하게 붙는" covariance 패턴은 유지한다.

`TorchOSNGSPipeline.initialize()`는 optional `covariance_scales`/`covariance_rotations` 입력을 받아 이를 보존한다. benchmark renderer export의 `point_cloud.ply`도 이 covariance를 그대로 사용한다.

## 실행

```powershell
.venv\Scripts\python.exe -m nurbs_constructor_benchmark
```

빠른 실행:

```powershell
.venv\Scripts\python.exe -m nurbs_constructor_benchmark --points 180 --skip-renderer-export --output C:\tmp\osn_gs_3d_benchmark_smoke
```

특정 장면:

```powershell
.venv\Scripts\python.exe -m nurbs_constructor_benchmark --scenes wave_annulus --points 600
```

## 해석 규칙

이 benchmark가 chart를 materialize했다는 사실은 quality 승인이나 production integration 허가를 뜻하지 않는다. 특히 새로운 3D scenes는 기존 XY projection·planar support·component topology 가정에서 false hole, component split, `review_required` 또는 `unsupported`를 노출할 수 있다. 이런 결과는 숨기지 않고 report에 남겨 Boundary-first hardening의 입력으로 사용한다.

현재 constructor의 active scope와 integration 금지는 [Urgent Work Master](../docs/Urgent_Work/OSN_GS_Urgent_Work_Master.md)를 따른다.