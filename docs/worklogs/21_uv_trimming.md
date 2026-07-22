# 21. UV 트리밍 (표면 지지 영역)

날짜: 2026-07-15

## 배경

`docs/worklogs/16_...`의 GT 지표에서 **Surface Support**의 over-support(extrapolation)가 정량화됐다: 사각 NURBS 패치가 관측 점군 footprint 밖(빈 UV 코너)까지 표면을 그려 `support_extrapolation_fraction`이 plane 0.239, sine 0.184까지 나왔다. `docs/nurbs_construction.md` 10절 "데이터 없는 UV 영역 trimming 없음"을 실제로 해소한다.

## 작업

패치별 **UV 지원(trim) 마스크**를 도입해, 관측 Gaussian이 실제로 차지하는 UV 영역으로 표면을 제한한다.

- `TorchNURBSSurface`에 `uv_support_mask`(R×R bool, `[0,1]^2`) 필드 + `support(uv)` 쿼리 추가. `None`이면 전 도메인 유효(하위호환).
- `TorchPipelineConfig`에 `surface_trim_resolution`(기본 24, 0=비활성), `surface_trim_dilation`(기본 1) 추가.
- `TorchOSNGSPipeline._assign_uv_support_masks`: `initialize()`에서 바인딩된 Gaussian UV로 patch별 점유 grid를 만들고, gap을 메우도록 max-pool dilation 후 마스크로 저장. 로컬 재분할(`_split_failed_patch`)로 생긴 새 patch도 마스크를 받는다.
- `nurbs_intermediate_payload`가 patch별 `uv_support`(resolution + mask)를 export → 렌더러가 trim된 표면만 그릴 수 있음. checkpoint v2도 마스크를 저장/복원(구 checkpoint는 `None`으로 degrade).
- 벤치마크: `sample_generated_surface(respect_trim=True)`가 trim 영역만 샘플 → support 지표가 **실제 trim된 표면**을 측정. `--trim-resolution`/`--trim-dilation` 노브 추가.

loss/anchor/평가 수학은 마스크를 쓰지 않으므로 학습 동작은 불변(순수 메타데이터 + 렌더러/지표 힌트).

## 결과 (600pts, seed0, lsq; trim 24/1)

| scene | extrapolation (before → after) | uncovered | chamfer |
|---|---|---|---|
| plane | 0.239 → **0.089** (−63%) | 0.021 (불변) | 0.0235 |
| sine | 0.184 → **0.092** (−50%) | 0.000 | 0.0228 |
| crease | 0.010 → **0.004** | 0.041 (불변) | 0.026 |
| density_gradient | 0.759 → 0.659 | 0.234 | 0.0275 |

- **uncovered가 전 scene 불변** = trimming이 빈 코너만 제거하고 **coverage 구멍을 새로 만들지 않음**. chamfer 영향도 미미.
- resolution 스윕(0/16/24/32 × dil 1/2): res↑일수록 extrapolation↓, dil=2는 overhang을 되살려 나쁨. res=24/dil=1이 균형점(res=32는 plane 0.044까지 되나 crease chamfer가 약간 증가).

## 남은 한계 / 후속

- **density_gradient의 0.66은 trimming이 아니라 metric tau 보정 이슈**다. `support_tau = 2.5 × median NN spacing`인데, 밀집 클러스터(70% 점)가 median을 지배해 tau가 과소 → 희박 주변부의 관측된 표면까지 extrapolation으로 오표기. 실제로 그 UV 셀은 점이 있어 마스크가 유지된다. 지역 밀도 적응형 tau(전역 median 대신 국소 spacing)로 지표를 보정하면 완화됨 — 별도 작업.
- **trim 마스크는 init 시점 UV 기준**이라 학습 중 Gaussian이 이동/재바인딩되면 stale해진다. `maintain_surface_from_certain`(UV refresh 지점)에서 주기적으로 재계산하는 것이 후속 개선.
- 렌더러(WebRenderer) 쪽에서 `uv_support` 마스크를 읽어 실제로 trim 그리는 것은 이 리포 밖 작업(payload는 준비됨).
