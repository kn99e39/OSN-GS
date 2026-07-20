# 20. Ground-Truth NURBS Metrics: Three Construction Concerns

날짜: 2026-07-15

## 배경 / 동기

`nurbs_constructor_benchmark`는 synthetic Gaussian scene을 쓰므로 **정답 표면을 안다**는 이점이 있는데, 기존 지표는 그걸 제대로 살리지 못했다. `surface_chart_rms`(생성표면 샘플의 analytic 표면까지 **수직** 거리, 편측)와 `fit_rms`(Gaussian→피팅표면)는 세 가지 서로 다른 안건을 하나로 뭉개서, 예컨대 `crease`가 4패치로 과분할돼도 각 조각이 국소적으로 잘 맞으면 residual은 좋아 보였다.

NURBS 표면 생성 품질은 실제로 세 축으로 나뉜다:
1. **Surface Fitting Accuracy** — 표면이 존재하는 곳에서 얼마나 정확한가.
2. **Surface Support** — 표면이 **올바른 위치**에 존재하는가(구멍/과확장).
3. **Patch Topology** — 패치 개수·경계가 정답과 맞는가.

## 작업

GT를 analytic 표면(정확)으로 두고 세 안건을 분리 측정하도록 벤치마크를 확장했다.

- `scenes.py`: `SyntheticGaussianScene`에 GT descriptor 추가 — `surface_fn(xy)→z`(정답 높이), `gt_patch_count`, `gt_patch_label(xy)`(위상; `crease`=2패치, 능선 x=0으로 분할, 나머지=1). 기존 `make_scene(name, count, seed, noise_std)` 시그니처는 불변.
- `ground_truth.py`(신규): 정답 표면 dense 샘플링(`gt_surface_points`, 관측영역 제한 `observed_gt_surface_points`)과 **GT NURBS 산출**(`gt_nurbs_payload`, degree-1, GT 패치별 control grid, 렌더러 포맷 = `nurbs_intermediate_payload`와 동일).
- `metrics.py`(신규): 세 안건 지표.
  - **Accuracy**: `accuracy_rms`(생성→정답표면, precision), `completeness_rms`(관측 정답표면→생성, recall), `chamfer_rms`.
  - **Support**: `support_coverage_uncovered_fraction`(관측영역 중 미복원 비율=구멍), `support_extrapolation_fraction`(생성표면 중 입력점에서 먼 비율=과확장; UV trimming 부재를 정량화). 임계는 입력점 median NN spacing의 배수.
  - **Topology**: `topology_gen/gt_patch_count`, `topology_label_ari`(입력 Gaussian의 생성 `cluster_ids` vs GT 라벨, 순열 불변 ARL, 과분할에 페널티).
- `runner.py`: 위 지표를 각 result의 `ground_truth` 블록으로 report에 포함, 출력 라인을 세 안건으로 재구성, `nurbs_surface_gt.json`도 함께 export. 안건별 회귀 게이트 추가: `--max-chamfer-rms`, `--max-extrapolation`, `--min-topology-ari`.

## 결과 (600 points, seed 0, lsq)

```text
plane:            patches=1(gt 1) chamfer=0.0235 acc=0.0065 | uncovered=0.021 extrap=0.239 | ari=1.000
sine:             patches=1(gt 1) chamfer=0.0228 acc=0.0104 | uncovered=0.000 extrap=0.184 | ari=1.000
crease:           patches=4(gt 2) chamfer=0.0254 acc=0.0087 | uncovered=0.041 extrap=0.010 | ari=0.223
density_gradient: patches=1(gt 1) chamfer=0.0275 acc=0.0152 | uncovered=0.234 extrap=0.759 | ari=1.000
```

세 안건이 깔끔히 분리되어 각 scene의 서로 다른 실패 모드가 드러난다:

- **crease → 위상 실패**: accuracy는 0.0087로 멀쩡한데 `ari=0.223`(4패치 과분할). 기존 chart_rms로는 안 보이던 문제.
- **density_gradient → support 실패**: `extrap=0.759`(밀집이 원점 근처라 패치의 76%가 데이터 밖) + `uncovered=0.234`(희박 배경 미복원). 위상은 정상.
- **plane/sine → 대체로 정상**, 다만 `extrap 0.18~0.24`로 사각 패치가 점군 footprint 밖으로 넘침(trimming 부재).

## 검증

- 벤치마크 4개 scene 정상 실행, `report.json`에 `ground_truth` 블록, `NURBS_output/<scene>/nurbs_surface_gt.json`(crease=2패치, degree-1) 생성 확인.
- 게이트 확인: `crease --min-topology-ari 0.9` → exit 1, `sine --min-topology-ari 0.9 --max-chamfer-rms 0.05` → exit 0.
- `tests/` 26개 통과(벤치마크는 unittest 스위트 밖, scenes 시그니처 불변이라 회귀 없음).

## 남은 것 / 후속

- 이제 세 안건이 정량화되므로, `TODO.md`의 aspect-ratio 수정 같은 개선을 이 지표로 before/after 비교 가능. 특히 `crease`의 과분할(ari↓)과 `density_gradient`의 extrapolation(↑)이 개선 우선 타깃.
- 렌더러에서 `nurbs_surface.json` + `nurbs_surface_gt.json` 오버레이로 시각 비교 가능(아직 실제 렌더러 확인은 미실시).
- support 임계 계수(`_OBSERVED_RADIUS_FACTOR=3.0`, `_SUPPORT_TAU_FACTOR=2.5`)는 median spacing 기준 휴리스틱 — 필요 시 scene 특성에 맞게 재보정.
