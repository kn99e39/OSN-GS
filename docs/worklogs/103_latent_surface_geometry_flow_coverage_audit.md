# Worklog 103 — latent surface geometry 흐름 복구 및 spatial coverage 실측 감사

## 상태

**완료 — 이 배치는 architecture 결정을 내리지 않는다.** Worklog 98이 도입한 tangent-frame component 구조가 Worklog 95의 latent surface 추정기(iterative weighted-PCA/MLS projection)의 실제 투영 좌표(`query.positions`)를 버리고 raw Gaussian center로 되돌아가 있던 것을 확인하고 정정했다. 이 divergence는 Worklog 98부터 102까지 전부에 파급됐다 — edge differential, 전역 UV 적분, fold 검출, chart 성장, patch identifiability, 최종 NURBS fit target까지 전부 raw Gaussian center 위에서 계산되고 있었다. 이번 배치는 (1) 이 divergence를 소스에서 수정하고, (2) Worklog 95 추정기가 실제로 만들어내는 latent surface의 spatial coverage를 어떤 다운스트림 승인 기준(chart validity, identifiability, NURBS 안전성)에도 구애받지 않고 있는 그대로 실측·export했다. **정성적/architecture적 결론은 이 worklog에 없다** — 결과 해석은 사용자가 직접 시각적으로 판단한다.

## 1. 확정된 divergence와 수정

`osn_gs/surface/torch_latent_surface_tangent_frame_field.py::build_tangent_frame_field`가

```python
query = support.query_batch(points)
normals = query.normals
...
component_points = points[node_list]   # <- 여기가 문제: query.positions가 아니라 raw points
```

였던 것을, kNN edge 후보 선정과 continuous-support 게이팅(Worklog 96의 기존 계약, 미변경)은 여전히 raw 좌표로 하되, **component에 최종 저장/사용되는 좌표만** latent-projected 좌표로 교체했다:

```python
projected_points = query.positions
...
component_points = projected_points[node_list]      # 이제 latent-projected
component_raw_points = points[node_list]            # provenance로만 보존
```

`k`, support radius, planarity threshold, MLS iteration 수, Gaussian kernel, region ownership, training, ADC, support-edge 판정 기준은 전혀 건드리지 않았다 — 이번 수정은 geometry provenance만 고친다.

## 2. Geometry provenance 계약

`TangentFrameFieldComponent`에 세 필드를 추가했다(기존 synthetic fixture 생성자와의 하위 호환을 위해 기본값 `None`):

- `raw_positions` — 투영 이전 원본 Gaussian center
- `projection_displacement` — `positions - raw_positions`
- `latent_supported` — 그 node 자신의 `query.supported` 플래그

`osn_gs/surface/torch_intrinsic_chart_atlas.py::_restrict_component`도 함께 고쳤다 — chart로 축소할 때 이 세 필드가 조용히 유실되고 있었다(Category C 발견, 이번 배치에서 수정).

## 3. 다운스트림 전수 감사

`component.positions`/`chart.component.positions`를 읽는 모든 지점(`torch_latent_surface_edge_differential.py`, `torch_global_differential_uv_integration.py`, `torch_parametric_domain_validity.py`, `torch_orientation_preserving_uv_integration.py`, `torch_intrinsic_chart_atlas.py`, `torch_chart_curve_lattice.py`, `torch_latent_surface_curve_lattice.py`, `torch_patch_identifiability.py`)를 grep으로 전수 확인했다. **`points[ids]`/`raw_positions[ids]`/`gaussian_xyz[ids]` 형태의 후속 재대체는 발견되지 않았다** — 전부 `component.positions`를 일관되게 읽는다(A류: latent-projected geometry 정상 사용). 소스 픽스 이후 이 지점들은 코드 변경 없이 자동으로 올바른 좌표를 받는다. 별도로 `_restrict_component`의 provenance 유실 1건(C류, 수정 완료)을 찾았다. Replay 스크립트들의 `evidence = points[selector]`, `seed.points[0/1]`은 latent-surface 처리 이전 단계(region-owned raw evidence 추출, seed anchor 힌트)에서 의도적으로 raw를 쓰는 지점으로 B류로 분류했다.

## 4. Latent surface coverage의 정확한 정의

- **RAW REGION EVIDENCE**: region 소유 raw Gaussian center 관측치 전체.
- **LATENT-SUPPORTED OBSERVATION**: 기존 Worklog 95 `query_batch`가 `supported=True`를 반환한 관측치.
- **LATENT-PROJECTED POSITION**: 그 관측치의 `query.positions` — 권위 있는 latent surface 좌표.
- **LATENT SUPPORT UNIT**: Worklog 98이 이미 쓰는 것과 동일한 continuously-supported kNN 그래프의 connected component — frame coherence(holonomy), UV validity, chart 소속, patch identifiability를 전혀 요구하지 않는다. Worklog 101의 "chart"와 동의어가 아니다.
- **DOWNSTREAM CHART / PRODUCTION NURBS PATCH**: 이후 단계의 개념이며, latent surface coverage와 동의어가 아니다.

모든 비율은 **evidence/node coverage**로만 표기했다 — surface-area coverage는 별도 면적 추정기 없이는 report하지 않는다(이번 배치에서 구현하지 않음).

## 5. 구현

- 신규 `osn_gs/surface/torch_latent_surface_coverage_audit.py::audit_region_latent_coverage` — region의 raw evidence 전체를 대상으로 supported/unsupported를 나누고, Worklog 98의 kNN+continuous-support 로직을 재사용해 (frame 없이) latent support unit으로만 조직한다. Convex hull/bounding box/PCA rectangle로 gap을 메우지 않는다.
- 신규 `osn_gs/surface/torch_latent_surface_visualization_nurbs.py::fit_visualization_nurbs` — 모든 latent support unit에 대해 고정된 해상도 사다리(2×2/degree1 → 3×3/degree2 → 4×4/degree2, replay로 튜닝하지 않음)로 시각화 전용 NURBS를 시도한다. `identifiability`/`chart`/`held-out`/`unsafe`/`extrapolative` 어떤 기준으로도 결과를 숨기지 않는다 — 수치적으로 fit이 안 될 때만 `VISUALIZATION_NURBS_MATERIALIZATION_FAILED`로 report한다.
- 신규 `scripts/devtools/latent_surface_coverage_export.py` — 7-region 실측을 돌려 A(FULL_SCENE)/B(REGION_EVIDENCE)/C(RAW_VS_LATENT)/D(ALL_LATENT_SURFACES)/E(DOWNSTREAM_COMPARISON) 5개 stage와 region별 독립 export를 만든다. `iteration_<N>` 디렉터리를 stage 구분자로 사용해(같은 WebRenderer 세션·같은 world 좌표계 안에서 iteration 전환만으로 stage를 비교할 수 있게) `RENDERER_INPUT_FORMAT.md`의 기존 `point_cloud.ply`+`nurbs_surface.json` 계약을 그대로 따른다. Worklog 102의 candidate-C 파이프라인은 E stage 비교용으로만 **그대로** 재사용했고, capacity/threshold 어느 것도 이 배치에서 바꾸지 않았다.

## 6. 검증

신규 focused 테스트 9개(`test_latent_surface_geometry_provenance.py`): component 좌표가 supported `query.positions`와 일치, raw Gaussian 좌표가 변경되지 않음, normal이 여전히 latent query에서 옴, chart 축소를 거쳐도 provenance가 보존됨, unsupported 관측치가 절대 latent coverage에 포함되지 않음, unsafe/extrapolative 라벨이 visualization NURBS를 가리지 않음, raw sample과 visualization NURBS가 서로 다른 representation kind를 씀, coverage audit이 frame coherence/chart를 요구하지 않음(AST), visualization NURBS가 identifiability/capacity 모듈을 import하지 않음(AST). 전체 회귀 1회 실행함(아래).

## 7. 실측: checkpoint `baseline_compatible/final` (실행 760초)

### 전역 집계

| 지표 | 값 |
|---|---:|
| 전체 Gaussian(모델 전체) | 1,685,549 |
| Region-owned raw evidence | 7,774 |
| Latent-supported | 4,599 (59.2%) |
| Latent-unsupported | 3,175 (40.8%) |
| Latent support unit 수 | 86 |
| Projected latent position 수 | 4,599 |
| Visualization NURBS 시도 | 86 |
| Visualization NURBS 성공 | 49 |
| Visualization NURBS 실패 | 37 |
| (비교 참고) Worklog 102 candidate-C patch 수 | 60 |

### Region별

| region | raw | supported | 비율 | unsupported | unit | viz 성공/실패 | displacement(median/p95/max, spacing 대비) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 929 | 493 | 53.1% | 436 | 15 | 11/4 | 0.227 / 0.861 / 1.450 |
| 1 | 310 | 187 | 60.3% | 123 | 6 | 3/3 | 0.239 / 0.774 / 1.244 |
| 2 | 1514 | 909 | 60.0% | 605 | 19 | 14/5 | 0.239 / 0.782 / 1.668 |
| 3 | 1674 | 977 | 58.4% | 697 | 13 | 3/10 | 0.303 / 0.960 / 1.479 |
| 4 | 1899 | 1189 | 62.6% | 710 | 13 | 5/8 | 0.305 / 0.931 / 1.602 |
| 5 | 745 | 415 | 55.7% | 330 | 12 | 8/4 | 0.225 / 0.881 / 1.326 |
| 6 | 703 | 429 | 61.0% | 274 | 8 | 5/3 | 0.221 / 0.848 / 1.384 |

Projection displacement는 local median spacing 대비 배수로, MLS 투영이 raw Gaussian center를 실제로 얼마나 움직였는지를 나타낸다(예: 0.227 = 그 지점의 local spacing의 22.7%만큼 이동).

## 8. Export 결과물

**(2026-08-19 정정)** WebRenderer는 `iteration_<N>` 폴더당 `point_cloud.ply` 정확히 1개만 읽는다(0개 또는 1개의 `nurbs_surface.json`을 동반) — 초판에서 폴더 하나에 여러 ply를 섞어 넣었던 것을 발견해 전부 폴더 하나당 ply 1개로 재구성했다.

- `output/osn_gs_scene_latent_coverage_audit/coverage_audit_report.json` — 위 표의 원본 전체(region별 세부 포함).
- `output/osn_gs_scene_latent_coverage_audit/full_scene/iteration_0000001/` — 전체 168만 Gaussian(`point_cloud.ply`만, json 없음).
- `output/osn_gs_scene_latent_coverage_audit/region_owned_evidence/iteration_0000001/` — 전 region raw evidence(`point_cloud.ply`만).
- `output/osn_gs_scene_latent_coverage_audit/latent_projected_samples_with_displacement/iteration_0000001/` — latent projected sample(`point_cloud.ply`) + `nurbs_surface.json`(`base_curves`로 4,599개 displacement 선분).
- `output/osn_gs_scene_latent_coverage_audit/latent_projected_samples_with_visualization_nurbs/iteration_0000001/` — 동일 latent sample(`point_cloud.ply`) + `nurbs_surface.json`(visualization NURBS 49개).
- `output/osn_gs_scene_latent_coverage_audit/full_scene_with_worklog102_nurbs/iteration_0000001/` — 전체 Gaussian(`point_cloud.ply`) + `nurbs_surface.json`(Worklog 102 candidate-C patch 60개).
- `output/osn_gs_scene_latent_coverage_audit/regions/region_<id>/{raw_region_evidence,unsupported_evidence,latent_projected_samples_with_displacement,latent_projected_samples_with_visualization_nurbs}/iteration_0000001/` — region별 독립 export, 각 폴더 ply 1개, world 좌표계 그대로.

Worklog 98~102의 기존 historical replay output(`output/extent_ab/val99~val102/*.json`)은 전혀 덮어쓰지 않았다 — 전부 이전 raw-position geometry 흐름 하에서 생성된 결과로 그대로 보존된다.

## 9. 재현 명령

```
python scripts/devtools/latent_surface_coverage_export.py \
    --checkpoint output/extent_ab/val64/baseline_compatible/final \
    --out output/osn_gs_scene_latent_coverage_audit \
    --device cuda --cap 2048
```

WebRenderer에서 필요한 폴더들을 함께 선택해 로드하면(예: `full_scene`과 `latent_projected_samples_with_visualization_nurbs`를 동시에 선택) 여러 iteration으로 나뉘어 브라우징하며 비교할 수 있고, `regions/region_<id>/`의 각 하위 폴더를 region별로 열면 개별 region의 raw/latent/unsupported/displacement/visualization NURBS를 world 좌표계에서 독립적으로 볼 수 있다.

## 결론 없음

이 worklog는 latent surface coverage가 충분한지 부족한지, chart 구성이 맞는지 틀렸는지, NURBS fitting이 맞는지 틀렸는지, Worklog 98~102 architecture를 유지할지 폐기할지에 대해 어떤 판단도 내리지 않는다. 위 실측·export 결과를 사용자가 직접 시각적으로 검토해 판단한다.
