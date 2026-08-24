# Worklog 112 — Renderer-Native Pixel Surface as NURBS Fitting Geometry

## 상태

**완료 — 실측 있음. 건축 판정: NO(대표-중심/픽셀-표면 불일치는 WL111 실패의 주 원인이 아니다).** Worklog 111을 정확히 그대로(chart 구성, image-space UV, 고정 8×4/degree-2 NURBS 설정) 보존하고, 오직 3D fitting 대상만 "대표 서펠 중심"에서 "렌더러가 실제로 계산한 픽셀별 median-depth를 언프로젝션한 표면 점"으로 바꿔 통제 비교(A/B)를 실행했다. **실측 결과: representative 멤버십 커버리지는 68.4%→71.5%, 렌더러 픽셀 커버리지는 93.1%→94.7%로 소폭 개선됐지만, 핵심 진단 지표인 컴포넌트 단위 커버리지 분포(중앙값·p95)는 WL111과 마찬가지로 정확히 0%로 전혀 개선되지 않았다. 게다가 fitting residual과 overlap 불일치의 최댓값은 오히려 극단적으로 악화됐다(residual 최대 7.9→1517.2, overlap 위치 불일치 최대 8.1→1514.4, 중앙값도 overlap 위치 불일치는 0.030→0.055로 악화).** directive 15절의 "materially improve" 기준을 충족하지 못했으므로, WL111의 실패가 대표-중심 기하 문제 때문이라는 가설은 기각한다.

## 1. 렌더러-네이티브 surface-depth 시맨틱스 (정확한 확인)

WL107-111이 이미 재사용해 온 벤더 2DGS forward 커널(`forward.cu`)을 다시 읽어 확인했다: `out_others[MIDDEPTH_OFFSET]`(`MIDDEPTH_OFFSET=5`, `auxiliary.h`)는 WL107의 `median_surfel_id` 캡처와 **정확히 같은 지점**(`if (T > 0.5) { median_depth = depth; median_contributor = contributor; }`, forward.cu:407-411)에서 기록되는 `median_depth` 값이다. 이 `depth`는 surfel 중심의 깊이가 아니라 **실제 ray-splat 교차점**의 카메라 공간 좌표(`depth = (s.x*Tw.x + s.y*Tw.y) + Tw.z`, forward.cu:369, `s`는 로컬 uv 평면 위 교차점)다 — 즉 median representative의 실제 표면 위치를 담고 있다. 이 배치는 이 값을 있는 그대로 재사용했을 뿐, 새로운 depth를 만들지 않았다(directive 2절). 이 채널은 진단 확장(`diff_surfel_rasterization_diag`)의 `render_with_pixel_representative`가 이미 `out_others`로 노출하고 있었으므로 **CUDA 재빌드가 전혀 필요 없었다.**

## 2. 정확한 픽셀→3D 언프로젝션

새 함수 없이, 학습 시 depth-normal consistency loss에서 이미 쓰이는 **기존, 공식 2DGS 코드 충실 포트**(`osn_gs/render/surfel_geometry.py::depths_to_points`, `OFFICIAL_CODE_FAITHFUL`)를 그대로 재사용했다: `depths_to_points(camera, median_depth.unsqueeze(0))`가 카메라 intrinsics/extrinsics로 픽셀+depth를 world-space 3D 점으로 변환한다. 새 메커니즘을 발명하지 않았다(directive 2/3절).

**실측 계약**: `_tilted_camera()`/`_single_surfel()` 고정 fixture로 같은 representative(대표 서펠 0번)가 커버하는 여러 픽셀의 언프로젝션된 3D 위치가 서로 다름을 실제 CUDA 실행으로 확인했다(`test_same_representative_yields_distinct_per_pixel_positions`) — 기울어진 평면 surfel의 footprint가 여러 카메라-공간 깊이에 걸쳐 있다는 2DGS의 당연한 기하학적 사실이며, 대표 중심 하나로 그 픽셀들을 대표시키는 WL111의 가정이 부정확했음을 직접 증명한다.

## 3. WL111 vs 새 통제 변수

| | WL111 (대표 중심) | WL112 (렌더러-네이티브 픽셀 표면) |
|---|---|---|
| chart 구성 | 동일 | 동일(무수정) |
| UV | 동일(픽셀 좌표) | 동일 |
| NURBS 설정 | 8×4, degree 2 | 동일 |
| 3D fitting 대상 | 대표 서펠 중심(뷰당 1개/대표) | 렌더러 median-depth 언프로젝션(뷰당 유효 픽셀 전부) |
| 최소 지지 조건 | distinct representative ≥32 | **valid PIXEL 샘플 ≥32**(directive 7/8절, 컴포넌트 크기와 무관) |

## 4. 픽셀-표면 chartability

| | WL111 | WL112 |
|---|---|---|
| raw chart 후보 | 1,163,380 | 1,163,380 (동일, chart 구성 무수정) |
| 유효 chart | 3,963 | **14,900** (3.76배) |
| representative 멤버십 커버리지 | 68.4% | **71.5%** |

pixel-count 기준 조건이 representative-count 기준보다 항상 느슨하므로(한 컴포넌트가 대표 수는 적어도 픽셀 수는 많을 수 있음, directive 8절) 유효 chart 수가 크게 늘어난 것은 예상대로다.

## 5. 렌더러 표면 픽셀 커버리지

`renderer_surface_pixel_coverage_fraction` = **94.7%**(유효 렌더러 픽셀 중 유효 chart에 속한 비율) — WL111의 `image_view_pixel_coverage_fraction_of_representative_pixels`(93.1%)보다 소폭 높다.

## 6. representative 커버리지

**71.5%**(WL111 68.4% 대비 +3.1%p) — 585,937개 대표 중 561,809개가 최소 1개 유효 chart에 속함.

## 7. 컴포넌트 가중 회계 (핵심 발견 — 개선되지 않음)

`per_component_chart_coverage_fraction_distribution`(155,457개 대표-보유 컴포넌트, **주의: 이 분포는 씬 표면적 지표가 아니라 컴포넌트 개수 기준 unweighted 지표다 — 작은 컴포넌트와 큰 컴포넌트가 동일한 가중치를 받는다**, directive 10절 지시대로 명시): 중앙값 **0.0**, p95 **0.0**(WL111과 정확히 동일), 평균 0.0105(WL111 0.0019 대비 소폭 상승), 최대 1.0. **소수의 거대 컴포넌트에 커버리지가 집중된다는 WL111의 핵심 발견은 이번 배치에서도 전혀 변하지 않았다** — pixel 기준으로 조건을 완화해도 애초에 대표 수 자체가 극히 적은(32개 미만) 파편화된 컴포넌트 다수는 여전히 chart를 만들 수 없다(단, directive 8절 지시대로 이를 "topology 실패"로 재분류하지 않는다 — 순수하게 지지 데이터 부족이다).

## 8. NURBS residual 비교

| | WL111 | WL112 |
|---|---|---|
| 중앙값 | 0.0319 | 0.0339 (거의 동일, 소폭 악화) |
| 평균 | 0.1142 | 0.1130 (거의 동일) |
| p95 | 0.5137 | 0.5054 (거의 동일) |
| **최대** | **7.915** | **1517.17** |

중앙값/p95는 사실상 변화 없다 — 긴 꼬리가 근본적으로 줄지 않았다. 오히려 최댓값이 극단적으로 악화됐다: 밀집 픽셀 샘플이 대표-중심 평균화가 우연히 완충하던 렌더러 depth의 국소 노이즈/특이점(예: 거의 접선 방향 시야각, chart 경계 부근 픽셀)을 직접 fitting 대상으로 노출시킨 결과로 해석된다.

## 9. Overlap 일관성 비교

| | WL111 | WL112 |
|---|---|---|
| 위치 불일치 중앙값 | 0.0297 | **0.0546** (+84%, 악화) |
| 위치 불일치 p95 | 0.4003 | 0.4179 (거의 동일) |
| **위치 불일치 최대** | **8.089** | **1514.42** |
| 법선 불일치 중앙값 | 5.04° | 5.37° (소폭 악화) |
| 법선 불일치 p95 | 57.93° | 61.06° (소폭 악화) |
| 법선 불일치 최대 | 179.97° | 179.88° (둘 다 이미 포화) |

**Overlap 일관성은 개선되지 않았다** — 중앙값이 악화됐고, 극단치는 훨씬 악화됐다. `OVERLAP_DISAGREEMENT`/`WL111_VS_PIXEL_SURFACE_RESIDUAL` export는 정규화가 단일 극단치에 지배돼 대부분 어둡게 나온다(그 극단치가 매우 국소적임을 시사).

## 10. 테이블 결과

table_top 75.5%(WL111 75.3%, 사실상 동일), table_legs 88.9%(동일).

## 11. 곡면 테이블 결과

table_side_curved **62.2%**(WL111 57.3% 대비 **+4.9%p**) — directive가 기대한 "곡면 영역 개선" 가설과 일치하는 유일하게 뚜렷한 긍정적 신호다. 그러나 이 커버리지 개선이 기하 품질(residual/overlap) 개선을 동반한다는 증거는 없다(7-9절 참고, 오히려 전체적으로 악화) — 커버리지와 fitting 품질은 별개다.

## 12. 패티오 결과

79.9%(WL111 77.8% 대비 +2.1%p).

## 13. 헤지/배경 결과

54.8%(WL111 49.0% 대비 +5.8%p) — 다섯 영역 중 가장 큰 상대 개선이지만 여전히 다섯 영역 중 최저 커버리지다.

## 14. 정확한 실패 귀속

**대표-중심/픽셀-표면 불일치는 WL111 실패의 주 원인이 아니다.** 근거:

1. 커버리지는 개선됐지만(멤버십 +3.1%p, 영역별 +0~5.8%p) 폭이 작고, 핵심 지표인 컴포넌트 단위 커버리지 분포(중앙값·p95)는 **전혀** 개선되지 않았다 — 여전히 대다수 컴포넌트가 0% 커버리지다.
2. fitting residual은 중앙값/p95에서 사실상 개선이 없고, 최댓값은 오히려 200배 가까이 악화됐다.
3. Overlap 일관성은 중앙값이 악화되고 최댓값이 190배 가까이 악화됐다 — 개선이 아니라 후퇴다.

directive 15절의 세 갈래 조건 중 어느 것에도 "materially improves"에 해당하지 않는다. WL111이 남긴 두 원인(위상 파편화, 미분화 거대 chart) 중 이번 배치가 통제한 변수(대표-중심 vs 픽셀-표면)는 residual/overlap 악화의 원인이지 개선의 원인이 아니었다 — 남아있는 실패는 여전히 **NURBS chart capacity/granularity**(거대 blob이 하나의 미분화 8×4 control grid로 강제 fit되는 문제, WL111이 이미 지목)이며, 이번 배치는 이를 더 많은/더 밀집된 데이터로 악화시켰을 뿐 해결하지 못했다.

## 15. 검토용 export 경로

`output/osn_gs_pixel_surface_nurbs/` 아래 11개 뷰(`iteration_0000001/point_cloud.ply`, `render.ppm`, `preview_png/render.png`, `README.md`): `ORIGINAL_2DGS_SCENE`, `WL111_REPRESENTATIVE_CENTER_NURBS`(WL111의 `VISIBLE_NURBS_PATCHES`를 그대로 복사, 재실행 아님), `RENDERER_NATIVE_PIXEL_SURFACE`, `PIXEL_SURFACE_CHART_DOMAINS`, `PIXEL_SURFACE_NURBS`, `PIXEL_SURFACE_COVERAGE`, `WL111_VS_PIXEL_SURFACE_RESIDUAL`, `OVERLAP_DISAGREEMENT`, `TABLE_PIXEL_SURFACE_NURBS`, `CURVED_STRUCTURE_PIXEL_SURFACE_NURBS`, `HEDGE_PIXEL_SURFACE_NURBS`. 전체 JSON 리포트(WL111 리포트를 읽어 직접 비교 포함): `output/osn_gs_pixel_surface_nurbs/renderer_native_pixel_surface_nurbs_report.json`.

## 16. 테스트

CUDA/production 코드는 무수정(`out_others`가 이미 필요한 채널을 노출하고 있어 진단 확장 재빌드조차 필요 없었다). `osn_gs/surface/torch_camera_observed_chart_domains.py`에 WL111 함수는 그대로 두고 새 함수(`build_view_chart_pixel_samples`, `valid_pixel_chart_mask`, `ViewPixelChartSamples`)만 추가했다.

- `tests/test_renderer_native_pixel_surface_chart.py`(신규, 9개): 연결요소 재사용/컴포넌트 비혼합, 픽셀 단위 비-collapse(직접 확인), 가려진 gap 비교차, pixel-count 기준 지지 회계(representative-count 아님), 결정론적 재생, 평면/곡면 pixel-surface fit 정확도, **실제 CUDA**로 median-depth 언프로젝션과 "같은 대표 서로 다른 3D 위치" 계약 검증.
- `.venv/Scripts/python.exe -m pytest tests/test_renderer_native_pixel_surface_chart.py tests/test_camera_observed_chart_domains.py tests/test_representative_only_visible_nurbs.py -q` → **24 passed**.
- 실측 스크립트는 `--max-views 6` smoke test로 파이프라인 전체가 오류 없이 끝까지 도는 것을 먼저 확인한 뒤, 전체 161개 뷰로 재실행했다(런타임 948.8초). 161개 뷰 위상 재생 결과가 WL107/109/111의 알려진 수치(36.77%/45.02%)와 다시 한번 정확히 일치함을 확인했다.
- 전체 pytest는 재실행하지 않았다(directive 지시: canonical/production 코드 무수정, 순수 실험 경로로 유지).

## 다음 배치로 넘길 사항

directive 15절 "NO" 갈래의 지시대로, 대표-중심/픽셀-표면 문제를 해결한 것으로 보고 chart 세분화로 넘어가지 않는다 — representation contract 자체(고정 8×4 NURBS가 매우 크고 비-disk인 chart를 강제로 하나의 control grid에 담는 문제)를 다시 검토해야 한다는 것이 이번 배치의 결론이다. 구체적 다음 단계 제안은 Master 문서 addendum에서만 다룬다(worklog 자체에는 넣지 않음, [[feedback_no_next_step_suggestions_in_worklogs]]).
