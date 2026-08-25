# Worklog 114 — Local Rank-Complete NURBS Chart Network

## 상태

**완료 — 실측 있음. 건축 판정: LOCAL_CHART_UNIT_NOT_VIABLE.** Worklog 107/109 canonical topology, Worklog 112 렌더러-네이티브 픽셀-표면 기하, per-view same-component blob 연결성, 고정 8×4/degree-2 NURBS 설정을 전부 동결하고, chart **단위**만 "camera-connected blob 하나 = NURBS chart 하나"(WL112)에서 "blob → 결정론적 local rank-complete 도메인 → 여러 개의 fixed 8×4 NURBS patch"로 교체했다. **8개 뷰(stride=20, 전체 161개 중 균등 표집, 위상/대표 회계는 여전히 전체 161개 뷰 기준 — §0 참조) 통제 비교 실측 결과: fitting residual과 domain shape(구멍·occupancy·aspect ratio)는 뚜렷이 개선됐지만, representative 커버리지가 11.7% 감소했고(88.3%로) overlap 법선 불일치가 크게 악화됐다(중앙값 5.8°→18.2°, p95 59.3°→96.5°)** — directive 16절이 요구한 다차원 판정 기준 중 residual/domain은 통과하지만 coverage/overlap이 실패해, **"fit 품질 개선이 자동으로 성공을 의미하지 않는다"**는 directive의 명시적 경고에 정확히 해당하는 결과다.

## 0. 범위 축소 (명시적 고지, 숨기지 않음)

위상/대표(median representative) sweep과 WL107/109 위상 재생은 **전체 161개 학습 뷰**로 수행했다(WL107-113과 완전히 동일, `median_surface_representatives=785,937`, `visible_component_count=559,989`, 최대 컴포넌트 36.77%, singleton 45.02% — 전부 정확히 일치 재확인). 그러나 chart 성장+fit이라는 비용이 큰 단계는 **8개 뷰**(stride=20, 인덱스 0/20/40/60/80/100/120/140)로 제한했다 — ARM A(WL112 베이스라인)와 ARM B(신규 local chart) **양쪽 다 동일한 8개 뷰 서브셋**에서 재계산해 공정한 통제 비교를 확보했다(WL112의 저장된 전체-161뷰 리포트를 재사용하지 않음 — 표본 크기가 다른 수치를 직접 비교하지 않기 위함). 이 축소의 이유: 독립 벤치마크에서 뷰 하나에 꽉 찬 27만 픽셀 blob 하나를 분해하는 데만 약 235초가 걸림을 실측했고(최초 시도는 BFS가 남은 blob 전체를 매번 처음부터 훑는 버그로 무한정 정체됐다가 증분 BFS로 수정, §1 참조), 161개 뷰 전체로 확장하면 예측 가능하게 수 시간이 걸릴 것으로 판단해 이번 세션 내에서는 다루지 않았다. 표본 크기 차이 자체를 architecture 결정에 영향을 주지 않도록, 위상/대표 계약은 전체 씬으로 유지하고 오직 새 chart 메커니즘 평가만 축소했다.

## 1. 정확한 local chart 구성

신규 모듈 `osn_gs/surface/torch_local_rank_complete_chart_growth.py`: 기존 WL111 `label_same_component_blobs`(무수정)로 얻은 각 camera-observed 같은-컴포넌트 blob 내부에서, **결정론적 "pole of inaccessibility" 시드**(blob 자신의 raster 경계로부터 최대 거리, `scipy.ndimage.distance_transform_edt`, 동률은 최소 (row,col)로 결정)에서 시작해 기존 image-space 4-neighbor 그래프로 BFS 성장시키며, **고정 8×4 degree-2 NURBS의 design matrix가 처음 full column rank(32)에 도달하는 시점**을 닫힘 조건으로 사용한다(directive 4절, scene-tuned 임계값 없음). 도달 못 하면(전체 연결 영역을 다 써도 rank<32) 그 영역은 fit하지 않고 `INSUFFICIENT_RANK_CLOSURE`로 남긴다(directive: "leave it unresolved, do not force a fit"). 서로 다른 canonical 컴포넌트는 blob 라벨링 자체의 구조적 보장으로 절대 같은 local chart를 공유할 수 없다(WL111과 동일한 보장 상속).

## 2. 정확한 rank-closure 시맨틱스

닫힘 판정은 오직 고정 모델의 tensor-product basis 자체(`resolution_u×resolution_v=32`개 basis function, 이 blob/뷰의 데이터와 무관하게 knot vector로 고정됨)에서 유도한다 — residual/픽셀수/면적/occupancy 임계값은 전혀 사용하지 않았다(directive 4/9절). 후보 크기는 32(최소 가능)부터 4씩 증가하며 매번 실제로 rank를 재계산한다(`_RANK_CHECK_STEP=4`, 순수 엔지니어링 상수로 명시적으로 공개, scene-tuned 아님). **중요한 성능 수정**: 최초 구현은 매 local chart를 뽑을 때마다 남은 blob **전체**를 BFS로 미리 다 훑어(`_bfs_order`) 그 중 앞부분만 썼는데, 이는 거대 blob에서 chart 하나당 O(blob 크기)의 낭비를 반복해 O(blob 크기²/chart 크기)로 폭증했다 — 실제 27만 픽셀 blob에서 무한정 정체되는 것으로 발견됐다. `_bfs_levels`(제너레이터, BFS 레벨 단위로 지연 평가)로 교체해 필요한 만큼만 성장하도록 고쳤고, 수정 후 같은 blob이 235초 만에 끝났다(1,784개 chart 추출). 랭크 체크 자체도 매 후보마다 CPU 텐서로만 수행하도록 고정했다(원래는 파이프라인 전체 device인 CUDA를 매번 왕복해 작은 행렬 연산 대비 전송 오버헤드가 지배적이었다) — 둘 다 순수 엔지니어링 수정이며 알고리즘의 의미는 바꾸지 않았다.

## 3. Chart 개수/복잡도

| | ARM A (WL112 베이스라인, 같은 8뷰) | ARM B (신규 local chart) |
|---|---|---|
| chart 수 | 889 | **14,137** (15.9배) |
| 뷰당 chart 수(중앙값) | — | 1,783 |
| 컴포넌트당 chart 수(중앙값/p95/최대) | — | 1 / 9 / **10,776** |
| chart당 샘플 수(중앙값/p95/최대) | — | 92 / 124 / 90,388 |
| chart당 representative 수(중앙값/p95/최대) | — | 35 / 63 / 32,132 |
| 10,000 커버 픽셀당 chart 수 | — | 88.1 |

15.9배는 directive가 경고한 "raster 스케일로의 폭발"(20배 이상을 자동 임계값으로 사용)에는 못 미치지만 결코 작지 않다 — 그리고 컴포넌트당 최대 10,776개라는 극단값은 전부 최대 컴포넌트(patio/hedge 거대 컴포넌트) 하나에 집중돼 있다(§9 참조). 전형적 chart(중앙값 92픽셀)는 합리적 크기이지만, 이는 전형적 chart의 이야기일 뿐 최대 컴포넌트에서 벌어지는 실제 폭증을 가리지 않는다.

## 4. 렌더러 표면 픽셀 커버리지 / 5. Representative 커버리지

| | ARM A | ARM B |
|---|---|---|
| 커버된 representative(같은 8뷰 서브셋 기준) | 328,679 | **290,369** |
| coverage 비율(B/A) | — | **0.883** |

representative 커버리지가 **11.7% 감소**했다 — chart 수가 15.9배 늘었음에도 오히려 더 적은 representative를 커버한다. 원인: local chart는 시드(pole)에서 컴팩트하게 성장하며 남는 얇은 경계 조각(§6)을 흔히 남기는데, WL112 베이스라인은 blob 전체를 통째로 하나의 chart로 fit하므로 그런 경계 조각도 그냥 포함시켜 버린다 — 즉 **국소화가 전형적 chart 품질은 개선하지만, 경계에서 좌초되는(stranded) 픽셀을 만들어 순 커버리지를 낮춘다.**

## 6. 미해결(unresolved) 가시 증거 커버리지

| 사유 | 개수 |
|---|---|
| `TOO_FEW_PIXELS`(연결 영역 자체가 32픽셀 미만) | 79,317 |
| `INSUFFICIENT_RANK_CLOSURE`(32픽셀 이상이지만 전체를 써도 full rank 불가) | 2,418 |
| `RUNTIME_CAP_SKIPPED`(엔지니어링 안전판, `max_patches_per_blob=2000`) | **0** — 이번 8뷰 실측에서 한 번도 발동하지 않음 |

**89,784개 representative가 오직 unresolved 영역에서만 등장**(directive 7절 지시대로, 이들은 여전히 VISIBLE TOPOLOGY EVIDENCE로 남으며 occluded/unknown/free-space/visible-termination으로 재해석하지 않는다). `RUNTIME_CAP_SKIPPED=0`은 이번 축소된 8뷰 실측에서는 안전판이 결과에 영향을 주지 않았음을 의미하지만, 전체 161뷰에서도 그럴지는 확인하지 않았다(§0의 범위 축소).

## 7. WL113 대비 도메인 모양 비교

| | WL112 베이스라인(같은 8뷰) | Local chart |
|---|---|---|
| occupancy_ratio 중앙값/p95 | 0.494 / 0.708 | 0.505 / **0.606** |
| hole_count 평균 / 구멍 있는 chart 비율 | 29.46 / **46.1%** | 0.69 / **15.7%** |
| aspect_ratio 중앙값/p95/최대 | 1.17 / 3.36 / 14.5 | **0.93 / 1.22** / 5.0 |

세 지표 모두 **뚜렷이 개선**됐다: 구멍 있는 chart 비율이 46.1%→15.7%로 3배 가까이 줄었고, aspect ratio는 훨씬 정사각형에 가까워졌으며(p95 3.36→1.22), occupancy의 극단(p95)도 개선됐다. **WL113이 지목한 실패 B(사각형 도메인/구멍 불일치)는 국소화로 실질적으로 완화된다.**

## 8. Residual / Overlap 비교 (WL112 대비)

| | WL112 베이스라인(같은 8뷰) | Local chart |
|---|---|---|
| residual 중앙값 | 0.0369 | **0.0041** (9배 개선) |
| residual p95 | 0.516 | **0.064** (8배 개선) |
| residual 최대 | 1517.2 | 1464.3 (거의 동일 — §10) |
| overlap 위치 중앙값 | 0.0552 | **0.0390** (개선) |
| overlap 위치 p95 | 0.357 | **0.207** (개선) |
| overlap 위치 최대 | 19.4 | 71.6 (**악화**) |
| overlap 법선 중앙값 | 5.77° | **18.15°** (3배 악화) |
| overlap 법선 p95 | 59.3° | **96.5°** (사실상 포화, 악화) |

residual과 overlap **위치**의 전형값(중앙값/p95)은 뚜렷이 개선됐지만, **overlap 법선 불일치는 크게 악화됐다** — chart 수가 15.9배 늘면서 서로 다른 chart 사이의 seam(경계) 수도 그만큼 늘었고, 각 chart가 독립적으로 fit되므로(WL111부터 유지된 "per-view, 병합 없음" 계약) 인접 chart끼리 법선이 어긋나는 경우가 훨씬 많아졌다 — directive가 중심 의도에서 명시한 "patch 경계는 representation seam"이라는 문장이 실제로는 **측정 가능한 대가**를 동반함을 실측으로 확인한 것이다.

## 9. D 이상치의 지속성 (건드리지 않음, 추적만)

WL113이 지목한 D(렌더러 median-depth 국소 수치 불안정)와 정확히 같은 패턴이 새 방법에서도 그대로 나타난다: `chart_id=1159`(hedge, component 0, view 0)는 1,852픽셀짜리 chart이면서도 `depth_std=32.5`, `residual_max=1464.3`, `max_overlap_position_discrepancy=71.6` — residual 최대값과 overlap 최대값 **양쪽 모두의 최상위 사례가 바로 이 하나의 chart**다. 이는 이번 감사([[project_design_intent_specification_implementation_traceability_audit]], Worklog 115)가 사전에 정확히 예측한 지점과 일치한다: **rank-closure는 대수적으로 full rank에 도달했음을 보장할 뿐, 그 chart가 진짜 depth-연속적인 하나의 물리적 표면 조각이라는 것은 보장하지 않는다** — 이 chart는 rank=32(full)이면서도 depth_std가 극단적으로 크다. 지시대로 이 샘플을 거부/clamp하지 않고 그대로 두었으며, D를 chart-architecture 결론에 섞지 않는다.

## 10. Table 결과 / 11. 곡면 결과 / 12. Patio 결과 / 13. Hedge 결과

| 영역 | 서브셋 내 visible 증거 | ARM A 커버 | ARM B 커버 | ARM B chart 수 | ARM B residual 중앙값 |
|---|---|---|---|---|---|
| table_top | 40,062 | 36,252 | 33,292 (−8.2%) | 2,741 | 0.00214 |
| table_side_curved | 68,171 | 53,599 | 46,634 (−13.0%) | 2,336 | 0.00489 |
| table_legs | 43,565 | 41,805 | 38,589 (−7.7%) | 1,924 | 0.00266 |
| patio | 168,041 | 153,066 | 136,721 (−10.7%) | 5,735 | 0.00377 |
| hedge | 60,314 | 43,957 | 35,133 (**−20.1%**, 최대 하락) | 1,401 | 0.00757 |

**모든 영역에서 예외 없이 커버리지가 감소했다** — §5의 전역 추세가 지역별로도 일관되게 재현된다. hedge가 상대적으로 가장 크게 감소했다(파편화가 이미 심한 영역일수록 국소화의 "경계 좌초" 효과가 더 크게 작용하는 것으로 해석). residual은 다섯 영역 모두에서 매우 낮다(0.002-0.008) — fit 품질 자체는 어디서나 좋아졌다. `LOCAL_OVERLAP_DISAGREEMENT` 시각 검토에서도 hedge 영역에 밝은 마젠타 클러스터가 집중돼 있음을 확인했다(§9의 수치와 일치).

## 14. 건축 판정

**LOCAL_CHART_UNIT_NOT_VIABLE.** Directive 16절의 판정 기준을 항목별로 대조:

| 기준 | 충족 |
|---|---|
| fit 품질이 WL112 대비 실질 개선 | ✅ residual/overlap-위치 뚜렷이 개선 |
| 거대 B/C residual 실패가 실질적으로 축소 | 부분적 — 전형값은 개선, 그러나 **최댓값**은 거의 그대로(1517→1464) |
| local chart 도메인 복잡도 감소 | ✅ 구멍/aspect ratio 뚜렷이 개선 |
| chart 수가 구조적으로 합리적 유지 | 부분적 — 15.9배(20배 자동 임계값은 안 넘었으나 작지 않음), 한 컴포넌트에 10,776개 집중 |
| table/patio/곡면 구조가 일관 유지 | ✅ residual은 어디서나 낮음 |
| topology 불변 | ✅ 완전 일치 재확인 |
| **coverage가 실질적으로 하락하지 않음** | ❌ **11.7% 하락, 다섯 영역 전부** |
| **overlap 법선 일관성이 악화되지 않음** | ❌ **중앙값 3배, p95 사실상 포화까지 악화** |

두 개의 핵심 기준(coverage 유지, overlap 법선 일관성)이 명확히 실패했다 — directive의 "fitting residual이 작아졌다고 자동으로 성공이라 부르지 말라"는 명시적 지시에 정확히 해당한다. **"Does replacing the camera blob with a mathematically local, rank-complete NURBS representation unit resolve the large/holey-chart failures without degenerating into pixel-scale patch fragmentation?"에 대한 답: NO** — pixel-scale로 완전히 붕괴하지는 않았지만(15.9배는 20배 미만), fit 품질 개선이 커버리지와 cross-chart 일관성이라는 실질적 대가 없이 오지는 않았다. Directive 결론 규칙대로 **stop and reassess NURBS representation itself before adding further mechanisms.**

## 15. 검토용 export 경로

`output/114_osn_gs_local_chart_network/`(번호 규약 적용) 아래 11개 뷰 폴더(`iteration_0000001/point_cloud.ply`, `render.ppm`, `README.md`) — `ORIGINAL_2DGS_SCENE`, `WL112_BASELINE_SAME_SUBSET`, `LOCAL_CHART_NETWORK`, `UNRESOLVED_VISIBLE_EVIDENCE`, `LOCAL_CHART_DOMAIN_OCCUPANCY`, `LOCAL_VS_WL112_RESIDUAL`, `LOCAL_OVERLAP_DISAGREEMENT`, `D_OUTLIER_PERSISTENCE`, `TABLE_LOCAL_CHART_NETWORK`, `CURVED_LOCAL_CHART_NETWORK`, `HEDGE_LOCAL_CHART_NETWORK`. 미리보기 PNG는 `preview_png/<뷰이름>.png` 한 폴더에 통합([[feedback_output_folder_numbering]] 규약). 전체 JSON 리포트: `output/114_osn_gs_local_chart_network/local_rank_complete_chart_network_report.json`.

## 16. 테스트

신규 `osn_gs/surface/torch_local_rank_complete_chart_growth.py`(순수 로직, CUDA 불필요) + `scripts/devtools/local_rank_complete_chart_network.py`(실측 스크립트). Canonical 위상(`torch_camera_induced_visible_adjacency.py`), WL111/112 chart 모듈, `torch_nurbs.py`는 전부 무수정(읽기전용 재사용).

- `tests/test_local_rank_complete_chart_growth.py`(신규, 9개, directive 15절 요구 계약 포함): 거대 평면 blob이 여러 full-rank local chart로 분해(A), 모든 반환 chart가 실제로 full rank(B), 결정론적 재현(F), 두 인접 컴포넌트가 절대 local chart를 공유 안 함(D), 32픽셀 미만 영역은 fit 없이 보존(F), collinear 1픽셀 폭 띠(픽셀 수는 충분하나 2D 퍼짐 없음)는 rank 도달 못 해 보존, 링 모양 blob의 내부 구멍에 chart 픽셀이 절대 생기지 않음(C), 전역 occluded gap을 절대 넘지 않음(E), `max_patches_per_blob` 안전판이 작동하고 별도로 보고됨.
- `.venv/Scripts/python.exe -m pytest tests/test_local_rank_complete_chart_growth.py -q` → **9 passed**.
- 실측 전 `--max-views 3 --chart-max-views 3`으로 스모크 테스트(2회 — 1차는 `distance_transform_edt`가 배열 전체를 덮는 마스크에서 이미지 경계를 배경으로 취급하지 않아 시드가 코너로 튀는 버그를 발견해 1픽셀 패딩으로 수정, 2차는 성공), 이후 `--chart-view-stride 20 --chart-max-views 8`로 §0의 축소된 실측을 실행했다(런타임 1878.3초). 위상 재생 수치가 WL107/109/113과 완전히 일치함을 재확인했다.
- 전체 pytest는 재실행하지 않았다(directive 지시: canonical/production 코드 무수정).

## 다음 배치로 넘길 사항

Directive 지시대로 다른 메커니즘(adaptive capacity, chart merging/stitching, 비-representative attachment, Trust, latent surface, occluded surface)은 이번 배치에서 구현하지 않았다. NOT VIABLE 판정에 따라 다음 단계는 "NURBS representation 자체를 재검토"해야 하지만, 구체적 다음 단계 제안은 Master 문서 addendum에서만 다룬다(worklog 자체에는 넣지 않음, [[feedback_no_next_step_suggestions_in_worklogs]]). 참고: 이 배치와 병행 진행된 [[project_design_intent_specification_implementation_traceability_audit]](Worklog 115)가 WL107-113의 실패 원인을 specification/control-experiment/데이터 계층으로 귀속했으며, 그 감사의 §7이 바로 이 배치의 rank-closure 가정(대수적 식별가능성 ≠ 기하적 chart 타당성)을 사전에 정확히 지적했고, §9의 D-지속성 실측이 그 예측을 직접 확인했다.
