# Worklog 111 — Representative-Only Visible NURBS Scaffold

## 상태

**완료 — 실측 있음. 건축 판정: NOT VIABLE (현재 chart 구성 방식으로는), 실패 원인 명확히 귀속.** Worklog 107/109의 canonical Renderer-Native Surface Representative Graph는 이번 배치에서 한 줄도 수정하지 않았고(`torch_camera_induced_visible_adjacency.py` 무수정, 재생 결과 최대 컴포넌트 36.77%/싱글톤 45.02%로 완전 일치 확인), Worklog 110의 AMBIGUOUS/LAYERED SUPPORT 판정에 따라 non-representative 증거는 fitting에 전혀 사용하지 않았다(오직 785,937개 MEDIAN_SURFACE_REPRESENTATIVE만 사용). 각 카메라 뷰 자신의 이미지-공간 픽셀 좌표를 chart UV 파라미터로 직접 사용하는(directive 4절) 새 chart 구성 방식으로 실측한 결과: representative 멤버십 커버리지는 68.4%, 픽셀 커버리지는 93.1%로 얼핏 양호해 보이지만, **컴포넌트 단위 커버리지 분포의 중앙값과 p95가 모두 0%**다 — 즉 커버리지는 소수의 거대 컴포넌트(테이블·패티오 등)에만 집중돼 있고, 155,457개 representative-보유 컴포넌트 중 절대다수는 유효 chart를 단 하나도 얻지 못했다. 게다가 커버에 성공한 소수의 거대 chart조차 fitting residual과 overlap 불일치가 심하게 꼬리를 끈다(최대 residual 7.9 scene 단위, 법선 불일치 최대 180°에 근접) — directive 3절이 명시적으로 경고한 "거대/비-disk 컴포넌트에 하나의 UV 도메인을 강제하지 말라"는 실패 양상이 그대로 재현됐다.

## 1. Chart-domain 구성 (정확한 방법)

새 모듈 `osn_gs/surface/torch_camera_observed_chart_domains.py`: 한 학습 뷰의 `(H, W)` representative map을 canonical `subset_ids`(WL107/109, 읽기전용)로 리맵한 component-id map 위에서, **같은 컴포넌트 id를 가진 이웃 픽셀만 연결하는 4-connectivity 연결요소(scipy.sparse.csgraph.connected_components, 정확한 알고리즘, 근사 아님)**로 "camera-observed chart candidate"(blob)를 만든다. 서로 다른 컴포넌트가 화면에서 인접해도 절대 같은 blob이 될 수 없다는 것은 이 labeling의 **구조적 보장**이다(directive 5절, 사후 필터가 아니라 라벨링 자체가 그렇게 동작 — `test_two_components_never_share_a_chart`, `test_two_different_components_adjacent_in_image_space_never_share_a_chart`로 검증).

## 2. UV 파라미터화

각 blob 안에서 대표 서펠 하나당 (u, v) 샘플 하나 — 그 뷰에서 그 서펠에 속한 모든 픽셀의 평균 (row, col)을, 그 blob 자신의 픽셀 bounding box로 `[0, 1]^2` 정규화한 값이다. 3D PCA나 raw-3D kNN이 아니라 **실제 카메라가 관측한 이미지-공간 좌표 그 자체**를 chart 파라미터로 사용한다(directive 4절 요구사항).

## 3. NURBS 지지(support) 수학적 최소 조건

`torch_nurbs.fit_torch_visible_surface_lsq`의 **이미 확립된 프로젝트 기본값**(`resolution_u=8, resolution_v=4, degree_u=degree_v=2`, 함수 시그니처 자체의 기본값, 이번 배치에서 새로 고른 값이 아님)을 그대로 사용했다. control grid 자유 control point 수 = 8×4 = 32개 — 이 fit이 regularizer가 아니라 데이터로 결정되려면 최소 32개의 독립 샘플이 필요하다는 것이 `MIN_CHART_MEMBERS=32`의 유일한 근거다(directive 7절, scene-tuned 아님).

## 4. 실측 회계 — representative chartability

| 항목 | 값 |
|---|---|
| 전체 학습 서펠 | 1,190,469 |
| MEDIAN_SURFACE_REPRESENTATIVE | 785,937 |
| raw chart 후보(blob) 총수 | 1,163,380 |
| 유효 chart(멤버 ≥32) 수 | **3,963** (raw의 0.34%) |
| 실제로 fit된 chart 수 | 3,963 (전부) |
| ≥1개 유효 chart에 속한 representative | 537,357 (68.4%) |
| 유효 chart가 하나도 없는 representative | 248,580 (31.6%) |
| 정확히 1개 유효 chart에 속한 representative | 28,327 |
| 2개 이상 유효 chart에 속한 representative | 509,030 (64.8%) |

raw chart 크기 분포는 극단적으로 치우쳐 있다: 중앙값 1(즉 대부분의 blob이 대표 1명짜리), 평균 11.77, 최대 74,608. 유효 chart(≥32)만 봐도 중앙값 60, 평균 3,050.7, 최대 74,608 — **소수의 거대 chart가 전체 멤버십의 절대다수를 차지**한다.

## 5. 실측 회계 — 컴포넌트 단위 chartability (핵심 발견)

`per_component_chart_coverage_fraction_distribution`(representative를 가진 155,457개 컴포넌트 전체): **중앙값 0.0, p95 0.0**, 평균 0.0019, 최대 1.0. representative 멤버십 커버리지(68.4%)나 픽셀 커버리지(93.1%)가 양호해 보이는 것은 소수의 거대 컴포넌트(테이블·패티오 등 WL107/109에서 이미 확인된 크게 연결된 컴포넌트) 때문이며, **개별 컴포넌트 기준으로 보면 절대다수(중앙값·p95 모두 0%)가 유효 chart를 단 하나도 얻지 못한다.** 이는 WL96-109에서 반복 확인된 컴포넌트 파편화(전체 559,989개 컴포넌트 중 45.02%가 싱글톤)의 직접적 귀결이다 — 32개 미만의 대표 서펠을 가진 컴포넌트는 어느 뷰, 어느 blob에서도 구조적으로 유효 chart를 만들 수 없다.

## 6. 이미지/뷰 커버리지

`image_view_pixel_coverage_fraction_of_representative_pixels` = 93.1%. 픽셀 기준으로는 매우 높다 — 그러나 5절과 마찬가지로 이는 화면 대부분을 차지하는 소수의 거대 컴포넌트에 의한 것이지, 노드(컴포넌트) 다양성에 의한 커버리지가 아니다. directive 13절이 명시적으로 경고한 "membership 회계만으로 scene coverage를 추정하지 말라"는 지시와 정확히 부합하는 사례 — 노드 멤버십(68.4%)과 픽셀 커버리지(93.1%)는 양호해 보이지만 컴포넌트 다양성(중앙값 0%)은 전혀 그렇지 않다.

## 7. NURBS fitting residual 분포

전체 12,090,033개 (chart, member) 쌍에서: 중앙값 0.0319, 평균 0.1142, p95 0.5137, **최대 7.915** (scene 단위). 중앙값은 양호하지만 꼬리가 매우 길다 — directive 3절이 경고한 "거대/비-disk 컴포넌트에 하나의 UV 도메인을 강제하지 말라"는 실패 양상과 일치한다: 74,608개 멤버를 가진 chart 하나가 8×4=32개 control point 하나로 표현되면, 표면이 평면이 아니거나 image-space에서 넓게 접혀 있을 경우 residual이 크게 벌어질 수밖에 없다.

## 8. Overlap 일관성

11,552,676개 연속-쌍 샘플(directive 9절, 전체 pairwise가 아닌 대표 샘플링, 명시): 위치 불일치 중앙값 0.0297, 평균 0.0953, p95 0.4003, 최대 8.089(scene 단위). 법선 불일치 중앙값 5.04°(양호), 평균 12.95°, **p95 57.93°, 최대 179.97°**(사실상 반대 방향). 국소적으로는(중앙값) 여러 chart가 대체로 일치하지만, 꼬리(대형/저품질 chart가 몰린 영역, 특히 헤지)에서는 서로 거의 반대 방향 법선을 내놓을 정도로 불일치한다.

## 9. 테이블 결과

table_top 62,608개 중 75.3% 커버. table_legs(포디움 기반) 67,157개 중 **88.9%**로 세 하위 영역 중 가장 높다. table_side_curved(테이블 rim, 곡면) 155,930개 중 **57.3%로 가장 낮다** — 곡면 표면이 하나의 image-space chart로 잘 펼쳐지지 않는 경향과 일치.

## 10. 곡면 구조 결과

table_side_curved 자체가 곡면 구조 대표 사례(9절 참고). 57.3% 커버, no_valid_chart 66,535개 — 세 테이블 하위 영역 중 미커버 인구가 절대적으로도 가장 크다. `CURVED_STRUCTURE_VISIBLE_NURBS` export에서 rim을 따라 얇은 빨간 띠로 시각 확인됨.

## 11. 패티오 결과

332,845개 중 77.8% 커버 — 다섯 영역(테이블 3개 하위 영역 + 패티오 + 헤지) 중 가장 큰 절대 인구, 두 번째로 높은 커버리지. WL107/109에서 확인된 36.77%짜리 거대 컴포넌트가 이 영역에 크게 걸쳐 있어 chartability가 좋은 것으로 보인다.

## 12. 헤지/배경 결과

167,397개 중 **49.0%로 다섯 영역 중 가장 낮다.** no_valid_chart 85,292개(절대 인구도 patio 다음으로 큼). WL96-109에서 반복 확인된 헤지의 volumetric/파편화 특성이 그대로 컴포넌트 크기 분포에 반영된 결과로 해석된다 — 개별 컴포넌트가 32개 미만이면 어느 뷰에서도 유효 chart를 만들 수 없다.

## 13. 미커버 representative 원인 귀속

directive 15절이 요구하는 명시적 분류:

- **CANONICAL_TOPOLOGY_ISSUE (주 원인)**: 동결된 WL107/109 위상 자체가 심하게 파편화돼 있다(45.02% 싱글톤, 다수 컴포넌트가 32개 미만). 32개 미만의 대표를 가진 컴포넌트는 이 chart 방식으로는 구조적으로 어떤 유효 chart도 만들 수 없다 — 이것은 이번 배치의 chart 구성 방법이 만든 결함이 아니라, 입력 위상이 가진 근본적 한계다.
- **CHART_PARAMETERIZATION_FAILURE (커버된 소수 인구의 부차 원인)**: 성공적으로 커버된 소수의 거대 컴포넌트에서도, 한 뷰의 blob이 통째로 하나의 8×4 control grid로 fit되면서(directive 3절이 경고한 "하나의 거대/비-disk 컴포넌트에 전역 UV 도메인 강제") residual과 overlap 불일치가 심하게 벌어졌다. 이번 배치는 큰 blob을 여러 sub-chart로 나누는 로직을 구현하지 않았다 — directive가 "임의 임계값/스윕 금지"를 지시했으므로, 이 세분화는 다음 배치의 architecture 결정 대상으로 남긴다.
- REPRESENTATIVE_SUPPORT_STARVATION은 아니다: representative 자체는 785,937개로 충분히 존재한다. 문제는 그 representative들이 canonical 위상 안에서 32개 이상 뭉쳐 있는 컴포넌트에 속하는가이다.
- NURBS_MODEL_CAPACITY_FAILURE와 OVERLAP_INCONSISTENCY는 CHART_PARAMETERIZATION_FAILURE와 같은 근본 원인(거대 blob을 하나의 control grid로 강제)에서 파생된 동일 현상의 다른 측면으로 본다 — 별개의 독립 원인으로 이중 계산하지 않는다.

## 14. 과거 NURBS 결과와의 비교

과거 NURBS/파라미터화 시대(worklog 79-102, 이번 direction으로 완전히 대체됨)의 재현 가능한 커버리지 수치를 참고용으로만 인용한다(재실행하지 않음, directive 14절): worklog 103/104는 latent-surface 공간 커버리지 59.2%, connectivity-only 세분화로 노드 표현 98.7%를 보고했으나 그 파이프라인은 서로 다른 표면 표현(잠재 표면 곡선망/NURBS)과 다른 위상 구성을 사용했고, 이번 배치의 canonical renderer-native 위상과는 직접 비교 가능한 대상이 아니다 — 다만 "membership/node 커버리지가 높아도 실제 시각적/컴포넌트 커버리지가 낮을 수 있다"는 동일한 경고가 이번 배치의 5-6절 발견과 질적으로 일치한다는 점은 주목할 만하다.

## 15. 건축 판정

**NOT VIABLE** (현재의 per-view-blob chart 구성 방식으로는). Completion condition에 대한 답: **"NO — representative-only 증거는 현재 chart 구성 방식으로는 scene-covering continuous visible NURBS를 만들기에 충분하지 않다."** 두 가지 독립적 이유가 함께 작용한다: (1) 동결된 canonical 위상 자체의 파편화로 인해 절대다수 컴포넌트가 chart 최소 요건(32개)을 구조적으로 만족하지 못함(중앙값/p95 컴포넌트 커버리지 0%), (2) 커버에 성공한 소수의 거대 컴포넌트조차 하나의 미분화 chart로는 fitting 품질(residual 최대 7.9)과 overlap 일관성(법선 불일치 최대 180°에 근접)을 만족스럽게 얻지 못함. NURBS readiness를 주장하지 않는다.

## 16. 검토용 export 경로

`output/osn_gs_rep_only_nurbs/` 아래 11개 뷰(각 `iteration_0000001/point_cloud.ply`, `render.ppm`, `preview_png/render.png`, `README.md`): `ORIGINAL_2DGS_SCENE`, `CANONICAL_REPRESENTATIVE_BACKBONE`, `CAMERA_OBSERVED_CHART_DOMAINS`, `REPRESENTATIVE_CHART_COVERAGE`, `VISIBLE_NURBS_PATCHES`, `NURBS_PATCH_BY_VISIBLE_COMPONENT`, `OVERLAPPING_CHART_DISAGREEMENT`, `UNCOVERED_REPRESENTATIVES`, `TABLE_VISIBLE_NURBS`, `CURVED_STRUCTURE_VISIBLE_NURBS`, `HEDGE_VISIBLE_NURBS`. 전체 JSON 리포트: `output/osn_gs_rep_only_nurbs/representative_only_visible_nurbs_report.json`.

## 17. 테스트

신규 순수 로직 모듈 `osn_gs/surface/torch_camera_observed_chart_domains.py`(scipy 기반 정확한 connected-components, 프로젝트 기존 의존성)와 `scripts/devtools/representative_only_visible_nurbs.py`(실측 스크립트)를 추가했다. `torch_camera_induced_visible_adjacency.py`, `torch_nurbs.py`, 벤더 CUDA, 트레이닝/파이프라인 코드는 전부 무수정(읽기전용 재사용만) — 순수 진단/실험 코드만 추가됐으므로 directive 지시대로 전체 pytest는 재실행하지 않았다.

- `tests/test_camera_observed_chart_domains.py` (신규, 9개): 연결요소 라벨링(단일/2-분리/무효-픽셀-비연결/대각-비연결), chart 후보 구성(컴포넌트 혼합 금지, UV 정규화, 빈 뷰, 픽셀 카운트 합), 유효 chart mask.
- `tests/test_representative_only_visible_nurbs.py` (신규, directive 11절 A-F 6개): A 평면 시트 → 정확한 fit, B 곡면 시트 → 회전하는 법선에도 연속적 fit, C 한 컴포넌트가 두 뷰에서 관측 → 두 chart·동일 component_id, D 겹치는 관측 → 겹침 영역에서 호환되는 형상, E 인접한 서로 다른 컴포넌트 → 절대 같은 chart로 fit 안 됨, F 가려진 gap으로 분리된 표면 → 두 컴포넌트, 어떤 chart도 gap을 가로지르지 않음.
- `.venv/Scripts/python.exe -m pytest tests/test_camera_observed_chart_domains.py tests/test_representative_only_visible_nurbs.py -q` → **15 passed**.
- 실측 스크립트는 먼저 `--max-views 6` smoke test로 파이프라인 전체(chart 구성 → fitting → 회계 → export → render)가 오류 없이 끝까지 도는 것을 확인한 뒤, 전체 161개 뷰로 재실행했다(런타임 367.7초). 161개 뷰 재생 결과가 WL107/109의 알려진 수치(36.77%/45.02%)와 정확히 일치함을 확인해 위상 재생의 정확성을 검증했다.

## 다음 배치로 넘길 사항

이 배치는 chart 세분화(거대 blob을 여러 chart로 나누는 규칙)를 구현하지 않았다(directive가 임의 임계값/스윕을 금지했고, 이번 배치의 목적은 "현재 방식으로 충분한가"를 먼저 측정하는 것이었다). NOT VIABLE 판정과 그 두 가지 원인(위상 파편화, 미분화 거대 chart)은 다음 아키텍처가 다뤄야 할 문제이지만, 구체적 다음 단계 제안은 Master 문서 addendum에서만 다룬다(worklog 자체에는 넣지 않음, [[feedback_no_next_step_suggestions_in_worklogs]]).
