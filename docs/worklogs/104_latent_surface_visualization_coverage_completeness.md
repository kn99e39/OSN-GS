# Worklog 104 — latent surface visualization coverage 완결성 보정

## 상태

**완료 — 이 배치도 architecture 결정을 내리지 않는다.** Worklog 103은 latent-supported 4,599개 sample을 전부 정확히 export했지만, 86개 latent support unit 중 37개가 시각화 전용 NURBS materialization에 실패해 `ALL_LATENT_SURFACES` NURBS view에서 사라져 보였다. 이번 배치는 latent surface 자체나 다운스트림 승인 기준을 전혀 건드리지 않고, **시각화 유닛(visualization unit)만** 바꿔 이 편향을 제거했다 — 하나의 latent support unit이 하나의 NURBS로 표현될 필요가 없다는 전제로, 연결성 기반 결정론적 subdivision을 도입했다.

## 1. 37개 실패 유닛의 실제 원인

7-region 실측 checkpoint(`baseline_compatible/final`)에서 원래 실패한 37개 unit 전부를 직접 재현·조사했다: **37개 전부가 `insufficient_points_for_any_surface`이며, 예외 없이 크기 1 또는 2 node짜리 unit이었다.** 표면은 최소 3개의 비공선(non-collinear) 점이 있어야 정의될 수 있으므로, 이건 fitting 로직의 결함이 아니라 **latent support graph 자체가 만들어낸, 더 나눌 수 없는 최소 크기의 고립 조각**이다. Subdivision으로 "구제"할 방법이 원천적으로 없다 — 이미 subdivision의 최소 단위보다 작다.

## 2. 결정론적 subdivision 방법

신규 `osn_gs/surface/torch_latent_surface_visualization_coverage.py::materialize_unit_with_subdivision`:

1. 먼저 주어진 조각이 자기 자신의 edge 목록 기준으로 이미 여러 connected component로 갈라져 있는지 확인한다 — 갈라져 있으면 fit을 시도하기 전에 즉시 그 조각들로 나눈다(**핵심 발견**: 구현 중 이 순서를 지키지 않으면 서로 연결된 적 없는 두 조각이 하나의 NURBS로 합쳐질 수 있음을 테스트로 발견해 수정했다 — fitter가 연결성을 신경 쓰지 않고 아무 점 집합에나 수치적으로 fit을 "성공"시킬 수 있기 때문).
2. 단일 connected 조각이면 기존 Worklog 103 해상도 사다리(`fit_visualization_nurbs`)로 fit을 시도한다.
3. 실패하면, 그 조각 자신의 그래프에서 BFS로 가장 먼 두 anchor를 찾아 각 node를 더 가까운 anchor 쪽에 배정하고(동률은 결정론적으로 낮은 index), 그 결과를 다시 connected-component 분석해 실제로 연결된 하위 조각들을 얻는다. Convex hull, bounding box, PCA rectangle, 임의의 Euclidean bridging은 전혀 쓰지 않는다.
4. 하위 조각마다 재귀적으로 반복한다. 조각 크기가 3 미만이 되거나 더 이상 나눌 수 없으면(예: 이미 1~2 node) `UNREPRESENTED_LATENT_FRAGMENT`로 정확한 source node ID와 사유를 report한다.

모든 node는 정확히 하나의 결과(materialized fragment 또는 unrepresented fragment)에만 속한다 — 이걸 매 실측마다 `node_accounting_ok` 필드로 기계적으로 검증했다.

## 3. 실측 결과: visualization representation coverage certificate

신규 `scripts/devtools/latent_surface_visualization_coverage_export.py`가 Worklog 103의 D stage(`ALL_LATENT_SURFACES`)와 region별 export를 재생성했다(A/B/C/E stage는 unit-level visualization 실패에 의존하지 않으므로 재생성하지 않음). 별도 출력 디렉터리(`output/osn_gs_scene_latent_coverage_audit_subdivided/`)에 써서 Worklog 103의 원본 export를 그대로 보존했다 — before/after를 직접 비교할 수 있다.

| 지표 | Worklog 103 | Worklog 104(subdivision 후) |
|---|---:|---:|
| Latent-supported node | 4,599 | 4,599(불변) |
| **Visualization으로 표현된 node** | (report 안 함) | **4,539 (98.7%)** |
| Visualization 미표현 node | (report 안 함) | 60 (1.3%) |
| Fully represented unit | (report 안 함) | 43 |
| Partially represented unit | (report 안 함) | 6 |
| Completely unrepresented unit | 37 | 37(불변 — 전부 1~2 node 고립 조각) |
| Visualization NURBS patch 수 | 49 | 59 |

Region별 represented 비율은 97.3%~99.2% 사이다. `visualization_coverage_certificate.json`의 `node_accounting_ok`는 7개 region 전부 `true`다.

## 4. Export

**(2026-08-19 정정)** WebRenderer는 `iteration_<N>` 폴더당 `point_cloud.ply` 정확히 1개만 읽으므로(0~1개의 `nurbs_surface.json` 동반), 대표성 있는 point set마다 별도 폴더를 쓰도록 재구성했다.

- `output/osn_gs_scene_latent_coverage_audit_subdivided/visualization_coverage_certificate.json` — 위 표의 원본 전체, region·unit·fragment 단위까지.
- `output/osn_gs_scene_latent_coverage_audit_subdivided/full_scene/iteration_0000001/` — 재생성 대상 전체 Gaussian(`point_cloud.ply`만).
- `output/osn_gs_scene_latent_coverage_audit_subdivided/latent_projected_samples_with_visualization_nurbs/iteration_0000001/` — latent projected sample(`point_cloud.ply`) + `nurbs_surface.json`(subdivision 이후 NURBS 59개).
- `output/osn_gs_scene_latent_coverage_audit_subdivided/unrepresented_latent_fragments/iteration_0000001/` — 미표현 fragment(`point_cloud.ply`만, json 없음).
- `output/osn_gs_scene_latent_coverage_audit_subdivided/regions/region_<id>/{raw_region_evidence,unsupported_evidence,unrepresented_latent_fragments,latent_projected_samples_with_visualization_nurbs}/iteration_0000001/` — region별 재생성, 각 폴더 ply 1개.
- Worklog 103의 원본 export(`output/osn_gs_scene_latent_coverage_audit/`)는 전혀 건드리지 않았다.

## 5. 재현 명령

```
python scripts/devtools/latent_surface_visualization_coverage_export.py \
    --checkpoint output/extent_ab/val64/baseline_compatible/final \
    --out output/osn_gs_scene_latent_coverage_audit_subdivided \
    --device cuda --cap 2048
```

## 6. 검증

신규 focused 테스트 10개: `test_latent_surface_visualization_coverage.py` 8개(모든 node가 표현되거나 명시적으로 미표현 처리됨, 표현/미표현 중복 없음, subdivision이 unit 밖 node를 만들어내지 않음, subdivision이 원래 연결된 적 없는 두 조각을 합치지 않음 — 이 테스트가 위 2번 항목의 순서 버그를 실제로 잡아냄, 최소 크기 미만 조각의 무한 재귀 방지, quality/safety/identifiability label이 geometry를 걸러내지 않음(AST), provenance 보존, Worklog 95 estimator/support를 재구성하지 않음), `test_latent_surface_visualization_coverage_export.py` 2개(certificate accounting identity, 1~2 node unit이 크래시 없이 unrepresented로 report됨). 전체 회귀 1회 실행함(아래).

## 완결 조건 충족 여부

"Worklog 103의 모든 latent-supported projected observation은 그 자신의 실제 supported geometry에서 유도된 visualization-only NURBS로 시각적으로 표현되거나, 사실에 근거한 수치적 사유와 함께 명시적으로 unrepresented fragment로 표시된다" — `node_accounting_ok=true`(7개 region 전부)로 기계적으로 검증됐다. 4,599개 중 4,539개(98.7%)가 표현되고, 나머지 60개(1.3%)는 정확한 source node ID와 함께 명시적으로 report된다 — 어떤 node도 조용히 사라지지 않는다.

## 결론 없음

이 worklog는 latent surface coverage가 충분한지, Worklog 98~102 architecture를 유지할지 폐기할지에 대해 어떤 판단도 내리지 않는다. 완결성 조건이 충족됐으므로, 이제 `ALL_LATENT_SURFACES` NURBS view를 사용자가 정성적으로 검토해도 된다는 사실만 보고한다.
