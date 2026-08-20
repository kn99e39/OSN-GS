# Worklog 99 — Interface-coherent Surfel Region merge

## 상태

**완료 — 실측 있음. 결과는 혼재(mixed)이며, 정직하게 보고한다.** Worklog 97(region-concentration, 실제 scene largest_fraction 20.84%, 그러나 곡률 있는 표면 과분열)과 Worklog 98(discontinuity-first, 곡률 표면은 완벽히 보존하지만 실제 scene에서 largest_fraction 94.51%로 percolation 재발)을 각각 최종 union rule로 채택하지 않고, "WL97의 안전하지만 과분열된 초기 region → region 간 interface 전체를 WL98의 미분 증거로 평가 → 광범위하게 지지되는 매끄러운 접점만 merge"라는 2단계 파이프라인으로 교체했다. 합성 fixture(원통·크리즈·평행 시트·zigzag 체인)에서는 설계 의도대로 전부 통과했고, 실제 scene에서 **테이블 상판의 곡면은 성공적으로 하나의 region으로 복원**됐다. 그러나 **배경(파티오+잔디+울타리)이 다시 하나의 거대 subset(53.86%)으로 합쳐졌다** — WL97의 20.84%보다는 나쁘고 WL98의 94.51%보다는 낫다. Architecture 채택 여부는 이 배치에서 결정하지 않는다.

## 아키텍처

```
학습된 2DGS surfel (t_w normal, 그대로)
    -> WL97 region-coherent partition (SAFE OVER-SEGMENTED 초기화 전용)
    -> REGION ADJACENCY GRAPH (초기 region 쌍 중 local candidate edge로 연결된 것)
    -> 각 인접 쌍의 전체 INTERFACE (edge 하나가 아니라 전부)
    -> WL98의 shape-operator residual + positional 연속성을, INTERFACE 전체에서 집계
    -> 지지(support) + 곡률-일관성 + positional-연속성을 모두 만족하는 interface만 merge
    -> 결정론적, 라운드 기반 반복 merge
    -> 최종 Coverage-first Surfel Subsets
```

## 1. 정확한 region-interface 정의

초기 region은 WL97의 `partition_surfels_region_coherent()` 결과 **그대로**(§2 참고 — WL98의 connected component로 재생성하지 않음). 두 초기 region `R_a`, `R_b`가 인접하다는 것은 **local candidate graph(WL96/97/98과 완전히 동일한 kNN + local-spacing gate)의 spatial edge 중 양 끝점이 서로 다른 region에 속하는 edge가 하나 이상 존재**한다는 뜻이다. 그 두 region 사이의 interface `I_ab`는 **그런 edge 전부**의 집합이다 — 최선의 edge 하나로 축약하지 않는다(§4 지시). 매 라운드마다 현재 DSU root 기준으로 이 interface 전체를 다시 계산한다(`osn_gs/surface/torch_interface_coherent_region_merge.py::partition_surfels_interface_coherent`).

## 2. WL97 초기화 자체의 수정 — 발견한 진짜 문제

WL97의 결과를 "그대로" 쓰려고 했으나, 구현 도중 **WL97 자신의 candidate edge 승인 기준에 positional 검사가 전혀 없다**는 사실을 발견했다: 정확히 같은 normal을 가진 두 개의 근접한 평행 시트(gap=0.15)를 WL97에 그대로 넣으면 **WL97 자체가 이미 둘을 하나의 region으로 합쳐버린다**(정상 정상 — normal이 같으므로 WL97의 유일한 기준인 orientation concentration은 절대 떨어지지 않는다). 우리 알고리즘은 merge만 하고 절대 split하지 않으므로, 이 초기화 실수는 이후 어떤 interface 평가로도 되돌릴 수 없다.

**정정**: `RegionCoherenceConfig`에 opt-in 필드 `require_positional_continuity`(기본값 `False` — 기존 WL97 standalone 동작·테스트는 전부 그대로 보존)를 추가했다. 이 값이 `True`이면 WL97의 accepted-edge 마스크에 WL98과 동일한(재사용, 새 임계값 아님) `normal_offset <= tangential_offset`(비율 1.0) 조건을 추가로 요구한다. 이 필드는 `positions`/`surface_normal`만 사용하므로 `SurfaceOrientationEvidence` 계약을 바꾸지 않는다. 본 배치의 초기화는 이 옵션을 `True`로 켠 WL97을 사용하며, WL97 standalone(기본값 `False`)의 15개 기존 테스트는 전부 변경 없이 통과한다(신규 회귀 테스트 1개 추가, §14).

## 3. 지지(support) / extent 통계 — 정확한 정의

Interface `I_ab`마다 다음을 기록한다:
- `edge_count` — interface edge 총 개수
- `unique_surfel_count_a`, `unique_surfel_count_b` — 양쪽 각각 접점에 실제로 관여한 서로 다른 surfel 수(edge 개수가 아님)
- `extent_in_spacing_units` — interface edge 중점들의 bounding-box 대각선을, 그 interface edge들의 평균 local spacing으로 나눈 값
- `fraction_smooth_continuation`, `mean_residual`, `max_residual`, `mean_normal_offset_ratio` (§4, §5)

**지지 floor는 새 독립 상수가 아니라 기존 상수에서 대수적으로 유도**했다:
- `min_unique_surfels_per_interface_side = local.neighbor_count`(=8) — interface 한쪽이 노드 하나 자신의 kNN 이웃보다 많은 surfel에 걸쳐 있어야 한다는 뜻. 이보다 좁으면 "공유된 면"이 아니라 노드 하나의 로컬 반경일 뿐이다.
- `min_interface_extent_in_spacing_units = local.spatial_connect_spacing_multiplier`(=2.0) — interface 자체의 공간적 폭이 candidate edge 하나가 도달할 수 있는 반경보다 넓어야 한다는 뜻. 이보다 좁으면 "고립된 sparse bridge"와 구별되지 않는다.

## 4. 곡률-일관성 interface 규칙 — 실측 전에 확정한 통계

지시(§6)의 요구대로, 어떤 통계를 쓸지 **실제 scene 결과를 보기 전에** 결정했다: interface의 **최소 residual**(WL98이 이미 실패한 방식 그대로 재현하게 됨)이 아니라 **`fraction_smooth_continuation`**(interface edge 중 WL98과 동일한 전역 임계값으로 "smooth"로 분류되는 비율)의 **과반(0.5) 이상**을 요구한다. `mean_residual`/`max_residual`은 진단용으로만 보고한다.

`edge_smooth = (edge_residual <= residual_threshold) & (edge_normal_offset_ratio <= parallel_sheet_normal_over_tangent_ratio)` — residual threshold와 ratio 임계값은 WL98과 **완전히 동일한 공식**(median + 3·MAD, ratio 1.0), scene 전체 spatial edge 모집단에서 **한 번만** 계산해 전 interface에 동일하게 적용한다(interface별로 다시 추정하지 않음 — interface마다 다른 기준을 쓰면 그 자체가 튜닝이 된다).

## 5. Positional 연속성 집계

`mean_normal_offset_ratio`(interface 전체 edge의 normal_offset_ratio 평균) `<= parallel_sheet_normal_over_tangent_ratio`(=1.0, WL98과 동일 임계값, 재사용)를 요구한다. 평행 시트처럼 normal은 거의 같지만 접평면 이탈이 큰 interface는 이 조건에서 걸린다.

## 6. 새로 도입한 유일한 자유 파라미터

**`interface_smooth_majority_fraction = 0.5`** 하나뿐이다(단순 과반, scene에 맞춰 스윕하지 않음). 그 외 모든 것은 WL97/WL98에서 **재사용**한 임계값이거나, §3에 적은 대로 기존 상수에서 **대수적으로 유도**한 floor다. (§2의 `require_positional_continuity`는 새 임계값이 아니라 기존 WL98 임계값을 WL97 초기화 단계에도 적용하는 on/off 스위치다 — 별도 항목으로 위에 명시.)

## 7. 결정론적 merge 알고리즘

라운드 기반 반복(최대 64라운드, 캡 초과 시 예외):

1. 현재 DSU root를 모든 초기 region에 대해 계산.
2. 현재 root가 다른 모든 spatial candidate edge를 현재 root 쌍으로 그룹화 → 각 그룹이 그 라운드의 interface.
3. §3~§5의 통계를 그룹마다 벡터화 계산, accept 여부 결정.
4. accept된 interface를 **`fraction_smooth_continuation` 내림차순, 동률은 (region_a, region_b) 오름차순**으로 정렬해 순차 처리(Worklog 97의 Kruskal 스타일 순차 union-find 전례를 그대로 따름). 이미 같은 root면 스킵(같은 라운드 내 transitivity로 이미 해소된 경우).
5. 라운드 내 merge가 하나도 없으면 종료. 실제 scene에서는 8라운드 만에 수렴(라운드별 merge 수: 4364 → 834 → 258 → 87 → 25 → 3 → 1 → 0).

## 8. 합성 fixture 검증

신규 focused 테스트 14개(`tests/test_interface_coherent_region_merge.py`) + WL97에 회귀 테스트 1개 추가, 전부 통과.

| Fixture | 기대 동작 | 실측 |
|---|---|---|
| 1/4 원통(WL97이 3개로 과분열) | 최종 1개 region으로 복원 | ✅ initial=3 → final=1 |
| 1/2 원통(WL97이 4개로 과분열) | 최종 1개 region으로 복원 | ✅ initial=4 → final=1, merges_applied = initial-1 |
| 90° 크리즈 | 분리 유지 | ✅ top-2 fraction > 0.85, 최소 1개 interface가 명시적으로 rejected |
| 평행 시트(gap=0.15, 동일 normal) | 분리 유지 | ✅ WL97 자체는 1개로 합치지만(§2), `require_positional_continuity=True` 초기화로 정확히 2개 region 유지 |
| Zigzag 4-plate 체인(각 접점은 narrow fold) | 체인 전체가 transitivity로 percolate하지 않음 | ✅ 첫/마지막 plate가 서로 다른 final region, final_region_count >= 3 |
| 단일 smooth edge 하나 | 지지 부족으로 merge 불가 | ✅ `min_unique_surfels_per_interface_side >= 2`(edge 1개=양쪽 각 1 surfel)로 구조적으로 불가능 |

**모든 합성 계약은 설계 의도대로 통과했다.**

## 9. 실제 scene 재실측 — 결과는 혼재

Checkpoint: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000`(Worklog 96/97/98과 동일, 1,190,469 surfel). Local candidate graph도 완전히 동일 파라미터로 재사용.

| | B. WL97 (standalone) | C. WL98 (standalone) | D. Interface-coherent (신규) |
|---|---:|---:|---:|
| Region/Subset 수 | 104,977 | 7,676 | 초기 114,420 → 최종 108,848 |
| Largest fraction | **20.84%** | **94.51%** | **53.86%** |
| Merge 적용 수 | — | — | 5,572 (8 라운드) |
| Coverage identity | True | True | True |

(초기 region 수가 WL97 standalone의 104,977이 아니라 114,420인 이유: §2의 `require_positional_continuity=True` 초기화가 WL97 자체보다 더 보수적으로 쪼개기 때문 — 이 초기화 자체의 largest_fraction은 20.62%로 WL97 standalone과 거의 동일해, 초기화 단계는 percolation-safe함을 별도로 확인했다.)

**긍정적 결과**: `INTERFACE_COHERENT_PARTITION` 뷰를 시각 검토한 결과 **테이블 상판 곡면이 하나의 단일 region으로 깨끗하게 복원**됐다(단색으로 렌더링). 다만 `MERGE_PROVENANCE_DEPTH` 뷰에서 테이블 영역은 낮은 depth(거의 merge 없이 초기 region 자체로 이미 하나)로 나타나, 이 특정 복원의 공은 이번 배치의 merge 메커니즘보다 §2의 초기화 정정(positional-gated WL97) 쪽에 더 크게 귀속될 가능성이 있다 — 정확한 귀속은 추가 조사 없이 확정하지 않는다.

**부정적 결과**: `INTERFACE_COHERENT_PARTITION` 뷰에서 파티오 바닥·잔디·울타리가 시각적으로 **동일한 단색**(largest subset, 53.86%)으로 렌더링됐다 — WL96/WL98이 보였던 것과 같은 종류의(정도는 약하지만) 거대 component 병합이다. `ACCEPTED_REGION_INTERFACE_MERGES` 뷰를 보면 accept된 5,572개 merge 중 다수가 울타리(hedge)의 조밀하고 텍스처가 많은 영역에 집중되어 있다 — 작은 파편들이 과반(0.5) 기준을 넘겨 연쇄적으로 합쳐진 것으로 보인다. **결론적으로 이번 배치의 주 목표(percolation 방지 + 곡률 복원 동시 달성)는 실제 scene에서 완전히는 달성되지 않았다** — WL97보다 largest_fraction이 악화됐다(20.84% → 53.86%).

## 10. Merge provenance

`merge_provenance`는 매 accept된 merge마다 라운드·양쪽 region id·edge_count·양쪽 unique surfel 수·extent·fraction_smooth·mean_residual·mean_normal_offset_ratio를 기록한다(`InterfaceCoherentPartition.merge_provenance`, `interface_coherent_accounting()`가 전체를 그대로 노출). `final_region_fragment_counts`는 각 최종 region이 몇 개의 초기 WL97 region으로 이루어졌는지 보여준다.

## 11. Coverage identity

`assigned == unassigned == 0`, `multiply_owned == 0`, `subset_sizes_match_ownership_map`, 전부 True — 실제 scene, 합성 fixture 모두 확인.

## 12. Review export

`scripts/devtools/interface_coherent_region_merge_export.py` → `output/osn_gs_interface_coherent_region_merge/`:

    A. ORIGINAL_2DGS_SCENE
    B. WL97_REGION_CONCENTRATION_PARTITION
    C. WL98_DISCONTINUITY_FIRST_PARTITION
    D. INTERFACE_COHERENT_PARTITION
    E. ACCEPTED_REGION_INTERFACE_MERGES
    F. REJECTED_REGION_INTERFACES
    G. MERGE_PROVENANCE_DEPTH

PNG preview: `output/osn_gs_interface_coherent_region_merge/preview_png/`.

## 13. 재현 명령

```
python scripts/devtools/interface_coherent_region_merge_export.py \
  --checkpoint output/arch_2dgs_coverage_first_surface/2dgs_run1/30000 \
  --out output/osn_gs_interface_coherent_region_merge \
  --device cuda \
  --source-path DATASET
```

## 14. 테스트

- 신규 focused: `tests/test_interface_coherent_region_merge.py` 14개, 전부 통과.
- WL97에 회귀 테스트 1개 추가(`test_parallel_sheets_fuse_by_default_but_separate_when_positional_continuity_required`) + 기존 "새 파라미터 1개" 테스트를 3개 필드(2개는 opt-in, 기본값 유지)로 업데이트, 16개 전부 통과.
- 전체 회귀: **1144 passed, 1 skipped, 1 warning, 18 subtests passed in 259.83s**(Worklog 98의 1129에서 +15).

## 결론 없음

이번 배치는 architecture 채택 여부를 결정하지 않는다. 실측 요약: 테이블 곡면 복원은 성공적이나(다만 정확한 공로 귀속은 미확정), 배경 percolation은 WL98보다는 개선됐지만 WL97보다는 악화됐다(20.84% → 53.86%). 과반(0.5) 기준이 조밀한 텍스처 영역(울타리)에서 너무 관대할 가능성이 있다는 것이 이번 배치에서 관찰된 사실이며, 특정 다음 단계를 제안하지 않는다.
