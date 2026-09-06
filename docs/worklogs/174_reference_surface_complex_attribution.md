# Worklog 174 — Real tabletop reference zero-set surface-complex 원인 분석

## 1. 의도 정렬

**최종 판정은 `MIXED_ATTRIBUTION`이다.** W172 tabletop selected support가 charting 이전에 이미 canonical zero-set 구조를 섞고 있는지, 아니면 W154 planar charting이 본래 coherent한 reference surface를 파괴하는지를 물었다. **양쪽 다 기여한다.**

W154 planar chart가 coherent한 reference surface를 실제로 파괴한다는 것은 직접 입증됐다. W173이 남긴 height-collision witness는 **하나의 연속 표면이 접힌 결과**다. 동시에 selected support 자체도 깨끗한 단일 sheet가 아니다. Charting 이전에 이미 **132개 triangle-connected reference component**와 국소적으로 모호한 cell들을 갖고 있다.

Region 1 분할, chart 재설계, support 보수, loop 병합·삭제, hole filling, NURBS fit은 하지 않았다. 이 batch는 원인 귀속에서 종료한다.

## 2. 구현 충실도

새 코드는 `worklog_174_surface_complex.py`(진단 유틸리티), `worklog_174_reference_surface_attribution.py`(본 분석), `worklog_174_review_exports.py`(시각화), `worklog_174_reextraction_feasibility_probe.py`(착수 전 타당성 검증)의 독립 진단이다. W154/native module, W171/W172/W173 산출물, production은 수정하지 않았다.

### 2.1 지시서 §3 전제와 저장소 현실의 차이

지시서는 "frozen raw zero-set **triangle** geometry"를 canonical reference로 쓰라고 했으나, **W171/W172 계보에 삼각형은 존재한 적이 없다.** 착수 전 정적 확인으로 다음을 확인했다.

- W171의 조상인 `candidate_f_tsdf_surface_samples.npz`의 키는 `source_cell_keys / cell_indices / world_xyz / normals / corner_values / corner_support_count`뿐이며 faces·verts가 없다. 셀당 정확히 1점(21,235,312 cell → 21,235,312 point).
- 이를 만든 `extract_tsdf_zero_surface_samples()`는 marching cubes가 아니라 12개 cube-edge 교점의 **평균 1점**을 내는 surface-nets 축약형이다. 저장소에 MC triangle table이 없다.
- 산출물과 코드 양쪽이 `"mesh_intermediate": false`로 명시한다.
- 저장소가 이미 이 한계를 기록해 두었다. `raw_visible_surface_replay_construction_provenance_audit.py`의 `"tsdf_cell_to_active_marching_cubes_cell": "DETERMINISTIC_IN_SOURCE_BUT_NOT_PERSISTED; extraction local cell ownership is discarded"`.

이는 W173이 §9에서 "native ordered 1D loop population이 없다"고 보고한 것과 **같은 근본 원인**이다. 사용자 승인 하에, 삼각형을 **동일 frozen corner scalar에서 결정론적으로 재계산**하는 경로로 진행했다. "저장된 것을 읽어오는" 것이 아니라 "같은 field에서 새로 계산"하는 것이며, 이 차이를 보고서 `reference_geometry_contract`에 명시했다.

### 2.2 Ownership이 construction-native인 이유

각 authoritative cell을 **자기만의 2×2×2 scalar block**으로 marching cubes에 넣는다. 따라서 triangle이 cell 경계를 넘을 수 없고, ownership이 기하가 아니라 **구조적으로 exact**하다. §3이 금지한 nearest-neighbor·radius·normal·threshold 매칭은 어디에도 쓰지 않았다.

Vertex welding은 global lattice-edge의 exact integer key(정수 corner + 양자화된 crossing parameter)로 한다. 공간 tolerance 병합이 아니다. 이 규칙이 맞아야 나머지 위상 전부가 유효하므로, 인접 두 cell이 하나의 성분으로 병합되고 떨어진 cell은 병합되지 않음을 합성 fixture로 먼저 검증한 뒤 실제 데이터에 적용했다.

### 2.3 착수 전 타당성 검증 결과

| 검증 항목 | 결과 |
| --- | ---: |
| 삼각형을 만든 cell | 15,189 / 15,189 |
| 총 삼각형 | 31,149 |
| **단위 cell 이탈 삼각형** | **0** |
| 재추출 centroid ↔ 저장 `world_xyz` 오차 median | 0.0121 h |
| 동 오차 max | 0.3310 h |

두 추출기의 cell 적격 규칙(8-corner authoritative + min≤0≤max)이 동일함을 코드로 확인했다. 오차가 0이 아닌 것은 정상이다. 저장된 `world_xyz`는 MC vertex가 아니라 cube-edge 교점의 평균점이라 애초에 다른 양이다. median 0.012 h는 같은 zero-set을 가리킨다는 증거다.

## 3. W172/W173 기준 보존

| 항목 | 값 |
| --- | ---: |
| W171 complete tabletop support | 17,965 |
| W172 selected support | 15,189 |
| Complete 대비 비율 | 84.5477% |
| Selected native components | 1 |
| W173 loops / chart holes | 134 / 134 |
| Historical h | 0.012105485424399376 |

W173의 `load_baseline()`을 그대로 호출해 complete/selected row hash, native component identity, 134 loop, chart origin/tangent/occupancy를 재현했다. 재추출에 쓴 corner values의 `cell_indices`와 `source_cell_keys`가 W172 저장 배열과 exact 일치함을 assert로 확인했다. W171/W172/W173 산출물 전체의 SHA-256 manifest를 실행 전후로 비교해 `frozen_inputs_unchanged_after_run = True`를 검증했다. Fitter는 호출하지 않았다.

## 4. Reference zero-set surface complex

| 항목 | 값 |
| --- | ---: |
| Triangle | 31,149 |
| Welded vertex | 17,177 |
| Edge | 48,330 |
| Euler characteristic | -4 |
| Triangle-connected component | **132** |
| Boundary edge | 3,213 |
| Boundary edge component | 189 |
| Non-manifold edge | **0** |
| Non-manifold vertex | 23 |
| Triangle을 못 낸 cell | 0 |
| 단위 cell 이탈 | 0 |

Edge degree는 1이 3,213개, 2가 45,117개로 **3 이상이 하나도 없다.** 즉 edge 수준에서는 manifold이며, 23개 welded vertex에서만 fan이 갈라지는 pinch가 있다. Euler/genus는 topology 집계일 뿐이며 물리적 의미로 읽지 않는다.

## 5. Source-cell ↔ triangle ownership

| 분포 | Count | Min | Median | Mean | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cell당 triangle 수 | 15,189 | 1 | 2 | 2.050760 | 3 | 9 |
| Cell당 국소 patch 수 | 15,189 | 1 | 1 | 1.028968 | 1 | 3 |

**427개 cell이 한 cell 안에 서로 분리된 patch를 2~3개 갖는다.** "cell 1개 = 표면 조각 1개"라는 현행 추상화가 모든 곳에서 성립하지는 않는다는 뜻이다. 다만 median이 1이므로 지배적 양상은 아니다.

### 5.1 Native 6-face 연결성 vs triangle 표면 연결성

| 항목 | 값 |
| --- | ---: |
| Native 6-face 인접 row 쌍 | 30,821 |
| Mesh edge를 공유하는 row 쌍 | 29,226 |
| **Native 인접이지만 표면 인접이 아님** | **1,595 (5.18%)** |
| 그중 여전히 같은 triangle component | 1,333 |
| 그중 다른 triangle component | 262 |
| **표면 인접이지만 native 인접이 아님** | **0** |

오류가 **일방향**이다. Native 6-face 연결성은 표면 연결성을 **과다추정**하기만 하고 과소추정하지는 않는다. W172가 "native component 1개"라고 보고한 support가 실제 표면으로는 132조각이라는 사실이 여기서 나온다.

## 6. W173 height-collision witness

| 항목 | 값 |
| --- | ---: |
| Row | 4043 / 4051 |
| Native cell index | `[44,148,81]` / `[45,119,65]` |
| **Native cell L1 거리** | **46** |
| 같은 chart bin | True (25,37) |
| World 거리 | 32.88738 h |
| Chart-normal 높이 차 | 32.88685 h |
| **같은 triangle component** | **True** |
| Surface path edge steps | **102** |
| Surface path 길이 | 57.238 h |
| 직선 대비 path 비 | **1.7377** |

**판정: `CHART_COLLAPSE_SAME_SURFACE`.**

두 row는 native 격자에서 46 cell 떨어져 있음에도 **같은 triangle-connected component에 속하며 102 edge step으로 이어진다.** Path/chord 비가 1.74이고 path의 world bbox가 z 0.800~1.019, y 1.439~1.802에 걸치는 것은 상판에서 옆면으로 내려가는 **연속 표면**임을 뜻한다. 거리 threshold를 도입해 "같은 sheet"를 정의하지 않았다. 판정은 오직 mesh edge 공유에 의한 인접성으로만 했다.

이로써 W173이 "정상 단일 sheet 확정을 막는 evidence"로 남겨 둔 witness의 성격이 규명됐다. 이것은 support가 이질적 구조를 섞은 증거가 **아니라** planar chart가 접은 증거다.

## 7. W173 loop / reference-surface 대응

| 모집단 | 개수 |
| --- | ---: |
| W173 chart loop | 134 |
| W173 chart hole | 134 |
| Reference-surface boundary edge component | 189 |
| Reference-surface triangle component | 132 |

**1:1 대응을 확립하지 않았고 주장하지도 않는다.** Chart hole은 quantized planar occupancy의 2D 빈 영역이고 boundary edge component는 3D triangle complex의 1D edge cycle이라 애초에 같은 종류의 대상이 아니다. Frozen 기록에는 두 모집단을 잇는 construction-native map이 없다.

확립된 것은 다음 두 가지다. selected support가 실제 surface boundary를 3,213 edge만큼 갖고 있으므로 chart hole 전부를 charting artifact로 돌릴 수 없다. 동시에 reference surface 자체가 132조각으로 갈라져 있으므로 일부 chart hole은 실제로 끊어진 표면 조각을 가른다. Per-hole causal provenance는 W173과 마찬가지로 여전히 복원 불가다.

## 8. Planar chart distortion

| 항목 | 값 |
| --- | ---: |
| Occupied chart bin | 6,609 |
| Multi-sample chart bin | 3,474 |
| **서로 다른 reference component를 섞는 bin** | **519** |
| Multi-sample bin height span median | 0.15626 h |
| 동 p95 | 9.40102 h |
| 동 max | **32.88685 h** |

W173의 32.88685 h witness가 max로 정확히 재현된다. 519개 bin이 서로 다른 reference component를 한 칸에 합치는 것은 projection의 결과이지 support의 결함이 아니다. 다만 나머지 bin의 median span이 0.156 h로 작다는 점도 함께 기록한다. Chart 붕괴가 전역적으로 균일하지는 않다.

## 9. Region-ownership 해석

| 해석 | 판정 |
| --- | --- |
| `REGION_SUPPORT_SINGLE_REFERENCE_SURFACE` | 부분 지지. 최대 성분이 triangle의 91.02%, support row의 90.55%를 차지하고 witness도 그 안에서 연결된다. 그러나 유일한 sheet는 아니다. |
| `REGION_SUPPORT_MIXES_REFERENCE_STRUCTURES` | 소수 잔여로서 지지. charting 이전에 이미 132 component이며 262개 native-인접 쌍이 서로 다른 component에 있다. |
| `REFERENCE_SURFACE_TOPOLOGY_AMBIGUOUS` | 국소적으로 지지. 427 cell이 다중 patch, 23 vertex가 non-manifold다. 다만 non-manifold edge는 0이다. |

**`region_id = 1` + W172 native component는 하나의 local reference-surface sheet를 나타내기에 충분하지 않다.** 다만 그 부족분은 소수다. 91%는 실제로 하나의 sheet다.

## 10. 시각 검토

[산출물 README](../../output/174_reference_surface_complex_attribution/README.md) · [전체 실측 JSON](../../output/174_reference_surface_complex_attribution/worklog_174_report.json) · [component 전체 목록](../../output/174_reference_surface_complex_attribution/loop_inventory.md)

| Family | 내용 |
| --- | --- |
| A reference surface complex world | 복원된 31,149 triangle을 면으로 렌더링, support 15,189점 중첩 |
| B surface complex components | 132 component 전부, 회색 dominant + 색상 131개, 숨김 없음 |
| C w173 height witness | witness surface path(빨강)와 chart 붕괴를 world/chart 나란히 |
| D world vs chart surface | world (Y,Z) / continuous chart / quantized occupancy 3면 비교 |
| E loop cause review | surface boundary 소유 row와 4개 모집단 개수 대비 |
| F rgb reference | DSC07960 / DSC08003 / DSC08043에 component 색 투영 |

총 **PNG 8개, PPM 0개, UTF-8 README 7개**다. 각 family README에 입력·색 범례·좌표계·한계와 **분석 및 평가** 절을 기록했다. Component filtering, small fragment 숨김, smoothing, gap filling, morphology를 적용하지 않았다.

직접 검토한 C 그림에서 빨간 경로가 상판에서 아래로 내려가는 하나의 연속 띠로 읽히고, 오른쪽 chart에서는 그 전체가 좁은 영역으로 접혀 두 점이 같은 bin에 겹치는 것이 보인다. B 그림에서 회색 dominant sheet가 테이블 상판 형태로 읽히고 색상 조각들이 그 안에 흩어져 있다. 이 배치 형태는 별개 물체보다 관측 결손으로 끊긴 조각에 가까워 보이지만, **그 구분은 현 증거로 확정되지 않으므로 판정에 쓰지 않았다.** RGB는 depth culling이 없어 image overlap을 topology 증거로 쓰지 않았다.

## 11. 아키텍처 원인 판정

**`MIXED_ATTRIBUTION`.** 완료 조건의 질문에 대한 답은 다음과 같다.

W154 planar chart는 coherent한 reference surface를 **실제로 파괴한다.** 감사한 witness는 하나의 연속 표면 경로가 단일 chart bin으로 접힌 것이고, 519개 bin이 서로 다른 reference component를 합치며, height span이 최대 32.89 h에 이른다.

동시에 selected support도 **깨끗한 단일 sheet가 아니다.** charting 이전에 이미 132개 reference-surface component를 담고 있고, native 6-face 인접 쌍의 5.18%가 표면상 인접이 아니며, 427 cell이 국소적으로 모호하다.

두 원인의 비중은 대칭이 아니다. 최대 성분이 91.02%를 차지하므로 support는 **지배적으로는** 하나의 sheet이고, 나머지 9%가 charting 이전에 이미 갈라져 있다. W173이 "정상 단일 sheet 확정을 막는다"고 본 그 witness는 이번에 chart 쪽 원인으로 귀속됐다.

## 12. 유지·미채택·미해결

- 유지: W171 tabletop selection, W172 exact component/rows, W173 chart·134 loop·topology 결과, native connectivity, W154 chart/boundary/fitter, historical TSDF/zero-set semantics, production.
- 미채택: Region 1 분할, region 정책 변경, chart 재설계, chart quantization 변경, normal/plane/surface-distance threshold, fragment 제거, dominant surface 선택, raw mesh 보수, smoothing, loop 병합, hole filling, multiply-connected domain 구현, NURBS fit/재설계, continuation, UNRESOLVED, Eligibility, inferred Gaussian.
- 미해결: 소수 component가 물리적으로 별개 물체인지 한 물체의 관측 결손인지 구분되지 않는다. 각 chart hole을 observation·construction·projection 중 무엇에 귀속할지에 대한 per-hole provenance가 여전히 없다. Reference는 frozen zero-set이며 physical ground truth가 아니므로 물리적 진리에 대한 주장은 하지 않는다.

**Structural Surface / Parametric Carrier ≠ Observed Evidence-Supported Domain.** 표면이 132조각이라는 사실이 곧 132개 물체를 뜻하지 않고, 91%가 한 sheet라는 사실이 곧 나머지를 무시해도 된다는 뜻도 아니다. 다음 단계 설계는 구현하지 않고 architecture review 지점에서 종료했다.

검증: W174 focused tests **17 passed**, W171/W172/W173 합산 회귀 **39 passed (6.08s)**로 historical 결과 불변을 확인했다. Full repository regression은 production 미변경이므로 실행하지 않았다. 모든 계산은 로컬 CPU였고 GPU workload나 replay-cache 복사는 없다. Source 코드·tests·Worklog는 commit하고, PNG/JSON/NPZ는 기존 정책대로 gitignored output에 남긴다.
