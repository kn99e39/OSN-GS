# Worklog 115 — Design-Intent / Specification / Implementation Traceability Audit (Worklog 107-113)

## 상태

**완료 — 감사(audit)만 수행, 구현/튜닝 없음(directive 지시).** Worklog 107-113의 인과 사슬(RESEARCH INTENT → AGENT SPECIFICATION → ACTUAL IMPLEMENTATION → OBSERVED RESULT)을 추적해, 관측된 실패들이 (a) 원래 연구 의도 자체의 문제, (b) 그 의도를 Agent directive로 번역하는 과정의 문제, (c) 실제 구현의 이탈, (d) 통제 실험이 의도적으로 얼린 조건의 예측 가능한 결과 중 어디서 비롯됐는지 구분했다. **결론: 감사 대상 범위(WL107-113)에서 순수 INTENT-level 실패(I)나 IMPLEMENTATION DEVIATION(III)은 발견되지 않았다.** 관측된 실패들은 압도적으로 **SPECIFICATION-INDUCED(II)** — chart 단위 정의(one-blob-one-chart), 사각형 UV 도메인, per-view 비병합이라는 번역 계층의 선택 — 이거나, 의도적으로 얼린 통제 조건(고정 8×4 NURBS 용량, IV)이거나, 렌더러 자체의 실제 현상(median-depth 국소 불안정, V)이다. Architecture의 canonical core(§11)는 그대로 유지되며, 재검토가 필요한 것은 specification 계층이지 연구 의도나 구현 충실도가 아니다.

## 1. 재구성한 상위 연구 의도

```
2DGS Renderer(관측된 image formation)
    -> Renderer-Native Surface Observation(렌더러 자신이 확인한 픽셀별 가시 표면)
    -> Visible Surface Topology(어떤 관측점들이 같은 연속 표면인가)
    -> Continuous Visible Surface Geometry
    -> NURBS Representation(그 기하의 구체적 parametric 인코딩)
    -> Representation Seam(NURBS patch 경계 — 인코딩의 artifact, 표면의 성질 아님)
    -> True Visible Termination(진짜 가시 표면이 끝나는 지점, chart가 그냥 멈추는 지점과 다름)
    -> Occluded Surface(향후 단계, WL113까지는 범위 밖)
```

의도 수준에서만(구현 요구사항 도출 이전) 회수한 불변 조건:

- Visible topology는 렌더러가 실제로 확인한 관측 연결성을 나타내야 한다 — 3D 근접성 가정이 아니다.
- **한 뷰의 비관측은 다른 뷰의 양성 관측에 대한 모순이 아니다.**
- **NURBS patch 경계는 representation seam이지, 물리적 표면 경계·visible-component 경계·occlusion 경계가 자동으로 아니다.**
- **Visible Surface Component ≠ Camera Blob ≠ NURBS Patch** — WL111/112가 암묵적으로 하나로 합쳤던 세 개의 서로 다른 대상.
- **Non-representative renderer contribution이 visible topology를 자동으로 정의하지 않는다.**
- **모호한 증거는 강제로 하나의 소유로 밀어넣지 않는다.**

이 재구성된 의도 어디에도 4-neighbor adjacency, 고정 8×4 그리드, 32-샘플 임계값, bounding-box UV 정규화, "blob 하나=chart 하나"는 명시돼 있지 않다 — 전부 하위 번역 계층의 선택이며, 이것이 아래 추적성 행렬이 보여야 할 핵심이다.

## 2-3. Intent → Specification 추적성 행렬 (핵심 항목만, 전체는 별첨 표 참조)

| 항목 | 상위 의도 | Directive 조작화 | 분류 |
|---|---|---|---|
| median representative | 렌더러 확인 관측점만 topology node | WL107: `T>0.5` crossing에서 캡처한 `median_surfel_id` | **CANONICAL SEMANTIC CONTRACT** |
| 4-neighbor adjacency | 카메라 자신의 렌더링이 직접 인접성 생성 | WL107 §5, 우측/아래 4-connectivity | 합리적 구현 선택(다른 선택도 가능했으나 이것이 GATE PASS를 받음) |
| 다중뷰 양성 union | 한 뷰의 비관측 ≠ 모순 | WL107 §9-10, threshold 없음, `UNRESOLVED_OBSERVATION_CONFLICT` 구조적으로 존재 불가 | **CANONICAL SEMANTIC CONTRACT** |
| 3D locality filter | 이미지-인접만으로는 실루엣-인접 오탐 가능 | 기존 WL96-106 candidate graph를 순수 필터로만 재사용 | 3D 필터의 **필요성**은 canonical, 기존 모듈을 그대로 재사용한다는 **구체적 선택**은 합리적 연속성 선택 |
| representative-only fitting | 모호한 증거를 강제로 밀어넣지 않음 | WL111: WL110 판정을 얼리고 non-representative 증거를 fitting에서 전면 배제 | 부분적으로 canonical(모호함을 강요하지 않음), 그러나 **전면 배제**는 의도가 요구하는 것보다 강한 조작적 선택 |
| **blob 하나 = chart 하나** | NURBS가 카메라 관측 연속 표면 조각을 표현 | WL111: 가장 단순한 첫 테스트로 명시적으로 선언(§11), subdivision은 의도적으로 미룸 | **실험적 조작 선택, WL111 스스로도 그렇게 명시.** WL114 directive의 "Component≠Blob≠Patch"는 사실상 이 등식의 철회 |
| image 좌표=UV | 실제 관측된 2D 투영에서 파라미터화 | WL111 §4 | **CANONICAL SEMANTIC CONTRACT** — WL97-102 시대 PCA/kNN 파라미터화의 이미 입증된 실패를 근거로 함 |
| bbox [0,1]² 정규화 | (의도 수준 근거 없음) | blob 자신의 픽셀 bbox로 정규화 | image-좌표=UV + tensor-product NURBS의 **기계적 필연**일 뿐, 독립적으로 요구된 바 없음 — WL113의 실패 B의 직접 원인 |
| 고정 8×4 degree-2 | (의도 수준 근거 없음) | `torch_nurbs.py`의 기존 함수 시그니처 기본값 재사용 | **실험적 조작 선택, 무관한 legacy 기본값 상속** — 이 architecture용으로 검증된 적 없음 |
| ≥32 샘플 요건 | (없음, 순수 파생) | 8×4=32에서 산술적으로 유도 | 8×4 선택이 주어지면 **논리적으로 필연** — 독립적 결정 아님 |
| per-view chart, 병합 없음 | 검증 안 된 병합 메커니즘 발명 금지 | 매 배치에서 반복 금지됨(merge/stitch) | 의도적·반복 재확인된 조작 선택 — 그러나 매 배치가 보고하는 **overlap 불일치의 직접 원인** |
| non-representative 부착 금지 | 모호한 증거를 강제 소유하지 않음 | WL108-113 전 배치 반복 확인 | **CANONICAL SEMANTIC CONTRACT**, WL110의 실측(truncation이 항상 보수적 방향)으로 이례적으로 잘 뒷받침됨 |
| rank-complete local chart(WL114, 진행중) | patch 경계는 representation seam | 고정 8×4 design matrix가 full column rank에 도달할 때까지 BFS 성장 | **신규 실험적 조작 선택 — §7에서 별도 감사** |

## 4. Specification → Implementation 충실도

WL107-113 전체에서 **IMPLEMENTED DIFFERENTLY FROM THE DIRECTIVE** 또는 미해결 **AMBIGUOUS/CANNOT VERIFY** 사례는 발견되지 않았다. 구현 충실도는 전반적으로 높다. 주목할 만한 예외 하나: `apply_secondary_geometric_gate`의 threshold 자체는 적응적으로 유도되지만(median+MAD), `residual_mad_multiplier=3.0`과 `parallel_sheet_normal_over_tangent_ratio=1.0`은 WL98-106 계보에서 상속된 고정 상수이며 이번 architecture용으로 재유도된 적이 없다 — **의미상 동등하게 구현됐으나 directive가 언급하지 않는 두 개의 숫자 손잡이가 존재**한다.

## 5. Agent가 도입한 가정 (실제로 코드에 존재하는 것만)

1. **cross-chart overlap을 대표당 평균(mean) 점/법선으로 집계** — directive가 요구하지 않음, median이나 first-observed도 동등하게 유효했을 것 — overlap 불일치 수치를 다소 완충시킬 수 있는 실질적 confound.
2. **연속-쌍(consecutive-pair) overlap 샘플링** — 매 배치 명시적으로 공개된 근사(전체 pairwise 아님), 3개 이상 chart가 닿는 대표의 불일치를 과소 계상할 수 있음.
3. **결정론적 tie-break 규칙**(pole seed, fragment 순서) — 이번 세션에 발명, floating-point 기하에서 measure-zero라 결과를 실질적으로 바꾸지 않음.
4. **`_RANK_CHECK_STEP=4`**(WL114) — 성장 루프의 후보 크기 증분 폭, chart 경계의 정확한 픽셀 경계에 영향.
5. **`max_patches_per_blob` 안전판**(WL114, 이번 세션 도입) — 순수 런타임 안전장치, architecture 파라미터 아님.
6. **rank/conditioning 진단의 2048-row 서브샘플링 캡**(WL113) — 공개됨, 가장 거대한 chart의 정밀도만 낮출 뿐 정성적 결론은 바뀌지 않을 가능성이 높음.

## 6. Prompt로 유발된 실패 양상 (반사실 추론)

**A. SUPPORT-LIMITED** — "완벽히 구현됐어도 발생했을까?" **그렇다**, 그러나 원인은 (blob=chart) + (per-view 비병합) + (32-샘플, 8×4에서 파생) 세 조건의 **결합**이다. 대표 인구(785,937) 자체는 작지 않다 — 문제는 그 관측이 어떻게 "뷰당·blob당" 단위로 쪼개지는가다. 이번 세션 WL114 스모크 테스트 자체가 이미 이를 부분 반증한다: 같은 증거를 어떻게 fitting 단위로 나누는지만 바꿔도(local rank-complete chart) 커버되는 대표 집합이 실질적으로 달라졌다. **주 provenance: II, 잔여 소량 IV/V.**

**B. RECTANGULAR-DOMAIN FAILURE** — 사각형 [0,1]² materialization은 원래 OSN-GS surface 의도가 요구한 바 없다. `image 좌표=UV`(canonical) + `blob별 bbox 정규화 하나의 도메인`(독립적으로 정당화된 적 없음)의 기계적 결과일 뿐이다. **`osn_gs/surface/torch_nurbs.py::TorchNURBSSurface`에는 이미 `uv_support_mask`(trimmed/masked 도메인 지원 필드)가 존재하지만 WL111-114 계보에서 한 번도 사용되지 않았다** — 이는 이번 감사의 가장 구체적인 발견 중 하나다. **Provenance: II.**

**C. FIXED NURBS CAPACITY FAILURE** — 8×4 degree-2가 최종 architecture로 의도됐는가, 통제 실험용으로만 얼렸는가? WL111 자체 텍스트가 후자임을 명시한다. 이번 세션 WL113 실측도 C를 좁게(소수 거대 컴포넌트에 국한) 뒷받침한다 — full-rank chart의 전형적 residual이 오히려 rank-deficient chart보다 낮았다(용량 부족이 아니라 표본 부족 쪽이 낮은 residual을 보였다는 뜻). **Provenance: IV, 깨끗함.**

**D. NUMERICAL/GRAZING FAILURE** — WL113의 정밀 outlier 추적(이번 세션)은 D가 작은 chart 내부에서 렌더러 median-depth 값이 극단적으로 튀는(예: 271픽셀 chart 안에서 depth 8.76~1723) **렌더러 자신의 픽셀별 depth 추출** 문제임을 보였다 — WL112/113이 발명한 것이 아니라 기존 `depths_to_points` 채널을 그대로 재사용한 결과다. **주 provenance: V.** 이번 감사에서 새로 짚은 2차 원인: WL107의 image-space adjacency 규칙은 "같은 representative 라벨 체인"만으로 topology를 정의하며, 라벨된 영역 **내부**의 depth-연속성은 전혀 점검하지 않는다 — 무효(-1) 픽셀을 가로지르지 않고도 진짜 depth 불연속이 한 컴포넌트 내부에 존재할 수 있다. **부차 provenance: II(약함).**

## 7. 제안된 rank-complete local-chart 아이디어 감사 (WL114, 구현하지 않음 — 이미 시작된 실측과 별개로 감사만 수행)

**핵심 질문**: "고정 8×4 design matrix가 full column rank에 도달"(대수적 식별가능성)이 "LOCAL GEOMETRIC NURBS REPRESENTATION UNIT"(기하적 chart 타당성)으로부터 논리적으로 따라오는가?

**아니다 — 이 간극은 실재하며 가설이 아니다.** Full column rank는 32개 파라미터 선형계가 풀 수 있고 퇴화하지 않았음만 보장한다 — 결과 patch가 컴팩트하고 disk에 가깝고 곡률상 일관된 지역 이웃인지는 전혀 보장하지 않는다. 이번 세션의 collinear-strip 테스트는 극단적 경우(1픽셀 폭 띠)만 올바르게 기각함을 보였을 뿐, "적당히 길쭉하지만 rank는 완전한" 영역이 기하적으로 타당함을 보증하지는 않는다. 스모크 테스트에서 관측된 양호한 aspect-ratio(중앙값 0.93 vs 베이스라인 1.56)는 **pole-of-inaccessibility seeding + BFS 방사 성장 휴리스틱의 우연한 emergent 성질**이지, rank-closure 기준 자체가 보장하는 성질이 아니다. **결론: full-rank closure는 좋은 fit의 필요조건이지 좋은 local chart의 충분조건이 아니다** — WL114의 residual/coverage 수치가 좋게 나오더라도, 이것만으로 이 간극이 해소됐다고 볼 근거는 없다. WL114가 이미 수집하는 aspect-ratio/occupancy 진단을 이 문제 전용으로 별도 확인해야 한다.

## 8. Failure-Provenance 요약표

| 실패 | I | II | III | IV | V |
|---|---|---|---|---|---|
| WL111 컴포넌트 커버리지 0% | 약한 부차 | **주** | — | 부차 | — |
| WL111/112 거대 chart residual/overlap 꼬리 | — | **주**(사각형 도메인) | — | 부차(고정 용량) | — |
| WL112 residual/overlap 악화 | — | 부차 | — | — | **주**(렌더러 depth 노이즈) |
| WL113-A | 약한 부차 | **주** | — | 부차 | 부차 |
| WL113-B | — | **주** | — | — | — |
| WL113-C | — | — | — | **주** | — |
| WL113-D | — | 부차 | — | — | **주** |

감사 범위(WL107-113) 전체에서 순수 **I(의도 실패)** 또는 **III(구현 이탈)**로 분류된 실패는 없다.

## 9. 반사실 테스트 요약

모든 주요 실패에 대해 "완벽히 구현됐어도 발생했을까?" → **A/B/C/D 전부 예** (구현 버그 아님을 확인). "조작 선택을 제거해도 논리적으로 필연적인가?" → **A/B는 아니다**(다른 specification이면 피할 수 있었음), **C는 그렇다**(어떤 고정 유한 용량 모델도 어떤 복잡도 상한은 가짐, 다만 "8×4"라는 구체적 숫자는 자의적), **D는 부분적**(렌더러 depth 노이즈는 의도와 무관하게 실재하지만, topology가 내부 depth-연속성을 침묵하는 것은 제거 가능한 진짜 간극).

## 10. 단순성 감사 / Design Debt

| 메커니즘 | 존재 이유 | 여전히 필요한가 | 제거 시 결과 |
|---|---|---|---|
| Per-view 전용 chart, 병합 없음 | 검증 안 된 병합 메커니즘 발명을 반복 금지 | 역사적 신중함, 증명된 필연 아님 | 제거하면 매 배치가 보고해 온 overlap 불일치가 줄어들 가능성이 높으나 새 architecture 결정이 필요 |
| bbox [0,1]² 사각형 UV 도메인 | image=UV 선택 후의 기계적 기본값 | **아니오** — `uv_support_mask`가 이미 존재, 미사용 | 실패 B의 직접 원인; 수정이 이미 코드베이스에 잠재해 있을 수 있음 |
| 고정 8×4/degree-2, 4개 배치 연속 동결 | 통제 비교 연속성(방법론적으로 타당) | 지금은 얼린 채 유지가 적절하나, 다시 검토 안 하면 영구화 위험 | 재검토 없이는 C가 구조적으로 영원히 미해결 상태로 남음 |
| `diff_surfel_rasterization_diag`(WL107 representative_id, WL109 forward_accepted, WL110 K=16 슬롯 누적) | canonical vendored 커널 불가촉 | **예, 여전히 필요** — WL110의 AMBIGUOUS/LAYERED 판정이 아직 미해결 | 제거하면 그 질문으로 돌아갈 유일한 경로가 사라짐 |
| `max_patches_per_blob`(WL114, 이번 세션) | 순수 런타임 엔지니어링 안전판 | architecture 파라미터로 취급하면 안 됨 | 실측에서 한 번도 발동 안 하면 문제 없음 |
| `_RANK_CHECK_STEP=4`(WL114) | 성장 루프 엔지니어링 세분도 | local-chart 단위가 채택되면 재검토 가치 있음 | 경계 픽셀이 미세하게 바뀔 수 있으나 정성적 결론은 안 바뀔 가능성 높음 |
| WL111 "CANONICAL_TOPOLOGY_ISSUE" 서술(코드 아님, 서사) | 당시 주 원인으로 명명 | 이번 감사로 "specification이 증폭한 topology 파편화"가 더 정확한 서술임이 드러남 | 향후 참조 시 "topology 자체를 고쳐야 한다"로 오독되지 않도록 교정 필요 |

## 11. 최소 근거-지지 canonical architecture

**증명된 CANONICAL CONTRACT**: (1) 렌더러 자신의 `T>0.5` median-crossing representative를 surface-observation primitive로 사용, (2) image-space 4-neighbor adjacency + 다중뷰 양성 union(부재≠모순)을 canonical Visible Surface Topology Backbone으로 사용(GATE PASS, bridge/frontier 강건성 감사 완료), (3) representative 인구와 non-representative(contributing/support) 증거는 항상 별도 카테고리로 추적, (4) AMBIGUOUS/LAYERED non-representative 증거는 강제 소유하지 않음(WL110, truncation이 보수적 방향으로 확인), (5) 카메라 관측 image 좌표를 chart UV로 사용(발명된 3D 파라미터화 아님), (6) Visible topology evidence와 NURBS-materialized evidence를 항상 분리 보고(WL114 directive가 이를 명시적 불변으로 재확인).

**미해결 representation 선택**: chart **단위** 자체(blob=chart는 기각됨, rank-complete local chart는 검증 중, §7의 대수/기하 간극 미해결), 사각형 vs trimmed(`uv_support_mask`) UV 도메인, 고정 8×4 용량이 최종인지 여부, per-view vs 병합된 chart 정체성, full rank가 기하적 chart 타당성의 적절한 대리 지표인지, non-representative 모호 증거를 언젠가 어떻게 표현할 것인지.

## 12. 다음 구현 배치 전 재검토할 사항

1. WL114 결과를 architecture 판정으로 신뢰하기 전에, rank-complete local chart의 aspect-ratio/compactness 분포를 §7의 대수/기하 간극 전용으로 별도 점검할 것.
2. 새 chart-도메인 메커니즘을 발명하기 전에 **이미 존재하는 `uv_support_mask` trimming**을 먼저 시도할 것 — 실패 B를 훨씬 작은 변경으로 해결할 수 있을 가능성.
3. Per-view 전용(비병합) chart 정체성을 그 자체로 독립적인 architecture 질문으로 재검토할 것 — 어떤 chart 단위를 택하든 남을 가능성이 높은 overlap 불일치의 가장 유력한 잔여 원인.
4. 고정 8×4/degree-2 용량이 기본값으로 영구화되지 않도록 주의할 것 — 지금까지 4개 배치 연속으로 정당하게 얼려왔으나, "이 architecture가 실제로 필요로 하는 용량이 무엇인가"라는 질문 자체는 한 번도 직접 던져진 적이 없다.
5. WL111의 "CANONICAL_TOPOLOGY_ISSUE" 서술을 향후 참조 시 "specification이 증폭한 topology 파편화"로 교정할 것.

## 완료 조건 답

현재 Visible-NURBS 실패는 **압도적으로 specification 계층(II)**에서 비롯됐다 — renderer-native topology 의도를 구체적 chart 구성 선택(blob=chart, 사각형 UV 도메인, per-view 비병합)으로 번역하는 과정에서. 여기에 분리 가능한 통제-실험 한계(IV, 고정 8×4 용량) 하나와 실재하는 데이터/렌더러 현상(V, 국소 median-depth 불안정) 하나가 더해진다. **감사 범위에서 구현 이탈(III)이나 순수 의도-수준 실패(I)는 발견되지 않았다.** Architecture의 canonical core(§11)는 그대로이며, 재검토가 필요한 것은 specification 계층이지 연구 의도나 구현 충실도가 아니다.

## 테스트

이 배치는 정적/의미론적 architecture 감사이며 전체 regression을 실행하지 않았다(directive 지시). 구현 충실도 확인을 위해 기존 코드(`torch_camera_induced_visible_adjacency.py`, `torch_camera_observed_chart_domains.py`, `torch_nurbs.py`, WL111-114 devtools 스크립트)를 직접 읽었을 뿐 어떤 코드도 수정하지 않았다.
