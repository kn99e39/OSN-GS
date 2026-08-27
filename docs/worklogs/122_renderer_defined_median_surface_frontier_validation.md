# Worklog 122 -- Renderer-Defined Median Surface Frontier Validation (Candidate B only)

브랜치: `arch/2dgs-coverage-first-surface`
보존 기준: Worklog 120 (`fdfb8ad60b6233ea8364a09ea3467c18e600a246`), Worklog 121 -- 둘 다 역사적 기록으로 그대로 두었고 수치를 다시 쓰지 않았다.

---

## 1. Agent Interpretation of Intent

**DIRECTION**: A(Surface-Hit)는 RETIRE, C(Geometric Visibility)는 RETIRE, D(Renderer Reachability)는 NOT VIABLE AS STATED. **B만** 검증한다. B를 최적화하지 않고, threshold를 비교하지 않으며, A/C/D를 되살리지 않는다.

**PURPOSE**: canonical renderer의 median-surface event가 OSN-GS의 **renderer-defined visible-surface observation frontier**로 쓰일 수 있는지 판정한다. 검증 대상 주장은 "median depth는 물리적 first ray/surface hit이다"가 **아니라** "renderer 자신이 선택한 visible-surface event가 camera-facing observed 영역과 behind-surface 영역을 가르는, 일관되고(coherent) 닫혀 있고(closed) 충분히 비모순적인(non-contradictory) frontier를 제공한다"이다. Soft alpha compositing이 frontier 뒤에 0이 아닌 기여/기울기를 남기는 것은 그 자체로 frontier를 무효화하지 **않는다**. 반대로 coverage만으로 검증되지도 **않는다**.

**CENTRAL INTENT**: median 규칙을 최적화하지 않는다. **추상화 자체가 유효한가**를 판정한다. 부정적 결과도 유효하며, 아키텍처 성공을 자동으로 주장하지 않는다.

### PRESERVE
학습 checkpoint, 161 학습 카메라, WL107/109 frozen visible topology, canonical renderer, **Candidate B `classify_view`(파일 무수정)**, frozen global aggregation, WL121의 true-fragmentation 컨텍스트 300개, 원본 4,712-query bank, 보충 908-query bank.

### CHANGE ONLY
median frontier 검증에 필요한 진단만.

### DO NOT (전부 준수)
B epsilon/tolerance 도입 / nextafter·ULP 조정을 production fix로 사용 / T > 0.5 변경 / median threshold sweep / B+D hybrid / A 부활 / C 패치 / D 물리적 재정렬 / **"D OCCLUDED가 physical-depth 재정렬에 불변"이라는 주장** / 3D-locality filter 수정 / visible component 병합·분할 / **B midpoint OBSERVED를 표면 연속성의 증거로 사용** / Occluded Surface / occluded NURBS / Trust / NURBS fitting 수정 -- **하나도 하지 않았다.**

### 지시 section 2 (WL121 정정) 준수
WL121의 "physical-depth 재정렬은 D OCCLUDED를 늘릴 수만 있다"는 진술을 **이번 배치로 이월하지 않았다**. WL121은 REACHED 해소 **이후**의 late-front 이벤트만 측정했고, 대칭 사례(해소/종료 **이전**에 처리되었으나 실제 per-pixel depth는 query보다 **뒤**인 이벤트 -- 이들은 termination 이전에 이미 T를 낮췄을 수 있다)는 측정하지 않았다. 따라서 D의 가상 물리-depth 교정의 **방향은 확립되지 않았다**. D는 이번 배치에서 수리하지 않으며, **B의 ground truth로 사용하지 않았고**, 역사적 맥락으로만 언급한다. 이 진술은 리포트 JSON `candidate_D_status`에도 그대로 기록되어 있다.

### PROMPT-REQUIRED DECISION
- B 결정 함수·frozen aggregation·topology·renderer 전면 보존, epsilon 금지.
- frontier self-closure를 "가능한 모든 median event"에 대해 수행하고, pixel 보존 여부·signed/absolute/relative delta·ULP·rho branch를 측정(section 3).
- tolerance 없이 dataflow를 분해해 closure 상실 지점을 귀속(section 4).
- 정확한 identity 표현이 존재하는지 진단(section 5).
- truncation 편향 없는 post-median 전수 회계와 A/B/C 의미 귀속(section 6/7).
- cross-view disocclusion(section 8), S1-S5(section 9), WL121 300 컨텍스트 재생(section 10), 영역별 분리 보고(section 11).

### AGENT-INTRODUCED OPERATIONAL CHOICE (전부 공개)
1. **CUDA 진단 필드(worklog 122, additive)**: 입력 `primitive_component`, `primitive_representative_class`; 출력 `post_median_counts`/`post_median_weights` (H,W,**10** 카테고리), `total_accepted_weight` (H,W), `post_median_depth_stats` (H,W,3 = sum/min/max). 기존 canonical 및 WL121 출력의 비트 불변을 테스트로 강제했다.
2. **post-median 판정**: `T <= 0.5` (acceptance 시점, pre-update) -- **WL110이 이미 쓰는 자기 자신의 post-median 검사**를 그대로 사용했고 새 정의를 만들지 않았다.
3. **10개 카테고리**: 0 all / 1 same component / 2 cross component / 3 unresolved component / 4 이 뷰의 median representative / 5 다른 뷰에서만 / 6 어디서도 아님 / 7 rho2d low-pass / **8 per-pixel depth가 median보다 앞** / **9 median 이후**. 8·9는 실행 중 depth offset의 최솟값이 음수임을 보고 추가했다 -- traversal-order post-median이 physically-behind를 뜻하지 않는다는 사실을 정량화하기 위해서이며, 어떤 결정도 이에 의존하지 않는다.
4. **closure 원인 분류**: `RASTER_PIXEL_REASSIGNMENT`(픽셀이 바뀌었는가 -- 정확한 정수 비교), `ROUND_TRIP_1_ULP` / `2-8 ULP` / `>8 ULP`(정확한 float32 ULP 거리). **이것은 측정된 사실의 분류이지 tolerance가 아니다** -- 어떤 것도 이 값 때문에 수용/기각되지 않는다.
5. **선택 stride(결정 무관)**: disocclusion anchor = 매 10번째 뷰(17뷰) x 뷰당 결정론적 raster stride 200개 = 3,400개; closure 정성 샘플은 매 20번째 뷰; post-median per-pixel 질량 비율 샘플은 소수 stride 977.
6. **synthetic fixture 상수**: S4 = 12 splat, spacing 0.004, opacity 0.25; S5 = layer gap 1.5, opacity 0.9; 전부 결과 방향을 보기 전에 확정.
7. **translucent fixture**는 **OUT-OF-SCOPE / AMBIGUOUS 의미 대조군**으로만 포함했고 어떤 아키텍처 결정도 여기서 끌어내지 않았다.

---

## 2. Historical B Invariance

Candidate B의 결정 함수 파일(`candidate_b_median_depth.py`)과 `shared.aggregate_global`은 **수정하지 않았다**(`git status`에 없음). 이번 배치의 모든 판정은 그 함수를 직접 호출해 얻었다.

테스트로 강제한 불변식:
- 결정 규칙 자체(`<=` OBSERVED / `>` OCCLUDED / sentinel UNRESOLVED)가 그대로다.
- **median 위 1 ULP는 여전히 OCCLUDED다** -- epsilon이 몰래 들어가지 않았음을 직접 확인.
- `candidate_b_median_depth.py`의 AST에 `0.0` 이외의 float 상수가 **없다**.
- `frontier_validation.py`는 어떤 상태도 스스로 대입하지 않고 `candidate_b.classify_view`를 호출한다.

동결 지문: median representative 합집합 **785,937** (WL119/120/121과 일치). WL107/109 topology 재생 = **component 559,989 / singleton 535,910 / largest fraction 0.36771306** (일치).

---

## 3. Frontier Self-Closure

**전수 측정**: 161개 학습 뷰의 **모든** 유효 median event -- **43,817,760개**(이 scene은 모든 픽셀에 유효 median event가 있다). 각 event를 WL119의 G2로 world 재구성 → **자기 자신의 source camera로 재투영** → frozen Candidate B로 분류.

| 항목 | 값 |
|---|---|
| 검사한 median event | **43,817,760** |
| 재투영이 relevant | 43,817,760 (100%) |
| source view **OBSERVED** | 35,660,438 (81.38%) |
| source view **OCCLUDED** | **8,157,322 (18.62%)** |
| source view UNRESOLVED | **0** |
| **closure 모순율** | **18.62%** |
| **source pixel 보존** | **43,817,760 / 43,817,760 = 100.00%** |
| source pixel 변경 | **0** |
| signed delta 평균 | +2.49e-09 |
| signed delta min / max | −3.05e-05 / +4.88e-04 |
| absolute delta max | 4.88e-04 |

rho branch 별 모순율: rho3d **18.53%**(6,456,982 / 34,846,910), rho2d **18.95%**(1,700,340 / 8,970,850) -- **branch와 사실상 무관**하다. 이는 low-pass provenance가 원인이 아님을 뜻한다.

**WL121 보충 endpoint 모순의 재현·설명**: WL121이 보고한 endpoint A 290 OBSERVED / 10 OCCLUDED, endpoint B 296 / 4를 이 배치가 **비트 단위로 그대로 재현**했다(section 10). 절대 오차가 작다는 이유로 넘기지 않고, 전수 측정으로 **모집단 모순율이 18.62%**임을 밝혔다 -- endpoint 표본(10/300 = 3.3%, 4/300 = 1.3%)이 오히려 모집단보다 낮았던 것이며, 의미론적 모순율은 절대 오차 크기와 별개의 양이라는 지시의 지적이 정확했다.

---

## 4. Numerical Boundary Attribution

tolerance를 **추가하지 않고** dataflow를 분해했다: canonical CUDA median depth → median s_u/s_v → G2 world 재구성 → world float32 저장 → camera transform → raster pixel → camera-space query depth → median 비교.

| 원인 | 건수 | 모순 중 비중 |
|---|---|---|
| CLOSED_OBSERVED | 35,660,438 | -- |
| **RASTER_PIXEL_REASSIGNMENT** | **0** | **0%** |
| half-pixel 규약 / 재투영 non-relevant | **0** | 0% |
| 재투영 픽셀에 유효 median 없음 | **0** | 0% |
| **ROUND_TRIP_1_ULP** | **7,906,869** | **96.93%** |
| ROUND_TRIP_2_TO_8_ULP | 250,448 | 3.07% |
| ROUND_TRIP_ABOVE_8_ULP | **5** | 0.00006% |

ULP 히스토그램(전체 43.8M): 0 ULP **27,717,804 (63.26%)**, 1 ULP **15,607,251 (35.62%)**, 2 ULP 483,048 (1.10%), 3 ULP 9,107, 4-9 ULP 합계 413, 그 위 극소.

**결론**: closure 상실은 **전적으로 world-좌표 왕복의 float32 반올림**이다. 픽셀 재배정은 **한 건도 없고**(100% 보존), half-pixel 규약 문제도 없으며, 8 ULP를 넘는 사례는 43.8M 중 **5건**이다. 1 ULP 떨어진 15.6M 이벤트 중 약 절반이 위쪽으로 반올림되어 `>` 비교에서 OCCLUDED가 되는 것이 18.62%의 실체다. **epsilon을 튜닝해 모순을 없애는 일은 하지 않았다.**

정성 사례(리포트 `qualitative_cases`):
- 닫힌 사례: view 0, pixel (212,204), rep 570083, stored median 3.49885082, 재투영 depth 3.49885058, margin **−2.38e-07**, ULP 1, OBSERVED.
- 모순 사례: view 0, pixel (0,1), rep 876288, stored median 10.33440590, 재투영 depth 10.33440685, margin **+9.54e-07**, ULP **1**, 픽셀 보존, OCCLUDED. **같은 1 ULP인데 부호만 반대다.**

---

## 5. Frontier Identity Analysis

**원리적으로 정확한 identity 표현이 존재한다.** query를 `(source camera id, source pixel id, renderer 자신이 저장한 median depth)`로 표현하고 B를 그대로 적용하면:

| | 값 |
|---|---|
| 검사한 event | **43,817,760** |
| OBSERVED | **43,817,760** |
| **모순** | **0** |

즉 "이 query가 renderer 자신의 median surface event **이다**"라는 관계는 표현 수준에서 **정확히 보존 가능**하다. 이는 진단이며 **Candidate B는 변경하지 않았다**.

따라서 지시의 질문에 대한 답은 **둘 다이고, 둘은 분리 가능하다**:
1. 관측된 source-frontier 모순은 **순전히 표현 왕복 문제**이며 원리적 정확 identity가 **존재한다**.
2. 그러나 **임의의 3D 점**으로 표현된 frontier 위의 query는 float32 반올림으로 양쪽에 갈리므로, 임의-3D median-frontier 분류는 **여전히 승인되지 않은 수치 정책을 요구한다**. 이 배치는 그 정책을 도입하지 않았다(section 15 금지).

---

## 6. Post-Median Contributor Accounting

**전수 회계.** WL110의 16-slot 배열(97.4% truncate)은 편향을 만들므로 **사용하지 않았고**, worklog 122의 additive CUDA 집계를 썼다.

| 항목 | 값 |
|---|---|
| median event가 있는 픽셀 | 43,817,760 |
| post-median 기여자가 1개 이상인 픽셀 | 43,724,631 (**99.79%**) |
| post-median accepted 기여자 총수 | **1,150,990,609** |
| median 픽셀당 평균 post-median 기여자 | **26.27** |
| 전체 accepted 기여 질량(= 1−T_final 누적) | 43,069,940.3 |
| **post-median 기여 질량** | 16,820,934.6 |
| **post-median이 전체 기여에서 차지하는 비율** | **39.06%** |

**픽셀당 post-median 질량 비율**: 중앙값 **0.4257**, p05 0.1601, p25 0.3436, p75 0.4697, p95 0.4938, **최댓값 0.49993**.

> **구조적 성질**: 이 비율은 **0.5를 넘을 수 없다**. median crossing이 `T > 0.5`에서 정의되므로 crossing 시점의 T는 0.5 이하이고, post-median 질량 = T_crossing − T_final < 0.5다. 즉 **frontier는 정의상 광선 총 기여 질량의 절반 이상을 항상 자기 앞에 둔다.** 이것은 튜닝된 성질이 아니라 canonical median 규칙 자체의 성질이며, frontier 일관성의 강한 근거다.

**depth offset(기여자 depth − median depth)**: 평균 +0.267. min −160,317.97 / max +1,256,555.5 -- 극단값은 WL121이 이미 규명한 **무한 평면 교차의 grazing-angle 산물**이며 대표성이 없다.

**핵심 교정(카테고리 8/9)**: post-median 기여자 중 **27.65%의 질량(248,820,747개)은 per-pixel depth가 median보다 오히려 앞**이다. traversal은 surfel **중심** depth로 정렬되므로 "traversal-order post-median"이 "physically behind"를 뜻하지 않는다.
→ **진짜로 frontier 뒤에 있는 질량 = 39.06% x 72.35% = 전체 기여의 28.26%.**

---

## 7. Post-Median Semantic Attribution

post-median 질량(=100%) 기준 분해:

| 범주 | 기여자 수 | 개수 비중 | **질량 비중** | 전체 scene 기여 대비 |
|---|---|---|---|---|
| **A. 같은 frozen visible component (중복 표현)** | 696,146,700 | 60.48% | **79.56%** | 31.07% |
| **B. 다른 visible component** | 454,843,909 | 39.52% | **20.44%** | 7.98% |
| **C. provenance 미해결** | **0** | 0% | **0%** | 0% |

representative provenance 분해(같은 post-median 질량 기준):

| | 질량 비중 |
|---|---|
| **이 뷰의 다른 픽셀에서 median representative** | **65.40%** |
| 다른 뷰에서만 median representative | 28.88% |
| **161뷰 어디에서도 median representative가 아님** | **5.72%** |

rho2d low-pass branch: post-median 질량의 24.84%.

**해석 -- 그리고 과대 해석 방지**:
- 범주 B(cross-component)를 **"visible secondary surface"라고 자동으로 부르지 않는다.** WL107/109 topology는 1,190,469 surfel을 **559,989 component**로 쪼개며 그 중 **535,910개가 singleton**이다. 즉 "다른 component"는 "다른 물리적 표면"의 **매우 약한 대리 변수**이며, 하나의 표면 안에서도 대부분 cross-component가 된다. 20.44%라는 수치는 그래서 독립 표면 증거로 읽을 수 없다.
- 훨씬 강한 신호는 representative provenance다: post-median 질량의 **94.28%**가 renderer 자신이 **어딘가에서 visible surface로 지정한** surfel에서 나온다(이 뷰 65.40% + 다른 뷰 28.88%). renderer가 **어디에서도** visible surface로 지정하지 않는 surfel에서 나오는 질량은 **5.72%**뿐이다.
- 이를 depth-side와 결합하면 상한이 나온다: **frontier 뒤에 있으면서 renderer가 어디서도 visible로 지정하지 않은 기여의 질량은 전체 scene 기여의 최대 2.24%** (= 39.06% x 5.72%). 이는 **주변부(marginal) 분포에서 얻은 상한**이며 결합분포로 측정하지 않았다(section 13의 INABILITY 참조).

---

## 8. Cross-View Disocclusion

3,400개 anchor(매 10번째 뷰 x 결정론적 200개). 각 anchor는 **자기 source view에서 renderer의 median surface event 그 자체**다. view-count threshold는 어디에도 없다.

| 항목 | 값 |
|---|---|
| anchor | 3,400 |
| source view B 상태 | OBSERVED **2,735** / OCCLUDED **665**(19.56%, section 3의 18.62%와 일치하는 왕복 반올림) |
| **1개 이상 뷰에서 가려짐** | **3,390 (99.71%)** |
| anchor당 OCCLUDED 뷰 (중앙값 / 평균 / 최대) | 52 / 58.4 / 160 |
| anchor당 OBSERVED 뷰 (중앙값 / 평균) | 24 / 30.2 |
| **가려진 anchor의 global OBSERVED 유지율** | **99.44% (3,371 / 3,390)** |
| **global OCCLUDED로 끝난 anchor** | **19 (0.56%)** |

**의도한 구분(view-local occlusion ≠ global occluded domain)이 실측으로 성립한다**: median event의 99.7%가 다른 뷰에서 가려지지만 99.44%가 global OBSERVED를 유지한다.

**잔여 의미론적 모순**: 19개 anchor(0.56%)는 renderer의 median surface event인데도 **global OCCLUDED**로 끝난다. 이는 source view의 왕복 반올림 모순이 발생했고 다른 어떤 뷰도 구제하지 못한 경우다. 영역별: patio 13, hedge 3, table_side_curved 2, table_legs 1, **table_top 0**.

정성 사례(예): anchor 443, source view 20(DSC07979), rep 656752, component 0, patio, relevant 9뷰 중 OBSERVED 3 / OCCLUDED 6 → global OBSERVED.

---

## 9. Synthetic Known-Geometry Contracts

| 계약 | 결과 |
|---|---|
| **S1** 단일 노출 표면 | **PASS** -- 카메라측 자유공간 2개 OBSERVED, frontier event OBSERVED(margin 정확히 0.0), 뒤쪽 2개 OCCLUDED |
| **S2** 완전히 가려진 후면 (2개 카메라 전부) | **PASS** -- 후면 3개 query 모두 global OCCLUDED |
| **S3** cross-view disocclusion | **PASS** -- per-view [OCCLUDED, OBSERVED] → global **OBSERVED** |
| **S4** 하나의 불투명 표면 = 12개 겹친 soft splat | frontier가 물리 표면 구간 [4.000, 4.044] **안**(median 4.008) ✓, post-median 9개 기여자 질량 **40.3%**, **same-component 100.0% / cross-component 0.0%**, depth 기준 9/9가 median 뒤 |
| **S5** 진짜로 다른 두 depth layer (간격 1.5) | first contributor 4.00, **median 4.00 = 근거리 visible layer 위** ✓, post-median 3개(질량 10.04%, cross-component share 9.96%), probe: 앞 OBSERVED / **layer 사이 자유공간 OCCLUDED** / 뒤 OCCLUDED |
| OUT-OF-SCOPE translucent 대조군 | 시트-표면 사이 OBSERVED, 불투명 표면 뒤 OCCLUDED -- **보고만 하고 어떤 결정도 유도하지 않았다** |

**S4가 section 7의 해석을 통제 조건에서 직접 뒷받침한다**: 하나의 물리적 표면을 여러 splat으로 표현하면 post-median 기여는 40.3%로 상당하지만 **100% 같은 component의 중복 표현**이며 독립 표면 증거가 아니다.

**S5의 해석 주의**: layer 사이 자유공간이 OCCLUDED로 나오는 것은 surface-observation frontier로서는 **옳다**(그 카메라에서 그 공간은 visible surface 뒤다). 이것은 물리적 공허함에 대한 주장이 **아니다**. median = physical first hit을 요구하지 않았고, 실제로 S5에서 median은 first contributor와 같은 layer에 놓였다.

---

## 10. Worklog 121 True-Fragmentation Replay

WL121의 정확한 300개 cross-component raster-adjacency 컨텍스트를 저장된 아티팩트에서 재생했다.

| 항목 | 결과 |
|---|---|
| 컨텍스트 | 300 |
| gating 귀속 | **3D-locality filter 거부 288 / 이차 기하 게이트 거부 12 / positive-edge-yet-split 0** (WL121과 동일) |
| **B global 상태가 WL121과 비트 단위 동일** | **True** |
| endpoint A | **290 OBSERVED / 10 OCCLUDED** (WL121 동일) |
| endpoint B | **296 OBSERVED / 4 OCCLUDED** (WL121 동일) |
| midpoint | **300 OBSERVED / 0 OCCLUDED** (WL121 동일) |
| 검증된 out-of-frustum control | 8 UNRESOLVED (동일) |

**해석 가드(지시 준수)**: `B(midpoint) = OBSERVED`는 **표면이 midpoint를 통과해 이어진다는 증명이 아니다** -- observed space는 자유 공간일 수 있다. 이를 visible-component 병합 기준으로 **사용하지 않았고**, topology는 변경하지 않았다. 유효한 독해는 오직 하나다: **midpoint 300개 중 global OCCLUDED는 0개**이므로, **B는 이 component 분리들이 global occlusion으로 설명된다는 증거를 제공하지 않는다.** (분리의 원인은 288/300이 3D-locality filter다.)

---

## 11. Region-Level Results

| 영역 | median event | closure 모순 | **모순율** | post-median 질량 비중 | **same-component 비중** | cross-component 비중 |
|---|---|---|---|---|---|---|
| table_top | 8,036,207 | 1,247,612 | **15.52%** | 39.91% | **87.15%** | 12.85% |
| table_side_curved | 6,810,842 | 1,259,691 | 18.50% | 39.47% | 76.99% | 23.01% |
| table_legs | 6,311,717 | 1,159,385 | 18.37% | 38.23% | 86.24% | 13.76% |
| patio | 18,734,754 | 3,690,311 | 19.70% | 39.00% | 78.22% | 21.78% |
| hedge/background | 3,924,240 | 800,323 | **20.39%** | 38.13% | **63.11%** | 36.89% |

cross-view disocclusion(anchor / global OCCLUDED): table_top 599/**0**, table_side_curved 531/2, table_legs 472/1, patio 1,465/13, hedge 333/3.
true-fragmentation(질의별 global B): table_top 184 OBS/0 OCC, curved 180/2, legs 170/2, patio 176/5, hedge 176/5.

**판독**: closure 모순율은 영역 간 15.5~20.4%로 **좁은 범위**이며 -- 왕복 반올림이 원인이므로 depth 크기에 완만하게 의존할 뿐 특정 영역의 병리가 아니다. post-median 질량 비중은 38~40%로 사실상 균일하다. 유일하게 뚜렷한 층화는 **same-component 비중**으로, 얇고 정연한 table_top(87.2%)에서 높고 얽힌 hedge(63.1%)에서 낮다 -- 다만 이는 topology 파편화 정도의 차이일 수도 있어 단독으로 표면 해석에 쓰지 않는다.

---

## 12. Qualitative Review

`output/122_osn_gs_median_frontier_validation/` 아래 3개 view(장면 전체를 near-black으로 깔고 probe를 색으로 표시), PNG는 공유 `preview_png/`에 모았다.

- `ORIGINAL_2DGS_SCENE` -- 표준 export.
- `FRAGMENTATION_B_GLOBAL_STATE` -- WL121의 908개 보충 query에 대한 B global 상태(녹=OBSERVED / 적=OCCLUDED / 회=UNRESOLVED).
- `DISOCCLUSION_ANCHORS` -- 3,400개 median-event anchor. **주황 = 1개 이상 다른 뷰에서 B=OCCLUDED**(= 실제로 disocclusion을 겪는 anchor), 나머지는 global 상태 색.

리포트 JSON에는 지시가 요구한 필드를 모두 갖춘 사례 기록이 있다:
- `qualitative_cases.frontier_closure_exact` (20건) / `.frontier_closure_contradiction` (60건): view id, source pixel, 재투영 pixel, pixel 보존 여부, representative id, world position, stored median depth, 재투영 query depth, signed margin, **ULP 거리**, rho branch, B 상태, closure 원인.
- `cross_view_disocclusion_cases` (24건): anchor index, source view id/name, representative id, **component id**, region, world position, source-view B 상태, relevant/OBSERVED/OCCLUDED 뷰 수, global 상태.
- `true_fragmentation_cases` (20건): context index, source view, 양쪽 representative id, **양쪽 component id**, gating 사유, endpoint A/B/midpoint의 B global, midpoint 평균 signed margin.

**외형만으로는 어떤 주장도 하지 않았다** -- 모든 시각 사례는 query id와 정량 레코드로 되짚을 수 있다.

---

## 13. Implementation Fidelity Statement

### Diagnostic-to-Code Ownership Map

| path | function | ownership | 계산하는 양 | state 변경 가능? | 테스트 |
|---|---|---|---|---|---|
| `observed_occluded/candidate_b_median_depth.py` | `classify_view` | **B (FROZEN)** | B의 결정 그 자체 | -- (무수정) | `TestHistoricalBInvariance`(4) |
| `observed_occluded/frontier_validation.py` | `float32_ulp_distance` | SHARED-VALUE | 정확한 float32 ULP 거리 | **NO** | `TestUlpAttribution`(3) |
| " | `evaluate_frontier_closure_for_view` | B-VALUE | 전수 self-closure + 원인 귀속 + identity 표현 | **NO**(frozen B 호출) | `TestFrontierSelfClosure`(3) |
| " | `ClosureAccumulator` | B-VALUE | 스트리밍 집계 | **NO** | 위와 동일 |
| " | `PostMedianAccumulator` | B-VALUE | post-median 전수 집계 | **NO** | `TestPostMedianAccounting`(8) |
| " | `region_table` | SHARED-VALUE | 영역별 표 | **NO** | -- |
| `observed_occluded/frontier_synthetic_contracts.py` | `build_s1..s5`, `build_translucent_control` | B-VALUE | S1-S5 + OUT-OF-SCOPE 대조군 | **NO**(frozen B 호출) | `TestSyntheticFrontierContracts`(7) |
| `observed_occluded_median_frontier_validation.py` | `main` | SHARED | 오케스트레이션·집계·export | **NO** | 실 scene 재생 |
| `diff_surfel_rasterization_qdepth/.../forward.cu` | worklog 122 post-median 블록 | B-VALUE | 10개 카테고리 count/mass, total mass, depth stats | **NO**(순수 additive) | `TestCanonicalEquivalenceUnderWorklog122Additions`(3) |

### 의도 → 해석 → 구현 → 측정 → 아키텍처 해석

| OUR INTENT | INTERPRETATION | IMPLEMENTATION | MEASURED | ARCHITECTURE READING |
|---|---|---|---|---|
| frontier가 자기 frontier의 관측측에 있는가 | 전 뷰 전 median event 전수 왕복 | `evaluate_frontier_closure_for_view` | 18.62% 모순, **픽셀 재배정 0**, **>8 ULP 5건** | closure는 의미론이 아니라 **표현 왕복**에서 깨진다 |
| tolerance 없이 원인 규명 | 정확한 정수/ULP 분류 | 위와 동일 | 96.93%가 1 ULP | epsilon 없이 원인 확정 |
| 정확한 identity가 존재하는가 | (camera, pixel, stored median) 표현 | 위와 동일 | **0 / 43,817,760 모순** | 원리적 정확 계약 **존재** |
| frontier 뒤 증거의 정체 | truncation 없는 전수 집계 + 3분류 | worklog 122 CUDA + `PostMedianAccumulator` | 39.06% 질량, 그 중 **72.35%만 실제로 뒤**, **94.28%가 renderer 지정 visible** | 대부분 **중복 표현**, 독립 표면 증거 상한 2.24% |
| view-local ≠ global | frozen aggregation, threshold 없음 | 3,400 anchor x 161 뷰 | 99.71% 가려짐, **99.44% global OBSERVED 유지** | 의도한 구분 성립, 잔여 0.56% |

### PROMPT-REQUIRED vs AGENT-INTRODUCED
section 1에 전부 열거했다. **새 threshold·tolerance·epsilon은 하나도 도입하지 않았다.**

### 실행 중 발견해 수정한 결함 1건
`frontier_synthetic_contracts.py`가 (H, W, **10**) post-median 집계를 `reshape(-1, 8)`로 읽어 카테고리가 어긋나 있었다(S4가 post-median 9개가 아니라 4개로 보고됨). 실 scene 경로는 `len(POST_MEDIAN_CATEGORIES)`를 쓰므로 **영향받지 않았고**, 잘못된 값이 보고서에 들어가기 전에 수정했다. 회귀 방지 테스트 `test_s4_category_widths_match_the_cuda_layout`를 추가했다(카테고리 폭 + 손계산 9개 post-median 확인).

### INABILITY TO REALIZE REQUESTED DIAGNOSTIC
**1건.** section 7의 세 범주(A/B/C)와 representative provenance, depth-side를 각각 **주변 분포로만** 측정했고 **결합 분포**(예: "median보다 실제로 뒤 **이면서** cross-component **이면서** 어디서도 representative가 아닌" 기여 질량)는 측정하지 **않았다**. 커널이 카테고리별 독립 누적기를 쓰기 때문이다. 따라서 "frontier 뒤의 독립적 visible-surface 증거"에 대해서는 **상한 2.24%(전체 기여 대비)** 만 말할 수 있고 정확한 값은 말할 수 없다. 근사값으로 대체하지 않았다.

---

## 14. Architecture Verdict

> **B. MEDIAN FRONTIER NUMERICALLY COHERENT BUT SEMANTIC VALIDITY REMAINS INCONCLUSIVE**

**확립된 것(강한 증거)**:
1. **closure 실패는 의미론적 실패가 아니다.** 43,817,760건 전수에서 픽셀 재배정 0건, 8 ULP 초과 5건, 96.93%가 정확히 1 ULP. rho branch와 무관.
2. **원리적으로 정확한 identity 계약이 존재한다** -- `(source camera, source pixel, stored median depth)` 표현에서 모순 **0 / 43,817,760**.
3. **frontier는 정의상 광선 총 기여의 절반 이상을 자기 앞에 둔다**(픽셀당 post-median 질량 비율 최댓값 0.49993). 이는 튜닝이 아니라 canonical median 규칙의 구조적 성질이다.
4. **frontier 뒤 증거는 압도적으로 중복 표현이다**: post-median 질량의 27.65%는 사실 frontier **앞**(traversal-order 산물)이고, 나머지 중 **94.28%가 renderer 자신이 어딘가에서 visible surface로 지정한** surfel에서 온다.
5. **알려진 기하에서 의미론이 성립한다**: S1/S2/S3 PASS, S4는 frontier를 물리 표면 안에 두고 post-median을 100% 같은 표면의 중복으로 유지, S5는 frontier를 근거리 visible layer에 둔다.
6. **view-local occlusion ≠ global occluded domain이 실측으로 성립한다**: 99.71%가 가려지되 99.44%가 global OBSERVED 유지.

**A(VIABLE)로 승격하지 못하게 막는 것**:
1. **OSN-GS의 Observed/Occluded 분해는 정의상 부피(임의 3D 점) 위에서 동작한다.** 그런데 frontier 자신의 정의 이벤트를 임의 3D 점으로 표현하면 **18.62%가 자기 source view에서 OCCLUDED로 뒤집힌다.** 이를 해결하려면 승인되지 않은 수치 경계 정책(epsilon, ULP 조정, 또는 exact-identity 표현으로의 전환)이 필요하고, section 15가 그것을 금지한다. **정확 identity가 존재한다는 사실은 정책이 승인되었다는 뜻이 아니다.**
2. **잔여 의미론적 모순이 0이 아니다**: renderer median event의 **0.56%(19/3,400)가 global OCCLUDED**로 끝난다.
3. **결합 분포 미측정**(section 13 INABILITY): "frontier 뒤의 독립적 visible-surface 증거"는 **상한 2.24%**만 확정되었고 정확한 값은 미확정이다.
4. cross-component 20.44%는 **독립 표면 증거로 읽을 수 없다**(topology가 1.19M surfel을 559,989 component, 그 중 535,910 singleton으로 쪼갠다) -- 즉 이 축으로는 A를 지지하지도 반박하지도 못한다.

**C(frontier 뒤 물질적 직접-표면 증거로 무효화)가 아닌 이유**: 뒤쪽 질량 28.26%는 물질적이지만, 그 94.28%가 renderer 자신이 visible로 지정한 surfel이고 S4가 통제 조건에서 같은 결론을 재현한다. 지시 자체가 "frontier 뒤의 0이 아닌 광도 민감도는 자동으로 무효화하지 않는다"고 명시한다.

**D(source-frontier closure 실패로 무효화)가 아닌 이유**: 모순의 100%가 ≤8 ULP 왕복이고 픽셀 재배정이 0건이며 정확 identity가 존재한다. frontier 자체는 닫혀 있고, 닫히지 않는 것은 **임의-3D float32 표현**이다.

**이 판정이 뜻하지 않는 것**: median depth = physical first hit. 그런 주장은 이 배치 어디에도 없다.

---

## 15. Remaining Architecture Question

**"OSN-GS의 Observed/Occluded 분해는 임의의 3D 점 위에서 동작해야 하는데, renderer-defined median frontier는 그 frontier 자신의 정의 이벤트조차 임의-3D float32 표현으로는 18.62%가 뒤집힌다. 그렇다면 OSN-GS는 (a) frontier 질의를 `(camera, pixel, stored median depth)` 같은 정확 identity 표현 위에서 수행하도록 아키텍처를 바꿔야 하는가, 아니면 (b) 임의-3D 분류를 위한 명시적 수치 경계 정책을 승인해야 하는가? 그리고 그 선택은 frontier 뒤에 남는 28.26%의 기여 질량(그 중 상한 2.24%가 renderer가 어디서도 visible로 지정하지 않은 증거)을 어떻게 취급하는지와 어떻게 연결되는가?"**

이 배치가 이 질문으로 좁힌 근거: 정확 identity가 존재한다는 사실(0/43.8M)과, 그럼에도 임의-3D 경로가 18.62%에서 실패한다는 사실이 **동시에** 참이며 서로 분리 가능하다는 것. 결합 분포를 측정하면 (b)의 필요 여부가 더 좁혀진다.

**이 배치는 여기서 멈춘다.** 새 threshold, hybrid classifier, topology 수리, Occluded Surface 구축, NURBS continuation 중 어느 것으로도 진행하지 않았다.

---

## 16. Exact Branch / Commit / Commands / Outputs

**브랜치**: `arch/2dgs-coverage-first-surface`
**보존된 역사적 기준 커밋**: `fdfb8ad60b6233ea8364a09ea3467c18e600a246` (Worklog 120).
**이 배치의 커밋 SHA**: `fafade1` -- Worklog 121과 함께 하나의 커밋에 담겼다(두 배치가 누적된 뒤 커밋되었기 때문이며, 이후 배치부터는 배치당 1커밋으로 진행한다). 이 배치는 WL121의 `value_space_supplemental_bank.npz`를 참조 전용으로 읽는다.

**추가한 파일**:
```
scripts/devtools/observed_occluded/frontier_validation.py
scripts/devtools/observed_occluded/frontier_synthetic_contracts.py
scripts/devtools/observed_occluded_median_frontier_validation.py
tests/test_observed_occluded_median_frontier_validation.py
docs/worklogs/122_renderer_defined_median_surface_frontier_validation.md
```

**수정한 파일(2개, 둘 다 순수 additive)**:
```
osn_gs/render/vendor/diff_surfel_rasterization_qdepth/   (config.h, forward.h, forward.cu, rasterizer.h, rasterizer_impl.cu, rasterize_points.h/.cu)
osn_gs/render/torch_surfel_query_depth_diagnostics.py
```
**Candidate B의 결정 함수, `shared.aggregate_global`, canonical renderer, WL107/109 topology, NURBS는 수정하지 않았다.**

**진단 CUDA 재빌드**(출력 arity가 바뀌면 stale `ext.o` 링크 오류가 나므로 build 디렉터리를 반드시 비운다):
```
rmdir /s /q %TEMP%\osn_gs_diff_surfel_rasterization_qdepth
scripts\build_surfel_extension_qdepth.bat 12.0
```

**테스트**:
```
scripts\run_with_msvc_env.bat .venv\Scripts\python.exe -m pytest tests\test_observed_occluded_median_frontier_validation.py -q
  -> 34 passed
scripts\run_with_msvc_env.bat .venv\Scripts\python.exe -m pytest tests\test_observed_occluded_median_frontier_validation.py tests\test_observed_occluded_value_space_comparison.py tests\test_observed_occluded_volumetric_audit.py -q
  -> 131 passed
```
production 동작이 바뀌지 않았으므로 full regression은 요구되지 않는다(section 16). 새 sibling 출력에 대한 canonical 정확-동등성 테스트는 필수 항목이며 `TestCanonicalEquivalenceUnderWorklog122Additions`가 12개 canonical 필드 + 9개 WL121 필드를 probe on/off 및 provenance 유무 조합에서 검사한다.

**실 scene 재생**:
```
scripts\run_with_msvc_env.bat .venv\Scripts\python.exe scripts\devtools\observed_occluded_median_frontier_validation.py ^
  --checkpoint output\arch_2dgs_coverage_first_surface\2dgs_run1\30000\checkpoint.pt ^
  --out output\122_osn_gs_median_frontier_validation ^
  --device cuda --source-path DATASET --images images_8
```
전체 실행 시간 **148.7초**(2회 스윕 161뷰, WL107/109 KNN 그래프, 43,817,760건 전수 closure, 전수 post-median 회계, 3,400 anchor x 161뷰, WL121 908 query x 161뷰, S1-S5).

**출력 경로**:
```
output/122_osn_gs_median_frontier_validation/median_frontier_validation_report.json
output/122_osn_gs_median_frontier_validation/median_frontier_validation.npz
output/122_osn_gs_median_frontier_validation/<VIEW_NAME>/iteration_0000001/point_cloud.ply
output/122_osn_gs_median_frontier_validation/<VIEW_NAME>/render.ppm
output/122_osn_gs_median_frontier_validation/preview_png/<VIEW_NAME>.png
output/confirmed/_run_logs/122_median_frontier_validation_run.log
```
WL120·WL121의 export는 규약대로 `output/confirmed/`에 있으며, 이 배치가 WL121의 `value_space_supplemental_bank.npz`를 참조 아티팩트로 읽는다.
