# Worklog 104 — Node-Level Observability Accounting과 Primitive/Visible-Topology 분리

## 상태

**완료 — 실측 있음. Branch A(대표적 결정)로 귀결, 그러나 renderer-native 신호와의 비교에서 나온 진짜 한계를 함께 정직하게 보고한다.** Worklog 103을 전혀 수정하지 않고 정확히 그대로 재실행(baseline replay)한 뒤, 새로운 **node-level**(surfel 자기 자신 단위) 관측성 회계를 추가했다. 결론: WL103의 singleton surfel 중 94.5%는 자기 CENTER가 161개 학습 카메라 어디에서도 `on_observed_surface`가 된 적이 없다 — 즉 이 batch는 **"pairwise 3D edge 판정이 너무 엄격해서"가 아니라 "surfel 자체가 애초에 positive observed-visible evidence가 아니어서" 대부분의 percolation-반대 실패(과소-연결)가 생긴다는 것을 확인했다(Branch A)**. 그러나 renderer-native `radii>0`(projection/culling 신호, occlusion-aware 아님)와 비교하면 이 "center-negative" surfel의 99.98%가 여전히 평균 48개 뷰에서 `radii>0`이었다 — 이는 이들이 완전히 존재하지 않는 것이 아니라 화면에 실제로 투영/기여는 하고 있다는 뜻이며, Phase-C의 point-sample 질의가 다수의 겹치는 2DGS surfel이 하나의 blended depth를 구성하는 상황에서 "이 surfel 하나가 실제로 최종 표면을 정의하는가"를 판별하기에 근본적으로 약한 proxy일 수 있다는 정직한 한계도 함께 기록한다. 이 batch는 그 한계를 인정하면서도, directive의 명시적 지시(§3 마지막 문장: "frustum inclusion을 실제 가시 기여로 부르지 말라")에 따라 radii>0만으로 Branch B로 뒤집지 않았다.

## 아키텍처

```
Worklog 103 그래프/파티션 (torch_positive_visible_adjacency.py, 수정 없음, 그대로 재실행)
    -> 신규: 매 surfel마다 canonical Phase-C 규칙(_per_view_status_codes, 재사용)으로
       161개 학습 뷰 전체에 대해 "내 CENTER가 on_observed_surface였던 뷰 수"를 집계
       (torch_node_level_observability_accounting.py)
    -> A/B/C/D 4-way node 분류 + WL103 singleton 전용 6-way cause 분류
    -> 신규: primitive ownership(전량 보존) vs visible topology membership(구조적 component만)
       분리 회계 (torch_primitive_ownership_visible_topology_separation.py)
```

새 모듈 세 개가 추가됐고, `torch_positive_visible_adjacency.py`/`torch_observation_evidence.py`/`torch_maximal_visible_connectivity.py`는 **한 줄도 수정하지 않았다** — 전부 이미 public인 함수(`build_candidate_graph`, `compute_positive_visible_adjacency_evidence`, `_per_view_status_codes`, `_project_to_camera`, `_connected_component_roots`)를 import해서 재사용했다.

## 1. Worklog 103 재현 검증

Export 스크립트가 자체적으로 `build_candidate_graph` + `compute_positive_visible_adjacency_evidence`를 직접 호출해 WL103의 connected-component 로직을 그대로 복제(동일 함수 재사용, 재구현 아님)한 뒤, 그 결과를 커밋된 WL103 리포트와 대조했다: **768,829 components, largest=0.1050 — 완전히 일치.** 관측 evidence(co_observed/positive/free/occluded per-camera 수치)도 카메라별로 WL103과 정확히 동일했다.

## 2. Node-level 관측성 — 전체 surfel

Canonical Phase-C 규칙(`_per_view_status_codes`, 신규 재구현 없음)을 각 surfel의 **자기 CENTER**에 대해 161개 뷰 전부와 대조해 집계했다(candidate graph/edge와 무관, 순수 per-node).

| 카테고리 | 개수 | 비율 |
|---|---|---|
| A. NEVER_POSITIVELY_OBSERVED_AT_NODE_LEVEL | 713,540 | 59.9% |
| B. OBSERVED_AT_NODE_LEVEL_NO_POSITIVE_EDGE | 40,786 | 3.4% |
| C. OBSERVED_AT_NODE_LEVEL_WITH_POSITIVE_EDGE | 435,481 | 36.6% |
| D. OBSERVED_AT_NODE_LEVEL_CONFLICT_ONLY | 662 | 0.06% |

## 3. Node-level 관측성 — WL103 singleton surfel(754,988개)만

| 카테고리 | 개수 | 비율 |
|---|---|---|
| A. NEVER_POSITIVELY_OBSERVED_AT_NODE_LEVEL | 713,540 | **94.5%** |
| B. OBSERVED_AT_NODE_LEVEL_NO_POSITIVE_EDGE | 40,786 | 5.4% |
| C. OBSERVED_AT_NODE_LEVEL_WITH_POSITIVE_EDGE | 0 | 0% (정의상 당연 — positive edge가 있으면 singleton일 수 없음) |
| D. OBSERVED_AT_NODE_LEVEL_CONFLICT_ONLY | 662 | 0.09% |

## 4. Phase-C center 시각성이 실제 surfel 시각성과 일치하는가

**일치하지 않는다 — 그러나 정확히 어느 방향으로 불일치하는지가 중요하다.**

- `center_positive_renderer_negative_count = 0`: center가 on_observed_surface인 surfel은 예외 없이 renderer의 `radii>0` 신호도 갖는다(당연한 방향 — depth test를 통과하려면 그 픽셀에 유효한 렌더 결과가 있어야 함).
- `center_negative_renderer_positive_count = 713,434` (전체의 59.9%, "한 번도 center-visible 아님" surfel의 99.98%): center는 한 번도 표면으로 인정받지 못했지만, `radii>0`(투영/컬링 신호)는 평균 48개 뷰(중앙값 33개 뷰)에서 발생했다.

## 5. 실제 사용 가능한 renderer-native visibility 증거 — 정확히 무엇이고 무엇이 아닌가

`osn_gs/render/surfel_rasterizer.py`(2DGS 공식 CUDA 커널을 벤더링한 그대로, 재구현 안 함)는 자기 docstring에서 이미 명시하고 있다: **논문 eq. 12-14의 per-pixel-per-surfel alpha-compositing weight(`omega_i = alpha_i * T_i`)는 벤더된 CUDA 커널 내부에만 존재하고 Python으로 절대 반환되지 않는다.** 이걸 노출하려면 벤더 커널 자체를 수정해야 하는데, 그러면 OFFICIAL_CODE_FAITHFUL 주장을 잃는다 — 이번 batch는 커널을 건드리지 않았다.

렌더러가 실제로 노출하는 것은 `radii`(per-surfel screen-space projection radius, `radii > 0` = `visibility_mask`/`visibility_filter`) 하나뿐이다. 이것은 **투영/컬링 신호**이지 **occlusion-aware 기여 신호가 아니다** — 다른 50개 surfel 뒤에 완전히 가려진 surfel도 `radii > 0`일 수 있다. 즉 지시 §3의 문구를 그대로 따르면: **"frustum inclusion을 실제 가시 기여의 증거로 부르면 안 된다"** — 이번 batch는 그 원칙을 지켜, `radii>0`를 "이 surfel이 최종 이미지에 실제로 기여했다"는 증거로 사용하지 않았다. 대신 정확히 무엇을 증명하고 무엇을 증명하지 않는지만 보고한다: 이 surfel들은 존재하지 않는 것도, 완전히 프러스텀 밖인 것도 아니지만, "최종 합성 이미지의 지배적 표면을 정의했다"는 더 강한 주장을 뒷받침할 렌더러 신호는 현재 API에 없다.

## 6. Branch 판단

Directive의 정확한 문구("majority of singleton surfels are individually observed-visible but fail because exact 3D candidate edges are not positively certified" — Branch B)와 대조하면: singleton의 94.5%가 **개별적으로도(node level) 한 번도 positively observed-visible이 아니었다.** 이는 "pairwise edge 판정이 지나치게 엄격해서"가 아니라 "surfel 자체가 (Phase-C가 가진 최선의 근거로 판단할 때) 애초에 positive observed-visible evidence가 아니어서"라는 뜻이다. **Branch A 채택.**

정직한 단서: renderer-native `radii>0`와의 비교(§4-5)가 보여주듯, 이 판단은 "surfel이 완전한 무(無)"라는 뜻이 아니라 "Phase-C의 point-sample 질의가 이 정도로 인정하지 않는다"는 뜻이다. 다수의 겹치는 2DGS surfel이 하나의 blended depth를 구성하는 상황에서, 개별 surfel 하나가 정확히 그 blended 값과 depth_epsilon 이내로 일치할 필요는 없다 — 이는 point-sample 질의 자체의 구조적 한계(surfel 표현 자체가 redundant/겹침을 전제로 학습되므로)일 수 있다. 그러나 이를 증명하거나 반증할 더 강한 renderer 신호가 API에 없으므로(§5), directive의 명시적 지시에 따라 이 불확실성을 이유로 Branch B로 전환하지 않았다. **Branch A를 채택하되, 이 한계를 다음 batch를 위해 명시적으로 열어둔다.**

## 7. Primitive Ownership vs Visible Topology Membership — 표현적 분리

새 모듈 `torch_primitive_ownership_visible_topology_separation.py`. 새 adjacency 없음, 새 threshold 없음, WL103 결과를 읽기 전용으로만 재분류한다:

- **Primitive Ownership**: 1,190,469개 전량 보존(`retained_surfels == total_surfels`, 항상). **아무것도 버리지 않는다.**
- **Visible Topology Membership**: component 크기 >= 2인 경우만 "구조적 Visible Surface Component 멤버"로 인정 — **435,481개(36.6%)**. 나머지 754,988개(63.4%, singleton)는 여전히 소유(retained)되지만 더 이상 "Visible Surface Component"라고 부르지 않는다 — 버려지지도, Trust나 미래의 latent-fitting 점수를 받지도 않는다.

## 8. 실제 scene 리뷰

동일 체크포인트, 동일 161개 학습 카메라.

- **테이블**: `WL103_PAIRWISE_POSITIVE_COMPONENTS`(재현 확인)에서 여전히 단일 component로 패티오와 분리. `NODE_OBSERVABILITY_CATEGORY_VIEW`에서 대부분 초록(C, 관측+연결).
- **패티오**: 넓은 단일 구조적 component 유지(WL103과 동일).
- **hedge/배경**: `SINGLETON_CAUSE_VIEW`에서 대부분 짙은 빨강(A, NODE_NEVER_POSITIVELY_VISIBLE) — 즉 hedge 파편화의 압도적 다수는 "카메라가 그 특정 pairwise 관계를 못 봐서"가 아니라 "그 개별 surfel 자체가 어느 뷰에서도 표면으로 인정된 적이 없어서"다.

## 9. Visible component 통계 / singleton 감소

이 batch는 WL103의 그래프/파티션을 전혀 바꾸지 않았으므로 **component 수·singleton 비율은 WL103과 완전히 동일하다**(768,829 / 63.4%) — 애초에 directive가 이번 batch의 목적으로 요구한 것이 아니다(Branch A는 "새 adjacency를 만들지 말고 멈추라"는 지시). 대신 그 63.4%의 **의미**를 primitive ownership과 분리해 재정의했다(§7).

## 10. occluded-gap 회귀 없음

WL103 자체를 재실행만 했으므로 wall+occluder 등 기존 계약은 그대로 유지된다(신규 focused 테스트로 재확인).

## 11. Review export

- `output/osn_gs_node_level_observability/{ORIGINAL_2DGS_SCENE, WL103_PAIRWISE_POSITIVE_COMPONENTS, SINGLETON_CAUSE_VIEW, NODE_OBSERVABILITY_CATEGORY_VIEW, RENDERER_PROJECTABILITY_VIEW}/`
- PNG 미리보기: `output/osn_gs_node_level_observability/preview_png/`
- 전체 리포트: `output/osn_gs_node_level_observability/node_level_observability_report.json`

## 12. 테스트

- `tests/test_node_level_observability_accounting.py` (14 tests)
- `tests/test_primitive_ownership_visible_topology_separation.py` (4 tests)
- 전체 regression: WL103의 1198 + 신규 18 = 1216 passed, 1 skipped (실행 결과는 커밋 메시지에 기록)

## 13. 결론

**Branch A.** WL103의 과소-연결(singleton 63.4%)은 주로 "3D pairwise edge 판정이 너무 엄격해서"가 아니라 "그 surfel 자체가 Phase-C가 가진 최선의 근거로도 positive observed-visible evidence가 아니어서" 발생한다(singleton의 94.5%). 이 batch는 새로운 adjacency 메커니즘을 발명하지 않고, 대신 **primitive ownership(100%, 전량 보존)과 visible topology membership(36.6%, 구조적 component만)을 별개의 계약으로 명시적으로 분리**했다 — WL103의 singleton-owned surfel은 버려지지 않지만 더 이상 "Visible Surface Component"라고 불리지 않는다. 다만 renderer-native `radii>0` 비교가 보여준 진짜 한계(§4-6) — 이 surfel들이 완전한 무(無)는 아니며, Phase-C의 point-sample 질의가 redundant/overlapping 2DGS 표현에 얼마나 잘 맞는 proxy인지는 현재 API로는 완전히 결론지을 수 없다는 것 — 를 다음 단계를 위해 정직하게 열어둔다.

## 참고

- 새 모듈: `osn_gs/surface/torch_node_level_observability_accounting.py`, `osn_gs/surface/torch_primitive_ownership_visible_topology_separation.py`
- Export 스크립트: `scripts/devtools/node_level_observability_export.py`
- 관련: [[project_positive_visible_adjacency]] (WL103, 이번 배치가 그대로 재실행한 baseline)
