# Worklog 98 — 전역 동기화 tangent frame 기반 curve lattice

## 상태

**완료 — 범위를 좁힌 결론(정정, 아래 "결과 해석 정정" 참고).** Worklog 96의 per-seed 독립 transversal 방향 선택을 latent surface 위 전역 동기화 tangent frame field로 대체했다. Worklog 97의 `inconsistent_transversal_curve_direction` 실패는 크게 줄었다(component 개수 기준 combined 91.7%→30.30% invalid, 2900: 17.14%, final: 45.16% — 아래 "결과 해석 정정" 표 참고). 그러나 parameterization이 성공한 component만 봐도 지금 경로(동기화 field→tree-integrated UV→고정 6×6 degree-2 tensor-product LSQ NURBS)는 여전히 대부분 extrapolative/unsafe다. **이것이 증명하는 것은 "NURBS representation 자체가 병목"이 아니라, 이 특정 fitting 경로(고정 capacity + 이 UV 구성 + 이 LSQ 절차)가 부족하다는 것뿐이다** — latent-surface support는 확보됐고, curve construction은 널리 가능해졌고, 전역 동기화된 tangent 방향이 Worklog 97의 독립-방향 실패를 상당 부분 해소한다는 것까지가 이 worklog가 실제로 증명한 범위다. 어느 구성 요소(patch 표현 자체/고정 capacity/generic LSQ 절차/UV 구성) 때문에 최종 fit이 여전히 부족한지는 이 배치만으로 분리되지 않는다. Visible Gaussian training, ADC, region ownership, Worklog 95 latent-surface estimator, supported-query 규약, continuous segment-support 요구, 기존 seed provenance, NURBS degree(2)·6×6 control grid, Worklog 97 external-UV/network-native fitting, held-out·safety 기준은 모두 미변경이다.

## 구현

### 1~2. Tangent plane과 parametric 방향 분리 + 전역 동기화 frame

신규 `osn_gs/surface/torch_latent_surface_tangent_frame_field.py`: latent surface estimator의 local PCA tangent axis는 tangent plane(normal)만 제공하고, 권위 있는 in-plane 방향으로 쓰지 않는다. Region train evidence 위에 Worklog 95의 k=16(`DEFAULT_K` 재사용) kNN 후보 edge를 만들고 각 edge를 `sample_segment_continuous_support`로 검증해 미지원 edge는 완전히 제거한다. 지지된 edge 그래프의 각 topological component마다: (1) anchor를 정하고(경계/typed seed가 있으면 그 시작점·접선을, 없으면 component centroid에 가장 가까운 node를 결정론적으로), (2) anchor의 `e_u`를 Gram-Schmidt로 고정하거나 anchor hint를 투영해 얻고 `e_v = normal × e_u`, (3) **3D 거리 기준 Dijkstra 최단경로**(단순 BFS hop count 아님 — 실측으로 BFS의 path-dependent drift가 커서 정정)로 spanning tree를 만들며 parent→child 전이마다 `propagate_tangent_onto_plane`(Worklog 96 tracer의 동일 함수 재사용)으로 frame을 물려받는다 — 매 node에서 독립적으로 축을 다시 고르지 않는다. Tree edge 적분으로 각 node에 chord-length 기반 `(u,v)` potential을 부여한다. Stable ID/입력 순서는 tie-break에만 쓰인다(centroid-최근접 등 geometry 기준이 항상 우선).

### 3. Boundary/feature semantics는 선택적 anchor

Typed(non-interior) seed가 있으면 그 시작점과 접선을 anchor로 쓰고 `anchor_seed_type`을 보존한다. 없으면 interior-only component는 순수 기하 기준(centroid 최근접 + 정준 Gram-Schmidt 축)으로 gauge를 고정한다 — interior gauge는 물리적 feature로 label하지 않는다.

### 4. Non-integrable block 검출(fail-closed)

Tree가 아닌(cycle-closing) 모든 지지 edge에 대해 한쪽 node의 `e_u`를 다른 쪽으로 전이한 결과와 그 node의 실제(tree로 이미 정해진) `e_u`를 비교한다 — cosine≤0(Worklog 97과 동일한 고정 기준)이면 orientation-inconsistent(holonomy violation)로 기록한다. Component 하나라도 holonomy-inconsistent cycle edge가 있으면 `coherent=False`로 fail-closed하고 PCA로 복구하지 않는다. Frame propagation이 새 tangent plane과 거의 평행해 방향을 정의할 수 없는 node는 singularity로 기록하고 그 지점에서 확장을 멈춘다(하위 트리는 다른 경로로 재도달하지 않는 한 component에서 제외됨 — 이 자체가 자연스러운 pre-fit 분해다). Angular disagreement(성공한 cycle edge의 평균/최대 각도), sign correction 수, 미지원 edge 수를 모두 report한다.

### 5~6. 동기화된 field로부터 curve 추적 + lattice 구성

신규 `osn_gs/surface/torch_latent_surface_curve_lattice.py`: `e_u`/`e_v` integral curve를 양방향으로 추적하되(Worklog 96 tracer 재사용), 매 step 방향은 **가장 가까운 field node에 이미 동기화된 `e_u`/`e_v`를 조회**해서 얻는다 — per-step 독립 재선택이 없다. 매 step은 latent surface 지지를 확인하고 미지원 즉시 종료한다(분리된 integral curve를 gap 너머로 잇지 않음). 그러나 실제 fit 입력은 traced curve만이 아니라 **동일 field가 이미 tree 적분으로 부여한 coherent component 전체의 `(u,v)`**다 — 이는 traced curve의 상위집합이며 수치적으로 일관된다(holonomy 검증을 통과한 동일 field에서 나옴). Worklog 96의 2×2 계약은 유지되지 않는다(사용하지 않음) — 대신 전역 동기화 field·holonomy 일관성이 실제 parametric 계약이다.

### 7. Worklog 97 network-native fitting 재사용

신규 `osn_gs/surface/torch_curve_lattice_native_fit.py`가 field에서 나온 `(points, uv)`를 Worklog 97의 `fit_torch_visible_surface_from_uv`(미변경)에 그대로 넘긴다. PCA는 어디에도 없다.

### 8~9. Paired 3-way 비교 + real replay

신규 `scripts/devtools/synchronized_frame_lattice_replay.py`가 동일 region train evidence에 대해 **A. Worklog 96 curve network + PCA-UV**, **B. Worklog 96 curve network(독립 방향) + Worklog 97 native fitting**, **C. 신규 동기화 field + curve lattice + native fitting**을 fallback 없이 실행한다. NURBS capacity·held-out·안전 기준은 세 경로 모두 동일하다(A/B는 Worklog 97 replay와 동일 함수 재사용, C는 같은 `classify_fitted_surface`를 pre-fitted surface에 적용).

## 검증

신규 focused 테스트 19개: `test_latent_surface_tangent_frame_field.py` 8개(곡률 있는 표면에서 coherent transport, 인접 sample 간 부호 동기화, 평평한 표면에서 강체 회전 불변성 — 곡률면에서는 Dijkstra tie-break가 회전에 민감해 정확한 재현이 아니라 상관관계로 검증하는 대신 zero-curvature fixture로 정확성 자체를 검증, boundary-anchored frame이 hint 방향과 정렬, interior gauge의 결정론성, hand-built 4-node loop에서 orientation reversal 검출, PCA 미사용, 계약 미충족 component 제외), `test_latent_surface_curve_lattice.py` 6개(coherent field에서 intrinsic UV 유효, point/uv 개수 일치, U/V curve 생성, incoherent component 거부, PCA 미사용, fit-driven 수정 없음), `test_curve_lattice_native_fit.py` 5개(동기화 lattice에서 정상 fit, capacity 고정, invalid lattice가 PCA로 대체되지 않음, PCA fit 함수 미호출, family별 잔차 분리 보고). Worklog 79~97 관련 focused 168개 통과. 전체 회귀 **1036 passed, 1 skipped**(448.9초).

## 실측: 7-region 실측(checkpoint 2900 / final)

`baseline_compatible` checkpoint 2900(held-out evidence 1915), final(held-out evidence 3920). 산출물: `output/extent_ab/val98/synchronized_frame_lattice_replay.json`, `..._final.json`.

| checkpoint | 경로 | attempted | invalid/failed(개수, raw %) | valid_supported(evidence-weighted) | extrapolative(evidence-weighted) | unsafe(evidence-weighted) | held-out p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2900 | A PCA_UV | 7 | 0/7 (0.00%) | 3.41% | 94.36% | 2.23% | 6.65 |
| 2900 | B 독립방향 native | 7 | 7/7 (100.00%) | 0.00% | 0.00% | 0.00% | — |
| 2900 | **C 동기화 field native** | **35** | **6/35 (17.14%)** | **5.67%** | 44.05% | 29.86% | 7.95 |
| final | A PCA_UV | 5 | 0/5 (0.00%) | 13.96% | 64.67% | 21.37% | 8.08 |
| final | B 독립방향 native | 5 | 4/5 (80.00%) | 0.00% | 21.37% | 0.00% | 4.66 |
| final | **C 동기화 field native** | **31** | **14/31 (45.16%)** | **2.38%** | 21.27% | 26.27% | 7.64 |
| combined | B 독립방향 native | 12 | 11/12 (91.67%) | — | — | — | — |
| combined | **C 동기화 field native** | **66** | **20/66 (30.30%)** | — | — | — | — |

**Component 개수 기준 raw invalid rate**(다음 두 줄이 실제 raw JSON `components_attempted`/`components_fit_failed_or_invalid`에서 직접 계산한 값이다 — 최초 배포판은 이 수치를 evidence-weighted `parameterization_invalid` fraction과 혼동해 2900을 20.4%, final을 50.1%로 잘못 기재했다. 두 수치는 서로 다른 지표(raw component count 기준 vs. held-out evidence 비중 가중)이며, raw 값이 "몇 개 component 중 몇 개가 실패했는가"의 정확한 답이다):

- 2900: 6/35 = **17.14%**
- final: 14/31 = **45.16%**
- combined: 20/66 = **30.30%**

`evidence_weighted_fractions`의 `parameterization_invalid`(2900: 20.42%, final: 50.08%)는 **다른 지표**다 — 각 실패 component가 그 region의 held-out evidence 중 자기 share(=held_out_evidence_size/len(components))만큼을 "parameterization_invalid" 분류에 기여한 값이며, 위 표의 valid_supported/extrapolative/unsafe와 같은 evidence-weighted 축에 있다. 두 지표를 섞어 쓰지 않는다.

**Worklog 97의 `inconsistent_transversal_curve_direction` 실패는 크게 줄어든다**: component 개수 기준 combined invalid rate가 B의 91.67%(11/12)에서 C의 30.30%(20/66)로 떨어진다 — region-level 전역 field가 seed마다 독립적으로 방향을 고르던 문제를 실제로 해소함을 실측이 보여준다. C는 또한 훨씬 많은 patch candidate(66개 component, worklog96의 12개 seed 대비)를 만든다 — 하나의 전역 field가 여러 coherent 영역으로 자연 분해되기 때문이다.

그러나 **parameterization이 성공한 component만 조건부로 봐도(evidence-weighted 값 기준) 최종 NURBS는 여전히 대부분 extrapolative/unsafe다**: 2900에서 조건부(비-invalid) valid_supported는 5.67%/(1-20.42%)=7.1%, extrapolative 55.4%, unsafe 37.5%; final에서 조건부 valid_supported는 2.38%/(1-50.08%)=4.8%, extrapolative 42.6%, unsafe 52.6%다. C의 valid_supported는 checkpoint마다 결과가 엇갈린다(2900: A 3.41%보다 높음, final: A 13.96%보다 낮음) — 두 checkpoint 모두에서 A보다 일관되게 낫다고 할 근거는 없다. 다만 extrapolative+unsafe 합계는 C가 두 checkpoint 모두 A보다 낮다(2900: 73.9% vs 96.6%, final: 47.5% vs 86.0%).

## 결과 해석 정정

이 배치가 실제로 증명하는 것은 다음 네 가지뿐이다: (1) latent-surface support는 확보돼 있다, (2) curve construction은 널리 가능해졌다(usable seed·coherent component 수 증가), (3) 전역 동기화된 tangent 방향이 Worklog 97의 독립-방향 실패를 상당 부분(component 기준 91.67%→30.30%) 해소한다, (4) 그러나 지금 경로(동기화 field→tree-integrated UV→고정 6×6 degree-2 tensor-product LSQ)는 parameterization이 성공한 component에서도 대부분 extrapolative/unsafe다. **NURBS representation 자체가 이미 병목으로 증명됐다고 단정하지 않는다** — 고정 6×6 capacity, generic point LSQ fitting, curve-network을 point cloud로 collapse하는 방식, tree-integrated UV 자체의 parametric domain 품질 중 어느 것이 원인인지는 이 배치만으로 분리되지 않았다. 후속 배치가 이를 bounded gate로 분리해야 한다.

## 결정

**결정 보류(범위 축소).** 이 worklog는 자체적으로 최종 architecture 결정을 내리지 않는다 — 원래 초안의 "Decision B: PARAMETRIC_PATCH_MODEL_LIMIT"는 과도한 일반화였다(위 "결과 해석 정정" 참고). Curve construction과 parameterization이 Worklog 97 대비 크게 개선됐다는 것과, 현재의 특정 fitting 경로가 여전히 대부분 extrapolative/unsafe를 낸다는 것 — 이 두 가지 좁은 사실만 확정한다. 어느 구성 요소(patch representation/고정 capacity/generic LSQ/UV 구성)가 원인인지 분리하는 것이 다음 bounded 배치의 과제다.
