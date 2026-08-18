# Worklog 100 — bounded intrinsic parameterization architecture gate

## 상태

**완료 — Decision C: COMPONENT_SCALE_NOT_GLOBALLY_PARAMETERIZABLE.** Worklog 99의 pre-fit validator에 두 가지 방법론적 confound가 있었다 — (1) local Jacobian 이웃을 UV-공간 kNN에서 뽑았다(UV가 이미 왜곡됐으면 실제로는 source-space 이웃이 아닌 점들이 연결돼 가짜 fold를 만들 수 있음), (2) orientation 기준을 독립적으로 부호가 정해지는 PCA/support normal로 삼았다(chart 전체의 단일 방향 반전은 gauge-equivalent일 뿐 실제 fold가 아님에도 그렇게 구분하지 못했음). 이 배치에서 두 confound를 같은 validator 안에서 직접 수정하고(별도 진단 worklog 없음), 그 위에서 A(Worklog 98 tree-integrated UV, 무변경)/B(전역 differential 동시 적분)/C(B에서 엄격히 초기화된 local-injectivity 보정)를 fallback 없이 실측 비교했다. 결과: 수정된 validator 기준으로 tree-integrated UV(A)의 domain-invalid율은 67.4%(31/46)로, Worklog 99가 보고한 confound 있는 80.4%보다는 낮지만 여전히 다수다. B는 A보다 소폭 개선(domain-valid 15→18/46, +6.5%p)하지만 대다수 실패를 없애지 못한다. **C는 B가 이미 domain-valid였던 component와 정확히 같은 18개에서만 valid였고(46개 전부에서 완전히 일치), 나머지 28개(60.9%) 전부 `not_globally_parameterizable_at_current_scale`로 fail-closed됐다 — 즉 고정된 local-injectivity 보정 스케줄이 단 하나의 component도 추가로 구제하지 못했다.** B의 global differential residual은 domain-invalid component에서도 절대값이 작다(RMS 대부분 0.001~0.03, cycle residual도 비슷한 크기) — 즉 fold는 큰 미분 불일치(noise)의 산물이 아니라, 해당 scale에서 synchronized field 자체가 단일 평면 chart로 embed될 수 없는 진짜 위상적/곡률적 구조다. Visible Gaussian training, ADC, region ownership, Worklog 95 latent-surface estimator, continuous support contract, Worklog 98 synchronized tangent-frame field·coherent component·curve construction, held-out evaluation, 기존 6×6/degree-2 NURBS fitter(`fit_torch_visible_surface_from_uv`)와 안전성 기준은 모두 미변경이다. Held-out 결과로 어떤 candidate도 튜닝하지 않았다.

## 구현

### 1. Pre-fit parametric domain validator 수정 (Worklog 99의 confound 2건)

기존 `osn_gs/surface/torch_parametric_domain_validity.py`를 같은 배치 안에서 직접 수정한다(별도 worklog 없음). 두 변경:

- **Local Jacobian 이웃을 source-graph adjacency로 교체**: 기존에는 `torch.cdist(uv, uv)`로 매번 새 UV-공간 kNN을 계산했다. 이제는 Worklog 98의 `build_tangent_frame_field`가 이미 continuous-support로 검증한 그래프(`component.tree_edges` ∪ `component.holonomy_edges`가 가리키는 non-tree edge)를 그대로 재사용한다 — 새 이웃 탐색을 하지 않는다.
- **Orientation 기준을 synchronized frame으로 교체**: 점 i에서 각 source-graph 이웃 j에 대해 `x_ij = [dot(p_j-p_i, e_u_i), dot(p_j-p_i, e_v_i)]`(그 점 자신의 synchronized in-plane 좌표)와 `y_ij = [u_j-u_i, v_j-v_i]`(UV 변화량)로 local 2×2 Jacobian을 least-squares 적합한다. 독립적으로 부호가 정해지는 `normals` 인자는 이제 필요 없다(module에서 완전히 제거). 먼저 component 전체 determinant 부호의 다수결로 **단일 전역 flip**을 canonicalize한 뒤(whole-chart 반전은 gauge-equivalent, fold 아님), 그 이후에도 **자신의 source-graph 이웃 중 하나라도 정반대 부호**를 가지면 그 점을 local fold로 표시한다(다수결 비교는 fold 경계 양쪽이 각자 내부적으로 일관돼 있어 오히려 fold 신호를 희석시킴을 실측 확인 후 정정).
- 새 report 필드: `global_orientation_flip_applied`(단일 전역 반전이 실제로 적용됐는지), `area_distortion_p95`(=|det J| 분포, singular value 곱), `shear_distortion_p95`(singular value 비율). 기존 `duplicate_incompatible_count`/`stretch_ratio_p95`/`mean_condition_number`/`max_condition_number`/`cycle_position_drift_p95_over_spacing`는 그대로 유지한다.

### 2. Edge differential constraints

신규 `osn_gs/surface/torch_latent_surface_edge_differential.py`: Worklog 98이 이미 continuous-support로 검증한 모든 edge(tree edge + holonomy edge 전부)마다, **양 endpoint의 synchronized frame을 각각 midpoint normal 평면에 transport하고 부호를 정렬해 평균**한 대칭 edge frame `(e_u_ij, e_v_ij)`을 만든다(한쪽 endpoint만 쓰는 tree propagation과 달리 endpoint 순서에 의존하지 않음). `du_ij = dot(p_j-p_i, e_u_ij)`, `dv_ij = dot(p_j-p_i, e_v_ij)`. Edge weight는 3D edge length 대비 component 자체 median spacing의 역제곱(`w_ij = (spacing/length)^2`, 짧은 edge 폭주 방지로 하한 clamp)이라는 고정 규칙 하나뿐이다 — replay나 held-out 결과로 조정하지 않는다(테스트로 확인).

### 3~4. Candidate B/C

신규 `osn_gs/surface/torch_global_differential_uv_integration.py::integrate_global_differential_uv`: 모든 edge differential을 동시에 만족하는 `min Σ w_ij[(u_j-u_i-du_ij)^2+(v_j-v_i-dv_ij)^2]`를 하나의 가중 least-squares로 푼다(u/v 좌표별로 독립적으로 분리되는 두 개의 선형계, `torch.linalg.lstsq`). Gauge는 한 점(첫 node)을 원점에 고정하는 것 하나뿐이며 그 외 어떤 재정렬도 하지 않는다. Overall/percentile/cycle-edge residual을 모두 report한다.

신규 `osn_gs/surface/torch_orientation_preserving_uv_integration.py::integrate_orientation_preserving_uv`: **B의 해에서 엄격하게 초기화**한다(다른 parameterization family가 아님). 고정 스케줄(최대 3회 반복, fold에 관여한 edge의 weight를 8배로 고정 boost) 동안 매 회 수정된 validator로 fold node를 찾아 그 node에 인접한 edge의 weight를 올리고 같은 목적함수를 다시 푼다. Held-out/fit 결과로 반복 횟수나 boost factor를 조정하지 않는다(고정값, AST/코드로 확인 가능). 스케줄이 끝났는데도 fold가 남으면 `not_globally_parameterizable_at_current_scale`로 fail-closed한다 — PCA 복구나 fit 기반 분할 없음.

### 5~7. Paired 비교 + real replay

신규 `scripts/devtools/intrinsic_parameterization_architecture_gate_replay.py`가 Worklog 98의 동일 coherent component evidence·synchronized field·supported edge·held-out evidence에 대해 A/B/C를 fallback 없이 각각 독립 실행한다. Domain-valid로 판정된 UV만 기존 `fit_torch_visible_surface_from_uv`(degree=2, 6×6, 기존 정규화 solver, 기존 안전 기준)에 넘기고 5개 범주(PARAMETER_DOMAIN_INVALID/FIT_FAILED/EXTRAPOLATIVE/UNSAFE/VALID_SUPPORTED)를 분리 보고한다.

## 검증

신규 focused 테스트 19개: `test_parametric_domain_validity.py`(재작성) 11개 — source-graph 이웃 사용(UV-kNN 아님), 단일 전역 axis flip에 대한 fold 판정 불변성, 실제 local fold 검출, synchronized frame 사용(독립 PCA normal 부호 뒤집어도 결과 불변), degenerate extent, area/shear distortion report, cycle drift, PCA 미사용, NURBS 비의존; `test_global_differential_uv_integration.py` 8개 — 평면 constant frame field의 정확한 적분, 여러 경로/cycle이 있는 그래프에서 global least-squares 회수, 완만하게 곡률 있는 integrable field에서 낮은 residual, edge 없음에서 fail-closed, clean flat field에서 candidate C가 0회 반복으로 candidate B와 일치, candidate C가 candidate B에서 엄격히 초기화됨(base_result 비교), base integration 실패 시 fail-closed, PCA 미사용; `test_latent_surface_edge_differential.py` 4개 — tree edge보다 많은 전체 supported edge 사용, weight의 결정론성, held-out/replay 의존 없음(AST), NURBS 비의존; `test_tree_path_drift_removed_by_global_integration` — 곡률 있는 표면에서 tree-only drift보다 global integration의 per-edge residual이 작음. Worklog 79~99 관련 focused 회귀 통과. 전체 회귀 실행함(아래).

구현 중 발견한 버그: fold 판정에 처음엔 "이웃 다수결"을 썼으나, 합성 fold fixture(좌우 절반이 각자 내부적으로 방향이 일관되고 경계 한 edge만 반대 부호인 경우)에서 다수결이 경계 신호를 완전히 희석시켜 fold를 놓쳤다 — "이웃 중 하나라도 반대 부호면 fold"로 정정했고, 같은 정정을 `torch_orientation_preserving_uv_integration.py`의 fold-node 탐지에도 동일 적용했다.

## 실측: 7-region 실측(checkpoint 2900 / final)

`baseline_compatible` checkpoint 2900(held-out evidence 1915, 실행 271초), final(held-out evidence 3173, 실행 420초). 산출물: `output/extent_ab/val100/intrinsic_parameterization_architecture_gate_replay_2900.json`, `intrinsic_parameterization_architecture_gate_replay_final.json`.

### Pre-fit domain validity (raw component-count 기준, 수정된 validator)

| checkpoint | coherent | A domain-valid | B domain-valid | C domain-valid(=B와 완전 일치) | C integration-failed |
|---|---:|---:|---:|---:|---:|
| 2900 | 29 | 10 (34.5%) | 11 (37.9%) | 11 (37.9%) | 18 (62.1%) |
| final | 17 | 5 (29.4%) | 7 (41.2%) | 7 (41.2%) | 10 (58.8%) |
| combined | 46 | 15 (32.6%) | 18 (39.1%) | 18 (39.1%) | 28 (60.9%) |

C의 domain-valid 여부는 46개 component 전부에서 B의 domain-valid 여부와 정확히 일치한다(46/46) — B가 이미 유효했던 component에서는 반복 0회로 그대로 통과하고, B가 무효였던 component는 단 하나도 추가로 구제하지 못하고 전부 `not_globally_parameterizable_at_current_scale`로 fail-closed됐다. 전역 orientation flip은 A에서 4회, B에서 2회 canonicalize됐다(모두 real flip이지 fold로 오분류되지 않았다). B의 domain-invalid component에서도 global differential residual 절대값은 작다(overall RMS 대부분 0.0001~0.03, cycle-edge RMS도 비슷한 규모, component 하나가 예외적으로 0.027) — fold가 큰 미분 불일치의 산물이 아니라 낮은 residual과 공존하는 진짜 국소 방향 반전이라는 뜻이다.

Worklog 99의 confound 있는 validator가 보고한 80.4%(37/46) invalid와 비교하면, 수정 후 A는 67.4%(31/46) invalid로 낮아졌다 — 두 confound가 실제로 fold율을 부풀렸음을 확인하지만, 다수 component가 여전히 무효라는 결론 자체는 뒤집지 않는다.

### NURBS 평가 (evidence-weighted, domain-valid 여부 공유 안 함 — A/B/C 각자)

| checkpoint | candidate | PARAMETER_DOMAIN_INVALID | FIT_FAILED | EXTRAPOLATIVE | UNSAFE | VALID_SUPPORTED | held-out p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2900 | A TREE_INTEGRATED_UV | 63.22% | 0.00% | 7.33% | 29.45% | 0.00% | 8.26 |
| 2900 | B GLOBAL_DIFFERENTIAL | 62.87% | 0.00% | 0.00% | 37.13% | 0.00% | 8.93 |
| 2900 | C ORIENTATION_PRESERVING | 62.87% | 0.00% | 0.00% | 37.13% | 0.00% | 8.93 |
| final | A TREE_INTEGRATED_UV | 76.74% | 0.00% | 7.00% | 16.26% | 0.00% | 4.74 |
| final | B GLOBAL_DIFFERENTIAL | 46.53% | 0.00% | 27.86% | 25.61% | 0.00% | 9.22 |
| final | C ORIENTATION_PRESERVING | 46.53% | 0.00% | 27.86% | 25.61% | 0.00% | 9.22 |

두 checkpoint, 세 candidate 모두 VALID_SUPPORTED는 0.00%다. B/C는 domain-invalid 비율을 A보다 낮추지만(특히 final에서 76.74%→46.53%), 그만큼 늘어난 domain-valid evidence는 EXTRAPOLATIVE나 UNSAFE로 넘어갈 뿐 VALID_SUPPORTED로 전환되지 않는다.

## 결정

**C. COMPONENT_SCALE_NOT_GLOBALLY_PARAMETERIZABLE.** B(전역 differential 동시 적분)는 A(tree 적분) 대비 domain-valid component를 15→18/46(+20% 상대)로만 늘렸다 — 대다수 실패를 제거하지 못했으므로 "tree 적분 경로 자체가 한계였다"(Decision A)는 성립하지 않는다. C(B에서 엄격히 초기화한 local-injectivity 보정)는 B가 이미 domain-valid였던 18개와 정확히 같은 18개에서만 성공했고 나머지 28개(60.9%) 전부 fail-closed됐다 — 고정된 local-injectivity 제약이 단 하나의 fold도 추가로 풀지 못했으므로 "local injectivity 제약이 필요하다"(Decision B)는 성립하지 않는다. B의 domain-invalid component에서도 global differential residual 자체는 작다(RMS 대부분 0.001~0.03) — fold가 큰 미분 불일치·noise의 산물이 아니라, 그 scale에서 synchronized field가 진짜로 단일 평면 chart에 embed될 수 없는 위상적/곡률적 구조라는 뜻이다. **결론: synchronized tangent field는 국소적으로 일관되지만(holonomy 통과), 현재 component scale에서 하나의 chart로 전역 적분될 수 없는 경우가 다수다.** 새 curve seed를 추가하지 않는다. 앞으로의 분해는 NURBS fitting 이전에 **intrinsic integrability·chart 구조 자체**가 이끌어야 한다.
