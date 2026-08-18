# Worklog 97 — curve-network-native NURBS fitting: parameterization 유효성 검증

## 상태

**완료 — Decision C: CURVE_NETWORK_PARAMETERIZATION_INVALID.** Worklog 96의 coherent curve-network block은 자기 자신의 2×2 correspondence 계약을 통과해도, family 간 transversal 방향이 일관되지 않아 intrinsic (u,v) 도메인을 정의하지 못하는 경우가 대다수(11/12, 91.7%)다. 이는 curve-network-native fitting 구현의 결함이 아니라 — 동일 fitting 코드가 방향이 일관된 합성 block에서는 정상 작동함을 테스트로 확인했다 — Worklog 96 curve-family/correspondence 구조 자체의 한계로 귀속한다. Visible Gaussian training, ADC, region ownership, Worklog 95/96 latent-surface support/seeding/tracing/continuous support/family construction/block decomposition, NURBS degree(2)·control grid(6×6)·held-out 규약은 모두 미변경이다.

## 구현

### 1~3. Curve-network 구조 보존 + network-derived parameterization + curve constraint fitting

신규 `osn_gs/surface/torch_curve_network_uv_parameterization.py`: PCA를 전혀 호출하지 않는다. Family V(transversal trace)마다 u는 그 trace가 시작한 seed curve 위 위치의 cumulative chord length를 [0,1]로 정규화한 값(trace 전체에서 고정) — `C_i(v) ≈ S(u_i, v)`. Depth별 v는 그 depth에 도달한 모든 trace의 (자기 chord length/자기 총 길이) 평균으로 reconcile해 U/V 두 family가 동일 parametric domain을 참조하게 만든다 — reconciliation은 모든 trace가 공유하는 depth 범위(`max_shared_depth`)에서만 이뤄져 단조성이 구성상 보장된다. Rung(family U, `D_j(u) ≈ S(u, v_j)`) 내부 점은 두 trace의 u값 사이를 rung 위 chord-length 위치로 선형 보간한다. Stable ID나 입력 순서는 동률만 깨고 geometry를 정의하지 않는다.

신규 `osn_gs/surface/torch_nurbs.py::fit_torch_visible_surface_from_uv`: 외부에서 준 UV로 IDW seed(`fit_torch_visible_surface`)와 단일 regularized LSQ solve(기존 `_solve_control_grid_lsq` 재사용)만 수행하고, foot-point 재투영(`project_torch_points_to_nurbs`)을 호출하지 않는다 — UV가 fitting 도중 3D geometry로부터 다시 추정되지 않는다.

신규 `osn_gs/surface/torch_curve_network_native_fit.py`: `fit_curve_network_native(block)`이 위 두 조각을 연결해 block마다 하나의 NURBS를 network-native하게 fit하고, U-family(trace)·V-family(rung) 잔차를 분리 보고한다. `fit_pca_uv(block)`은 동일 3D 표본에 대해 기존 `fit_torch_visible_surface_lsq`를 그대로 호출하는 baseline A다. 두 경로 사이에 fallback은 없다(AST 검사로 `fit_curve_network_native`가 `fit_torch_visible_surface_lsq`를 호출하지 않음을 확인).

### 5. Correspondence 유효성 검증(fail-closed)

`build_curve_network_uv`는 seed chord length가 0에 가까우면(`degenerate_seed_chord_length`), family V curve가 2개 미만이면(`insufficient_family_v_curves`), u 순서가 단조증가·비중복이 아니면(`nonmonotonic_or_duplicate_u_family_ordering`), u/v extent가 0에 가까우면(`degenerate_*_parameter_extent`), 공유 correspondence depth가 2 미만이면(`insufficient_shared_correspondence_depth`), 같은 (u,v)에 다른 geometry가 매핑되면(`duplicated_parameter_location_incompatible_geometry`) fail-closed한다. **구현 중 실측으로 발견한 추가 실패 모드**: 인접한 두 transversal trace(rung으로 연결되는 유일한 쌍)가 서로 반대 방향으로 걸으면(cosine similarity ≤ 0) `inconsistent_transversal_curve_direction`으로 fail-closed한다 — Worklog 96의 per-seed-sample inward-hint 방향 선택이 인접 sample 간에 일관된다는 보장이 없다는 것을 real-data replay에서 처음 발견해 이 배치의 correspondence 검증에 추가했다(Worklog 96의 curve tracing 자체는 미변경). 어떤 경우도 PCA로 복구하거나 fit 결과로 재정렬하지 않는다.

### 4/6/7/8. Constraint semantics, baseline 비교, evaluation

Curve type별 수동 가중치는 도입하지 않았다(지시대로). 모든 block에 A. `PCA_UV_POINT_FIT`과 B. `CURVE_NETWORK_NATIVE_FIT`을 fallback 없이 동일 3D 표본·동일 6×6/degree-2 capacity·동일 안전 기준(EXTRAPOLATION_BOUND=4.0, Jacobian near-degenerate, local fold)으로 paired 실행했다(`scripts/devtools/curve_network_native_fit_replay.py`).

## 검증

신규 focused 테스트 18개: `test_nurbs_from_uv.py` 4개(외부 UV가 그대로 solver에 도달, PCA 미재계산, capacity 불변), `test_curve_network_uv_parameterization.py` 8개(chord-length parameterization, 단조 transverse parameter, rung intersection에서 u 일관성, 전체 curve network를 강체 회전해도 UV 불변, PCA 미의존, 모순된 방향 거부, degenerate seed chord length 거부, 계약 미충족 block 거부), `test_curve_network_native_fit.py` 6개(방향이 일관된 합성 block에서 정상 fit, PCA/native가 동일 3D 표본 사용, capacity 고정, family별 잔차 분리 보고, 계약 미충족 block이 PCA로 대체되지 않음, native 함수가 PCA fit 함수를 호출하지 않음). Worklog 79~96 관련 focused 149개 통과. 전체 회귀 **1017 passed, 1 skipped**(256.5초).

## 실측: 7-region 실측(checkpoint 2900 / final)

`baseline_compatible` checkpoint 2900(evidence 1526 held-out), final(evidence 2600 held-out). 산출물: `output/extent_ab/val97/curve_network_native_fit_replay.json`, `..._final.json`.

| checkpoint | attempted block | native parameterization_invalid | native valid_supported | pca valid_supported | native held-out p95(가중) | pca held-out p95(가중) |
|---|---:|---:|---:|---:|---:|---:|
| 2900 | 7 | **7/7 (100%)** | 0.00% | 3.41% | — | 6.65 |
| final | 5 | **4/5 (80%)** | 0.00% | 13.96% | 4.66 | 8.08 |

두 checkpoint 합쳐 12개 block 중 **11개(91.7%)가 `inconsistent_transversal_curve_direction`으로 fail-closed**된다. 유일하게 유효한 parameterization을 만든 1개 block(final checkpoint region 4)도 valid_supported가 아니라 extrapolative로 분류됐다(그 block 단독 paired p95는 native 4.66 대 pca 8.03으로 native가 더 나았지만, 표본 1개뿐이라 일반화할 수 없다). **Curve-network-native 경로는 두 checkpoint 어디에서도 단 하나의 valid_supported patch도 만들지 못했고**, 같은 block에 대한 PCA_UV 경로는 두 checkpoint 모두 nonzero valid_supported를 유지했다.

## 결정

**C. CURVE_NETWORK_PARAMETERIZATION_INVALID.** Worklog 96의 coherent curve-network block(자기 2×2 correspondence 계약은 이미 통과한 block)의 91.7%가 intrinsic (u,v) 도메인조차 정의하지 못한다 — 원인은 이번 배치가 새로 추가한 부호 일관성 검증이 아니라(이 검증은 발견된 문제를 노출했을 뿐, 만든 것이 아니다), **Worklog 96 curve-family 구조 자체**(seed sample마다 독립적으로 고르는 inward-hint 기반 transversal 방향 선택이 인접 sample 간 일관성을 보장하지 않음)다. 이는 PCA fallback으로 숨기지 않았고 그대로 보고한다. Curve-network-native fitting 자체는 방향이 실제로 일관된 block에서는 정상 동작함을 합성 테스트로 확인했으므로, fitting/parameterization 구현의 결함이 아니라 **입력 curve network 구조의 불충분함**이 병목이다. 지시대로 이 결과를 근거로 curve-seed heuristic이나 PCA/UV variant를 추가로 조정하지 않는다.
