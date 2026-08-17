# Worklog 94 — bounded surface-evidence representation architecture gate

## 상태

**완료 — Decision 3. 네 representation 모두 coherent evidence 대다수를 unresolved/unsafe로 남긴다. Constructor-level 재설계는 중단하고, 다음 architecture target은 학습 중 visible geometric evidence 생성 자체(upstream)로 옮긴다.** Visible Gaussian training, ADC, region ownership, Worklog 79 coverage, PCA-UV, 6×6 NURBS fitting, held-out evaluation, 기존 safety criteria는 모두 미변경이다. 이 배치는 root-cause 진단(worklog 90~93)에서 architecture 결정으로 전환하는 단 하나의 bounded 비교이며, 또 다른 고립된 representation 진단으로 잇지 않는다.

## 배경

Worklog 93은 Worklog 92의 thick/curved center band(Worklog 90 `MULTILAYER_OR_VOLUMETRIC` evidence 대부분)가 curvature를 보존하는 recoverable latent 2D surface를 갖는다고 확인했다(Decision A: LATENT_SURFACE_RECOVERABLE). 이번 배치는 그 진단 결과를 **동기로만** 사용하고, real replay outcome으로 representation parameter를 튜닝하지 않았다.

## 최소 공통 adapter

신규 `osn_gs/surface/torch_surface_evidence_representation_gate.py`는 각 representation을 `(raw_positions, raw_covariance) -> (adapted_positions, adapted_covariance)` 함수 하나로 정의한다. `build_chart_unit_face_topology_context`(Worklog 89)와 `build_full_region_surface_face_topology`가 내부적으로 `extract_covariance_frame(covariance)`만을 유일한 기하 입력으로 쓴다는 사실을 확인한 뒤, 이 두 값만 교체하고 나머지 constructor 체인(Worklog 82 micro-component → Worklog 83 assembly → Worklog 89 face-incidence boundary → Worklog 79 coverage(경계 검증 내부) → PCA-UV → 6×6 NURBS → held-out → `valid_supported`/`extrapolative`/`unsafe_geometry`/`unresolved` 분류)은 전부 기존 함수(`materialize_chart_unit_cut_boundaries`, `evaluate_fit`)를 그대로 호출한다. Region 형성/ownership은 raw evidence로 **한 번만** 계산하고 네 representation 모두 동일하게 재사용한다 — representation은 region-owned `(evidence, evidence_covariance)`를 constructor에 넘기기 직전에만 교체된다.

### A. RAW_CENTER_BASELINE
Worklog 89의 기존 center 기반 representation. Pure pass-through(`positions`/`covariance` 무변경).

### B. CENTER_LATENT_SURFACE
Worklog 92/93과 동일한 local kNN(k=8) diagnostic 평면(covariance 미사용)으로 각 점의 독립 투영을 먼저 계산한 뒤, **cross-neighborhood consistency를 명시적으로 강제**하는 Jacobi 스타일 1-pass consensus averaging(각 점을 자기 kNN 이웃들의 독립 투영 평균 쪽으로 50%만 이동)을 적용한다 — 독립적 per-point projection을 그대로 최종 representation으로 쓰지 않는다(`tests/test_surface_evidence_representation_gate.py::test_center_latent_surface_is_not_independent_per_point_projection`이 직접 검증). Covariance는 이동한 위치의 local tangent frame에 맞춰 재구성하고(raw covariance 방향은 버림), normal-direction thickness는 local spacing의 5%로 collapse한다.

### C. COVARIANCE_SURFEL_SUPPORT
Position과 covariance 방향을 그대로 두되(covariance normal을 ground-truth surface geometry로 가정하지 않음), footprint의 observed reach(`equivalent_tangent_scale`)만 `support_radius_scale`로 disclosure한다. 기존 constructor가 이미 `normal_candidate`만 읽으므로, 이 representation의 차별점은 별도 지표로만 보고되고 constructor의 normal 소스를 몰래 바꾸지 않는다.

### D. HYBRID_LATENT_PLUS_SUPPORT
위치/topology는 B와 동일한 latent surface를 쓰고, covariance는 raw footprint의 in-plane 크기(support extent)만 가져오되 방향은 B와 동일한 latent tangent frame을 쓴다(raw covariance 방향은 여전히 버림).

## Fallback 없음

각 representation은 독립적으로 전체 파이프라인을 통과한다. 한 representation에서 실패한 unit이 다른 representation으로 재시도되는 코드 경로는 없다 — `analyze_representation`은 매번 raw region context에서 새로 시작한다.

## 검증

`tests/test_surface_evidence_representation_gate.py` 7개: A는 순수 pass-through, C는 position/orientation 불변(지지 반경만 disclosure), B는 점을 실제로 옮기고 covariance orientation을 재구성, B가 독립 per-point projection과 다름(consensus 강제 확인), D가 B와 동일 위치를 쓰면서 raw footprint support extent를 추가로 갖는지, 4개 representation 모두 shape 정합, raw 입력 tensor 불변 전부 통과. Worklog 79~93 관련 focused 70개 통과. 전체 회귀 **969 passed, 1 skipped**(272.6초).

## 실측: 7-region 실측(checkpoint 2900 / final)

`baseline_compatible` checkpoint 2900(3526 evidence)와 final(7774 evidence), cap=2048. 산출물: `output/extent_ab/val94/surface_evidence_representation_gate_replay.json`, `..._final.json`.

| checkpoint | representation | coherent | recoverable | valid_supported | extrapolative | unsafe | unresolved | held-out p95 | displacement/spacing | runtime | peak mem |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2900 | A RAW_CENTER_BASELINE | 88.15% | 0.170% | 0.000% | 0.000% | 0.170% | 87.98% | 2.67 | 0.000 | 24.7s | 5.0MB |
| 2900 | B CENTER_LATENT_SURFACE | 86.67% | 0.482% | 0.000% | 0.113% | 0.369% | 86.19% | 18.54 | 0.593 | 30.7s | 18.2MB |
| 2900 | C COVARIANCE_SURFEL_SUPPORT | 88.15% | 0.170% | 0.000% | 0.000% | 0.170% | 87.98% | 2.67 | 0.000 | 24.6s | 4.1MB |
| 2900 | D HYBRID_LATENT_PLUS_SUPPORT | 86.67% | 0.482% | 0.000% | 0.113% | 0.369% | 86.19% | 18.54 | 0.593 | 30.1s | 4.1MB |
| final | A RAW_CENTER_BASELINE | 84.82% | 0.090% | 0.051% | 0.000% | 0.039% | 84.73% | 3.90 | 0.000 | 66.2s | 23.3MB |
| final | B CENTER_LATENT_SURFACE | 84.11% | 0.463% | 0.154% | 0.090% | 0.219% | 83.65% | 7.66 | 0.586 | 79.8s | 8.1MB |
| final | C COVARIANCE_SURFEL_SUPPORT | 84.82% | 0.090% | 0.051% | 0.000% | 0.039% | 84.73% | 3.90 | 0.000 | 70.1s | 7.7MB |
| final | D HYBRID_LATENT_PLUS_SUPPORT | 84.11% | 0.463% | 0.154% | 0.090% | 0.219% | 83.65% | 7.66 | 0.586 | 88.1s | 8.1MB |

C는 두 checkpoint 모두에서 A와 수치가 정확히 동일하다 — constructor가 covariance normal만 읽고 footprint 크기는 아직 어디에서도 소비하지 않기 때문에, C의 disclosure-only 설계가 의도대로 동작함을 그대로 보여준다. D는 두 checkpoint 모두에서 B와 거의 동일하다(topology/coverage 결정이 position에 의해 지배되고 covariance in-plane 크기 차이는 구조를 바꾸지 못함).

B/D는 recoverable evidence를 A/C 대비 2.8~5.1배로 늘리고(0.170%→0.482%, 0.090%→0.463%), final checkpoint에서는 valid_supported도 A/C보다 3배(0.051%→0.154%) 크지만, **네 representation 모두 valid_supported는 여전히 0.2% 미만**이고 **unresolved는 83.6~88.0%로 압도적**이다. B/D는 또한 held-out p95가 A/C 대비 2.0~6.9배 악화되고(예: 2900에서 2.67→18.54) geometry displacement가 local spacing의 0.59배에 달해, 회수한 소량의 추가 evidence가 fit 품질 저하와 실질적 geometry 이동을 대가로 얻어진다. Runtime은 B/D가 A/C보다 20~33% 더 걸리고, peak memory는 B/D가 2900에서 A/C 대비 최대 4.4배 크다(final에서는 비슷).

## 결정

**3. 모든 representation 후보가 coherent evidence의 대다수를 unresolved 또는 unsafe로 남긴다.** A/C는 83.7~88.0% unresolved, B/D는 83.7~86.2% unresolved다 — B/D가 최대 recoverable을 5배 늘려도 unresolved 감소폭은 1.5%p 안팎이고 valid_supported는 어느 representation에서도 0.2%를 넘지 않는다. Constructor-level 재설계(representation 교체)는 이 4개 후보 범위에서 문제를 풀지 못한다.

**Constructor-level 재설계를 중단한다.** 다음 architecture target은 boundary/representation이 아니라 **training 중 visible geometric evidence가 어떻게 생성되는지(upstream)**로 옮긴다. 이는 지시대로 이 배치의 결론이며, 또 다른 고립된 representation 진단은 이어가지 않는다.
