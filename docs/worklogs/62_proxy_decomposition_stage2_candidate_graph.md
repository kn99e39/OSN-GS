# Proxy-Based Surface Decomposition Stage 2 후보 그래프

## 범위

Stage 2는 diagnostics-only spatial candidate generation까지만 구현했다. Component merge, admissibility threshold, weighted score, production integration은 구현하지 않았다. Candidate graph는 scene 이름, GT topology, GT component count를 입력으로 받지 않으며, 기존 production component membership을 읽거나 변경하지 않는다.

## 구현

- `osn_gs/surface/torch_surface_candidate_graph.py`
  - adaptive leaf AABB diagonal을 local scale로 사용하는 `aabb_distance <= radius_factor * max(scale_a, scale_b)` 조건
  - expanded-AABB sweep-and-prune broad phase와 exact AABB distance 검사
  - canonical pair ordering, duplicate 제거, deterministic optional degree cap
  - pair별 centroid/AABB/raw support gap, scale-normalized gap, candidate source 기록
  - face/edge/corner/overlap/disjoint는 accepted edge의 diagnostics로만 계산
  - pair 수, complete-graph 비율, node별 degree/histogram/min/median/p95/max/mean, recall, source/tag 집계
- `nurbs_constructor_benchmark/boundary_first.py`, `runner.py`
  - `--bf-candidate-diagnostics`, `--bf-candidate-radius-factor`, `--bf-candidate-max-neighbors` 추가
  - diagnostics 결과만 report에 넣고 production component/fitting에는 전달하지 않음
- `scripts/devtools/analyze_surface_candidate_graph.py`
  - 회전, point count, adaptive leaf resolution, density gradient, parallel-layer distance, disconnected-close, crease sweep
  - GT label은 graph 생성이 끝난 뒤 false candidate 분류에만 사용
- `tests/test_surface_candidate_graph.py`
  - 누락 pair, face-smooth recall, ordering/duplicate, contact provenance, payload, degree cap, production 비간섭 검증

## 핵심 결과

기본 설정은 `radius_factor=0.25`, `max_neighbors=0`이다.

| 조건 | nodes | edges | complete 대비 | degree mean | p95/max | smooth reference recall |
|---|---:|---:|---:|---:|---:|---:|
| curved_annulus 기본 | 12 | 39 | 0.591 | 6.50 | 9/9 | 1.000 |
| 회전 variants | 6–13 | 15–40 | 0.513–1.000 | 5.00–6.50 | 5–9 | 1.000 |
| points 300/600/1200 | 4/12/24 | 6/39/100 | 1.000/0.591/0.362 | 3.00/6.50/8.33 | 3/9/13 | 1.000 |
| voxel max count 75/150/300 | 18/12/4 | 65/39/6 | 0.425/0.591/1.000 | 7.22/6.50/3.00 | 11/9/3 | 1.000 |

- `curved_annulus` 기존 누락 smooth pair는 4/4, 기존 face-smooth pair는 14/14 포함됐다.
- Radius sweep 0/0.1/0.25/0.5/1.0에서 edge 수는 33/35/39/50/62였고 두 recall은 모두 1.0이었다.
- 회전·point count·adaptive leaf resolution·density gradient 전체 sweep의 평가 reference recall은 모두 1.0이었다.
- Candidate source는 모든 edge에서 `scale_aware_expanded_aabb_sweep`으로 기록됐다.

## False candidate 및 explosion 분석

GT-cross edge를 false candidate로 분류했다. `nonface_cross_component`는 누락 smooth pair도 포함할 수 있으므로 false로 취급하지 않았다.

| 유형 | 조건 | false candidate 비율 | graph |
|---|---|---:|---:|
| crease cross | 기본 crease | 57.1% | 8 nodes / 28 edges |
| parallel-layer cross | gap 0.03–0.48 | 57.1% | 모두 8 / 28 |
| disconnected-region cross | gap 0.02–0.20 | 27.3% | 모두 10 / 22 |

Coarse 4–8 leaf 조건에서는 complete graph가 자주 발생한다. 다만 최대 측정 조건인 24 nodes에서는 100/276 edges, degree mean 8.33, max 13으로 절대 edge/degree는 아직 제한적이었다. Parallel/disconnected distance 변화는 candidate membership을 바꾸지 않았지만 cross-edge median `support_gap`은 parallel 0.0818→0.4860, disconnected 0.0946→0.2606으로 단조 증가했다. 따라서 이 graph는 high-recall broad phase로는 적합하지만 false candidate 제거를 담당할 수 없다.

## 실제 pipeline benchmark와 회귀

다음 실제 통합 명령을 실행했다.

```text
.venv\Scripts\python.exe -B -m osn_gs.cli benchmark --constructor boundary_first --scenes curved_annulus mild_curved_sheet crease close_parallel_sheets density_gradient --points 600 --bf-candidate-diagnostics --skip-renderer-export --output artifacts\proxy_decomposition_stage2_unified_benchmark.json
```

`curved_annulus` production 결과는 기존과 동일하게 2 components (`disk_like`, `complex`)였고 diagnostics graph만 추가됐다. 직접 off/on 테스트에서도 `cluster_ids`, `surface_uv`, component count와 fitting payload가 동일했다.

검증 결과:

- Stage 2 집중 테스트: 11 passed
- 전체 테스트: 182 passed, 1 skipped
- Sweep artifact: `artifacts/proxy_decomposition_stage2.json`
- Unified benchmark: `artifacts/proxy_decomposition_stage2_unified_benchmark.json/report.json`

## 결론과 다음 gate

1. Missing smooth pair recall: 1.000 (4/4).
2. Candidate 규모: 기본 39 edges, degree mean 6.50, p95/max 9; 최대 측정 24 nodes에서 100 edges, max degree 13.
3. False 유형: crease 57.1%, parallel layer 57.1%, disconnected-close 27.3%.
4. 안정성: 모든 orientation/scale sweep에서 recall 1.0. Axis-aligned hierarchy의 leaf count 변화 때문에 graph density는 0.362–1.0으로 변동했다.
5. Stage 3 판단: **diagnostics-only agglomeration prototype으로 조건부 진행 가능**. Candidate recall은 충분하지만 dense coarse-leaf graph와 높은 GT-cross 비율 때문에 Stage 3는 proxy/support-gap/layer-separation admissibility를 반드시 검증해야 한다. Production 채택 근거는 아직 없다.

Stage 2에서 멈춘다. 사용자 승인 없이 Stage 3를 구현하지 않는다.
