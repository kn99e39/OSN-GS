# Representative-sector dual-scale lineage gate

## 작업 범위

Representative-sector termination scale 변경을 검증하기 위해 production 동작을 바꾸지 않고 read-only replay를 확장했다. 동일한 representatives, reliability states, regions, accepted topology, canonical frames, seed admission, directed ordering configuration을 고정한 뒤 `equivalent_tangent_scale` branch와 `candidate_scale` branch를 나란히 평가했다.

이번 배치에서는 termination semantics, neighborhood multiplier, seed admission, directed ordering rule, continuation logic, NURBS fitting criteria, threshold를 변경하지 않았다. 생산 경로의 의미 변경 없이 검증 harness와 targeted test만 추가했다.

## 구현 내용

- `scripts/devtools/replay_termination_neighborhood_scale.py`를 dual-scale audit harness로 확장했다.
- 각 termination candidate에 대해 source representative stable ID, supporting representative stable IDs, region ID, extraction scale/radius, raw/normalized candidate ID, typed reason, angular evidence, directed-ordering input ID, compatibility edge IDs, ordered component ID, component state, seed admission result, NURBS materialization result, rejection reason을 기록한다.
- footprint branch와 candidate branch 사이의 stable-ID diff를 raw candidate ID 기준으로 출력한다.
- 기존 analytic/topology fixture를 `--fixtures` 경로에서 실행할 수 있게 하고 downstream counter, false-support classification, production-path composition summary를 같은 JSON에 포함했다.
- `_representative_knn_spacing`의 `M == 1` finite fallback(`1e-9`)과 `M == 2` pair distance는 기존 구현 상태를 확인하고 targeted replay test로 함께 고정했다.

## 검증 결과

Targeted validation:

```text
.venv\Scripts\python.exe -m pytest -q tests\test_termination_neighborhood_scale_contract.py tests\test_representative_graph_scale.py tests\test_termination_neighborhood_scale_replay.py
11 passed in 2.47s
```

Focused fixture audit 요약:

| Fixture | candidate recall | candidate raw/norm | physical | closed | open | branch | seed | NURBS | classified FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| close parallel sheets | 1.0000 | 48/48 | 48 | 2 | 0 | 0 | 2 | 2 | 0 |
| U-shaped concavity | 1.0000 | 44/44 | 44 | 2 | 0 | 0 | 2 | 2 | 0 |
| narrow neck | 1.0000 | 32/32 | 32 | 1 | 0 | 0 | 1 | 1 | 0 |
| thin strip | 1.0000 | 48/48 | 48 | 2 | 0 | 0 | 2 | 2 | 0 |
| high-valence / branching topology | 1.0000 | 124/124 | 110 | 5 | 2 | 0 | 5 | 5 | 0 |
| accepted-topology bridge contamination | 1.0000 | 124/124 | 110 | 5 | 2 | 0 | 5 | 5 | 0 |
| box faces and corners | 1.0000 | 124/124 | 110 | 5 | 2 | 0 | 5 | 5 | 0 |
| cylinder | 1.0000 | 78/78 | 74 | 3 | 2 | 0 | 3 | 3 | 0 |
| sphere | 1.0000 | 0/0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Full repository suite는 focused gate 통과 뒤 한 번만 실행했다.

```text
.venv\Scripts\python.exe -m pytest -q
713 passed, 1 skipped, 1 warning, 8 subtests passed in 182.52s
```

## 판정

Candidate-scale representative-sector termination neighborhood contract는 이번 focused gate에서 승인한다.

근거:

- accepted-neighbor coverage가 모든 fixture에서 candidate branch `1.0`으로 복구됐다.
- analytic negative control에서 classified in-region false support 증가가 없었다.
- branch component count와 false closure count가 footprint branch 대비 회귀하지 않았다.
- downstream stable-ID lineage가 raw candidate부터 normalized candidate, directed ordering, seed admission, NURBS materialization까지 연결된다.
- Box face/corner behavior는 `box_faces_and_corners` fixture에서 명시적으로 보고했고, 124 raw/normalized, 110 physical, closed 5, open 2, branch 0, NURBS 5로 유지됐다.

## 남은 위험

현재 fixture audit의 production-path composition은 sector-only이다. analytic fixtures는 downsampled full-cloud continuation path를 만들지 않으므로 continuation-only / both-path 후보는 0으로 보고된다. Full-cloud continuation composition은 harness 구조상 기록되지만, real downsampled checkpoint replay에서 한 번 더 확인하는 것이 더 강한 증거다.

일부 fixtures는 manual label 없이 충분한 ground truth predicate가 없어 precision은 `n/a`로 남는다. 이번 batch에서는 false-support histogram이 비어 있음을 negative-control evidence로만 사용했다.
