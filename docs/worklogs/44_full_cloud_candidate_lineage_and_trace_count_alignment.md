# Full-cloud candidate lineage 및 trace count 정합화

## 작업 범위

Real 3k/5k/10k checkpoint에서 full-cloud continuation을 포함한 production candidate 경로를 source representative stable ID와 candidate ID 양쪽으로 연결했다. Representative-sector raw, full-cloud continuation raw, normalized, directed-ordering input, component, materialized boundary까지 동일 frozen input에서 계측했다.

Production termination semantics, continuation semantics, histogram threshold, seed admission, Hungarian solver, NURBS fitting은 변경하지 않았다.

## 확인한 결함

두 trace의 count 차이 원인은 production 결함이 아니라 diagnostic trace stage 혼용이었다.

- `trace_real_snapshot_boundary_waterfall.py`는 production `boundary_halfedge_candidates`의 normalized physical candidate를 세어 153/181/121을 보고했다.
- `trace_physical_termination_gates.py`는 같은 production candidate set을 읽어놓고도, 해당 candidate를 세기 전에 representative-sector local-neighbor gate를 먼저 재연했다.
- Full-cloud continuation으로 이미 생성된 physical candidate 일부가 representative-sector local support에서는 `no_neighbor_support` 등으로 먼저 빠져서 136/167/106으로 과소 집계됐다.

추가로 `frozen_core_seeding_replay.py::replay_boundary_candidates()`는 full-cloud continuation replay를 만들면서 termination scale 인자로 `state.rep_frame.equivalent_tangent_scale`을 넘기고 있었다. Production 본체는 이미 `candidate_scale`을 전달하지만, frozen replay helper는 contract와 달라 replay 간 불일치 원인이 될 수 있었다.

## 적용한 수정

- `scripts/devtools/trace_full_cloud_candidate_lineage.py`를 추가했다. 동일 frozen checkpoint에서 다음 stage를 한 JSON으로 연결한다.
  - representative-sector raw
  - full-cloud continuation production raw
  - sector-only / continuation-only / both physical source IDs
  - normalized / duplicate suppression removed IDs
  - typed provenance
  - directed-ordering input
  - component / materialized boundary
- `scripts/devtools/trace_physical_termination_gates.py`에서 authoritative production typed candidate membership을 representative-sector replay gate보다 먼저 반영하도록 수정했다.
- `scripts/devtools/frozen_core_seeding_replay.py::replay_boundary_candidates()`가 full-cloud continuation replay에서도 `state.candidate_scale`을 termination neighborhood scale로 전달하도록 수정했다.
- `tests/test_termination_neighborhood_scale_replay.py`에 continuation-backed production candidate가 sector no-neighbor gate로 오분류되지 않는 regression test를 추가했다.

## Real checkpoint 전후 수치

| checkpoint | 기존 physical gate trace | corrected physical gate trace | production waterfall / lineage | 원인 |
|---|---:|---:|---:|---|
| 3k | 136 | 153 | 153 | continuation-backed physical candidates를 sector local-neighbor failure로 선분류 |
| 5k | 167 | 181 | 181 | 동일 |
| 10k | 106 | 121 | 121 | 동일 |

Unified lineage 결과:

| checkpoint | footprint recall | candidate recall | footprint no-neighbor | candidate no-neighbor | sector physical source | continuation physical source | both | normalized physical | ordering input | closed | materialized |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3k | 0.0523 | 1.0000 | 701 | 2 | 326 | 159 | 47 | 153 | 153 | 0 | 0 |
| 5k | 0.0171 | 1.0000 | 741 | 2 | 389 | 184 | 96 | 181 | 181 | 0 | 0 |
| 10k | 0.0184 | 1.0000 | 657 | 2 | 326 | 124 | 54 | 121 | 121 | 0 | 0 |

Normalization/duplicate suppression 결과:

| checkpoint | production raw physical | normalized physical | removed raw candidate IDs | duplicate physical source IDs | ordering-input/component mismatch |
|---|---:|---:|---:|---:|---:|
| 3k | 159 | 153 | 6 | 0 | 0 |
| 5k | 184 | 181 | 5 | 0 | 0 |
| 10k | 124 | 121 | 5 | 0 | 0 |

모든 candidate는 최종 typed state를 정확히 하나 가진다. Normalized physical candidate object count와 source stable-ID count는 153/181/121로 exact match였다.

## Negative controls

Representative-sector fixture replay 기준:

| fixture | physical | closed | open | branch | materialized | classified FP |
|---|---:|---:|---:|---:|---:|---:|
| sphere | 0 | 0 | 0 | 0 | 0 | 0 |
| cylinder | 74 | 3 | 2 | 0 | 3 | 0 |
| close parallel sheets | 48 | 2 | 0 | 0 | 2 | 0 |
| thin strip | 48 | 2 | 0 | 0 | 2 | 0 |
| accepted-topology bridge contamination | 110 | 5 | 2 | 0 | 5 | 0 |
| box faces/corners | 110 | 5 | 2 | 0 | 5 | 0 |

Full-evidence production path smoke controls also kept sphere physical 0 and floater/isotropic contamination construction stable.

## 검증

Focused tests:

```text
.venv\Scripts\python.exe -m pytest -q tests\test_termination_neighborhood_scale_contract.py tests\test_representative_graph_scale.py tests\test_termination_neighborhood_scale_replay.py tests\test_full_cloud_continuation_shell.py tests\test_boundary_topology_safety.py
28 passed in 12.02s
```

Repository-wide pytest:

```text
.venv\Scripts\python.exe -m pytest -q
714 passed, 1 skipped, 1 warning, 8 subtests passed in 183.56s
```

## 결론

Full-cloud continuation 포함 production candidate lineage는 stable-ID 기준으로 정합화됐다. 136/153, 167/181, 106/121 차이는 source ID와 candidate ID 중복 문제가 아니라, production continuation candidate를 sector-only replay gate가 먼저 탈락시키던 diagnostic trace 결함이었다. Frozen replay helper의 scale 전달 결함도 수정해 candidate-scale production contract와 맞췄다.
