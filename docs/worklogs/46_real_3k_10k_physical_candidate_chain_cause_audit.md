# Real 3k/10k physical candidate chain 원인 감사

## 수행 내용

- frozen real checkpoint `3000`, `5000`, `10000`을 representative cap `2048`로 replay하고, physical candidate의 raw -> normalized -> directed-ordering 경로를 stable ID 기준으로 추적했다.
- `scripts/devtools/trace_real_physical_candidate_chains.py`를 추가했다. 이 도구는 region별 candidate spatial chain, compatibility in/out degree, open-chain endpoint, endpoint의 가장 가까운 missing interval, 첫 topology/geometry gate, raw/normalized lineage, 주요 surface region의 continuation state를 기록한다.
- 성공 대조군인 5k의 closed region `133`, `143`과 3k/10k의 open region을 비교했다.
- candidate intermediary를 통한 2-hop topology support를 허용하는 반사실 replay를 실행했다. 이는 production 변경이 아니며, 기존 Y-branch safety 조건을 제거했을 때 어떤 loop가 생기는지만 확인했다.

## 원인

| checkpoint | raw physical | normalized physical | raw closed region | normalized closed region | 결론 |
|---|---:|---:|---|---|---|
| 3k | 159 | 153 | 없음 | 없음 | physical candidate chain 미생성/fragmentation |
| 5k | 184 | 181 | 133, 143 | 133, 143 | 대조군 loop 유지 |
| 10k | 124 | 121 | 없음 | 없음 | physical candidate chain 미생성/fragmentation |

3k/10k의 normalized 이전 raw candidate set에도 closed loop가 없었다. 따라서 continuation provenance, candidate identity, normalization duplicate merge가 chain을 끊은 원인은 아니다. Normalization으로 줄어든 candidate 수는 각각 6/3이지만, raw stage에서도 3k/10k는 closed component가 0개이고 5k의 두 closed component는 그대로 유지됐다.

주요 region의 첫 병목은 full-cloud continuation의 `no_gap`이다. 예를 들어 3k의 member 21 region은 physical 3 / `no_gap` 18, member 20 region은 physical 0 / `no_gap` 19였고, 10k의 member 28 region은 physical 3 / `no_gap` 25였다. 생성된 소수 physical candidate도 region 내부에서 topology-support가 이어지지 않아 open chain 또는 isolated state가 됐다.

5k의 성공 대조군 region `143`은 physical candidate 4개 중 3개가 directed cycle을 만들었고 compatible edge는 5개였다. 반면 3k/10k의 큰 region은 0-5개 physical candidate에 머물렀으며, endpoint의 첫 실패는 주로 `topology_support_missing`, 나머지는 non-forward/lateral geometry였다.

## 반사실 안전성 검증

candidate intermediary도 2-hop support로 허용하면 3k에서 길이 3/4 cycle 2개, 10k에서 길이 3 cycle 1개가 추가됐다. 그러나 이는 `boundary_candidate_ids`를 통해 candidate intermediary를 제외하는 기존 Y-junction/branch contamination 방어를 해제한 결과다. gap 보간, open-chain 강제 폐쇄, topology threshold 완화에 해당하므로 production에 적용하지 않았다.

따라서 이번 감사에서 확인된 사실은 3k/10k의 남은 원인이 implementation defect가 아니라 full-cloud physical evidence의 `no_gap` 우세와 region-local topology fragmentation이라는 것이다. 확인되지 않은 topology 완화를 적용해 짧은 loop를 만드는 것은 완료 기준과 safety contract에 맞지 않는다.

## 결과와 검증

- frozen rigid-transform 관련 focused regression:

```text
79 passed in 29.96s
```

- 전체 pytest:

```text
715 passed, 1 skipped, 1 warning, 8 subtests passed in 184.20s
```

- Synthetic contract는 기존 수치를 유지했다: Box 6 closed, Cylinder 3 closed, Sphere physical candidate 0, Thin slab 분리 유지.

## 남은 단일 병목

Real 3k/10k에서 full-cloud continuation이 physical termination으로 분류하는 source가 큰 region perimeter를 이룰 만큼 충분하지 않다. 다음 단계는 directed ordering을 완화하는 것이 아니라, frozen full-cloud support evidence에서 `no_gap`으로 판정된 boundary-proximate representative의 관측 support가 실제로 연속 surface support인지 별도 data/evidence audit으로 검증하는 것이다.
