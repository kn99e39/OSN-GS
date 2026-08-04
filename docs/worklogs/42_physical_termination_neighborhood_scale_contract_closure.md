# Physical termination neighborhood scale contract closure

## 상태

RepresentativeGraphScale 기반 termination neighborhood를 canonical contract로 유지한다. 이번 batch는 cross-region continuation, seed admission, directed ordering, NURBS fitting을 변경하지 않았다.

## Scale 계약

`candidate_scale`은 representative topology의 support radius, region graph spacing, termination angular observation에만 쓴다. Gaussian `equivalent_tangent_scale`은 primitive footprint/anisotropy/thickness에 남긴다. legacy caller는 affinity graph와 같은 `tangent_major_scale` default를 명시적으로 쓴다.

## Frozen A/B (representative-sector 경로)

full-cloud continuation은 의도적으로 제외하고, 같은 representative/reliability/region/accepted topology에서 extraction scale만 바꿨다.

| snapshot | footprint recall | candidate-scale recall | degree>=2 중 footprint <2 | candidate <2 | physical A/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3k | 5.23% | 100% | 709/724 | 0/724 | 2 / 326 |
| 5k | 1.71% | 100% | 708/711 | 0/711 | 0 / 389 |
| 10k | 1.84% | 100% | 623/624 | 0/624 | 0 / 326 |

footprint scale은 region graph가 이미 승인한 same-surface accepted neighbor를 termination gate에서 다시 소실시킨다. candidate-scale은 반경 확대만으로 out-of-region node를 physical support로 승인하지 않으며, sector path의 support는 여전히 accepted region topology에서만 집계된다.

## Candidate lineage

각 A/B에서 raw emission, normalized candidate, typed provenance, physical `observed_support_termination`, ordering input을 stable-ID hash로 기록했다. 세 snapshot 모두 raw=normalized+removed, typed reason 정확히 하나, ordering input=physical assertion을 통과했다. normalization 제거는 0개였다.

기존 136/153 등의 수치는 full-cloud continuation을 포함하는 production construction trace와, continuation을 제외한 representative-sector A/B replay를 같은 stage처럼 비교한 결과였다. 이번 A/B raw/normalized/physical 수치는 그 production final count와 직접 비교하지 않는다.

## 검증

- focused scale/invariance/analytic safety controls: `42 passed in 21.42s`
- prior repository-wide pytest: `709 passed, 1 skipped, 1 warning, 8 subtests passed in 182.51s`

## 결론

candidate-scale 전달은 threshold/radius multiplier/seed/solver를 바꾸지 않은 scale-source correction이며, frozen representative topology에서 accepted-neighbor coverage를 복구한다. full-cloud continuation lineage의 별도 report와 Box face precision trace는 다음 batch의 단일 병목으로 남긴다.
