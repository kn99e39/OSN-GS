# Worklog 177 — Intrinsic-Normal Structural Local Surface Decomposition: Controlled Contract Composition Audit

## 1. Intent alignment

이번 batch의 질문은 `W150 form_surface_regions`의 structural safeguards를 보존하면서 learned intrinsic Gaussian normal `t_w`만 surface-normal authority로 바꿀 수 있는지였다. W97/W154와 W10/W31–W38/W150을 각각 baseline으로 보존했고, production module·checkpoint·TSDF ownership·NURBS·visualization은 변경하지 않았다.

`t_w` 치환 전에 dependency audit를 먼저 수행했으며, clean composition gate가 실패했으므로 Candidate C 구현·replay·real-scene 시각화를 진행하지 않았다.

## 2. W154/W155 Baseline A — W97 intrinsic-normal path

W154의 active Gaussian decomposition은 `devtools/demo/candidate_f_gaussian_region_owned_tsdf.py:449`의 `partition_surfels_region_coherent(active_orientation, RegionCoherenceConfig(), ...)`이다. 이는 W96/W97 learned intrinsic `t_w`와 region-coherence contract다.

W155 exact checkpoint는 `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000`이며, 보존된 scene-wide 회계는 active surfel `1,190,469`, raw region `104,977`, accepted region `64,892`, accepted population `1,146,852`, isolated fallback `40,085`이다. largest-region fraction은 `0.216319`, top-1/5/10 fraction은 `0.216319 / 0.330285 / 0.358279`이다.

## 3. W150 Baseline B — covariance structural path

W150 constructor의 source path는 다음과 같다.

```text
extract_covariance_frame
  → evaluate_structural_reliability
  → build_manifold_affinity_graph
  → form_surface_regions
```

실제 `form_surface_regions` 호출은 `osn_gs/surface/torch_visible_surface_construction.py:207`이며, W154 candidate runner에는 해당 호출이 없다. 동일 W154/W155 checkpoint에 W150의 covariance frame, reliability, affinity가 동일 row order로 저장되어 있지 않으므로 W150 quantitative replay는 `NOT_COMPARABLE`이다. 누락된 입력을 approximation하지 않았다.

## 4. Historical contract separation

W96/W97은 `t_w`를 orientation authority로 사용하지만 W97에는 W150의 shared-neighbor consensus, bridge veto, tangent/path transport가 없다. 반대로 W10/W31–W38/W150은 covariance eigenframe의 normal·tangent·scale과 reliability/affinity를 함께 사용한다.

따라서 “W97 identity가 W154/W171까지 보존되었다”는 W175 결과는 “W97과 W150이 같은 structural constructor다”라는 증명이 아니다.

## 5. Dependency audit 범위

다음 quantity를 semantic role, source, covariance/full-frame dependency, `t_w` 치환 후 의미 보존 여부로 감사했다.

| Quantity | 판정 | 핵심 근거 |
|---|---|---|
| normal | `GENERIC_NORMAL_CONSUMER` | sign-independent normal alignment 자체는 `t_w`로 대체 가능 |
| tangent axes | `COVARIANCE_FRAME_DEPENDENT` | `tangent_u/v`는 covariance eigenvectors |
| spatial relation | `COVARIANCE_FRAME_DEPENDENT` | candidate/radius가 tangent footprint scale 사용 |
| tangent-plane displacement | `COVARIANCE_FRAME_DEPENDENT` | normal projection과 covariance residual scale의 결합 |
| normal separation | `COVARIANCE_FRAME_DEPENDENT` | gap을 `normal_thickness`로 정규화 |
| reliability | `COVARIANCE_FRAME_DEPENDENT` | shape/conditioning/scale과 neighborhood evidence |
| affinity | `COVARIANCE_FRAME_DEPENDENT` | residual, footprint, anisotropy, thickness를 함께 분류 |
| parallel conflict | `SEMANTICALLY_INCOMPATIBLE_WITH_SIMPLE_SUBSTITUTION` | close-parallel 분리의 thickness-normalized veto |
| shared-neighbor consensus | `COVARIANCE_FRAME_DEPENDENT` | counted edge와 reliability가 covariance affinity에 의존 |
| seed classification | `GENERIC_NORMAL_CONSUMER` | branch는 generic하지만 입력 reliability/degree가 변경됨 |
| component-pair support | `NORMAL_SOURCE_INDEPENDENT` | cross-edge 개수 집계 자체는 독립적이나 edge identity는 아님 |
| bridge veto | `COVARIANCE_FRAME_DEPENDENT` | local divergence, support, cut, path evidence 결합 |
| path/tangent transport | `SEMANTICALLY_INCOMPATIBLE_WITH_SIMPLE_SUBSTITUTION` | normal rotation과 tangent scale transport가 함께 필요 |
| ambiguity | `COVARIANCE_FRAME_DEPENDENT` | ambiguity policy와 ambiguity evidence를 구분해야 함 |

정확한 파일·line reference와 각 row의 설명은 [W177 report](../../output/177_intrinsic_structural_dependency_audit/worklog_177_report.json)에 기록했다.

## 6. Normal authority audit

`GaussianCovarianceFrame.normal_candidate`는 covariance의 `lambda3` eigenvector다. W96/W97 `t_w`는 표면 방향 authority로서 normal alignment를 계산하는 부분에는 의미를 유지할 수 있다. 그러나 W150 API가 받는 것은 normal 하나가 아니라 `GaussianCovarianceFrame`, `StructuralReliabilityResult`, `ManifoldAffinityGraph`다.

## 7. Tangent-frame dependency

`GaussianCovarianceFrame`은 `tangent_u`, `tangent_v`, `tangent_major_scale`, `tangent_minor_scale`, `equivalent_tangent_scale`, `normal_thickness`를 같은 covariance eigenframe에서 만든다. `t_w`만 `normal_candidate` 위치에 넣고 나머지를 유지하면 하나의 coherent orthonormal frame이라는 전제가 깨질 수 있다.

반대로 `t_w`에서 tangent basis와 scale까지 새로 만들면 normal source만의 substitution이 아니며, 역사적 covariance scale과 threshold semantics를 새로 정의하게 된다.

## 8. Reliability dependency

`evaluate_intrinsic_reliability`는 covariance conditioning, planarity/isotropy, tangent scale, normal thickness를 자체 evidence로 사용한다. `evaluate_contextual_consistency`도 neighbor normal agreement와 tangent-major-scale normalized residual을 함께 사용한다.

따라서 intrinsic `t_w`를 normal로 넣는 것만으로는 W150 reliability의 입력 의미를 유지할 수 없다. 새로운 intrinsic scale/conditioning contract를 만들지 않는 한 clean substitution이 아니다.

## 9. Affinity and parallel-conflict dependency

`build_manifold_affinity_graph`의 pair metrics는 다음을 함께 계산한다.

```text
normal_alignment
mutual_tangent_residual
tangent_direction_displacement_ratio
normal_direction_separation_over_thickness
tangent_footprint_ratio
tangent_anisotropy_ratio
```

특히 aligned normal pair도 `normal_direction_separation_over_thickness`가 크면 `parallel_but_separate`로 분리된다. `t_w`는 historical Gaussian `normal_thickness`의 보수적 physical bound를 제공하지 않으므로 이 veto의 의미는 simple normal substitution으로 보존되지 않는다.

## 10. Consensus, seed, component-pair support

`form_surface_regions`의 shared-neighbor consensus와 seed classification은 자체적으로는 graph 집계/DSU 정책이다. 그러나 그 graph의 same-surface edge와 endpoint reliability는 covariance-dependent affinity/reliability에서 온다.

component-pair cross-edge 개수 자체는 `NORMAL_SOURCE_INDEPENDENT`인 집계 연산으로 분류했지만, 어떤 edge가 포함되는지는 독립적이지 않다. 따라서 이 quantity 하나만으로 W150 전체를 normal-independent라고 해석하지 않는다.

## 11. Bridge veto and path/tangent transport

W38에서 보존된 bridge semantics는 shared support, local cut, tangent-frame divergence, parallel conflict, path evidence를 사용한다. `_evaluate_path_consistency`는 `frame.normal_candidate`의 hop rotation뿐 아니라 `frame.tangent_major_scale`에 의한 tangent-plane displacement consistency도 계산한다.

이는 W150 structural safeguard가 normal scalar seam으로 분리되지 않는 직접적인 근거다. `t_w`만 교체하면 정상적인 single-frame transport가 아니며, 모든 tangent quantities까지 intrinsic으로 바꾸면 새 architecture contract가 된다.

## 12. Parameter and mathematical identity contract

역사적 parameter 값은 변경하지 않았다. 현재 source의 역할은 다음과 같다.

```text
candidate_scale default = frame.tangent_major_scale
residual_scale default  = frame.tangent_major_scale
footprint criterion     = frame.equivalent_tangent_scale
parallel gap scale      = frame.normal_thickness
normal relation         = abs(dot(normal_candidate[a], normal_candidate[b]))
path scale              = frame.tangent_major_scale
```

대표적인 mathematical identity는 `d_T = d - (d·n)n`, `|d·n_bar| / tau_n`, `||d|| / s_t`이다. Candidate C를 만들지 않았으므로 이 identities에 intrinsic scale이나 새 threshold를 삽입하지 않았다.

## 13. Clean composition gate

Gate 결과는 **`NO_CLEAN_NORMAL_SUBSTITUTION`**이다.

정확한 blocker는 다음 두 가지다.

1. covariance tangent/scales를 유지하고 `normal_candidate`만 `t_w`로 바꾸면 서로 다른 orientation/scale authority를 섞은 frame이 된다.
2. `t_w`로 tangent basis/scale까지 다시 만들면 normal authority만 바꾸는 것이 아니며 historical covariance metric과 threshold semantics를 재정의한다.

그 결과 W150의 essential safeguards인 parallel conflict와 tangent/path transport의 meaning preservation을 보장할 수 없다.

## 14. Candidate C status

Candidate C는 별도 module로 구현하지 않았다. W97, `form_surface_regions`, W154 production path를 수정하지 않았다. clean gate가 실패했으므로 historical planar, smooth curved, parallel-sheet, phase-alias/shortcut, weak-bridge, crease/discontinuity synthetic fixture도 Candidate C arm으로 실행하지 않았다.

이는 synthetic fixture를 실패했다고 판정한 것이 아니라, 해당 fixture를 실행할 자격이 되는 clean contract가 없다는 뜻이다.

## 15. W174 witness and fixed tabletop accounting

W174 witness는 Baseline A만 확인했다.

| Row | Stable Gaussian ID | W97 subset | Role | Ambiguous | Stored path subset crossing |
|---:|---:|---:|---|---|---:|
| 4043 | 4,937,175 | 1 | core | false | 0 |
| 4051 | 3,929,355 | 1 | core | false | 0 |

핵심은 서로 다른 ID가 아니라 두 owner가 이미 W97 existing core membership에 있고 stored path의 subset boundary crossing이 0이라는 점이다. W150/C 비교는 `NOT_COMPARABLE` / `NO_CANDIDATE_C`다.

고정 W171 tabletop AABB에서도 W97 baseline은 complete support `17,965`, selected support `15,189`, support subset `1`, distinct owner Gaussian `3,315 / 2,887` (complete/selected)이다. TSDF component, W174 triangle, W173 loop를 identity source로 사용하지 않았다.

## 16. Zero-set, scene-wide, curved control, visualization

- Zero-set: 기존 W154 stored nearest-Gaussian association을 재사용하는 규칙만 확인했으며, Candidate C subset projection은 `NOT_RUN_NO_CANDIDATE`다. nearest recomputation과 TSDF connectivity 기반 identity 변경은 없었다.
- Scene-wide: Baseline A의 `104,977` subsets, `40,085` singleton/isolated fallback, largest/top-k distribution을 보존해 기록했다. Baseline B는 `NOT_COMPARABLE`, Candidate C는 `NOT_RUN_CLEAN_GATE_FAILED`다.
- Historical lineage: parallel sheet, phase alias, weak bridge는 W10/W31–W38 evidence로만 보존했다. W99/W100 patio→hedge exact stable mapping은 이번 arm에서 `NOT_COMPARABLE`이다.
- Curved control: W171 curved/vase는 valid grounded real curved positive control이 아니므로 `REAL_CURVED_POSITIVE_CONTROL_UNAVAILABLE`이다. synthetic historical control만 cross-reference했다.
- Visualization: Candidate C가 real-scene에 도달했을 때만 필요한 A–G set은 clean gate failure로 생성하지 않았다. W154/W155/W171–W175 기존 artifacts는 변경하지 않았다.

## 17. Final verdict — retained / rejected / open

최종 판정은 **`NO_CLEAN_NORMAL_SUBSTITUTION`**이다. 기존 historical SDF/TSDF나 W97 identity가 실패했다는 뜻이 아니라, W150 structural contract가 covariance normal만의 wrapper가 아니라 covariance full-frame·scale·reliability·affinity에 결합되어 있다는 뜻이다.

Retained:

- W97/W154/W155 exact identity lineage와 W174 witness baseline
- W10/W31–W38/W150 covariance structural lineage
- Gaussian `WHO`와 TSDF `WHERE`의 분리
- historical parameter와 production behavior

Rejected:

- W150 + `t_w` automatic composition
- largest-component/trusted-subset filtering
- epsilon/tolerance/threshold sweep
- TSDF component을 identity로 승격
- Candidate C 또는 architecture의 자동 promotion

Open:

- genuinely intrinsic-normal structural contract를 새 architecture question으로 정의할지 여부
- same-checkpoint W97↔W150 covariance/reliability/affinity crosswalk 확보
- historical contract를 주장하지 않고 intrinsic tangent/scale semantics를 별도로 설계할지 여부

## 검증

- W177 focused tests: `5 passed`
- static audit CLI: `NO_CLEAN_NORMAL_SUBSTITUTION`, Candidate C `NOT_IMPLEMENTED_CLEAN_GATE_FAILED`
- production replay/GPU/large regression: clean gate failure로 실행하지 않음
- `git diff --check`: 통과

