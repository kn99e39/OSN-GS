# Worklog 167 — Raw SDF/TSDF Zero-Set Surface as Explicit Camera-Ray Blocker

## 1. 의도 정렬

완료 상태는 `REAL_SCENE_REVIEW_REQUIRED`이다. 이번 작업은 historical raw SDF/TSDF zero-set mesh가 arbitrary-point camera ray의 명시적 blocker로 쓸 수 있는지 확인하는 diagnostic-only audit이다. production visibility/occlusion 동작, Candidate-B, NURBS, Gaussian Regions, Boundary First, continuation, Eligibility는 변경하지 않았다.

## 2. 구현 충실도

- W166의 `historical_sdf_zero_surface_raw.npz`를 primary source로 사용했다. W153 historical `h=0.012105485424399376`, `mu=0.03631645627319813` 및 vertex/face row order, world geometry, connectivity를 보존했다.
- two-sided Möller–Trumbore exact ray-triangle test를 구현했다. `t > 0`, barycentric bounds, degenerate triangle 제외, `(depth, triangle_id)` deterministic tie-break을 사용한다.
- 상태는 `HIT`, `NO_HIT`, `AMBIGUOUS` 세 가지이며, 각 ray에 first-hit camera depth/world XYZ/triangle ID/component ID/barycentric/positive-depth intersection count를 기록한다. exact coplanar non-degenerate ray는 `AMBIGUOUS`로 남긴다.
- real scene broad phase는 projected triangle AABB screen tile만 사용하며, 후보는 synthetic과 동일한 exact primitive으로 재검사했다. determinant epsilon은 `0.0`이고 sweep하지 않았다.
- query ladder는 before/surface/behind 세 점을 사용했다. surface query는 first-hit point 자체이며 Candidate-B median depth를 blocker로 사용하지 않았다.

## 3. 역사적 zero-set 보존 및 component attribution

W166 raw NPZ SHA-256는 `7e5df59a09877fcc1eebd1ab12d4c43a12d3dd352869fdb846c82d64678b495a`이고, vertex `28,694,040 × 3`, face `45,116,659 × 3`이다. W153 faces-adjacency accounting과 동일한 방식으로 real mesh component label을 계산했으며 `582,646` component로 일치했다. mesh repair, smoothing, filtering, hole filling, component 제거는 하지 않았다.

real attribution에서 major는 exact vertex count 상위 20개뿐이며, 나머지는 semantic size threshold 없이 disconnected fragment로 기록했다. fragment count만으로 false blocker를 주장하지 않았고, real hidden-space physical ground truth는 `NONE_NON_ORACLE_REVIEW_REFERENCE`로 명시했다.

## 4. Synthetic analytic blocker 결과

동일한 historical-style `SparseProjectiveTSDF` construction과 historical all-eight-corner zero-level extraction을 fronto-parallel plane, oblique plane, curved sphere에 적용했다.

| fixture | analytic hit | zero-set hit | hit rate | interior miss | support/silhouette miss | false blocker | p95 depth error / h |
|---|---:|---:|---:|---:|---:|---:|---:|
| fronto-parallel plane | 2,500 | 2,500 | 1.0000 | 0 | 0 | 0 | 0.0000 |
| oblique plane | 2,362 | 2,208 | 0.9348 | 0 | 154 | 0 | 2.65e-7 |
| curved sphere | 1,356 | 1,356 | 1.0000 | 0 | 0 | 0 | 0.0507 |

따라서 synthetic gate는 통과했다. oblique plane의 154 miss는 support/silhouette boundary에만 남았고 interior miss는 0이다. sphere에서는 zero-set first hit가 analytic surface보다 지연되는 raw voxelization 분포가 보존됐다(`p95=0.0507h`, `max=0.4331h`; 1 grid-cell 초과 vertex 4,620). 이를 보정하거나 숨은 geometry로 해석하지 않았다.

## 5. Frozen real-scene replay

W153 canonical camera metadata와 W162–W165의 frozen ROI를 사용해 `DSC07960.JPG`, `DSC08003.JPG`, `DSC08043.JPG`를 pixel stride 4로 재생했다. 총 1,259 sampled ROI rays에서 `HIT=1,259`, `NO_HIT=0`, `AMBIGUOUS=0`이었다.

| camera | rays / hits | major | fragment | median first-hit depth | query ladder |
|---|---:|---:|---:|---:|---|
| `DSC07960.JPG` | 436 / 436 | 423 | 13 | 6.2356 | 436 before / 436 surface / 436 behind |
| `DSC08003.JPG` | 399 / 399 | 396 | 3 | 2.7154 | 399 / 399 / 399 |
| `DSC08043.JPG` | 424 / 424 | 423 | 1 | 3.3217 | 424 / 424 / 424 |

모든 유효 ray에서 query ladder는 `IN_FRONT_OF_ZEROSET_SURFACE`, `ZEROSET_FIRST_SURFACE`, `BEHIND_ZEROSET_SURFACE`를 각각 결정했고 `NO_DECISION=0`이었다. 이는 blocker primitive과 query relation이 구현상 동작한다는 뜻이지, real scene의 hidden-surface truth를 증명하는 것은 아니다.

## 6. 시각화 및 검증

필수 여섯 종류를 PNG primary로 저장했다: `raw_zero_set_mesh`, `first_hit_surface`, `query_ladders`, `blocker_relation`, `component_provenance`, `common_world`. 모든 visualization directory와 nested camera/fixture directory에 UTF-8 `README.md`를 두었고, 최종 산출물은 [`output/167_raw_zero_set_ray_blocker_audit`](../../output/167_raw_zero_set_ray_blocker_audit/)에 있다. PNG 75개, JSON 9개이며 `W153 replay_cache`는 temp mirror에서 제외했다.

focused verification은 `tests/test_worklog_167_raw_zero_set_ray_blocker_audit.py`의 `11 passed`이다. exact hit/barycentric, deterministic first hit, degenerate/coplanar contract, analytic plane/oblique/sphere, query ladder, ROI mask alignment, disconnected component attribution, batch ray contract, projection-once broad phase를 고정했다.

## 7. 결론과 남은 위험

synthetic에서는 historical raw zero-set이 interior camera-ray blocker로 충분히 작동했다. real replay에서도 세 fixed camera와 ROI에서 exact first-hit와 query ladder가 재현됐지만, non-oracle mesh의 fragment hit가 의도한 physical surface와 일치하는지는 사람이 확인해야 한다. 그러므로 최종 판정은 `REAL_SCENE_REVIEW_REQUIRED`이며, 이 결과만으로 global Observed/Occluded state, NURBS handoff, hidden-surface identity, 또는 production architecture promotion을 승인하지 않는다.

