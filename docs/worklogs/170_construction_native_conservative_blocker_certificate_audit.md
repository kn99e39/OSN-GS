# Worklog 170 — Construction-Native Conservative Blocker Certificate Audit

## 1. Intent alignment

Historical projective SDF/TSDF construction이 `BEHIND_ZEROSET(v,x)`보다 강한 non-tuned conservative per-view physical blocker certificate를 제공하는지 감사했다. W167 raw zero-set blocker, W168 strict failure, W169 `MIXED` attribution, historical `h`/`mu`/iso-value/TSDF construction, 모든 component, strict `BEHIND_ZEROSET` relation 및 W161 pause를 변경하지 않았다. epsilon, margin, threshold sweep 또는 correction은 도입하지 않았다.

## 2. Available construction-native evidence

`SparseProjectiveTSDF`가 저장하는 field는 정확히 `keys`, fused mean `value`, `support_count`, `h`, `mu`뿐이다. `ExtractedSurface`는 `vertices`, `faces`, nearest-lattice-corner 기반 `vertex_support_count`/`vertex_field_value`, `h`, aggregate `stats`를 저장한다. per-view `phi_v`, source camera/view ID, projective depth sample, contributor별 weight·extrema는 저장하지 않는다. Fusion 시 각 authoritative observation의 weight는 고정 `1`이지만 결과에는 합산 count만 남는다.

W168 네 fixture를 exact replay하고 모든 valid zero-set first hit `9,428`개에서 extractor ownership rule로 hit cell을 복구했다.

| fixture | first hits | unique triangles | unique cells | interior / boundary | premature interior / boundary |
|---|---:|---:|---:|---:|---:|
| fronto-parallel plane | 2,500 | 2,450 | 2,025 | 2,304 / 196 | 84 / 0 |
| oblique plane | 2,208 | 2,130 | 1,795 | 2,158 / 50 | 1,276 / 30 |
| curved sphere | 1,356 | 1,313 | 1,125 | 1,192 / 164 | 0 / 0 |
| layered two-sheet | 3,364 | 3,026 | 2,025 | 3,136 / 228 | 970 / 68 |

모든 hit cell은 eight-corner authoritative이면서 `min(phi)<=0<=max(phi)`였고 corner `support_count`는 전부 `1`이었다. 총 triangle vertex `28,284`개를 모두 grid-edge interpolation으로 복구했다. Fronto/layered edge fraction residual은 exact `0`; oblique/sphere의 float reconstruction residual은 각각 최대 약 `4.63e-7`/`4.60e-7` lattice coordinate였다. 이 값은 marching-cubes representation accounting이며 uncertainty margin이 아니다.

W168 fixture는 source-view fusion이 아니라 analytic signed distance를 float32 `SparseProjectiveTSDF` carrier에 직접 넣는다. 따라서 fixture의 `support_count=1`은 camera provenance가 아니고, projective depth observation/source-camera support는 존재하지 않는다.

## 3. One-sided semantic justification

| quantity | availability | defensible one-sided physical relation |
|---|---|---|
| hit cell identity | triangle에서 복구 가능, surface에는 미저장 | 없음 — representation의 spatial ownership일 뿐이다. |
| eight corner TSDF values | 있음 | 없음 — clipped per-view 값의 mean이며 lower/upper envelope가 아니다. |
| corner `support_count` | 있음 | 없음 — view identity, sign agreement, extrema를 포함하지 않는다. |
| marching-cubes interpolation | 있음 | 없음 — stored scalar level zero에는 정확하지만 physical surface enclosure가 아니다. |
| projective depth observations | fusion 호출 중에만 입력 | 없음 — voxel별로 저장되지 않고 renderer median 자체도 certified physical first hit이 아니다. |
| source-camera support | fusion loop 중에만 알 수 있음 | 없음 — identity와 contributor value가 summation 뒤 폐기된다. |
| `h` | 있음 | 없음 — median footprint 기반 sampling scale이지 worst-case physical error bound가 아니다. |
| `mu` | 있음 | 없음 — authority/truncation band이지 zero-set의 one-sided displacement bound가 아니다. |
| cell diagonal / ray chord | 계산 가능 | 없음 — cell extent만 제한하며 physical surface 또는 target-ray hit을 제한하지 않는다. |
| iso-value `0` | 있음 | 없음 — fused representation level에 physical enclosure semantics가 없다. |
| W169 maximum error | 측정값 있음 | 없음 — fixture empirical maximum은 construction theorem이 아니다. |

동일한 stored mean/count가 서로 다른 per-view histories를 나타낼 수 있다. 예를 들어 `[-0.5,+0.5]`와 `[0,0]`은 mean `0`, count `2`로 축약되지만 per-view extrema와 sign agreement가 다르다. 따라서 stored state만으로 per-view one-sided envelope나 source identity를 복원할 수 없다.

## 4. Certificate definition

결론은 **`NO_CONSTRUCTION_NATIVE_BLOCKER_CERTIFICATE`**다. one-sided physical meaning이 construction semantics에서 따라오는 quantity가 하나도 없으므로 `CERTIFIED_BLOCKED(v,x)`의 mathematical predicate를 정의하지 않았다. `c*h`, `mu`, cell exit, W169 max error 또는 renderer median ordering을 fallback margin으로 사용하지 않았다.

## 5. Synthetic soundness

Certificate가 존재하지 않으므로 `CERTIFIED_BLOCKED => analytic GT_BLOCKED` implication, false positives, false negatives를 평가하지 않았다. W168 analytic surface를 certificate 정의에 사용하는 것은 construction-native evidence가 아니라 oracle 사용이므로 제외했다. all-false predicate도 “certificate”로 포장하지 않았다.

W168 fixture/extraction은 모두 exact 재현됐고 W168/W169 report 및 ray artifact SHA-256는 실행 전후 동일했다. 이는 audit 입력 보존을 검증할 뿐 physical blocker soundness 성공을 뜻하지 않는다.

## 6. Coverage / abstention

정의된 certificate가 없으므로 certification coverage와 abstention rate는 `N/A`다. 임의로 `0% coverage`인 classifier를 만들지 않았다. 대신 representation evidence accounting으로 first hit `9,428`개, interior `8,790`, boundary `638`, W168 premature interior `2,330`, boundary `98`을 전수 보존했다. 어느 population도 trusted subset으로 승격하거나 제외하지 않았다.

## 7. Architecture result

질문에 대한 답은 **NO**다. Existing historical SDF/TSDF construction은 `BEHIND_ZEROSET`보다 강한 non-tuned conservative physical per-view blocker certificate를 제공하지 않는다.

`BEHIND_ZEROSET(v,x)`은 계속 유효한 representation-level geometric fact다. 그러나 fused mean/corner sign change/marching-cubes hit은 physical first blocker의 one-sided enclosure가 아니므로 `DirectObservationUnavailable`과 자동 등치할 수 없다. W161 spatial-domain pause는 유지한다.

## 8. Retained / rejected / open

- retained: W167 raw zero-set blocker, W168 strict failure, W169 `MIXED`, historical construction과 모든 component, strict `BEHIND_ZEROSET`, W161 pause.
- rejected: `c*h`, W169 error margin, threshold sweep, zero-set offset/erosion, component filtering/trusted subset, TSDF 변경, NURBS, global state, Eligibility/continuation.
- open: 향후 physical blocker certificate에는 semantically justified one-sided evidence 또는 새로운 construction output이 필요하며, 이는 별도 승인된 architecture batch의 범위다.
- output: `output/170_construction_native_conservative_blocker_certificate_audit/`
- focused tests: `4 passed`; stored schema/provenance loss, fusion non-injectivity, trilinear corner evaluation, frozen four-fixture evidence audit와 input hash 보존을 확인했다.
- visualization은 생성하지 않았고 PNG/PPM 및 W153 replay-cache temp mirror도 만들지 않았다.
