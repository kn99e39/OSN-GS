# Worklog 173 — Real tabletop multi-loop observed-support domain 원인 분석

## 1. 의도 정렬

**최종 판정은 `MIXED_ATTRIBUTION`이다.** W172 tabletop의134 traced loop를 모두 조사했다. Observed-support hole이 실제 chart occupancy에 존재하고, W154의 exactly-one-loop 조건이 직접적인 rejection gate임은 확인됐다. 그러나 이 입력은 정상 simple outer 하나에 정상 inner hole들만 있는 domain이 아니다. Self-contact, loop 간 접촉, native/chart face-connectivity 차이와 world-height conflict도 함께 남아 있다.

따라서 “134 loop이므로134 physical surface”도, “multi-loop만 허용하면 하나의 정상 NURBS carrier가 된다”도 입증되지 않았다. 이 batch는 attribution에서 종료한다. Support 보수, loop 선택·병합·삭제, NURBS fit, domain 재설계는 하지 않았다.

## 2. 구현 충실도

새 코드는 `worklog_173_loop_topology.py`, `worklog_173_tabletop_multi_loop_domain_attribution.py`, `worklog_173_review_exports.py`의 독립 진단이다. W154/native module과 W171/W172 candidate는 수정하지 않았다. Topology 계산은 기존 integer chart lattice의 exact edge/vertex 관계를 사용하며 plane/normal/area/perimeter tolerance를 추가하지 않았다.

Diagnostic containment witness의 quarter-cell 위치는 unit cell의 안/밖을 구별하는 exact lattice 표본 위치다. Support를 이동시키거나 acceptance margin을 추가하지 않는다. Perimeter/area의 크기는 기술 통계 및 full population의 size-rank 표시에만 사용한다. Native cube complement의 allocation guard는 메모리 보호이며 geometry acceptance 기준이 아니다.

새 utility tests는 ID invariance, closure, annulus/nesting, independent outers, self-crossing/self-contact, invalid enclosing contour, cubical ring/cavity, frozen world/chart 대응과 PNG 존재를 검사한다. Synthetic fixture는 utility test에만 사용했다. W171 synthetic fitting은 요청된 기존 focused-test 회귀 확인으로만 실행했으며 새 architecture experiment로 사용하지 않았다. Curved/vase는 새 진단·사례 탐색 대상에서 제외했다. 기존 W172 test의 historical case 재확인은 새 positive evidence로 사용하지 않는다.

## 3. W172 기준 보존

| 항목 | 값 |
| --- | ---: |
| W171 complete tabletop support | 17,965 |
| Complete native components | 313 |
| W172 selected support | 15,189 |
| Complete 대비 비율 | 84.5477% |
| W172 original component ID / region ID | 0 / 1 |
| Selected native components | 1 |
| Historical h | 0.012105485424399376 |
| Authoritative loops | 134 |

W171에 저장된 tabletop AABB와 region organization으로 해당 case만 다시 읽었다. W172 `complete_row_indices`, source-cell keys, native indices, world XYZ, normals, region IDs, row order를 원본 complete support와 exact 비교했다. 전체/선택 support hash, 최대 component identity,134개 loop의 vertex 배열/offset 및 chart origin/tangents/occupancy가 일치했다. W154 boundary를 그대로 호출해 동일134 loop와 동일 rejection reason을 재현했다. Fitter는 호출하지 않았다.

W171/W172 code·output과 기존 boundary/fitter의 실행 전후 SHA-256 manifest를 보존했다. Historical h/mu/zero-set construction 및 extraction을 재설정하거나 변경하지 않았다. Empty corner provenance placeholder도 W172 상태를 유지했다. 원본 output은 재생성하지 않았다.

## 4. W154 loop의 정확한 의미

| 단계 | 기존 구현 의미 |
| --- | --- |
| Native connectivity | 같은 region의 source cell을 native 6-face neighbor로 연결한다. |
| Chart | Support mean과 평균 normal 기반 canonical tangent frame을 사용한다. Longer span을 u로 두는 기존 규칙을 유지한다. |
| Occupancy | Projected point를 `floor(projected/h + 0.5)`로 quantize한다. 해당 integer index의2D pixel을 occupied로 표시한다. |
| Boundary edge | Occupied pixel의4개 변 중 face-neighbor가 unsupported 또는 배열 밖인 변. 진행 방향 왼쪽에 occupied pixel을 둔다. |
| Tracing | 남은 edge 중 lexicographically minimum edge에서 시작하고, 매번 minimum outgoing endpoint를 따라간다. 시작점 복귀 또는 dead end에서 멈춘다. |
| Loop 보존 | 시작점으로 돌아오고 chain length가4 이상일 때만 저장한다. 마지막의 중복 start vertex는 배열에서 생략한다. |
| Loop 정렬 | Absolute signed area 내림차순, 그다음 첫 vertex 좌표 순서다. W173은 원래 순서도 보존한다. |
| Case `closed` | `bool(loops) and len(loops)==1`이다. 개별 loop closure의 AND가 아니다. |
| Geometry | Integer chart pixel corners를 chart plane에 embed한 world 좌표다. Native zero-surface의 실제 boundary edge를 직접 추출한 것이 아니다. |

Retained trace는 **개별적으로 닫혀 있음이 보장**되지만, 반복 vertex나 point-contact가 없는 simple polygon인지는 보장하지 않는다. Greedy outgoing 선택은 branching vertex에서 compound closed walk를 만들 수 있다. 일반적으로 dead-end chain은 버려질 수 있으나 이번 입력은 boundary edge **1,290개 모두가 정확히 한 번씩**134 trace에 포함돼 누락 edge는0개다.

기존 occupancy pixel은 index에서 `[index,index+1]`의 corner 범위를 사용한다. Sample rounding 위치와 boundary corner 사이에 별도 `-0.5` recentering은 없다. W173은 이 규약도 보존했다. 따라서 loop world 위치를 actual surface edge로 해석하지 않는다.

## 5. Tabletop loop 전체 집계

| 항목 | 결과 |
| --- | ---: |
| 전체 loop / individually closed / open | 134 / 134 / 0 |
| Simple valid / non-simple | 132 / 2 |
| Self-contact loop / proper self-crossing loop | 2 / 0 |
| Repeated-vertex loop / duplicate-edge loop | 2 / 0 |
| Inter-loop contact pairs / proper crossing pairs | 9 / 0 |
| 전체 edge / 누락 edge / 중복 사용 edge | 1,290 / 0 / 0 |

`L000`은500-edge closed trace로 `(117,7)`에서 vertex를 재방문한다. `L009`는16-edge trace로 `(23,11)`을 재방문한다. 둘 다 open polyline이 아니며, strict crossing 없이 point-contact로 이어진 non-simple closed walk다. 나머지132개는 loop 자체로는 simple polygon이다. Loop 사이 contact 여부와 전체 nesting tree의 유효성은 별도 검사다.

| 분포 | Count | Min | Median | Mean | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge / stored vertex count | 134 | 4 | 4 | 9.626866 | 12 | 500 |
| Perimeter `/h` | 134 | 4 | 4 | 9.626866 | 12 | 500 |
| Perimeter world | 134 | 0.04842194 | 0.04842194 | 0.11653788 | 0.14526583 | 6.05274271 |
| Signed algebraic area `/h²` | 134 | -105 | -1 | 49.320896 | -1 | 6,965 |
| Absolute algebraic area `/h²`, invalid 포함 | 134 | 1 | 1 | 54.649254 | 5.35 | 6,965 |
| Absolute simple-polygon area `/h²` | 132 | 1 | 1 | 2.666667 | 5 | 105 |
| Absolute simple-polygon area world² | 132 | 0.00014654 | 0.00014654 | 0.00039078 | 0.00073271 | 0.01538699 |

Invalid loop의 algebraic winding area는 기록하되 simple-polygon 면적과 구분했다. Loop마다 stable ID, 원래 rank/index, edge/vertex/unique-vertex 수, duplicate/repeated topology, closure, signed area, perimeter, chart/world bbox·extent·vertex centroid, 가까운 실제 support까지 거리와 plane residual을 JSON에 보존했다. Loop vertex는 chart-cell corner이므로 source support sample count와 혼동하지 않는다. `support_sample_count`는 해당 의미가 없어 null이다.

[전체 loop 목록](../../output/173_tabletop_multi_loop_domain_attribution/loop_inventory.md)은 size-ranked head와134개 전체 tail을 빠짐없이 보여 준다. Stable ID는 시작 vertex·방향에 독립적인 canonical integer-cycle ordering으로 정했다. 크기는 acceptance에 쓰지 않았다.

## 6. Containment와 nesting

Outer/inner는 면적순으로 결정하지 않고, boundary contact 검사와 polygon containment 및 양쪽 occupancy로 조사했다. **정상 simple outer 하나 + 정상 inner들로 이루어진 strict nesting tree는 인증되지 않는다.** Main exterior를 포함하는 L000 자체가 non-simple이며,9쌍의 inter-loop point contact도 있다.

L000의 generalized even-odd interior 안에 있는 관계는 기술할 수 있다. Contact-free generalized containment가 검출된 loop는131개이고, contact-free enclosing loop가 없는 trace는3개다. 이3개를 독립 physical outer family3개라고 해석해서는 안 된다. Self-contact 또는 경계 접촉으로 strict parent를 정할 수 없는 경우도 포함하기 때문이다.

JSON의 strict `outer_family_count=0`은 **인증된 simple-polygon root가0개**라는 뜻이다. Exterior가 없다는 뜻이 아니다. 전체 valid outer-family 수는 null, 모든134개 loop의 전역 nesting certification은 ambiguous로 기록한다. Invalid outer 안에 들어 있는 simple hole을 독립 outer로 잘못 세지 않는다. 각 loop의 generalized parent/depth/children와 invalid-container 이유를 별도로 기록했다. Strict nesting-depth 통계는 count0 / null이며 generalized relation을 유효 tree로 승격하지 않았다.

Closed occupied-pixel union은 vertex-connectivity로1개다. Face-connectivity로는6,608-cell component와1-cell component의 **2개**다. 따라서 두 face-family가 보인다고 곧바로 두 physical sheet라는 결론도 내리지 않는다.

## 7. Support occupancy와 hole 원인

현재 chart occupancy는6,609 occupied cells와 **134 bounded unsupported components**를 갖는다. 각각의 빈 영역은 chart상 관측 support가 없다는 뜻이다.

| Loop 유형 | Loop 수 | Exact occupancy 대응 |
| --- | ---: | --- |
| Simple unsupported interior-hole loop | 131 | 각 loop가 하나의 bounded empty region에 대응 |
| Simple exterior loop L023 | 1 | Chart `(37,65)`의1 occupied cell 경계 |
| Compound L000 | 1 | Exterior에 인접한496 edge + hole ID21에 인접한4 edge |
| Compound L009 | 1 | Hole ID24의12 edge + hole ID30의4 edge |

따라서134 trace는 **131개 simple hole trace + exterior1개 + compound2개**이며,134개의 bounded empty regions는 **131개 + L000에 붙은1개 + L009에 붙은2개**다. Loop count와 hole count가 우연히 같아도 일대일 대응은 아니다. L000은 외부 경계와1-cell hole을 point-contact로 한 trace 안에 포함한다. L009는 두 빈 영역을 point-contact로 함께 돈다. 이를 분해하거나 다시 연결하지 않았다.

모든 directed edge의 왼쪽은 occupied, 오른쪽은 unsupported였다. 전체 signed algebraic area 합6,609는 occupied cell 수와 일치한다. Hole size의 min/median/mean/p95/max는 **1 / 1 / 2.671642 / 5 / 105 cells**다. 크기에 상관없이 모두 남겨 두었다.

요청된 occupancy 분류 A는131 loop, B는1 loop, D는 compound2 loop다. C인 “missing native cell이 원인인 local defect”는 chart occupancy만으로 개별 loop에 확정할 수 없어 별도 원인 미확정으로 남겼다. 각 chart hole이 missing observation, native zero-set sampling 또는 projection/quantization 중 무엇 때문에 생겼는지에 대한 per-hole causal provenance는 현 sample records만으로 복원되지 않는다. 빈 영역 내부를 physical surface, occluded evidence 또는 latent continuation evidence로 채우지 않았다.

## 8. World-space physical-sheet 일관성

Frozen chart mean-normal plane에 대한 기술 통계를 계산했다. 새 plane fit이나 cutoff는 없다. Local 값은 native face-adjacent sample의 변위를 각 endpoint normal plane에 평가한 거리다.

| 기술량 | Median | Mean | p95 | Max |
| --- | ---: | ---: | ---: | ---: |
| Absolute global plane residual, world | 0.04167870 | 0.04570732 | 0.10911322 | 0.21205441 |
| Absolute global plane residual `/h` | 3.44296 | 3.77575 | 9.01354 | 17.51722 |
| Local endpoint-plane residual `/h` | 0.05965 | 0.15266 | 0.62337 | 2.28817 |
| Normal angle to chart normal, degrees | 16.11489 | 40.07875 | 153.18675 | 178.57884 |
| Unoriented normal angle, degrees | 14.97499 | 24.25433 | 74.81442 | 89.99960 |
| Neighbor normal angle, degrees | 10.33249 | 24.04661 | 103.99529 | 178.92689 |
| Multi-sample chart-bin height span `/h` | 0.15626 | 2.35212 | 9.40102 | 32.88685 |

Signed global plane residual 범위는 **-17.51722h ~ +16.43147h**, chart normal과 내적이 음수인 sample은2,664개다. Native neighbor normal 내적이 음수인 edge는2,007/30,821개다. Normal sign variation은 reconstruction orientation의 문제일 수도 있으므로 곧바로 서로 다른 physical surface라는 판정에 사용하지 않는다.

동일 chart bin `(25,37)`의 row4043/4051은 continuous chart 평면상 거리 **0.00227072 world = 약0.18758h**인 반면, normal 방향 높이 차는 **0.39811128 world =32.88685h**다. 이 witness와 전체 height 분포는 현재 chart가 support의 서로 다른 높이를 같은 occupancy로 합치는 상황을 보여 준다. Quantized collision 자체가 exact continuous fold-over를 증명하는 것은 아니지만, “같은 높이의 정상 단일 sheet에 hole만 있다”는 확정을 막는 실측 evidence다.

All-loop world geometry는 애초에 하나의 chart plane에 embed되므로 loop 자체가 coplanar라는 사실은 physical-sheet 일관성 증거가 아니다. 각 loop의 nearest actual support 거리·signed plane residual을 함께 저장했고, world/chart 동시 그림에서는 실제 sample 높이를 색으로 구분했다. Continuous projected sample의 exact duplicate는0개지만 이것도 연속적인 surface chart의 injectivity를 보장하지 않는다.

## 9. Chart-projection 충실도

**Native 단계에는 비교 가능한 ordered1D boundary-loop population이 없다.** W154 입력은 source-cell sample과6-face graph이며, native surface triangle/2-complex의 경계가 아니다. 따라서 “native에서134 loop였는데 chart에서도134”라는 동일성을 주장할 수 없고, native→chart loop/containment/orientation 변화량은 직접 정의되지 않는다.

| 비교 | 결과와 의미 |
| --- | --- |
| Native face-connected support | 1 component |
| Chart occupied face-components | 2: 6,608 cells +1 cell |
| Chart occupied vertex-components | 1: point-contact까지 허용하면 연결됨 |
| Native adjacent edge 중 서로 다른 chart face-component로 대응 | 3개 |
| Chart integer loops → 저장된 world embedding | 134개 모두 exact correspondence; closure/contact/nesting/orientation는 같은 planar embedding의 관계 |
| Native 3D cube-union topology | V30,493 / E75,712 / F60,313 / C15,189, Euler=-95, β0=1, β1=126, β2=30 |

세 native edge는 모두 chart bin `(36,64)`와 `(37,65)` 사이의 diagonal 관계로 바뀐다. 핵심 row는6,545이며 native cell `(56,119,97)`이다. 인접 row6,288 /6,551 /6,544와의 native face 연결은 그대로 있지만 occupied chart pixels에서는 point-contact만 남는다. 이 **face-connectivity 변화는 chart projection/quantization에서 생긴 것**이다. Closed pixel union 자체가 두 개로 분리됐다고 과장하지 않는다.

Native cube union의126 tunnels/30 cavities는 sample cell support 자체도 topologically trivial하지 않음을 보여 주는 보조 진단이다. 이는 raw physical zero-surface의 hole 수가 아니며, chart의134 hole과1:1 대응시키지 않는다. Native graph cycle rank15,633도 lattice cycle을 포함하므로 physical hole 수로 쓰지 않는다.

Chart에는 self-contact2 loop와 inter-loop contact9쌍이 있고 proper crossing은0이다. 저장된 world loop는 원래 integer loop를 rank-2 affine plane에 embed한 결과이며 vertex/edge 대응을 exact 확인했다. 이 embedding 단계에서 새 closure/contact/containment/orientation 변화는 없다. 반면 native surface complex가 없어 native physical boundary의 crossing 또는 fold-over가 새로 생겼는지는 확정할 수 없다. 현재 확인된 것은 discrete face-connectivity 변화, non-simple trace 및 높이 정보의 occupancy collapse다.

## 10. 시각 검토

[산출물 README](../../output/173_tabletop_multi_loop_domain_attribution/README.md) · [전체 실측 JSON](../../output/173_tabletop_multi_loop_domain_attribution/worklog_173_report.json) · [전체134 loop 목록](../../output/173_tabletop_multi_loop_domain_attribution/loop_inventory.md)

| Family | 내용 |
| --- | --- |
| A complete support world | 전체17,965 row; 진단 component cyan, 나머지312 component gray |
| B selected component world | 선택15,189 row +134개 planar loop 위치, equal XYZ |
| C all loops chart | 전체 occupancy와134 loop, stable ID 전부 표시 |
| D containment/nesting | 전체 domain 및 L000 self-contact/L023 point-connected cell 확대 |
| E loop-scale distribution | 전체134 rank의 edge/perimeter/area; invalid algebraic area도 표시 |
| F world vs chart correspondence | 같은 support의 height residual, 대응 ID,32.88685h bin-height witness |
| G RGB reference | DSC07960 / DSC08003 / DSC08043의 원본 RGB + support/loop projection |

총 **PNG9개, PPM0개, UTF-8 README9개**다. 각 visualization family와 parent/root README에 입력·색·좌표·한계를 각각 기록했다. 모든 row와 loop를 유지했으며 small-loop display filtering도 없다. Camera PNG는 family 바로 아래 `<camera_name>.png`다. Root JSON/NPZ/loop inventory는 full tail과 world/chart 대응을 제공한다.

직접 검토한 chart 그림은 많은 작은 hole뿐 아니라 L000/L009 point-contact와 L023의 별도 face-component를 보여 준다. World/chart 그림은 support 높이의 여러 분포가 하나의 planar occupancy에 합쳐지는 위치를 보여 준다. RGB는 tabletop 위치 확인용 secondary reference다. Depth culling이 없어 image overlap을 physical topology 증거로 쓰지 않으며, RGB 사진이므로 Gaussian state visualization을 새로 만들지 않았다.

## 11. 아키텍처 원인 판정

| 가설 | 판정 |
| --- | --- |
| H1 `MULTIPLY_CONNECTED_OBSERVED_SUPPORT` | Chart observed-support에 많은 hole이 있다는 부분은 확인. 하나의 정상 physical carrier라는 전제까지는 인증되지 않음. |
| H2 `BOUNDARY_CONTRACT_LIMITATION` | Exactly-one-loop 조건이 직접적인 rejection gate임을 확인. 그러나 chart 자체의 contact/face-connectivity/height conflict도 남아, 그 조건만 불필요하게 강한 유일 원인이라고 확정할 수 없음. |
| H3 `NON_SINGLE_CHART_STRUCTURE` | 현재 chart/trace 계약의 문제를 뒷받침하는 evidence가 있음. 이것을 true physical multi-sheet 또는 모든 parametric carrier의 불가능성으로 일반화할 수 없음. |

**`MIXED_ATTRIBUTION`**:134 loop는 주로 unsupported chart holes를 반영하지만, compound tracing과 chart connectivity 변화도 함께 반영한다. W154 exactly-one-loop 조건은 architecture bottleneck 중 하나이며, 현재 evidence로는 유일한 bottleneck이라고 할 수 없다. 이 topology가 모든 single structural parametric surface를 불가능하게 한다는 결론 역시 나오지 않았다.

요청의 A/B/C에 대해 A의 support-hole 부분, B의 direct gate 부분, C의 current-chart 결함 부분이 동시에 관찰된다. “물리적으로 정상 단일 sheet인 입력이 오직 loop 개수 때문에 부당하게 reject됐다”는 강한 B 해석은 이번 입력에서 입증되지 않았다.

## 12. 유지·미채택·미해결

- 유지: W171 tabletop selection, W172 exact component/rows,134 loop 전부, native connectivity, W154 chart/boundary/fitter, historical TSDF/zero-set semantics, production, W161 pause.
- 미채택: loop/component filtering, size threshold, trusted subset, outer-only 선택, compound trace 분해/병합, hole filling, morphology, smoothing, NURBS rescue, multi-loop domain 구현, continuation, UNRESOLVED, Eligibility, inferred Gaussian.
- 미해결: Native surface2-complex와 boundary correspondence, continuous chart injectivity/physical-sheet certificate, 각 unsupported region의 observation-vs-construction-vs-projection 원인 provenance가 없다. 이 evidence 없이는 하나의 continuous physical carrier인지와 genuinely non-single-chart physical geometry인지를 완전히 구분할 수 없다.

**Structural Surface / Parametric Carrier ≠ Observed Evidence-Supported Domain.** Unsupported inner hole은 occluded surface evidence도 latent continuation evidence도 아니다. 다음 domain 설계는 구현하지 않고 architecture review 지점에서 종료했다.

검증: W1739개 + 관련 W1715개 + W1725개 + render-projection3개를 합친 **22 passed (7.03s)**. W171의 기존 skimage/NumPy deprecation warning80개만 있었다. 마지막 시각화 배치 정리와 cubical-cavity unit check 이후 W173 focused tests도 **9 passed (3.18s)**다. Full repository regression은 실행하지 않았다. 모든 계산은 로컬 CPU였고 GPU workload나 replay-cache 복사는 없다. Source 코드·tests·Worklog는 commit하고, PNG/JSON/NPZ는 기존 repository 정책대로 gitignored output에 남긴다.
