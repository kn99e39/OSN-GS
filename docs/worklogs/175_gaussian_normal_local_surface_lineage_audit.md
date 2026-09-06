# Worklog 175 — Gaussian-normal Local Surface identity 계보 감사

## 1. 의도 정렬

**판정: `CANONICAL_LOCAL_SURFACE_IDENTITY_PRESERVED` — W96/W97 intrinsic-normal coverage-first 계열의 동일 W154/W155 checkpoint에 한정한다.**

W171 tabletop의 `region_id=1`은 W97 decomposition의 `subset_ids=1`을 그대로 받았다. W154 이후 다른 canonical subset을 하나의 Region으로 병합한 결과가 아니다. W174 witness 4043/4051의 owner도 이미 Gaussian branch에서 같은 subset의 core였으며, 저장된 reference path의 owner에서 subset boundary crossing은 0이다.

다만 **W150이 canonical로 기록한 `form_surface_regions`와 이 W97 경로는 서로 다르다.** W154는 W150의 mutual-tangent·multi-edge consensus·bridge/path 계약을 호출하지 않는다. 이번의 ID 보존 판정은 그 더 풍부한 계약까지 구현됐다는 뜻이 아니다. 이를 숨겨서 모든 historical `region_id`를 동치로 취급하지 않는다.

이번 배치는 읽기 전용 계보 진단이다. Region/zero-set ownership/geometry/chart/boundary/production 변경, NURBS fit, continuation은 하지 않았다. 착수 시 기존 W174 staged 변경이 있었으며 그 파일과 staging은 보존했다.

## 2. Canonical Local Surface Decomposition 식별

### 실제 intrinsic-normal 계열과 다른 historical 계약의 구분

| 소스·기록 | 정확한 역할 | W154에서 실행 여부 |
| --- | --- | --- |
| [W96](96_2dgs_coverage_first_surfel_partition.md), `torch_surfel_surface_orientation.py` | 신규 canonical 방향을 2DGS intrinsic `t_w` + coverage-first subset으로 명시 | 실행 |
| [W97](97_region_coherent_surfel_partition.md), `torch_region_coherent_surfel_partition.py::partition_surfels_region_coherent` | W96 local graph에 region-level orientation coherence를 추가한 subset 생성 | **기본 설정으로 실행** |
| [W98](98_discontinuity_first_surfel_partition.md) / [W99](99_interface_coherent_region_merge.md) / [W100](100_bilateral_interface_region_merge.md) | discontinuity·positional gate·interface merge의 별도 실험 계열 | 호출 안 함. W154 뒤에 merge로 붙은 것이 아님 |
| [W150](150_boundary_first_local_surface_decomposition_contract_trace.md), `torch_gaussian_surface_region_formation.py::form_surface_regions` | covariance-frame reliability/affinity, consensus, bridge/path를 사용하는 기존 constructor 계약 | **호출 안 함** |
| [W155](155_intrinsic_normal_gaussian_region_viability_audit.md) | W154와 동일 checkpoint·기본 W97 partition을 두 번 재실행하고 stable-ID mapping을 저장 | 이번 감사의 직접 frozen mapping |

W154의 정확한 호출 지점은 [candidate runner](../../devtools/demo/candidate_f_gaussian_region_owned_tsdf.py)의 `run`: `derive_surface_orientation_from_surfel` → active rows → `partition_surfels_region_coherent(active_orientation, RegionCoherenceConfig())`다. TSDF sample을 읽기/추출하기 전에 Gaussian identity를 만든다. W97 구현의 도입 commit은 `c6a697f`, 후속 `845eccd`는 positional gate를 opt-in으로 추가했으나 기본값은 False로 보존했다.

W150이 가리킨 [기존 constructor](../../osn_gs/surface/torch_visible_surface_construction.py)의 실제 호출은 `extract_covariance_frame` → reliability → manifold affinity → `form_surface_regions`다. Source의 오래된 Worklog 번호나 docstring 명칭만 보고 이것을 intrinsic `t_w` W97 호출로 해석하지 않았다. 같은 checkpoint의 두 경로 사이 per-Gaussian crosswalk는 W154/W155에 없다. 따라서 **W150 subset이 몇 개 병합됐는지는 null/미확보이지 1개 또는 0개라는 실측이 아니다.** W97에 대한 정확한 1개 판정과 구분한다.

### 실제 W97 same-surface 의미

| 요구 항목 | 실제 코드 의미 |
| --- | --- |
| Local geometric neighborhood | `build_candidate_graph`: 8-NN를 무방향 deduplicate. 각 row의 8거리 median을 spacing으로 삼고 `distance <= 2*min(spacing_i,spacing_j)` |
| Intrinsic Gaussian normal | `TorchGaussianSurfelModel.get_normal`의 rotation matrix 세 번째 열 `R[:,:,2]` → `surface_normal`; covariance normal 재유도 없음 |
| Sign-independent normal consistency | `abs(dot(t_w_i,t_w_j)) >= 0.85`; region scatter도 `n nᵀ`라 sign-independent |
| Tangent / mutual tangent | **membership에 사용 안 함.** `t_u/t_v/s_u/s_v`는 운반되지만 W97 판단에 소비되지 않음 |
| Spatial separation | 위 local center-distance gate만 활성. normal-offset/parallel-sheet separation의 `require_positional_continuity=False` |
| Multi-edge consensus | W10/W150 shared-neighbor consensus **없음**. 아래 region concentration과는 다른 계약 |
| Bridge/path consistency | W10/W150 tangent transport, shortcut, bridge veto **없음**. 대신 concentration-gated union과 non-bridging one-hop propagation |
| Region coherence | structural-core의 `M_R=Σ n_i n_iᵀ`, `C_R=λmax(M_R)/trace(M_R)`; union 시 `(1+0.85)/2=0.925` 이상만 허용 |
| Ambiguity | 둘 이상의 structural root에 닿는 propagated row는 flag를 남기고 deterministic raw owner 1개를 선택. W154 accepted ownership에서는 제외 |

Accepted edge는 alignment 내림차순, 동률 `(left,right)` 오름차순으로 처리한다. Union root는 최소 row index다. Singleton은 accepted edge 중 최고 alignment, 동률 최소 target index로 structural region에 한 홉 attach한다. Attach는 region scatter를 갱신하거나 region을 연결하지 않는다. 연결 대상이 없으면 자기 singleton subset을 유지한다. W175는 이 규칙·threshold·KNN을 변경하거나 재실행하지 않았다.

## 3. Canonical subset identity

직접 artifact는 `output/confirmed/155_intrinsic_normal_gaussian_region_viability_audit/gaussian_id_region_status_mapping.npz`다. 필드는 `stable_gaussian_id`, `region_id`, `membership_status`, `partition_role`, `ambiguous_multi_region`이다. 여기의 `region_id`는 **W97 output `subset_ids`를 저장한 것**이다.

| 항목 | 실측 |
| --- | ---: |
| Active Gaussian / raw assigned Gaussian | 1,190,469 / 1,190,469 |
| Raw subset 수 | 104,977 |
| Raw subset ID 미할당 | 0 |
| Core / attached | 1,115,158 / 31,694 |
| Ambiguous / rejected | 3,532 / 0 |
| W154 unassigned 상태(isolated fallback) | 40,085 |
| W154 accepted Gaussian | 1,146,852 |

`unassigned` status 40,085는 **raw subset ID가 없다는 뜻이 아니다.** Isolated fallback은 자기 singleton ID를 가지지만 W154에서는 accepted owner가 아니다. Ambiguous도 raw owner ID와 flag를 별도 보존한다.

W155의 두 replay mapping digest와 이번 재검산 결과가 모두 `06c9e1cbc730f06581895b32ad683e8822c7626eb3de9017fa8f83aaf0248bce`다. 방식은 stable ID 정렬 후 ID/region/status 각각의 dtype·shape·bytes SHA256이다. Checkpoint hash도 W155 기록 `4e49b916d668acf5eb0a1cc31b979caa86c5fd8743ac9632f36d7e9c1c72c9e2`와 재대조했다.

최종 subset ID는 size 내림차순, 동률 root index 순서다. **같은 입력과 row order에서 deterministic하지만 temporal ID나 임의 row permutation 불변 ID는 아니다.** Stable Gaussian ID join은 mapping 저장 순서와 무관하게 검증했다. 최초 W97 문서의 population은 1,197,331로 현재와 다르므로 당시 숫자 `subset 1`을 현재 `1`과 직접 동일시하지 않는다. W154는 최초 W97 artifact를 복사한 것이 아니라 같은 구현을 자기 checkpoint에서 재계산했고, W155가 그 동일 입력 mapping을 검증·저장했다.

## 4. W154로의 계보

| 변환 | 분류 | identity 판정 |
| --- | --- | --- |
| Checkpoint → non-uncertain active rows | filtering-only | Stable IDs 유지. 이번 checkpoint는 전부 active |
| Local graph + intrinsic `t_w` → W97 `subset_ids` | recomputation / 최초 identity 생성 | Gaussian branch에서 생성 |
| `subset_ids` → `GaussianRegionMembership.region_ids` | identity-preserving | `_as_cpu_tensor(partition.subset_ids)`로 직접 변환. merge/relabel 없음 |
| Role/ambiguity → `accepted_mask` | filtering-only | core/attached만 accepted, raw ID 자체는 유지 |
| Raw TSDF sample → `nearest_gaussian_index/id` | projection-only / association | Euclidean nearest center. Distance/normal rejection gate 없음 |
| Nearest Gaussian → `nearest_region_id` | identity-preserving | 기존 Gaussian ID의 subset lookup |
| `nearest_region_id` → `owned_region_id` | identity-preserving + filtering-only | Accepted면 동일 ID, 아니면 -1 |
| Owned region → native support component | one-to-many support split | 같은 region 내부 6-face connectivity. Region ID를 새로 만들지 않음 |

**파일명 전제 정정:** `candidate_f_tsdf_surface_samples.npz`에는 `region_id`가 없다. `source_cell_keys/cell_indices/world_xyz/normals/corner_values/corner_support_count`만 있다. Ownership은 같은 sample row index의 **`candidate_f_region_owned_support.npz:owned_region_id`**, owner Gaussian은 **`candidate_f_association.npz:nearest_gaussian_index/nearest_gaussian_id`**에 있다.

전체 **21,235,312 sample**에서 다음을 exact 검증했다. Selected positive case만 검사한 것이 아니다.

- `active_stable_ids[nearest_gaussian_index] == nearest_gaussian_id`
- `W155.region_id[nearest_gaussian_index] == nearest_region_id`
- `W155.membership_status ∈ {core,attached}` lookup이 `accepted_mask`와 일치
- `where(accepted, W155.region_id[nearest_gaussian_index], -1) == owned_region_id`

W97 subset의 downstream many-to-one Region merge는 **0**이다. 여러 Gaussian/zero-set sample이 같은 subset ID를 공유하는 것은 원래 membership이지 subset 간 merge가 아니다. Nearest association의 tie rule은 source 그대로 `smallest_stable_gaussian_id_among_returned_exact_minima`이며 KD-tree가 반환한 최대 8개 후보 범위의 tie 처리다. 이번 감사에서는 nearest search를 새로 수행하지 않고 저장된 association을 사용했다.

## 5. W171 tabletop으로의 계보

W171은 기존 W145 DSC07960 event AABB와 prior dominant `region_id=1`을 frozen selection으로 사용했다. AABB의 accepted support 20,935개 중 다른 기존 region의 2,970개는 **selection 밖**이지 Region 1에 병합된 것이 아니다. W172는 그 뒤 native largest component를 diagnostic control로만 택했다.

| 모집단 / 범위 | W97 subset ID 수 | Gaussian 수 또는 distinct owner 수 | Owned zero-set support 수 |
| --- | ---: | ---: | ---: |
| 전체 scene subset 1 | 1 | 65,471 | 92,684 |
| Frozen AABB 안 subset 1 Gaussian | 1 | 8,077 | 공간 crop 기준은 W171 행에 별도 적용 |
| W171 complete tabletop | **1 (ID 1)** | **3,315 distinct owner Gaussian** | **17,965** |
| W172 selected component | **1 (ID 1)** | **2,887 distinct owner Gaussian** | **15,189** |

전체 subset 1의 status는 core 63,120 / attached 2,245 / ambiguous 106 / rejected 0 / unassigned 0이다. W171 support의 owner status를 **sample-weighted**로 세면 core 17,257 / attached 708 / ambiguous·rejected·unassigned 각각 0이다.

선택 과정에서 보이지 않는 상태도 별도 회계했다. Frozen AABB의 모든 TSDF sample은 21,604개이며 nearest-owner status는 core 20,214 / attached 721 / ambiguous 150 / fallback 519이다. 이 중 accepted 20,935개가 기존 W171 envelope와 일치한다. Ambiguous/fallback을 정상 owner로 승격하지 않았다.

시각 review context는 AABB의 non-uncertain Gaussian 9,243개에 AABB 밖 complete-support owner 124개를 더한 9,367개다. **344 subset 전부**의 global/crop/review Gaussian 수와 support 수를 [전체 CSV](../../output/175_gaussian_normal_local_surface_lineage_audit/subset_inventory.csv) 및 JSON에 기록했다. 크기 threshold, top-K subset 생략은 없다. Subset 1 review Gaussian은 8,201개(core 7,910 / attached 279 / ambiguous 12)다.

## 6. W174 height-witness provenance

| 필드 | Witness 4043 | Witness 4051 |
| --- | --- | --- |
| W172 selected-local row | 4043 | 4051 |
| W171 complete-local row | 4226 | 4234 |
| W154 global sample row | **9,008,662** | **9,017,086** |
| Source cell | `[44,148,81]` | `[45,119,65]` |
| Active Gaussian index | 1,154,360 | 740,505 |
| Stable Gaussian ID | **4,937,175** | **3,929,355** |
| W97 subset / W154 owned region | **1 / 1** | **1 / 1** |
| Membership / ambiguous flag | core / False | core / False |
| Gaussian `t_w` | `[0.8614034,-0.2554164,-0.4390293]` | `[-0.8615543,-0.1989502,0.4670578]` |
| Stored nearest distance | 0.0523979291 world / **4.328445 h** | 0.00459436234 world / **0.379527 h** |

판정은 **`SAME_CANONICAL_LOCAL_SURFACE_SUBSET`**, W97 계열 기준이다. Normal은 부호를 수정하지 않았고 unsigned angle은 26.3134°다. 이것을 두 Gaussian이 직접 이웃이라는 증명으로 쓰지 않았다. Nearest distance도 새로운 acceptance margin으로 쓰지 않는다.

W174 저장 `witness_path`를 그대로 읽어 triangle → `triangle_owner_row` → W172 row → W154 sample → stored nearest Gaussian → W155 subset으로 합성했다. 결과는 **103 triangle / 73 distinct support row / 23 distinct Gaussian owner**, 전부 subset 1 core다. Path-ordered subset crossing은 **0**이다. Full correspondence는 [tabletop_lineage.npz](../../output/175_gaussian_normal_local_surface_lineage_audit/tabletop_lineage.npz)에 있다.

복원한 owning Gaussian은 **W154가 지정한 단일 nearest-center owner**다. TSDF fusion에 실제 기여한 Gaussian 전체 또는 renderer contributor 목록이 아니다. 해당 causal provenance는 이 association에 없다. 그 의미에서 owner ID는 충분하지만 physical-generation provenance는 불충분하다.

## 7. Tabletop-to-side transition의 historical 처리

이 witness의 양 끝과 reference path에 대응하는 owner를 **이미 W97이 같은 subset의 structural core로 취급했다.** W154/W171이 서로 다른 W97 identity를 뒤에서 합쳐 만든 경로가 아니다. Raw core membership은 concentration-gated union으로 형성되므로 source semantics상 accepted Gaussian graph 내부 연결은 존재한다. 그러나 W174 triangle path를 그 Gaussian graph의 edge path라고 부를 수는 없다.

W155에는 full accepted graph와 chronological union trace가 없고 `normal_cut_edges` 및 `region_conflict_edges`만 있다. 따라서 정확히 어떤 Gaussian edge sequence가 이 두 owner를 이었는지, 특정 edge를 union한 시점의 region state는 복원하지 않았다. Graph를 새로 만들거나 parameter를 바꾸지 않았다.

기존 cut 기록을 AABB 양 끝 내부 edge로 한정해 읽으면 normal-cut 7,154개(다른 final subset 간 4,346 / 같은 subset 내부 2,808), concentration-rejected merge 2,416개(다른 subset 간 1,195 / 같은 subset 내부 1,221)다. **Cut edge가 있다고 final subset boundary라는 뜻은 아니다.** 다른 경로를 통한 연결 또는 ownership propagation이 있을 수 있으므로 individual rejected edge를 subset 분할로 재해석하지 않는다.

Subset 1 core 63,120 normal의 scatter concentration을 진단용 float64로 평가한 값은 **0.9250007213**이다. 이는 historical aggregate rule을 설명하지만 max pairwise angle, planar chart injectivity, physical single-sheet를 보증하지 않는다. User가 지칭한 tabletop-to-side reference path는 이 의미에서 **같은 historical subset으로 처리됐다**고 답할 수 있다. 모든 물리적 tabletop/side transition이 의도적으로 하나라고 일반화하지 않는다.

## 8. Zero-set ownership 계약

실제 계보는 다음과 같다.

```text
W97 intrinsic-normal Gaussian subset identity (WHO, 먼저 생성)
  → W154 accepted Gaussian membership
  → global TSDF zero-set sample의 nearest Gaussian lookup
  → 동일 subset-owned support (WHERE의 사후 association)
  → 같은 region 내부 native support components
  → component boundary / structural materialization gate
```

**Coarser Region으로의 downstream ID merge는 발견되지 않았다.** 다만 TSDF는 subset별 source observation으로 따로 구성한 field가 아니라 global historical field다. 따라서 `subset-owned zero-set`은 **저장된 attribution 계약**이지 subset별 causal field construction의 증명이 아니다. Gaussian identity를 geometry에서 새로 만든 것은 아니지만, nearest lookup이 그 identity를 잘못된 물리적 support에 전달할 가능성을 본 감사로 배제하지 못한다.

Native TSDF connectivity는 region ID를 새로 생성하지 않는다. 그러나 W154에서 component별 boundary/fit 단위를 만들고 W171에서 fragmentation precondition을 검사하므로 **실무적으로 두 번째 downstream structural-unit decomposition/gate** 역할을 했다. 이것을 두 번째 Gaussian Local Surface identity로 승격하면 안 된다. W172 largest-component selection, W174 triangle components도 diagnostic support organization이며 identity가 아니다.

## 9. 시각 검토

[산출물 README](../../output/175_gaussian_normal_local_surface_lineage_audit/README.md) · [전체 실측 JSON](../../output/175_gaussian_normal_local_surface_lineage_audit/worklog_175_report.json)

| Family | 의미 |
| --- | --- |
| A Gaussian local subsets | 9,367 Gaussian center / 344 raw subset을 world에서 표시. Deterministic 64개 intrinsic `t_w`와 YZ normal-direction 패널 |
| B Region vs subset | 동일 Gaussian의 W97 ID와 W154 membership ID를 동일 camera/frame/color로 나란히 표시 |
| C Zero-set ownership | Complete 17,965 / selected 15,189점을 owner의 subset ID로 표시. 전부 ID 1이고 TSDF component 색을 사용하지 않음 |
| D Height witness | 4043/4051, owner stars/IDs/core, 원래 `t_w`, W174 path와 owner 연결을 YZ/XZ로 표시 |
| E RGB reference | DSC07960/DSC08003/DSC08043 원본 사진과 canonical subset center projection을 비교 |

총 **PNG 9개, PPM 0개, UTF-8 한글 README 10개**다. A–D 4개, E 3개, mandatory historical pair 2개다. Fixed-camera PNG는 `<camera>.png`로 flat 저장했다. 모든 artifact directory는 자체 입력·state·palette·조건·한계·평가 설명을 가진다. Scientific figure skill에 따라 generated imagery 대신 정량 scatter/normal/path figure를 썼다. A–E 모든 family를 직접 시각 검토했고 normal 확대 패널과 겹치지 않는 축/주석을 확인했다.

필수 Original Scene / Observed-Occluded pair는 W154/W155의 **DSC07957, 648×420, iteration 30000, 동일 1,190,469 Gaussian, black background, OSNSurfelRasterizer** PNG를 byte-identical 복사했다. Original learned SH와 geometry는 변경하지 않았다. 당시 state 767,400 OBSERVED / 423,069 OCCLUDED / 0 UNRESOLVED를 그대로 보존하며 **W168–170 이후 physical blocker 재인증으로 부르지 않는다.** 새로운 state나 UNRESOLVED를 구현하지 않았다.

A–E는 subset-ID 진단이며 상태 색의 대체가 아니다. RGB는 원본 사진이고 Gaussian render가 아니다. Canonical half-pixel projection을 사용하지만 depth culling은 하지 않으므로 RGB overlap을 ownership 증거로 사용하지 않는다. Complete overview에서 excluded TSDF components를 숨기지 않았으며 모든 contextual subset을 보존했다.

## 10. 아키텍처 원인 판정

완료 질문에 대한 답은 **“현재 W96/W97 intrinsic-normal decomposition의 identity는 W171 ownership까지 보존됐다”**이다. Witness는 downstream Region merge counterexample이 아니다. 그 Gaussian owner들은 이미 historical subset 1 core였고 W174 reference path도 stored owner 기준으로 그 subset 안에 남는다.

따라서 주 판정은 `CANONICAL_LOCAL_SURFACE_IDENTITY_PRESERVED`다. `LOCAL_SURFACE_IDENTITY_COARSENED_DOWNSTREAM`은 W97→W154→W171에 대해 기각한다. 반면 `CANONICAL_DECOMPOSITION_ITSELF_TOO_COARSE_FOR_CURRENT_STRUCTURAL_USE`는 이번 감사만으로 확정하지 않는다. 하나의 Gaussian subset이 하나의 planar chart여야 한다는 계약은 W97에 없고, nearest-owner support가 물리적으로 올바르게 전달됐는지도 별도 문제다.

**“W150의 더 풍부한 Local Surface 계약이 그대로 소비됐는가?”에는 NO**다. 정확한 차이는 W154 `run()`이 `form_surface_regions` 대신 W97 default partition을 생성하는 지점이다. 이는 저장된 여러 W150 IDs의 합병이 확인됐다는 뜻이 아니라 **다른 constructor 경로의 선택**이다. W150 identity를 기준으로 한 per-Gaussian 비교는 crosswalk 부재라는 provenance gap으로 남긴다. 이 차이를 수치상 W97 identity 보존으로 감추지 않는다.

## 11. 유지·미채택·미해결

- 유지: intrinsic `t_w`, historical decomposition·parameters·artifacts, W154 global TSDF/sample/ownership, W171 selection, W172 selected rows, W173/W174 진단, production.
- 미채택: 신규 decomposition, threshold/KNN/bridge 변경, Region 1 split, subset merge/remove/filter, TSDF/triangle connectivity를 WHO로 사용, boundary/chart 재설계, NURBS fit, continuation, UNRESOLVED 구현.
- 미해결: W150 richer-contract와 W97의 same-checkpoint crosswalk, exact accepted Gaussian edge/union path, nearest-center owner와 실제 renderer/TSDF contributor의 일치, physical single-sheet/structural-carrier 적합성. 본 batch는 이를 자동 수정하거나 새로운 설계로 진행하지 않는다.

검증은 신규 lineage 9개 + 기존 W97/W174 focused tests로 **42 passed, 75 warnings, 9.78s**다. Warning은 기존 W174 skimage/NumPy deprecation과 pytest class-fixture deprecation이며 실패가 아니다. 원본 W154/W155/W171–174와 audited source **1,505개 파일 SHA256**를 실행 전후 및 테스트에서 대조했다. 전체 repository regression은 실행하지 않았다.

```powershell
.venv\Scripts\python.exe -B devtools/demo/worklog_175_gaussian_normal_lineage_audit.py
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_worklog_175_gaussian_normal_lineage.py tests/test_region_coherent_surfel_partition.py tests/test_worklog_174_surface_complex.py
```

Local CPU에서 저장 artifact join과 작은 figure export만 수행했다. Graph/partition 재생성, CUDA job, heavy replay, replay-cache 복사는 없다. Code/tests/Worklog/문서 인덱스는 W175 범위로 commit하고 PNG/JSON/NPZ는 기존 정책의 gitignored `output/175_gaussian_normal_local_surface_lineage_audit/`에 둔다. 다른 작업의 W174 staged 변경은 commit에 섞거나 지우지 않는다.
