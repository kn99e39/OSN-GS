# Worklog 149 — Physical-Sheet Evidence vs Chart-Extent Failure Attribution

## 상태: 완료 — 격리된 비정규 attribution 진단

WL148의 committed baseline을 변경하지 않고, 거대한 chart extent가 evidence
population, PCA orientation, coordinate extrema, 또는 support materialization 중
어디에서 발생하는지 분리해 기록했다. 이 작업은 filter, refit, robust PCA,
chart trimming, Surface Membership, continuation, Occluded Surface를 실행하지
않았다.

## WL148 exact baseline 재현

- event union: `1586`
- camera count: `DSC08043.JPG=754`, `DSC07960.JPG=330`, `DSC08003.JPG=502`
- event union SHA-256: `79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78`
- frozen representative: `96 x 40 = 3840`
- representative XYZ SHA-256: `5fe79de62c6842cb02a99fa940f2e3ffa7fcd4c51165db606c693029aa59c941`
- representative normals SHA-256: `e01b51ddc2bc43586fdb16f1772c0f3a4e99ec46a81bfbb8202f8c78377ad05b`
- support: `314 / 3840`, unsupported `3526 / 3840`
- support-mask SHA-256: `23d00a22ae5ffc307ac3d5772c63c271291f535d2d383c63d68139708a6401d9`
- all-four materializable cells: `211`
- `h=0.012105485424399376`, `mu=0.036316456273198128`는 WL148 값 그대로 보존했다.

WL148 temp replay NPZ와 report를 다시 읽어 event union, representative XYZ/
normals, support mask, materializable cell mask를 모두 exact 비교했다.

## Chart 좌표와 extrema ownership

현재 chart는 world XYZ에서 PCA covariance용 centroid를 계산한 뒤, centroid를
빼지 않고 `world_xyz @ basis_axes`를 사용한다. 즉 chart의 `u/v/n`은 별도의
Scene 좌표계가 아니라 Gaussian Scene world XYZ에 대한 고정 PCA basis 투영이다.

- PCA centroid: `[1.3399900217, 1.5395132999, 0.9801192508]`
- basis columns:
  - `u=[0.6953601954, 0.7176686624, -0.0377609539]`
  - `v=[-0.3339030975, 0.3691584702, 0.8673123690]`
  - `n=[0.6363826837, -0.5904859989, 0.4963298953]`
- projected bounds:
  - `u=[-0.3689544253, 1.3587535437]`, span `1.7277079690`
  - `v=[-0.5984138976, 1.2633397036]`, span `1.8617536011`
  - `n=[1.4449360893, 1.8346735369]`
  - rectangular chart area: `3.2165665329`
- exact owner event IDs:
  - `u_min: 947`, `u_max: 795`
  - `v_min: 1527`, `v_max: 1104`

owner event provenance는 source camera, source pixel, renderer median-event depth,
world XYZ, normal, chart `u/v/n`으로 내보냈다. 네 owner의 상태는 모두
`AMBIGUOUS`이다. 저장된 provenance만으로는 primitive/contributor identity나
ground-truth physical-sheet label을 확인할 수 없기 때문이다. 따라서 owner를
wrong-sheet 또는 intended-tabletop evidence라고 자동 판정하지 않았다.

## 영향도 분해

모든 `1586` event에 대해 두 가지 leave-one-out 진단을 계산했다.

- fixed-axis LOO: 원래 PCA axes를 고정하고 해당 점을 제외한 extrema/span/area만
  재계산했다. rectangular-area influence는 median `0`, p95 `0`, max
  `2.2730185370`이며 `1582 / 1586`점은 near-zero였다.
- full-PCA LOO: 한 점을 제외할 때마다 PCA를 다시 계산하고 right-handed sign
  ambiguity를 정렬했다. joint axis rotation은 median `0.0159302°`, p95
  `0.0853413°`, max `2.0767824°`였다. origin shift는 median
  `0.00009047`, p95 `0.00020884`, max `0.00100989`이다.
- fixed-axis와 full-PCA 모두 ranking은 diagnostic-only이며 keep/reject
  threshold는 없다. 영향도가 큰 점을 입력에서 제거하지 않았다.

## Synthetic contract

- Fixture A: compact same-plane population + far point에서 fixed-axis extent
  leverage를 검출했지만 자동 reject하지 않았다.
- Fixture B: 고정 축 area 영향과 full-PCA orientation 영향을 분리했고 joint
  rotation `3.285336°`를 확인했다.
- Fixture C: 개별 extrema 지배점이 없는 경우 false rejection을 만들지 않았다.

세 fixture 모두 통과했다. 이는 real-scene physical identity를 증명하지 않으며,
진단 코드의 mechanical contract만 검증한다.

## 실제 Scene review와 구현 fidelity

동일 checkpoint/renderer/camera에서 mandatory `Original Scene`과
`Observed/Occluded` pair를 생성했다. Gaussian row count는 `1,190,469`이고
marker Gaussian은 `0`, geometry unchanged는 `true`, colour-only override는
`true`이다. exact owner별 source-camera render, clean oracle overlay, local
crop, cross-view projection, common-world context를 함께 export했다.

- 수동 선택: 기존 WL145 frozen tabletop broad-planar case와 세 camera.
- heuristic: extrema ownership, leave-one-out ranking, source-camera review
  상태 표기.
- full reference 사용: WL148 replay와 provenance 검증, owner review/plot의
  평가 target으로만 사용했다. 입력을 filtering하거나 refit하는 데 사용하지
  않았다.
- final paper method에서 허용되지 않는 것: 저장된 ground-truth identity가 없는
  상태에서 owner를 제거하거나 chart를 줄이는 것, 이 진단 결과를
  Surface Membership 또는 Occluded Surface로 승격하는 것.
- canonical production code, checkpoint, 161 cameras, Candidate B,
  historical topology, WL127/WL148 output은 변경하지 않았다.

## 판정

### ARCHITECTURE VERDICT: `UNRESOLVED`

chart mechanical sensitivity는 확인됐지만, influential event가 의도한 physical
sheet인지 다른 structure인지 판정할 증거가 없다. 따라서 architecture ordering을
결정할 수 없다. 다음에 필요한 것은 physical primitive/contributor identity 또는
ground-truth physical-sheet label이며, 그 전에는 membership-first와 chart-first
중 어느 쪽을 선택하지 않는다.

true-occluded prototype은 이 unresolved gate 때문에 실행하지 않았다.

## 검증과 산출물

- `tests/test_physical_sheet_evidence_vs_chart_extent_failure_attribution.py`:
  `4 passed`
- WL149 module syntax check: 통과
- scene run: `COMPLETED_ISOLATED_NON_CANONICAL_DIAGNOSTIC`
- output: `output/149_physical_sheet_evidence_vs_chart_extent_failure_attribution/`
- 보존 복사본: `temp/149_physical_sheet_evidence_vs_chart_extent_failure_attribution/`
- 주요 파일: `baseline_reconciliation.json`, `chart_attribution.json`,
  `full_per_point_influence.json/.csv/.npz`, `renderer_event_provenance.json`,
  `chart_space_attribution.png`, `common_world_provenance/`,
  `extrema_owner_reviews/`, `mandatory_gaussian_visualization_pair/`
