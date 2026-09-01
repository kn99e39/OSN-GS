# Worklog 143 — Renderer median-depth 의미론 및 multi-view evidence aggregation 감사

## 상태

완료. 이 배치는 새 `Surface Membership`, 새 `mu`, representative fitting, continuation 또는 Occluded Surface를 구현하지 않은 격리 진단이다. WL142의 mask-only support와 canonical renderer/checkpoint를 read-only로 재생했다.

## 구현

- `devtools/demo/multi_view_support_lifting_depth_semantics_evidence_aggregation.py`를 추가했다.
- `osn_gs/render/surfel_rasterizer.py`와 vendored CUDA `forward.cu`의 실제 `depth_median` 경로를 추적했다.
- renderer-native event를 `pixel x/y + depth_median`에서 정확한 camera/world 좌표로 복원한 뒤 같은 camera로 재투영하고 depth를 다시 계산했다.
- 모든 per-view 관계를 `NEAR_MEDIAN`, `BEFORE_MEDIAN`, `AFTER_MEDIAN`, `NO_VALID_DEPTH`, `OUTSIDE_MASK`로 보존했다. `BEFORE/AFTER`는 물리적 모순 판정이 아닌 renderer-relative ordering state다.
- 선택을 위한 새 membership rule은 만들지 않고, 진단용 `D1 >= 1`, `D2 >= 2`, `D3 = 3` near-view population만 산출했다.
- WL141의 고정 3-camera control과 WL142의 `MASK_ONLY_BASELINE`을 세 ROI에서 row-ID count/hash로 exact replay했다.

## Renderer depth 의미론 검증

canonical surfel renderer의 `depth_median`은 CUDA에서 `depth=(s.x*Tw.x+s.y*Tw.y)+Tw.z`로 계산된 renderer event의 camera/view-space z다. Gaussian center의 view-space z, camera에서 event까지의 Euclidean ray length, normalized/inverse depth와 동일하다고 해석하지 않는다.

6개 deterministic camera(`DSC07957.JPG`, `DSC07993.JPG`, `DSC08030.JPG`, `DSC08066.JPG`, `DSC08103.JPG`, `DSC08139.JPG`)에서 각 6,000개 valid pixel을 샘플했다. 모든 camera가 측정되었고, reprojection pixel residual p95는 `8.04e-14`~`1.14e-13`, absolute renderer-z residual p95는 `0`~`1.78e-15` world unit이었다. 고정 identity gate를 통과하여 `DEPTH_QUANTITY_IDENTITY_PASS`로 판정했다.

WL139에서 물려받은 고정 scale은 `h=0.012105485424399376`, `mu=0.03631645627319813`이다. 첫 self-consistency camera에서 exact event의 median residual은 `0h`, `+/-h` displacement는 `+/-1h`, `+/-mu` displacement는 `+/-3h` (`+/-1mu`)로 재현되었다. 이 값들은 scale sanity이며 threshold tuning이 아니다.

## ROI 결과

| ROI | candidate | MASK_ONLY | near count histogram (0/1/2/3) | D1 / D2 / D3 | absolute residual median / p95 (h) |
|---|---:|---:|---|---:|---:|
| tabletop top | 1,554 | 1,367 | 1,367 / 0 / 0 / 0 | 0 / 0 / 0 | 596.226 / 818.870 |
| curved table rim | 20,181 | 17,842 | 17,166 / 676 / 0 / 0 | 676 / 0 / 0 | 71.382 / 149.775 |
| paver ground | 7,552 | 6,220 | 6,145 / 75 / 0 / 0 | 75 / 0 / 0 | 376.961 / 792.094 |

WL141/WL142 historical replay는 세 ROI 모두 row-ID count/hash 기준 exact reproduction이었다. 따라서 support population이 바뀐 결과가 아니다. Curved rim은 한 camera에서만 676개가 `NEAR_MEDIAN`이었고, 세 camera 중 두 개 이상에서 near인 point는 0개였다.

## WL142 all-zero 원인 귀속

세 ROI 모두 다음 값이 0이었다.

- hard veto 이전 `near >= 2` requirement: `0`
- `BEFORE_MEDIAN` 때문에 제거된 near-support: `0`
- `AFTER_MEDIAN` 때문에 제거된 near-support: `0`
- hard-veto 재계산 survivor: `0`

즉 WL142의 all-zero 결과는 hard zero-before/zero-after veto가 후보를 제거했기 때문이 아니다. 그 veto에 도달하기 전부터 두 개 이상의 camera에 직접 depth-consistent한 support가 없었다. 이 결과의 attribution은 다음으로 고정한다.

**C. DEPTH REPRESENTATION AND AGGREGATION ARE VALID, BUT THE HISTORICAL MASK SUPPORT HAS LITTLE DIRECT DEPTH CONSISTENCY**

이는 physical-sheet identity가 해결되었다는 뜻이 아니며, D1/D2/D3를 final support로 승격하지 않는다.

## 산출물 및 검증

산출물은 `output/multi_view_support_lifting_depth_semantics_evidence_aggregation/`에 격리했다.

- `depth_self_consistency/`: renderer median event sample PLY/NPZ 및 camera별 residual report
- 각 ROI의 `camera_overlays/`: original scene, historical mask-only baseline, per-view state, D1/D2/D3 raw overlays
- 각 ROI의 `geometry/`: mask-only 및 D1/D2/D3 PLY/NPZ
- 각 ROI의 `3d_review/`: 동일 physical chart viewpoint의 raw diagnostic PNG
- root 및 case별 JSON report: input path/hash, camera state matrix, histogram, replay hash, attribution

PNG 63개, JSON 13개, PLY 16개를 생성했고 JSON parse 및 PNG verify를 모두 통과했다. focused tests는 다음과 같이 통과했다.

```text
14 passed in 1.44s
```

문법 검사도 통과했으며, 실제 고정 CUDA 재실행 결과는 `DEPTH_QUANTITY_IDENTITY_PASS`, `failures=[]`였다.

## 구현 충실도와 남은 위험

- 수동/상속 선택: WL141의 3개 camera polygon과 ROI control을 그대로 사용했다. self-consistency camera 6개는 sorted camera-name quantile 규칙으로 deterministic하게 선택했다.
- full reference 사용: WL127 raw Visible Surface는 read-only candidate-row source와 diagnostic reference로만 사용했고, 새 fitter나 withheld fitting input은 없다. WL139/WL141/WL142 report는 입력 hash를 함께 기록했다.
- heuristic/비최종: D1/D2/D3는 attribution용 population일 뿐이며, 새 depth tolerance나 membership rule을 선택하지 않았다.
- 실행하지 않은 것: representative/NURBS/SH replay, continuation, true-occluded prototype, Candidate B 변경, canonical production 변경.
- 남은 위험: renderer event의 numeric identity는 닫혔지만, 동일 physical sheet의 multi-view correspondence와 최종 support aggregation은 여전히 미해결이다. `BEFORE/AFTER`를 곧바로 occlusion/geometry contradiction으로 해석하면 안 된다.

## 결론

이번 배치의 verdict는 **C**다. `depth_median` 수량 자체는 renderer convention에 맞고 왕복 identity도 통과했지만, 현재 WL141/WL142 historical mask support에는 multi-view direct depth consistency가 충분히 존재하지 않는다. 이 진단만으로 Surface Membership, Visible Surface representative, 또는 Occluded Surface architecture를 주장하지 않는다.
