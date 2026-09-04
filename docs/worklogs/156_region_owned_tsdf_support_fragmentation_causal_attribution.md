# Worklog 156 — Region-Owned TSDF Support Fragmentation Causal Attribution Audit

상태: 완료

## 작업 목적

W154/WL155에서 하나의 plausible Gaussian Region이 수만 개의 native TSDF support component로 나타나는 원인을 진단했다. 이번 배치는 진단 전용이며 TSDF field, zero-surface 판정, Gaussian ownership, native component membership, connectivity, Boundary First, NURBS를 변경하지 않았다.

## 의도 정렬과 구현 충실도

- 주 대상은 WL155 tabletop review candidate 목록에서 고정한 Region `0`이다. 이는 physical-sheet ground truth가 아니다.
- 통제 대상은 frozen historical high-TSDF Region `2`, `5`다.
- WL153 field, W154 Candidate F의 samples/association/ownership/component IDs, WL155 Gaussian ID-region-status mapping만 읽었다.
- W154와 동일한 source-cell key encoding 및 exact six-face adjacency를 재생했다. 새 connectivity graph, bridge radius, smoothing, dilation, ownership 재할당은 만들지 않았다.
- 모든 real-scene view는 동일 checkpoint `30000`, camera, resolution `(648,420)`, black background, `OSNSurfelRasterizer` 조건을 사용했다.

## 정합성 결과

- W153 authoritative field: `76,720,314` voxels, `h=0.012105485424399376`, `mu=0.03631645627319813`.
- W154 TSDF zero-surface samples `21,235,312`, accepted-owned `20,426,913`, unowned `808,399`, native components `495,970`가 frozen 입력과 일치했다.
- W155 mapping hash 재계산/파일 비교: exact `true`.
- W154 nearest Gaussian stable-ID join: exact `true`; Region/status join: exact `true`.
- WL153 closure는 aggregate round 정보만 있고 voxel별 authoritative-addition lineage가 없어, closure 원인 귀속은 `NOT_RECOVERABLE_UNDER_EXISTING_CONTRACT`로 보류했다.

## 실제 정량 결과

| Region | 역할 | owned samples | components | frontier faces | A field 밖 | B zero-surface 아님 | C 다른 Region | D unowned/ambiguous | E same-region 미연결 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | primary tabletop candidate | 3,133,747 | 41,319 | 6,480,446 | 2,084,144 (32.16%) | 4,055,975 (62.59%) | 262,039 (4.04%) | 78,288 (1.21%) | 0 |
| 2 | high-TSDF control | 976,227 | 11,704 | 1,946,886 | 452,971 (23.27%) | 1,451,098 (74.53%) | 38,921 (2.00%) | 3,896 (0.20%) | 0 |
| 5 | high-TSDF control | 738,742 | 11,008 | 1,540,108 | 733,078 (47.60%) | 786,160 (51.05%) | 17,776 (1.15%) | 3,094 (0.20%) | 0 |

A/B/F field-starvation group은 Region 0에서 `6,140,119 / 6,480,446 = 94.75%`, C/D ownership-discontinuity group은 `340,327 = 5.25%`였다. exact same-region eligible-but-not-connected(E)는 세 Region 모두 `0`이므로 native face adjacency의 기계적 분리 오류 증거는 없다.

Region 0 component size 분포는 min/median/p75/p90/p95/p99/max가 `1/2/3/6/10/32/2,968,372`이고, singleton `20,408`, `<=2` `28,987`, `<=4` `35,027`, `<=8` `38,631`, `<=16` `40,210`이다. largest 1/5/10/100 component의 owned-sample fraction은 각각 `94.72%/95.26%/95.39%/95.88%`다. small(`<=8`) 및 singleton subset에서도 각각 field-starvation이 frontier의 `97.18%`, `96.62%`를 차지했다.

Field starvation의 직접 증거로 Region 0 A frontier의 neighbor-corner authoritative presence는 4–7/8, median 7/8이었다. B frontier의 corner TSDF min/median/max는 `-1.0000/-0.4292/0.8348`, max는 `-0.8482/-0.0113/1.0000`, 최소 support는 `1/8/100`(min/median/max)으로 기록됐다. 즉 B는 field가 없어서가 아니라 authoritative field 안에서 zero crossing을 형성하지 못한 경우다.

## Grid gap·ownership·status 감사

16-step diagnostic probe에서 Region 0은 1-step `10.83%`, 2-step `7.80%`, 3-step `4.94%`, 5+ step `13.27%`, probe 내 same-region component 미발견 `63.15%`였다. 이는 분포 측정이며 bridge radius나 연결 보정값을 선택하지 않았다.

Region 0 C의 주요 ownership transition은 `0→107: 1,233`, `0→71: 961`, `0→617: 712`, `0→314: 687`, `0→21: 602`였다. D의 W155 membership status는 `ambiguous: 5,625`, `unassigned: 72,663`이며, core/attached/rejected는 이 D 집계에서 나타나지 않았다. 이 숫자는 boundary strip인지 large band인지 판정하거나 attachment를 수행하는 데 사용하지 않았다.

## 합성 계약과 정성 검토

연속 same-region, authoritative band 부재, alternating ownership, unowned strip, same-region face-adjacent, replay-volume boundary, 기타 contract reason의 A–G 합성 계약을 모두 통과했다. 합성 통과는 real architecture 성공을 의미하지 않는다.

각 target에 A–H matched PNG를 생성했다: original Gaussian scene, frozen Gaussian Surface Region, all associated TSDF support, native components, largest component, frontier cause, ownership transitions, field starvation. canonical W155 `Original Scene`/`Observed-Occluded` pair도 PNG로 보존했고 W156 복사본에서는 PPM을 제외했다. 모든 산출물 디렉터리와 nested view/camera/frontier 디렉터리에 README를 추가했다.

## Architecture 결과

`TSDF_FIELD_STARVATION_DOMINANT`.

주 대상 frontier의 94.75%가 authoritative field 부재 또는 authoritative-but-not-zero-surface이고, ownership discontinuity는 5.25%, exact same-region native connectivity failure는 0이다. 따라서 이번 증거는 “native face adjacency 자체가 실패했다”가 아니라, frozen TSDF evidence가 local continuity를 제공하지 못한 것이 component fragmentation의 지배적 원인임을 지지한다.

## 유지 / 거부 / 미해결

- 유지: frozen W153/W154/W155 계약, Region 0 primary 및 Region 2/5 controls, exact native component 결과, diagnostic frontier accounting, matched PNG와 README.
- 거부: connectivity repair, component merge/split, bridge/dilation, ownership 변경, field 재생성, NURBS/Boundary First 변경, synthetic marker Gaussian.
- 미해결: WL153 closure의 voxel별 round lineage가 없어 field hole의 시간적 생성 원인은 추가로 귀속할 수 없다. physical-sheet 여부도 이 배치에서 주장하지 않는다.

상세 JSON과 raw frontier records는 [W156 output README](../../output/156_region_owned_tsdf_support_fragmentation_causal_attribution/README.md), [worklog_156_report.json](../../output/156_region_owned_tsdf_support_fragmentation_causal_attribution/worklog_156_report.json)에서 확인한다.
