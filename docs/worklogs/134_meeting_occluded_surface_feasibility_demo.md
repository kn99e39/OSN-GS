# Worklog 134 — Meeting Feasibility Demo: Occlusion-Bounded Surface Continuation

## INTENT ALIGNMENT

이번 배치는 최종 OSN-GS `Occluded Surface` 구현이 아니라, 고정된 실제 장면에서
Visible Surface의 continuation 가능성만 확인하는 비정규 feasibility track으로
실행했다. Worklog 127의 reconstructed Visible Surface mesh를 read-only로 사용했고,
Worklog 127–133, canonical renderer/checkpoint, 161 cameras, Candidate B와 기존
NURBS fitter의 production 동작은 변경하지 않았다.

## IMPLEMENTATION FIDELITY

새 코드는 `devtools/demo/meeting_occluded_surface_feasibility.py`에만 추가했다.
실험은 retained frontier의 PCA tangent/normal과 deterministic ruled linear strip,
그리고 관측 top/side angle을 transfer하는 고정 `+/- theta` branch를 사용한다.
second-order/third-order continuation, symmetry prior, neural completion, Trust,
whole-scene Occluded Surface는 사용하지 않았다.

전체 reference mesh는 고정 spatial mask를 적용하고 withheld evaluation target을
분리하는 데 사용했다. withheld XYZ/normal/endpoint/distance는 prediction,
branch selection, extent selection에 들어가지 않았고, withheld geometry는
evaluation·overlay·error 계산에만 사용했다. `h`와 `mu`는 WL127 cache에서 읽었으며
재조정하지 않았다. true-occluded prototype은 controlled gate 실패로 실행하지 않았다.

## MANUAL DEMO CONFIGURATION

manual demo-only 선택은 table crop, top-like/side-like/leg-brace box, local axes,
H1/H2 one-sided cut, pseudo-occluded volume, Candidate-B local volume이다.
주 primary side/rim ROI는 `u=[-6.6,-4.7]`, `v=[3.0,4.2]`, `n=[1.10,1.70]`,
`u_cut=-5.498`이며 continuation extent는 `0.798` world unit으로 고정되었다.
mesh subsampling 후 side ROI는 24,000 points, retained 13,443 (56.01%), withheld
10,557 (43.99%)였다. `h=0.0121054854`, `mu=0.0363164563`이다.

## OBSERVED TABLE SURFACE PATTERNS

top-like 10,372 points, side/rim 24,000 points, leg/brace 10,111 points를
고정 box에서 읽어 robust normal, tangent, angular dispersion, plane residual과
spatial extent를 측정했다. 이 측정은 continuation 입력에서만 수행했고 withheld
side target을 참조하지 않았다.

## MEASURED TOP-SIDE JUNCTION ANGLE

관측 top-like/side-like patch의 median normal로 계산한 angle은 `1.4197 deg`였고,
pairwise dispersion은 median `13.7292 deg`, p95 `32.5333 deg`였다. 90도나 다른
right-angle을 hard-code하지 않았다.

## H1 SELF-CONTINUATION CONTROLLED HOLDOUT

boundary-attached `u <= u_cut` retained surface만으로 frontier를 추정하고, local
frontier tangent 방향의 first-order ruled strip을 `u > u_cut` 쪽으로 연장했다.
생성 surface는 832 points / 1,550 triangles / area `0.975553`였고, withheld
reconstructed visible-surface reference에 대해서만 평가했다.

## H2 JUNCTION-PATTERN / BRIDGE CONTROLLED HOLDOUT

동일 fixed side holdout의 target frontier에 관측 top-side angle을 transfer하고
`+theta`와 `-theta`를 모두 fixed pseudo-volume/back-through 검사했다. 두 branch가
모두 valid하여 선택 결과는 `AMBIGUOUS`였고, 안전한 단일 prediction은 생성하지
않았다. 따라서 bridge/closure를 만들지 않았다.

## KNOWN-FREE-SPACE VIOLATION ACCOUNTING

controlled case에서는 manually supplied pseudo-occluded volume 밖을 허용하지
않는 것으로 계산했다. H1 prediction에서 volume 밖 point 103개(12.38%), touched
area 약 `0.118233`이 발생했다. 별도의 explicit free proxy point와 직접 겹친
point는 0개였지만, pseudo-volume 위반을 포함하면 gate 조건을 만족하지 않는다.
Candidate B를 사용한 true prototype용 proxy는 `R4_FRONT_OF_SURFACE_PROBE`와
`global_B == 1`만 free proxy로 취급하도록 구현했으며, 모든 OBSERVED query를
free로 간주하지 않았다.

## CONTROLLED HOLDOUT QUANTITATIVE RESULT

| case | visible / withheld | median / h | p95 / h | coverage <= h / <= 2h | normal median / p95 | boundary position gap | boundary normal gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 curved side/rim | 56.01% / 43.99% | 6.988 | 13.488 | 1.57% / 8.81% | 26.65° / 80.81° | 0.000h / 0.000h | 30.24° / 83.20° |
| H2 junction transfer | 56.01% / 43.99% | not generated | not generated | not generated | unavailable | not generated | not generated |

H1 평가는 withheld reconstructed visible-surface reference point에 대한 point-to-
generated-surface distance이다. observed fitting residual을 성공 근거로 사용하지
않았다. H2는 branch ambiguity 때문에 metric을 만들지 않았다.

## CONTROLLED HOLDOUT QUALITATIVE RESULT

H1 raw fixed-view overlay에서 cyan continuation은 boundary에 붙지만 withheld green
surface를 따라가지 않고 거의 일정 방향으로 벗어난다. H2 overlay는 두 magenta
candidate가 동시에 남아 있어 단일 continuation으로 해석할 수 없다. 따라서 비평면
side/rim의 controlled continuation은 visually plausible하거나 non-catastrophic한
결과가 아니었다.

## CONTROLLED FEASIBILITY GATE

고정 결과를 직접 확인한 뒤 manual gate를 `CONTROLLED_HOLDOUT_FAILS`로 선택했다.
이 선택은 결과를 본 뒤 geometry parameter를 바꾼 것이 아니며, numeric threshold를
fit하거나 parameter sweep하지 않았다. 두 controlled 결과가 gate를 통과하지
못했으므로 conditional true-occluded prototype은 실행하지 않았다.

## TRUE-OCCLUDED PROTOTYPE RESULT

`NOT EXECUTED — controlled gate failed.` 따라서 Candidate-B evidence를 이용한
실제 occluded geometry, visible-plus-predicted view, novel view, closure metric을
생성하거나 주장하지 않았다. 이는 true occluded surface의 성공/실패를 의미하지
않는다.

## FIGURE EXPORTS

발표용 polish 대신 advisor가 바로 열어볼 수 있는 raw fixed-view PNG와 geometry를
우선 출력했다.

- `output/134_meeting_occluded_surface_feasibility/table_demo_scene_overview.png`
- `output/134_meeting_occluded_surface_feasibility/H1_self_continuation/raw_fixed_view_overlay.png`
- `output/134_meeting_occluded_surface_feasibility/H2_junction_transfer/raw_fixed_view_overlay.png`
- 각 case의 `controlled_geometry.npz`, retained/withheld point PLY, generated/branch PLY
- `table_demo_geometry.npz`와 top/side/leg/table point PLY
- 전체 provenance/metrics: `meeting_occluded_surface_feasibility_report.json`

## MEETING VERDICT

**C. CONTROLLED HOLDOUT FAILS**

고정된 non-planar real-scene holdout에서 first-order continuation이 withheld
reconstructed Visible Surface를 회복하지 못했고 H2는 branch ambiguity가 남았다.
따라서 내일의 메시지는 “Visible Surface representation의 continuation feasibility가
이 설정에서 입증되었다”가 아니라, “이 단순 continuation은 controlled setting에서도
충분하지 않았으며, continuation extent/termination/occlusion evidence를 원리적으로
정의해야 한다”이다. canonical Occluded Surface architecture로 자동 진행하지 않는다.
