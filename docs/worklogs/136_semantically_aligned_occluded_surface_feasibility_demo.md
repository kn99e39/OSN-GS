# Worklog 136 — 의미 정렬된 Occluded-Surface feasibility demo

## INTENT ALIGNMENT

이번 배치는 canonical OSN-GS Occluded Surface architecture를 구현하는
작업이 아니다. frozen Worklog 127 Visible Surface에서 실제 table geometry의
의미가 맞는 두 영역을 수동으로 고정하고, 관측 영역에 붙은 한쪽 holdout을
만든 뒤, retained geometry만으로 짧은 continuation을 구성하는 격리된
meeting demo이다.

- H1: 실제 `thin_table_leg_brace` 위치의 leg/brace self-continuation
- H2: 별도 visible tabletop/side source 접합의 각도를 다른 target 접합부에
  전달하는 junction-pattern transfer
- H1/H2의 withheld XYZ는 prediction/fitter에 들어가지 않고 evaluation,
  visualization, quantitative error에만 사용했다.
- WL127의 `h=0.012105485424399376`, `mu=0.03631645627319813`은 읽기 전용이며
  조정하지 않았다.

## IMPLEMENTATION FIDELITY

새 구현은 [semantically_aligned_occluded_surface_demo.py](../../devtools/demo/semantically_aligned_occluded_surface_demo.py)와
전용 출력 `output/semantically_aligned_occluded_surface_demo/`에만 있다.
기존 WL134 구현과 output은 재실행·수정하지 않았다.

수동 선택은 leg/brace ROI, source/target tabletop-side box, world
`u/v/n` axes, holdout cut, permitted volume이다. ROI는 full WL127 mesh를
직접 보고 의미를 정한 뒤 고정했으며, withheld error로 선택하지 않았다.
continuation은 기존 retained-frontier first-order ruled primitive와
측정된 source angle의 deterministic ±branch transfer를 사용한다. H2
extent는 선언한 target ROI continuation extent의 고정 0.50이다.

PNG는 형상을 직접 읽기 쉽게 하기 위한 고정 `0.02` world-unit voxel display
thinning만 적용한다. 저장된 NPZ/PLY geometry와 metric에는 적용하지 않는다.
manual ROI, semantic label, frontier heuristic, Candidate B proxy는 최종
paper method에서 허용되지 않는 demo-only shortcut이다.

## WHY WL134 DID NOT TEST THE INTENDED SEMANTICS

WL134 H1은 curved side/rim self-continuation이었다. WL134 H2도 실제
tabletop과 side의 접합이 아니라 같은 curved-side ROI 주변의 adjacent
upper/lower strips를 source pair처럼 사용했고, 약 `1.4197°`의 angle을
측정했다. 따라서 WL134는 intended actual-leg H1 또는 actual-tabletop ↔
side H2 semantics를 거부한 결과가 아니라, 질문이 다른 feasibility test였다.
이번 배치에서는 WL134를 다시 돌리거나 parameter를 튜닝하지 않았다.

## ACTUAL TABLETOP / SIDE RELATION

`actual_top_side_junction.png`에서 source top은 green, source side/rim은
blue, target top은 orange, target side/rim은 red로 표시된다. PCA 기반 semantic
audit은 source와 target 모두 boundary-coincident point와 충분한 local
surface extent를 확인한 뒤 H2를 시작했다.

- source visible junction angle: `77.6347°`
- target pair angle audit: `47.3371°`
- source/target local plane residual과 normal dispersion은 report에 모두
  기록했으며, 곡면 side의 높은 dispersion을 숨기지 않았다.
- H2는 target side의 `u=x`, `u_cut=-9.50`에서 `u<=cut` retained,
  `u>cut` withheld인 boundary-attached holdout이다.

## H1 — LEG / BRACE SELF-CONTINUATION

leg/brace ROI는 world x `[-0.30,0.40]`, y `[0.48,1.08]`, z `[0.70,1.35]`이며
`u=y`, `u_cut=0.75`로 고정했다. visible retained fraction은 `35.80%`,
withheld fraction은 `64.20%`, continuation extent는 `0.33` world unit이다.

withheld reference only 결과:

- median error / h: `4.6842`
- p95 error / h: `10.3943`
- coverage `<=h / <=2h`: `3.12% / 11.98%`
- normal median / p95: `49.75° / 86.48°`
- boundary position gap median / p95: `0.00h / 0.00h`
- boundary normal discontinuity median / p95: `70.39° / 85.98°`

raw PNG에서 generated sheet가 termination에 붙어 있는 것은 확인되지만,
withheld brace의 실제 분포와 충분히 맞지 않는다. 따라서 H1은 prediction
object가 `VALID`이라는 사실만으로 positive signal로 승격하지 않았다.

## H2 — REAL JUNCTION-PATTERN TRANSFER

실제 source/target semantic pair가 먼저 확인되었으므로 H2를 실행했다.
source tabletop/side의 measured angle `77.6347°`를 target retained-only
frontier에 전달하고 `plus_theta`와 `minus_theta`를 각각 fixed permitted
volume에서 검사했다.

- `plus_theta`: permitted volume 밖 점 `216`
- `minus_theta`: permitted volume 밖 점 `241`
- valid branch: `0`
- status: `NO_VALID_TRANSFER`

따라서 H2는 withheld prediction metric을 만들어 성공처럼 보이지 않았고,
두 branch geometry와 rejection reason만 NPZ/PLY/report에 남겼다. branch
selection에는 withheld reference가 사용되지 않았다.

## TRUE-OCCLUDED PROTOTYPE

실행하지 않았다. Candidate B archive에는 fixed leg/brace volume 안에
`global_B == 2` query가 있었지만, controlled H1이 useful/non-catastrophic
하지 않았으므로 conditional gate를 열지 않았다. H2 target volume에는
Candidate-B occluded query support도 없었다. 그러므로 fake true-occluded
metric, whole-scene completion, novel-view photometric rendering을 만들지
않았다.

## PROMOTED

없음. 어떤 geometry도 canonical Occluded Surface 또는 Candidate B로
promote하지 않았다.

## RETAINED

실제 semantic ROI audit, boundary-attached holdout 계약, retained-only
continuation trace, withheld-only metric, raw PNG/NPZ/PLY는 advisor discussion
용 isolated evidence로 보존한다.

## REJECTED

WL134의 H1/H2를 이번 intended semantics의 evidence로 재해석하지 않았다.
또한 H1 generated surface의 존재만으로 feasibility success라고 주장하지
않았다.

## OPEN

continuation extent, geometric termination, physical correspondence,
occlusion evidence, confidence와 publishable method 정의는 여전히 open이다.
이번 결과를 이용해 canonical Occluded Surface architecture로 자동 진행하지
않는다.

## MEETING VERDICT

**C. NEGATIVE FEASIBILITY RESULT — semantically correct primitives fail.**

실제 leg/brace와 실제 tabletop↔side relation을 사용한 controlled setting에서
현재 first-order self-continuation은 quantitative·qualitative positive가
아니었고, measured junction transfer는 branch rejection으로 끝났다. 따라서
이번 배치는 표면이 생성됐다는 사실이 아니라, 의도한 semantic primitive가
아직 usable evidence를 주지 못했다는 결론이다.
