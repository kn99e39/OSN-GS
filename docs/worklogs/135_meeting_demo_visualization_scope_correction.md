# Worklog 135 — Meeting Demo Visualization Scope Correction

## 작업 목적

사용자 검토에서 기존 H1/H2 그림이 point cloud와 전역 축 범위 때문에 일직선처럼
보여 실제 surface 형태와 continuation 방향을 읽기 어려웠다. 실험 계약, H1/H2
construction, gate, metric은 변경하지 않고 advisor가 geometry를 직접 판독할 수
있도록 visualization/output만 보정했다.

## 변경 내용

`devtools/demo/meeting_occluded_surface_feasibility.py`에 local `u/v/n` display
frame을 추가했다. `u=0`은 visible termination, `u>0`은 continuation side이며,
retained/withheld point cloud에는 얇은 triangulated surface skin을, generated
continuation에는 실제 ruled grid surface를 그린다. 기존 fixed-view 비교는 같은
local camera/framing을 사용한다.

`controlled_pseudo_occlusion.png`는 4개 local geometry panel로 다시 생성된다.
`raw_fixed_view_overlay.png`에는 local 3D surface, `u-v` footprint, `u-n` side
profile을 함께 출력해 termination과 prediction/reference 차이를 바로 확인할 수
있다. NPZ/PLY geometry artifact와 기존 metric은 그대로 유지한다.

## 검증

- 기존 fixed negative-gate 실행 재생성: `verdict=C`, H1 `median=6.988h`, H2 `AMBIGUOUS`
- visualization은 H1 prediction을 cyan surface로 보이게 하지만 withheld target을
  construction에 전달하지 않는다.
- focused tests: `6 passed`
- canonical production code, Worklog 127–134, renderer, Candidate B, geometry,
  evaluation metric은 변경하지 않았다.

## 해석

새 그림은 실패를 숨기지 않는다. H1은 visible boundary에 붙은 cyan ruled surface가
green withheld surface의 실제 높이 분포를 따라가지 못하는 모습을 더 명확히 보여준다.
따라서 visualization 개선은 결과 해석 가능성을 높였을 뿐, feasibility verdict를
`C. CONTROLLED HOLDOUT FAILS`에서 바꾸지 않았다.
