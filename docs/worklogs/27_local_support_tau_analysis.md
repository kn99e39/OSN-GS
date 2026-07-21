# 27. Local Support-Tau 분석

날짜: 2026-07-15

## 작업

synthetic constructor benchmark에 local-density support 분석을 추가했다.

- metrics.py가 각 generated sample에서 가장 가까운 input Gaussian의 local nearest-neighbour spacing을 사용해 local support tau를 기록한다.
- 기존 global extrapolation metric은 비교를 위해 유지했다.
- runner.py가 extrapolation_global과 extrapolation_local을 함께 출력한다.

## 결과

CPU benchmark 명령:

    .venv\Scripts\python.exe -m nurbs_constructor_benchmark --scenes density_gradient --points 600 --seed 0 --output C:\tmp\osn_gs_support_tau_check --skip-renderer-export

측정값:

- extrapolation_global: 0.659
- extrapolation_local: 0.094
- uncovered: 0.234
- chamfer_rms: 0.027456
- topology ARI: 1.000

## 평가

큰 global extrapolation 값은 주로 dense cluster가 만든 global median-spacing artifact다. local 비교는 UV trimming이 이 장면의 지배적인 실패 원인이 아님을 확인한다.

## 남은 작업

- sensitivity sweep을 수행한 뒤에만 기본/gating support tau를 결정한다.
- 요청된 occupancy artifact와 non-rectangular support scene을 추가한다.
- 비교 기간에는 global과 local 값을 함께 유지한다.
