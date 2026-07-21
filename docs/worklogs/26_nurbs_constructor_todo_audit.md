# 26. NURBS Constructor TODO 감사

날짜: 2026-07-15

## 검토 근거

이전 construction worklog 01~20과 현재 benchmark code를 검토했다.

- nurbs_constructor_benchmark/metrics.py
- diagnostics.py
- scenes.py
- runner.py

## 확인 후 TODO에서 제거한 항목

- synthetic rectangular plane, sine, density-gradient scene.
- input Gaussian-to-fitted-surface RMS.
- ground-truth accuracy, completeness, bidirectional Chamfer RMS.
- Jacobian area-degeneracy 및 fold-over diagnostics.
- expected/generated patch-count와 label-ARI metric.

위 항목은 production constructor benchmark에 구현되어 있으며 worklog 근거도 기록되어 있다. metric ownership은 fitting accuracy, support, topology로 분리돼 있다.

## 계속 열려 있는 항목

- elongated plane과 mild-curved-sheet coverage.
- Jacobian condition number 및 seed/config stability sweep.
- local-density-adaptive support threshold, occupancy artifact, non-rectangular support scene/metric, support-mask lifecycle refresh.
- chartability, topology boundary scoring, multi-patch 개선.

이 감사는 NURBS fitting이나 training code를 변경하지 않았다.
