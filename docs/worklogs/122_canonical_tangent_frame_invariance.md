# Worklog 122 — Canonical Tangent-Frame Invariance Repair

## 목표

Gaussian-only `smooth_curved_sheet`의 rotation, uniform-scale, input-order shuffle, covariance sign-equivalent representation에서 local support-termination sector frame의 비정규성을 제거한다.

## 범위

- accepted local topology만 사용한 canonical region seed와 tangent-frame transport
- transported frame 기반 sector occupancy, cyclic missing-run, directed ordering 재생성
- clean plane·curved sheet invariance 및 negative control 회귀
- Worklog 121 orchestration을 유지한 통합 검증

## 금지 범위

fixture별 dispatcher, global world axis fallback, raw eigenvector sign, phase-alias/crease/parallel/rejected edge 기반 transport, production path 변경은 사용하지 않는다.
