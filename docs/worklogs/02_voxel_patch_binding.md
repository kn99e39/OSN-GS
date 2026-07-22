# 2. 복셀 경계 패치 결속

## 작업 내용

- Voxel 6-neighbor graph에서 normal angle threshold를 넘는 edge를 끊고 connected patch ID를 생성했다.
- Region/point patch ID를 voxel 결과와 Gaussian persistent binding에 저장했다.
- Base curve fitting을 전역 PCA 한 번이 아니라 patch별 local PCA curve로 분리했다.
- Patch별 visible NURBS surface를 생성하고 Gaussian별 local UV로 surface loss를 평가한다.
- 전체 NURBS control point budget을 65,536으로 제한해 scale 또는 patch 수 증가 시 자동 축소한다.

## 결과

- 단일 대표 surface는 기존 stream/export 호환용으로 유지된다.
- 실제 optimization은 모든 patch control grid와 weight를 대상으로 수행된다.
- CPU pipeline smoke test가 통과했다.

## 평가

Normal boundary가 이제 단순 visualization flag가 아니라 curve, NURBS, Gaussian binding을 나누는 topology 신호다. Connected-component 계산은 rebuild 시 작은 voxel graph를 CPU에서 처리하므로 장기적으로 GPU graph component 구현 또는 캐시가 필요하다.
