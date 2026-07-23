# Worklog 71: visible Gaussian 블러의 근본 원인 — scene_extent 기준(basis) 불일치

날짜: 2026-07-23

상태: **원인 확정, 수정은 보류(다음 작업으로 이월). 나중에 다듬어서 다룬다.**

## 배경

Worklog 70에서 opacity_lr을 baseline과 맞췄지만 visible Gaussian 블러는 그대로였다. iteration 3000에서 baseline과 OSN-GS의 point cloud를 직접 비교해 원인을 찾았다.

## Gaussian scale 분포 비교 (iteration 3000, 동일 카메라·동일 조건)

| | OSN-GS | baseline |
|---|---:|---:|
| Gaussian 개수 | 1,830,437 | 1,827,606 |
| scale 평균 | 0.01973 | 0.01573 |
| scale 중앙값 | 0.01044 | 0.00865 |
| scale p90 | 0.04993 | 0.03348 |

Gaussian 개수는 거의 같은데 OSN-GS scale이 전 구간에서 20~50% 더 크다. Gaussian이 크면 splat이 넓게 퍼져 디테일이 뭉개지므로, 이게 블러의 직접 원인이다.

## 근본 원인: `scene_extent`의 측정 기준이 baseline과 다름

- baseline `cameras_extent`(`gaussian-splatting/scene/dataset_readers.py::getNerfppNorm`): **카메라 위치**들의 평균 중심으로부터 최대 거리 × 1.1. 이 데이터셋에서 **4.94**.
- OSN-GS `_scene_extent()`(worklog 63에서 수정한 버전): **sparse point cloud**를 mean-center 후 90th percentile 거리 × 1.1. 이 데이터셋에서 **12.31**.

OSN-GS 로더가 읽은 카메라로 baseline과 동일한(카메라 위치 기반) 공식을 직접 계산해보면 **4.92**가 나와 baseline의 4.94와 사실상 일치한다. 즉 두 값이 다른 건 계산 실수가 아니라 **"무엇을 측정 기준으로 삼을지"가 애초에 다른 설계**다.

이 데이터셋은 정원을 걸어다니며 촬영한 walkthrough 씬이라, 카메라는 좁은 경로를 따라 움직이지만 관측된 point cloud(배경 나무·담장 등)는 카메라 경로보다 훨씬 멀리까지 뻗어 있다. 그래서 두 기준이 2.5배 차이 난다.

## 문제의 본질: 서로 다르게 설계된 로직을 같은 파이프라인에 결합함

- `scene_extent`는 `spatial_lr_scale`(xyz LR 배율), ADC의 clone/split 임계값(`percent_dense * scene_extent`), world-size pruning 임계값(`max_scale_ratio * scene_extent`)에 전부 쓰인다.
- 이 상수들(`percent_dense=0.01`, `xyz_lr=1.6e-4` 등)은 baseline이 **카메라 기반 extent**를 전제로 튜닝한 값이다.
- OSN-GS는 (SfM point cloud의 outlier에 강건하기 위해) **point cloud 기반 percentile extent**로 바꿨는데, 정작 그 값을 baseline과 동일한 상수들에 그대로 넣었다.
- 두 방식 모두 각자의 맥락에서는 합리적이다 — point cloud 기반은 outlier에 강건하고(camera 기반은 애초에 bundle adjustment로 노이즈가 적어서 이 강건성이 필요 없었을 뿐), camera 기반은 baseline의 다른 모든 상수와 스케일이 맞아떨어진다. 문제는 **기준이 다른 두 설계를 섞어서 하나의 파이프라인에 박아 넣은 것**이지, 어느 한쪽이 명백히 틀린 게 아니다.

## 다음 방향 (보류, 이번 세션에서는 미착수)

단순히 baseline 방식(카메라 기반)으로 되돌리는 게 가장 빠른 수정이지만, 사용자는 이 지점을 "잘 다듬으면 더 나은 결과가 나올 수도 있다"고 판단해 즉시 되돌리지 않고 향후 별도 작업으로 남겨두기로 했다. 예를 들어 point cloud 기반 extent를 계속 쓰되 percent_dense/xyz_lr 등 하위 상수들을 그 기준에 맞게 재보정하는 방향, 또는 두 추정치를 결합하는 방향 등을 고려할 수 있다. 이번 세션에서는 구현하지 않는다.
