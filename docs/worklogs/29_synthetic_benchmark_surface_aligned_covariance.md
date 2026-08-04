# Worklog 29 — 벤치마크 Synthetic Gaussian Scene에 Surface-Aligned Covariance 변형 추가

## 배경

세션 중 실제 학습 데이터(colab_train_3dgs.ipynb, 3k~10k iteration)에서 `osn-gs benchmark`/canonical construction 병목을 재점검하다가, Worklog 25~28의 "학습이 더 진행되면 genuine termination candidate가 늘어날 것"이라는 가설이 이 real DATASET 구간(3k~5k iter, 1.6M~1.9M gaussian)에서는 성립하지 않음을 확인했다(`reliable_count`가 0~1에서 벗어나지 않음, 10k iter는 covariance eigen-decomposition NaN으로 아예 크래시). 이 발견과는 별개로, 지금 `osn-gs benchmark`가 쓰는 synthetic box/cylinder/sphere scene의 Gaussian covariance에 "surface-aligned" 변형을 추가해달라는 별도 요청을 받아 진행했다.

## 기존 상태

`nurbs_constructor_benchmark/scenes.py`의 `_baseline_like_surface_covariance`는 이미 tangent-frame 정렬(local z축이 정확히 analytic surface normal)이지만, `output/graphdeco_ab_3k` 실측 baseline 3DGS 통계를 흉내내기 위해 anisotropy ratio(median 5.44, p25/p75 3.14/10.09, 1.5~32x 범위)와 tangent 크기에 log-normal noise를 추가로 섞는다. 사용자 확인 결과, 원하는 "Surface Align" 버전은 이 noise를 제거하고 모든 Gaussian을 균일하게 얇은 tangent-plane disk로 강제하는 "이상적인" 대비군이었다(AskUserQuestion으로 3가지 후보 중 확정).

## 구현

- `scenes.py`: `_surface_aligned_covariance(points, normals)`를 추가. `_baseline_like_surface_covariance`와 동일한 `_tangent_frame_quaternion` rotation을 쓰되, ratio를 고정 상수 `_SURFACE_ALIGNED_RATIO = 12.0`(baseline p75 근처)으로 두고 tangent_major=tangent_minor=local spacing으로 결정론적으로 설정 — per-point noise 없음.
- `_make_covariance(covariance_mode, points, normals, generator)` 디스패처를 추가해 `"baseline_noisy"`(기존 기본값) / `"surface_aligned"` 두 경로를 선택.
- `make_scene()` 및 `_make_box_scene`/`_make_cylinder_scene`/`_make_sphere_scene`, legacy height-field 분기에 `covariance_mode: str = "baseline_noisy"` 파라미터를 추가 — 새 파라미터는 마지막 위치이자 기본값이 기존 동작과 동일하므로, 리포 전체 18개 `make_scene(...)` 호출부(테스트/devtools 스크립트 포함, 전부 4개 이하 위치 인자 또는 keyword 사용)는 수정 없이 그대로 동작.
- `runner.py`: `build_parser()`에 `-surf`/`--surf`(store_true) 플래그 추가, `main()`에서 `covariance_mode = "surface_aligned" if args.surf else "baseline_noisy"`를 계산해 canonical/boundary_first 두 경로의 `make_scene(...)` 호출에 모두 전달.

## 검증

- `_baseline_like_surface_covariance` vs `_surface_aligned_covariance` 직접 비교: box 60점 기준 anisotropy ratio가 baseline은 1.5~32x(median 4.4)로 흩어지고, surf는 거의 정확히 12.0으로 고정됨(단, spacing이 극도로 작은 경우 `clamp_min(5e-5)` floor에 걸려 ratio가 12보다 작아지는 것은 baseline과 동일한 기존 floor 동작).
- `python -m osn_gs.cli benchmark -surf ...` / `--surf ...` 둘 다 정상 동작 확인.
- `--constructor boundary_first`는 covariance를 전혀 소비하지 않는 경로(raw point/normal만 사용, `boundary_first.py` 확인)이므로 `-surf` 유무와 관계없이 출력이 동일한 것이 정상.
- `--constructor canonical`(covariance_scales/rotations를 실제로 pipeline에 전달)은 `-surf` 유무와 무관하게 box에서 `no_admissible_region`으로 실패 — 이는 이 작업과 무관한 기존에 알려진 결함(closed multi-face topology에서 canonical 기본 constructor가 하드 실패, memory `project_osn_gs_benchmark_volumetric_dataset` 기록)이며 이번 변경으로 새로 생긴 것이 아님을 baseline/-surf 양쪽에서 동일하게 재현해 확인했다.
- `pytest tests/test_synthetic_gaussian_dataset.py tests/test_gt_nurbs.py tests/test_boundary_component_recovery.py` 14개 전부 통과.

## 범위 밖

canonical constructor의 `no_admissible_region` 자체를 고치는 것, real DATASET의 `reliable_count` 붕괴/`B_candidate_linking_failed`/10k iter NaN eigen-decomposition 크래시 조사 — 전부 이번 작업과 별개이며 손대지 않았다.
