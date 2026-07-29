# Worklog 114 — Trimmed Component Jacobian Test-health 부채 해소

## 상태

구현·검증 완료. 반복되던 `tests/test_trimmed_component_fitter.py`의 두 실패를 해소했고, 전체 pytest는 green 상태로 복구됐다.

## 원인

flat-plane fixture는 control grid collapse가 아니고 `jacobian_min = 0.003688097`으로 양수였다. 하지만 기존 `degenerate_fraction`은 median Jacobian의 0.1% 이하인 sample을 모두 hard degeneracy로 분류했다. 24×24 sample 중 UV 경계의 단 하나가 이 상대 임계치에 걸려 `1 / 576 = 0.001736111`이 되었고, `degenerate_fraction == 0.0` 기대가 실패했다.

이는 true singularity와 relative low-area boundary warning을 같은 상태로 취급한 진단 의미론 부채였다.

## 변경

- `torch_trimmed_component_fitter._jacobian_metrics()`에서 hard `degenerate_fraction`을 scale-aware numerical singularity 기준(`max(1e-8, median × 1e-6)`)으로 정의했다.
- 기존의 상대 0.1% tail 신호는 `near_degenerate_fraction`으로 별도 보존했다.
- 두 threshold를 payload에 노출해 소비자가 판정 기준을 감사할 수 있게 했다.
- 완전히 붕괴한 v 방향 NURBS fixture를 추가해 hard degeneracy가 여전히 `1.0`으로 검출됨을 고정했다.

## 검증

- `tests/test_trimmed_component_fitter.py`: 8 passed.
- append/proposal/model 회귀: 68 passed.
- 전체 pytest: `536 passed, 1 skipped, 1 warning, 8 subtests passed in 120.70s`.

전체 실행 wrapper는 120초 제한 때문에 exit 124를 반환했지만, pytest 출력은 100% 진행 후 위의 정상 최종 결과를 기록했다. warning은 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion 기존 경고 1건이며 이번 Jacobian 변경과 무관하다.

## 남은 위험

`near_degenerate_fraction > 0`은 더 이상 hard failure가 아니지만 사라진 정보도 아니다. Boundary-first quality/review gate는 이 값을 별도 warning/review signal로 사용할 수 있으며, 실제 `degenerate_fraction`은 true singularity 검출 용도로 유지한다.