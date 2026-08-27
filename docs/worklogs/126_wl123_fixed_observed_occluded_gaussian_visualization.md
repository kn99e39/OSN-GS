# Worklog 126: WL123 Fixed Observed/Occluded Gaussian Visualization

상태: **완료**
범위: Worklog 125의 고정 visualization 계약을 Worklog 123의 frozen world-space query contract에 적용한 실제 장면 export
코드 변경: 진단 export 1개와 focused contract test 1개. Candidate B·renderer·topology·production path는 변경하지 않음.

## 목적과 고정 조건

기존 Worklog 123의 `EVENT_IDENTITY_EFFECT`는 query marker Gaussian을 추가한 historical diagnostic이므로 Worklog 125 계약의 canonical visualization이 아니다. 이 배치는 그 historical 산출물을 재작성하지 않고 `output/confirmed/123_osn_gs_volumetric_frontier_query_contract/`에 보존했다.

새 export는 다음을 고정했다.

- checkpoint: `output/arch_2dgs_coverage_first_surface/2dgs_run1/30000/checkpoint.pt`
- camera set: Worklog 123과 같은 train 161개 (`DATASET`, `images_8`, `sparse/0`, `llffhold=8`)
- preview camera: name-sorted train camera `DSC07957.JPG`
- per-view: Worklog 123 frozen Candidate B의 canonical stored median depth 비교
- global: frozen ANY-OBSERVED aggregation
- 수치 정책: epsilon, ULP band, nextafter, view-count rule **없음**

## Gaussian 질의 의미

checkpoint의 1,190,469개 Gaussian **중심** 각각을 arbitrary world-space `x`로 전수 평가했다. Gaussian 중심은 renderer가 생성한 pixel median event 자체가 아니므로 event provenance를 임의로 부여하지 않았다. 즉 source-event identity 예외를 이 visualization에 사용하지 않았다.

따라서 색상은 각 중심의 frozen Candidate-B global query state일 뿐이며, physical first hit, surface ownership, trust, surface continuity, 또는 독립적인 hidden-surface evidence를 뜻하지 않는다.

## 결과

| global state | Gaussian 수 |
|---|---:|
| `OBSERVED` (green) | 798,304 |
| `OCCLUDED` (red) | 391,457 |
| `UNRESOLVED` (gray) | 708 |
| 합계 | 1,190,469 |

WL119/WL123의 frozen median representative union도 다시 785,937로 일치했다.

## Worklog 125 동일성 계약 검증

`ORIGINAL_SCENE`와 `OBSERVED_OCCLUDED`는 모두 정확히 1,190,469개 Gaussian row를 사용했다. 두 view에서 position, scale, rotation, opacity의 SHA-256 fingerprint가 export 전후 동일함을 확인했다.

- 추가 marker Gaussian: 0
- 조명/광원 추가: 없음
- overlay 추가: 없음
- geometry/opacity 변경: 없음
- `OBSERVED_OCCLUDED`에서 바뀐 값: Gaussian 색상뿐

원본 view는 checkpoint의 학습 SH appearance로 렌더링했고, state view는 고정 palette를 표현하기 위해 같은 Gaussian의 DC 색상만 state 색으로 대체하여 렌더링했다. 이는 state visualization에서 의도한 appearance-only override이며 원본 checkpoint를 저장하거나 수정하지 않는다.

## 산출물

`output/126_wl123_fixed_observed_occluded_gaussian_visualization/`:

- `ORIGINAL_SCENE/render.ppm` 및 `preview_png/ORIGINAL_SCENE.png`
- `OBSERVED_OCCLUDED/render.ppm` 및 `preview_png/OBSERVED_OCCLUDED.png`
- 각 view의 동일-row PLY와 Korean README
- `gaussian_center_global_states.npz`: Gaussian 중심별 global state와 aggregation accumulator
- `wl123_fixed_observed_occluded_visualization_report.json`: camera, state count, hash, output provenance

## 검증

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_wl123_fixed_observed_occluded_visualization.py
```

결과: `3 passed`.

실제 CUDA export는 `scripts/run_with_msvc_env.bat`로 실행했고, qdepth diagnostic extension은 기존 local cache를 사용했다(`ninja: no work to do`). 출력 중 `vswhere.exe` compiler-version probe 경고는 extension build probe의 환경 경고였으며 export는 exit 0으로 완료됐다.

## 남은 범위

이 배치는 visualization 계약의 구현만 닫는다. Occluded Space를 별도 volumetric representation으로 materialize하지 않았고, 그 공간이 존재하는 것처럼 marker를 추가하지 않았다. Candidate B, event-provenance query contract, visible topology, Occluded Surface/NURBS에는 후속 변경을 하지 않는다.