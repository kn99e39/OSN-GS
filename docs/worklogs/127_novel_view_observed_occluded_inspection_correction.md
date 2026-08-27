# Worklog 127: Novel-View Observed/Occluded Inspection Correction

상태: **완료**
범위: Worklog 123 global state를 보이는 방식의 correction. Candidate B 및 classification camera set은 변경하지 않음.

## 문제 교정

Worklog 126은 Worklog 125의 Gaussian-row/color-only 계약은 지켰지만, preview를 frozen 161 query camera 중 `DSC07957.JPG`로 렌더링했다. 이는 global `OCCLUDED` Gaussian을 점검할 view로 부적절하다. 그 Gaussian은 바로 그 capture view의 앞쪽 Observed Gaussian에 다시 가려질 가능성이 높으므로, 보이는 red는 leak 또는 view-local 표현의 혼동처럼 보인다.

따라서 Worklog 126 output은 historical export로 `output/confirmed/126_wl123_fixed_observed_occluded_gaussian_visualization/`에 보존하고, 현재 review output을 novel inspection camera로 교체했다.

## 고정된 분류와 분리된 review camera

다음은 그대로 유지했다.

- Worklog 123 frozen 161 train camera set
- canonical stored median depth 및 Candidate B `classify_view`
- frozen ANY-OBSERVED global aggregation
- Gaussian 중심 1,190,469개의 provenance 없는 arbitrary world-space query 의미
- state count: `OBSERVED` 798,304 / `OCCLUDED` 391,457 / `UNRESOLVED` 708
- epsilon, ULP band, view-count rule: 없음

rendering에만 frozen query camera set에 속하지 않는 deterministic outer-orbit camera 후보 16개(8 azimuth × elevation 12°/28°)를 만들었다. 중심은 scene median, 반경은 capture/scene radius보다 바깥으로 고정했다. 이 pose들은 state 판정에는 입력되지 않는다.

고정 후보 중 red-dominant visible pixel 수가 가장 큰 `NOVEL_OUTER_ORBIT_e12_a00`을 review camera로 선택했다. score는 218,096 pixel이며, 이 값은 red state가 새 관점에서 실제로 드러나는지를 위한 presentation-only 선택값이다. 분류 또는 수치 경계 정책에 사용하지 않았다.

## Worklog 125 visualization 계약 검증

`ORIGINAL_SCENE`와 `OBSERVED_OCCLUDED`는 새 novel camera를 공통으로 사용한다.

- Gaussian row 수: 둘 다 1,190,469
- position, scale, rotation, opacity: 동일 fingerprint
- 바뀐 값: `OBSERVED_OCCLUDED`의 Gaussian 색상만
- marker Gaussian: 0
- 조명/광원/shading/emissive 추가: 없음
- overlay: 없음

색상은 green=`OBSERVED`, red=`OCCLUDED`, gray=`UNRESOLVED`다. red는 frozen camera set에 대한 renderer-defined global state이지 physical first hit, hidden physical surface, ownership, trust, surface continuity를 뜻하지 않는다.

## 산출물

현재 결과:

- `output/127_wl123_novel_view_observed_occluded_visualization/ORIGINAL_SCENE/render.ppm`
- `output/127_wl123_novel_view_observed_occluded_visualization/OBSERVED_OCCLUDED/render.ppm`
- `output/127_wl123_novel_view_observed_occluded_visualization/preview_png/ORIGINAL_SCENE.png`
- `output/127_wl123_novel_view_observed_occluded_visualization/preview_png/OBSERVED_OCCLUDED.png`
- `wl123_fixed_observed_occluded_visualization_report.json`: 16 novel pose score, chosen pose, row/hash/state evidence

## 검증

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_wl123_fixed_observed_occluded_visualization.py
```

결과: `4 passed`.

실제 CUDA export: exit 0. qdepth diagnostic extension은 local cache로 실행됐다 (`ninja: no work to do`).

## 범위 밖

이 correction은 visualization camera만 바꾼다. Occluded Space를 새 Gaussian 또는 volumetric geometry로 만들지 않았고, Candidate B, provenance, topology, Occluded Surface/NURBS, Trust를 수정하지 않는다.