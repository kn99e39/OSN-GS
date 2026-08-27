# Worklog 125: Fixed Gaussian Visualization Contract

상태: **완료**
범위: Gaussian visualization의 고정 산출물·색상·동일성 계약 문서화
코드 변경: 없음 (문서와 색인만 변경)

## 결정

Gaussian visualization은 시각적으로 보기 좋은 별도 장면을 만드는 작업이 아니다. **해당 visualization 환경에 이미 존재하는 Gaussian들의 색상만 바꾸어 상태를 표시하는 작업**이다. 조명, 광원, shading, emissive 효과, 추가 marker Gaussian, geometry 변형은 사용하지 않는다.

## 모든 visualization에 고정적으로 포함할 두 산출물

모든 Gaussian visualization batch는 아래 두 결과를 같은 checkpoint·iteration·camera·resolution·background·renderer 조건으로 반드시 만든다.

| 고정 결과 | 내용 |
|---|---|
| `Original Scene` (`ORIGINAL_SCENE/render.ppm`) | 해당 환경의 동일 Gaussian set을 원래 학습된 색상/SH, position, scale, rotation, opacity 그대로 렌더링한다. 색상 덮어쓰기, Gaussian 추가, 조명 추가를 금지한다. |
| `Observed/Occluded` (`OBSERVED_OCCLUDED/render.ppm`) | `Original Scene`과 **동일한 Gaussian row와 geometry**를 사용하고 색상만 상태별로 덮어쓴다. Gaussian이 Observed Space에 있으면 Observed 색, Occluded Space에 있으면 Occluded 색을 사용한다. |

고정 색상표는 다음과 같다.

- `OBSERVED`: green `(0.10, 0.85, 0.35)`
- `OCCLUDED`: red `(0.92, 0.18, 0.18)`
- `UNRESOLVED` 또는 상태 미결정: gray `(0.60, 0.60, 0.62)`

상태를 계산하지 못한 Gaussian을 임의로 Observed 또는 Occluded로 칠하지 않는다. 상태가 `UNRESOLVED`이면 반드시 gray로 남기고 보고서에 그 수를 기록한다.

## Occluded Space 표현 규칙

현재 validated Occluded Gaussian/volumetric representation이 없는 경우, Observed/Occluded 결과에서 Occluded 공간을 marker로 발명하지 않는다. 즉, **추가 Gaussian을 만들어 Occluded Space가 있는 것처럼 보이게 하지 않는다.** 별도의 validated volumetric representation이 실제로 존재하고 승인된 batch에서만 `OCCLUDED_VOLUMETRIC` 산출물을 추가할 수 있다. 그 경우에도 고정 두 결과를 대체하지 않고 추가한다.

## 고정 실행·비교 규칙

- 두 결과는 같은 view, camera, checkpoint, iteration, resolution, background, renderer와 같은 Gaussian 개수를 사용한다.
- `Observed/Occluded`에서는 position, covariance/scale, rotation, opacity, ordering, camera, renderer를 변경하지 않는다. 색상 입력만 상태 map으로 교체한다.
- 각 output directory에는 색상 의미와 Gaussian-row 동일성 검증을 담은 README를 둔다.
- 추가 진단 그림(`frontier`, `topology`, `identity`, `residual` 등)은 허용하지만, 위 두 결과를 항상 함께 포함한 뒤에만 추가할 수 있다. 추가 그림마다 별도 legend와 상태 정의를 기록한다.
- 배치마다 임의로 다른 항목 조합·색상표·marker 정책을 선택하지 않는다.

## WL123 출력에 대한 교정

이전 `EVENT_IDENTITY_EFFECT/render.ppm`은 canonical visualization이 아니다. 해당 진단 스크립트는 장면 Gaussian을 어두운 색으로 덮고, query 위치에 높은 opacity와 별도 색상의 marker Gaussian을 추가했다. 따라서 화면의 녹색 번짐은 광원 추가가 아니라 **합성 marker Gaussian의 splat/alpha compositing 결과**였다. 이 historical output은 재작성하지 않지만, 앞으로 `Original Scene` 또는 `Observed/Occluded`의 기준 산출물로 사용하지 않는다.

이번 문서는 visualization contract만 확정한다. 기존 renderer와 production model은 변경하지 않았고, 다음 visualization implementation batch가 이 계약을 구현해야 한다.
