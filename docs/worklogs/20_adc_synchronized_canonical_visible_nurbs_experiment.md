# Worklog 20 — ADC 동기화 canonical visible NURBS 실험

## 목표

기본 `initialize` 스케줄은 유지하면서, 구조적 ADC commit 이후의 현재 Gaussian만으로 visible NURBS를 재구축하는 격리 실험 경로를 검증했다. legacy/voxel/IDW/local split fallback은 복원하지 않았다.

## 구현

- `--visible_nurbs_update_schedule {initialize,adc_post_commit,disabled}`를 추가했다. 기본값은 기존 `initialize`다.
- `adc_post_commit`은 `visible_nurbs_state=unavailable_until_adc`, `patches=[]`, `coverage_semantics=reliable_core_only`로 Gaussian 학습을 시작한다.
- clone/split/prune가 발생한 ADC와 Gaussian optimizer step이 끝난 뒤에만 detached/no-grad canonical 재구축을 실행한다. opacity reset만으로는 실행하지 않는다.
- 종료 시 source fingerprint가 마지막 시도와 다를 때만 terminal 재구축한다.
- uncertain 및 occluded-chart-owned Gaussian을 constructor 입력에서 제외한다.
- clone/split/prune/checkpoint를 통과하는 단조 증가 `stable_gaussian_ids`를 추가했다.
- patch registry, 전체 membership, UV, ownership, trim mask를 임시 상태에서 계산한 뒤 성공 시 commit한다. review/failure/0 surface면 stale patch·surface optimizer·visible binding을 비우고 ADC Gaussian 상태는 유지한다.
- JSONL은 event/ADC count/fingerprint, constructor 진단, sample/full/opacity coverage, UV validity, spatial occupancy, runtime, CUDA allocated/reserved/peak memory를 기록한다.
- CUDA canonical reconstruction에서 contextual reliability의 CPU boolean mask를 CUDA neighbor index로 접근하던 오류를 수정하고 GPU 회귀를 추가했다.

## 자동 검증

- `tests/test_adc_synchronized_visible_nurbs.py`는 deferred 상태, one-way/RNG 불변성, stale 제거, stable ID, empty checkpoint, opacity-reset-only 비트리거, controlled multi-ADC 대조를 검증한다.
- controlled curved sheet에서는 materialized NURBS와 sample/full coverage `1.0`을 확인했다. 동일 seed/config의 `disabled`와 `adc_post_commit` Gaussian trainable tensor는 bitwise 동일했다.
- CUDA structural reliability 회귀와 ADC-synchronized 회귀: `14 passed`.
- 인접 회귀 묶음: `130 passed`.
- repository-wide pytest: `578 passed, 1 skipped, 1 warning, 8 subtests passed in 132.85s`.

## 실제 DATASET 다중-ADC 평가

설정: RTX 5080, `max_images=1`, `image_downscale=8`, `train_resolution_scale=4`, 6 iterations, ADC iteration `2/4/6`, `densify_grad_threshold=0.001`, `adc_max_gaussians=150000`, opacity/screen/world pruning 비활성화.

- Gaussian-only baseline은 세 구조적 ADC에서 split child `784`, `588`, `570`을 생성해 `138,766 -> 139,737` Gaussian이 됐다. 독립 baseline 재실행은 `139,713`으로 끝나 CUDA rasterizer/ADC가 이 조건에서 bitwise 재현적이지 않음을 확인했다.
- `adc_post_commit` 재실행은 세 event에서 split child `784`, `542`, `494`, 최종 `139,676` Gaussian이었다. baseline 대조와의 정확한 model equality는 이 실데이터 CUDA 비결정성 때문에 판정하지 않았고, CPU controlled 회귀의 bitwise equality를 one-way 계약 근거로 사용한다.
- 각 post-commit reconstruction은 기술 오류 없이 `no_admissible_region`으로 fail-closed했다. 모든 895 canonical sample이 ambiguous였고 `reliable_count=0`, `region_count=0`, `materialized_surface_count=0`이었다. 따라서 stale NURBS 없이 Gaussian 학습과 ADC는 계속됐다.
- event runtime은 `0.881`, `0.874`, `0.902`초(누적 `2.657`초)였고 peak CUDA allocated memory는 약 `273–280 MB`였다.
- commit 없는 cap sensitivity(`512/1024/2048`)도 각각 sample `265/465/895`에서 모두 `no_admissible_region`, reliable `0`, materialized `0`이었다. 이 실제 장면의 결과는 cap 하나의 우연이 아니다.

## 결론

**PARTIALLY_SUPPORTED**

ADC-synchronized lifecycle, atomic/fail-closed semantics, CUDA execution, diagnostics, stable ID, checkpoint, controlled one-way Gaussian invariance와 전체 pytest는 지원된다. 그러나 현재 실제 `DATASET`은 어떤 평가 cap에서도 reliable canonical region을 만들지 못해 visible NURBS product를 materialize하지 않는다. 즉 실험 스케줄은 지원되지만 이 복합 장면에서의 canonical NURBS product는 아직 지원되지 않는다.