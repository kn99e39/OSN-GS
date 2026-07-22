# 6. ADC 동등성 확장

## 작업 내용

공식 Graphdeco `gaussian-splatting`의 `train.py`와 `scene/gaussian_model.py`를 기준으로 OSN-GS ADC를 다시 대조했다.

- Densification 경계를 원본과 동일한 `iteration > from`, `iteration < until`로 수정했다.
- Screen/world oversized prune를 opacity-reset 이후에만 함께 활성화한다.
- Opacity reset을 ADC interval 블록에서 분리해 독립 schedule로 실행한다.
- Gaussian optimizer step을 backward -> ADC/prune -> optimizer 순서로 이동했다.
- Shape 변경 시 기존 Gaussian gradient row를 보존하고 새 child row는 zero gradient로 시작한다.
- Position LR에 scene spatial scale을 적용했다.
- Opacity LR 기본값을 원본의 0.05로 맞췄다.
- `percent_dense`, opacity threshold, split sample 수, screen/world size 기준을 CLI와 notebook에 노출했다.
- OSN-GS의 UV, patch ID, confidence metadata는 clone/split에서 계속 상속한다.
- Uncertain-to-certain promotion은 계속 금지한다.

## 결과

- 기존 강제 quantile fallback 없이 원본 gradient threshold를 따른다.
- 초기 iteration의 공격적인 world-scale prune가 제거됐다.
- ADC iteration에서도 backward gradient와 Adam row-state가 유지된다.
- ADC 관련 회귀 테스트를 포함해 전체 10개 테스트가 통과했다.
- Notebook JSON과 Python syntax가 통과했다.

## 평가

Certain Gaussian ADC의 핵심 clone/split/prune schedule과 optimizer lifecycle은 원본 3DGS에 상당히 근접했다. OSN-GS가 의도적으로 추가한 Gaussian cap, UV/patch binding, uncertain prune-only 정책은 유지한다. CUDA sparse Adam, exposure optimizer, white-background 전용 초기 opacity reset은 현재 OSN-GS 학습 계약에 없으므로 포함하지 않았다.
