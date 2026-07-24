# Worklog 73-A: anisotropy 격차 추가 조사 — 단일 원인 특정 실패, 확인된 사실만 기록

날짜: 2026-07-23

상태: **조사 완료, 명확한 단일 원인은 못 찾음. 추가 fix 없이 현재 상태(worklog 72) 유지.**

## 배경

Worklog 72의 `scene_extent`/`calibration_extent` 분리 수정 이후, Gaussian scale 크기(magnitude) 분포는 baseline과 거의 일치했지만 **anisotropy(max/min축 비율)는 여전히 큰 차이**가 있었다(OSN-GS 평균 4.80 vs baseline 15.10). 육안 crop 비교로도 baseline이 여전히 더 선명했다. 이 격차의 원인을 추가로 조사했다.

## 확인 1: iteration당 gradient 크기는 거의 동일함

`_scaling`/`_rotation`/`_xyz`의 `.grad.abs().mean()`을 두 프레임워크에 임시로 찍어 iteration 100~500 구간을 직접 비교했다(진단 코드는 측정 후 원복). 세 값 모두 매 iteration 자릿수까지 거의 동일했다. **"OSN-GS 쪽 학습 신호가 약하다"는 가설은 기각.**

## 확인 2: dense_extent(clone/split 임계값)는 이제 완전히 일치함

baseline의 `densify_and_prune`에 clone/split 후보 카운트 진단을 임시로 추가해 실측(측정 후 원복):

```
baseline: percent_dense=0.01 extent=4.922929286956787 dense_extent=0.04922929033637047
```

OSN-GS 로그의 `threshold`/내부 계산과 대조하면 **`dense_extent` 값이 소수점까지 baseline과 완전히 일치**한다 — worklog 72의 calibration_extent 수정이 의도대로 절대 임계값을 맞췄음을 재확인했다.

## 확인 3: clone/split 카운트는 비슷한 규모지만 OSN-GS가 소폭 더 많음

population이 비슷한 지점(~120만 개) 기준:
- baseline: clone_candidates≈77,400, split_candidates≈6,100 (합계 ≈83,500)
- OSN-GS: cloned≈87,200, split≈12,100 (합계 ≈99,300)

OSN-GS가 매 step 합계 기준 약 20% 더 많이 늘어난다 — 이게 iteration 3000 시점 최종 Gaussian 수 차이(210만 vs baseline 182만, +15%)와 대략 맞아떨어진다. 다만 이 차이 자체의 단일 원인은 특정하지 못했다.

## 참고: 이번 3000-iteration 비교에서는 양쪽 다 screen-size pruning이 꺼져 있음

`screen_size_prune_from_iter` 기본값이 3000이고 조건이 `iteration > 3000`(strict)이라, **3000 iteration짜리 실행에서는 baseline·OSN-GS 둘 다 screen-size pruning이 단 한 번도 발동하지 않는다.** 두 프레임워크에 동일하게 적용되는 제약이라 비교 자체를 왜곡하진 않지만, 이 실험의 Gaussian 개체수 자체가 "정상 稳定 상태"가 아니라 "아직 pruning 브레이크가 안 걸린 상태"라는 점은 감안해야 한다.

## 결론

- scale 크기(magnitude) 격차는 worklog 72로 해결됨 — 확실한 원인(camera vs point-cloud extent 기준 불일치)이 있었고 고쳤다.
- anisotropy 격차는 **단일하고 명확한 코드 버그로 환원되지 않는다.** gradient 크기도 같고 threshold도 이제 완전히 같은데도 남아있는 걸 보면, 더 미세한 "개체군 동역학"(예: 더 많은 중복 clone이 지역별 gradient를 쪼개서 개별 Gaussian의 flattening 수렴이 느려지는 효과) 문제로 추정되지만 확증하지 못했다.
- 이 이상 파려면 개별 Gaussian을 anisotropy 궤적까지 추적하는 훨씬 무거운 계측이 필요하다. 이번 세션에서는 여기서 멈추고, scale magnitude 개선(worklog 72)을 현재 상태로 채택한다.
