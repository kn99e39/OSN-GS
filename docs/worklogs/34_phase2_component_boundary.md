# 29. Phase 2 컴포넌트 단위 경계 추출

날짜: 2026-07-20

## 수행 내용

- Final Boundary governing plan을 기준으로 Phase 1 component builder와 진행 중인 component-level boundary extractor를 점검했다.
- 필수 6개 장면의 Phase 1 benchmark와 Phase 2 boundary benchmark를 실행했다.
- 64x64 support resolution에서 significant loop 기본 임계값을 4셀에서 20셀로 조정했다. 이 값은 mask 영역을 채우거나 삭제하지 않으며, loop를 topology hole이 아닌 tiny diagnostic artifact로 분류하기 위한 값이다.

## 결과

- plane, sine, planar_hole, crease, close_parallel_sheets에서 예상 component 수와 assignment ARI 1.0을 복원했다. density_gradient는 component 하나를 유지했지만 3.7%의 sample이 inactive/unassigned 상태로 남았다.
- planar_hole의 significant hole(262셀)은 보존했고, density_gradient의 17셀 gap은 tiny artifact로 분류하여 significant hole count를 0으로 유지했다.
- 위 구분을 검증하는 regression test를 추가했다.

## 평가

Phase 2 결과물은 benchmark 전용이며 legacy 또는 voxel_patch_stage1 constructor, trainer, ADC 동작을 변경하지 않는다. artifact threshold는 diagnostics에 표시되고 component-report CLI에서 설정할 수 있으며, morphology 연산이나 hole count 강제 규칙이 아니다.

## 남은 위험

- 20셀 기본값은 64x64 benchmark mask에 맞춰 조정된 값이다. support resolution이 크게 바뀌면 재조정 또는 재검증이 필요하다.
- density_gradient는 여전히 3.7%의 sample이 inactive leaf에 속하므로 실제 데이터에 대한 support coverage eligibility 문제를 완전히 해결한 것은 아니다.
- 이 보고서는 Phase 3 진행을 승인하지 않는다. governing plan에 따라 Phase 2 결과 검토 후 사용자의 승인을 받아야 한다.
