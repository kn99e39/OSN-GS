# 8. Persistent Surface Lifecycle

날짜: 2026-07-10

## 작업

- adaptive voxel을 초기 initialization bootstrap으로 한정했다.
- 주기적인 global voxel, base-curve, NURBS 재구축을 제거했다.
- 기존 interval을 persistent NURBS patch 품질 검사 주기로 변경했다.
- patch별 Gaussian-to-NURBS residual을 scene extent로 정규화해 기록한다.
- 연속 실패 횟수, 최소 Gaussian 수, 최소 component 크기를 만족할 때만 failed patch를 local voxelization한다.
- 기존 patch는 유지하고 분리된 유의미 component만 새 NURBS patch로 추가한다.
- 새 patch parameter만 기존 Adam optimizer에 등록해 이전 control-point state를 보존한다.
- maintenance residual, patience counter, topology version을 checkpoint와 stream/output metadata에 포함했다.
- notebook Train 셀에 update interval과 local correction 설정을 노출했다.

## 결과

- state.voxel_regions의 객체와 initial topology는 training 동안 고정된다.
- NURBS control points와 weights는 기존 surface loss를 통해 매 iteration 계속 학습된다.
- 안정적인 patch에서는 maintenance가 구조나 binding을 변경하지 않는다.
- 문제가 지속되는 patch만 국소적으로 분할되며 ADC child의 parent binding 상속은 유지된다.
- global rebuild에서 발생하던 UV 재투영, patch ID 재할당, surface optimizer reset이 제거됐다.

## 평가

Voxel은 NURBS의 경쟁 표현이 아니라 초기 domain partition 및 드문 topology correction 도구가 됐다. NURBS가 persistent canonical geometry로 유지되므로 architecture의 surface-centric 원칙과 실제 training lifecycle이 더 일치한다.

## 검증

- 전체 Torch smoke/regression test 15개 통과.
- local correction 단독 회귀 테스트 통과.
- 안정 maintenance에서 voxel/patch identity 보존 확인.
- 새 patch optimizer 등록 후 기존 Adam step 보존 확인.

## 남은 위험

- 현재 품질 신호는 geometric Gaussian-to-surface residual이다. image residual을 patch basis weight로 backtracking하는 고도화가 남아 있다.
- local correction은 split만 수행하며 patch merge와 orphan patch 정리는 아직 없다.
- initial voxel snapshot은 export에 유지되지만 local correction cell graph 자체는 별도 전역 voxel payload로 병합하지 않는다.
