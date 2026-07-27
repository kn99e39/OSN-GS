# Worklog 93 — Boundary-first 복구 스윕 게이트

## 상태

- Boundary-first 격리 경로의 복구 판별 보강과 회귀 스윕 검증을 완료했다.
- 기존 benchmark dispatcher와 학습/렌더링 기본 경로는 변경하지 않았다.
- 기존 dispatcher 통합 준비도는 약 50%다. 격리된 curved annulus 경로는 완료됐지만, 일반 disk·비볼록·다중 루프 입력의 interior support 정책과 실제 품질 기준은 아직 남아 있다.

## 수행 내용

- 분리된 raw component 사이의 복구 edge가 다음 조건을 모두 만족할 때만 허용되도록 했다.
  - local nearest-neighbor spacing으로 정규화한 support witness 거리가 임계값 이내다.
  - PCA normal의 부호 불확실성을 고려해 absolute normal agreement를 사용한다.
  - AABB contact 차원이 1 이하라서 면적 겹침을 나타내지 않는다.
- 기본값은 `max_normalized_support_distance=2.0`, `minimum_abs_normal_agreement=0.9`, `max_aabb_contact_dimension=1`로 유지했다.
- `curved_annulus`는 분리 component를 하나의 복구 region으로 합쳐 outer/hole boundary를 다시 얻고, cyclic multi-patch Boundary-first 표면으로 진행한다.
- `close_parallel_sheets`는 경계처럼 보이는 근접 AABB라도 2차원 면적 접촉이므로 복구하지 않는다.

## 검증

- density/seed 조합: point count `400`, `600` × seed `0`, `1`, `2`.
  - `curved_annulus`: 모든 조합에서 복구 edge 허용 및 annulus multi-patch 경로 확인.
  - `close_parallel_sheets`: 모든 조합에서 복구 edge 거부 확인.
- 표적 테스트: `tests.test_boundary_component_recovery`, `tests.test_boundary_first_support_pipeline` — 8 passed.
- 공유 작업 트리 전체 pytest: `411 passed, 1 skipped, 1 warning`.
  - warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion 경고다.

## 판단

- 단순 거리만으로 raw component를 합치는 것은 저밀도 parallel sheet에서 오검출 위험이 있다.
- normal 부호는 PCA 방향에 따라 뒤집힐 수 있으므로 signed dot만 사용하면 curved annulus를 잘못 거부할 수 있다.
- 따라서 proximity, absolute normal agreement, 비면적 AABB contact를 함께 요구하는 현재 계약이 이 격리 경로의 최소 안전 조건이다.

## 남은 위험과 다음 단계

- disk/비볼록/다중 루프 입력에서 outer boundary와 interior support를 어떻게 생성할지 일반 정책이 필요하다.
- topology 성립뿐 아니라 control grid 품질, 경계 오차, curvature/normal 연속성을 포함한 benchmark 품질 게이트가 필요하다.
- 위 조건이 갖춰진 뒤 feature-gated dispatcher 연결을 검토한다. 기본 dispatcher 교체, Phase G 관련 경로, 학습/production integration은 이 Worklog 범위에 포함하지 않는다.