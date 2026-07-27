# Worklog 95 — Boundary-first 전체 scene review export

## 상태

- 격리된 Boundary-first support 경로를 renderer 형식으로 내보내는 전용 runner를 추가했다.
- 기존 `osn-gs benchmark --constructor boundary_first` dispatcher는 변경하지 않았다.
- review runner는 annulus 성공만을 성공 조건으로 사용하지 않는다. 요청된 모든 scene의 `constructed` 또는 `unsupported` 상태와 사유를 report에 남긴다.
- 기존 dispatcher 통합 준비도는 약 65%다.

## 산출물

- 실행 명령: `.venv\Scripts\python.exe -B -m nurbs_constructor_benchmark.boundary_first_support_runner --output artifacts\boundary_first_support_review_20260727`
- report: `artifacts/boundary_first_support_review_20260727/report.json`
- renderer 입력: `artifacts/boundary_first_support_review_20260727/NURBS_output/<scene>/`
  - 모든 scene: `point_cloud.ply`, `boundary_first_support_status.json`
  - constructed scene: 추가로 `nurbs_surface.json`
  - 모든 scene: `<scene>_gt/nurbs_surface.json` ground-truth overlay

## 전체 scene 관찰 결과

- constructed (현재 explicit paired-boundary 경로): `crescent`, `planar_hole`, `mild_curved_sheet`, `planar_hole_elliptical`, `planar_hole_density_gradient`, `curved_annulus`.
- unsupported (누락 근거를 숨기지 않음):
  - `plane`, `close_parallel_sheets`: `hole_area_ratio_too_small`
  - `crease`, `density_gradient`, `triangle`, `u_shape`: `interior_support_network_required`
  - `sine`, `elongated_plane`, `planar_hole_offcenter`: `multi_loop_pairing_deferred`
- 따라서 현재 구현은 annulus 전용이 아니지만, 모든 topology를 포괄하는 universal Boundary-first constructor도 아니다.

## 검증

- runner는 generated/ground-truth renderer directory, point cloud, NURBS JSON, status JSON, report JSON을 생성한다.
- runner와 quality/recovery/pipeline 표적 회귀: 11 passed.
- 이 Worklog 이후 전체 pytest는 별도 Agent가 작업 중인 append adapter 실패군이 해소된 뒤 다시 실행해야 한다. 직전 전체 결과는 `386 passed, 27 failed, 1 skipped, 1 warning`이며 실패 27개는 모두 append adapter 테스트다.

## 다음 단계

- outer boundary만 있는 disk/triangle/crease/density-gradient 계열의 interior support-network 정책을 설계·구현한다.
- multi-loop pairing의 correspondence 정책을 명시한다.
- constructed 6개 scene에 대해 원본 관측 경계 전체 대 surface 거리, normal, curvature fidelity gate를 추가한다.
- feature-gated dispatcher 연결은 위 일반화·품질 조건 이후에만 검토한다.