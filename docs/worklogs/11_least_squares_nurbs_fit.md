# 11. 파라미터 보정을 포함한 최소제곱 NURBS 피팅 (Phase B)

날짜: 2026-07-11

## 작업

- `fit_torch_visible_surface_lsq()`를 구현했다. 기존 inverse-distance(IDW) 휴리스틱을 초기값(seed)으로 사용하고, UV 고정 시 control grid에 대해 선형인 정규화 최소제곱 시스템 `(B^T W B / N + λ_s L + λ_t I) P = B^T W X / N + λ_t P_seed`를 풀며, foot-point UV 재투영과 교대로 반복한다 (표준 parameter correction).
- 정규화는 두 항이다: `L`은 control grid 2차 차분 thin-plate 페널티(축별 kron 구성), Tikhonov 항은 원점이 아니라 seed grid에 anchoring하여 데이터가 없는 control point가 부드러운 seed를 따라가게 한다.
- patch fitting 시 voxel region density를 weighted LSQ의 point weight로 사용한다.
- 초기 Gaussian binding(`project_points_to_patches`)이 PCA 평면 투영에서 foot-point projection으로 교체됐다. local correction patch fitting도 같은 LSQ 경로를 쓴다.
- degree 노출: `TorchNURBSSurface`의 degree_v 기본값이 1(구간별 선형)에서 2로 올라갔고, `surface_degree_u/v`, `surface_fit_mode`(lsq/idw), `surface_fit_smoothness`, `surface_fit_tikhonov`, `surface_fit_rounds`, `surface_projection_iterations`가 config, 두 CLI(train.py, scripts/train_osn_gs_torch.py 공용 헬퍼), notebook Train 셀(OSN_SURFACE_* 변수)에 노출됐다.
- 정규화 기본값은 해석적 sine sheet 스윕으로 결정했다: `λ_s=1e-4, λ_t=1e-4` (1e-3은 과도한 평활화로 확인).

## 결과

- 해석적 sine sheet(1,200 point, 10x8 control, degree 2/2) 기준 정규화 RMS 표면 거리: IDW seed 0.0087 → LSQ 0.0028 (scene extent 대비, 약 3배 개선). 테스트 임계 0.005 고정.
- LSQ가 반환하는 최종 foot-point UV anchor의 평균 거리 비율 < 1%.
- control point 수가 point 수보다 많은 극단 케이스에서도 seed-anchored 정규화 덕분에 발산 없이 유한한 표면을 유지한다.

## 평가

Visible surface의 parametric representation이 이제 "초기화 수준 휴리스틱"이 아니라 실제 fitting 오차를 최소화하는 최소제곱 해다. rational weight가 전부 1인 fitting 시점에는 표면이 control point에 대해 정확히 선형이므로 LSQ가 exact하고, 학습 중 weight 최적화는 기존 gradient 경로가 그대로 담당한다. IDW는 seed와 CLI fallback(`--surface_fit_mode idw`)으로 유지된다.

## 검증

- 신규 fidelity 테스트 4개: LSQ가 IDW를 이기고 절대 임계 통과, foot-point UV anchor 품질, 과소 데이터 안정성, config→patch degree 전파.
- 전체 25개 테스트 통과. 두 CLI `--help`에서 신규 인자 등록 확인, notebook JSON 유효성 확인.

## 남은 위험

- LSQ는 fitting 시 rational weight=1을 가정한다. checkpoint 재개 후 weight가 1이 아닌 patch를 다시 fit하는 경로는 현재 없지만, 향후 local correction이 학습된 patch를 재fit하게 되면 weight를 고려해야 한다.
- 정규화 기본값은 합성 표면 기준이다. 실제 COLMAP 장면에서 `surface_fit_smoothness` 재검증이 필요하다.
- UV domain trimming(빈 UV 영역 마스킹)은 여전히 Phase C 범위로 남아 있다. 데이터 없는 영역의 control point는 seed-anchored로 안정화될 뿐 관측 표면을 의미하지 않는다.
