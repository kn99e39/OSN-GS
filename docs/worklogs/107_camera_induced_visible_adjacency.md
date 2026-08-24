# Worklog 107 — Camera-Induced Visible Adjacency

## 상태

**완료 — 실측 있음. WL103/WL106 대비 뚜렷이 개선된, 그러나 최종 architecture 채택을 자동으로 주장하지 않는 결과.** WL106이 "3D candidate edge에 카메라가 승인/모순 판정"이라는 아키텍처를 반증한 것을 받아, 이번 배치는 방향을 완전히 뒤집었다: **카메라 자신의 렌더링된 표면이 직접 surfel 인접성을 생성**하고(image-space adjacency), 3D candidate graph는 순수 국소성 제약으로만 쓰이며(§6), 전역 topology는 여러 뷰의 양성 관측의 **합집합**(union)이다 — 한 뷰의 occlusion은 다른 뷰가 관측한 양성 관계를 절대 부정하지 않는다(§9-10, 이 배치의 핵심 의미론적 교정). **실측: singleton 비율 63.4%(WL103)/83.8%(WL106) → 45.0%로 대폭 개선, 최대 component 비율 10.5%/2.9% → 36.8%.** 시각 검토 결과 이 36.8%는 무분별한 percolation이 아니라 **패티오 바닥 자체가 이루는 하나의 정당한 넓은 연속 표면**으로 보이며, **테이블은 여전히 패티오와 뚜렷이 분리된 단일 component**로 남아 있다(픽셀 색상 직접 대조로 확인). Hedge/배경은 완전히 해결되지는 않았지만(대부분 파편화 유지) 극단적 단절에서는 벗어났다.

## 아키텍처

```
매 학습 뷰:
    renderer 공식 median-depth(T>0.5) representative (WL105 backward-pass 진단을 넘어,
    이번 배치는 실제 CUDA 진단 빌드로 픽셀별 GLOBAL surfel ID를 노출)
        -> 4-connectivity(우측/아래) raster 이웃에서 서로 다른 representative ID 쌍 = 이미지-공간 양성 관계
    -> [필터 1] 기존 3D candidate graph(WL96-106, 수정 없음)에 실제로 존재하는 spatial edge인가 (국소성 제약, topology 생성 아님)
    -> [필터 2] 2차 기하 게이트(WL98/102/103/106 재사용: shape-operator residual + 수정된 signed positional-offset)
    -> 전역 그래프 = 모든 뷰에 걸친 위 결과의 **합집합** (퍼센트 threshold 없음, occlusion이 다른 뷰의 양성 관계를 부정하지 않음)
    -> connected components
```

Worklog 103(`torch_positive_visible_adjacency.py`), 104, 105, 106(`torch_renderer_grounded_visible_adjacency.py`)은 전부 한 줄도 수정하지 않았고 그대로 재현 가능하게 남아 있다.

## 1-4. Renderer-native pixel surface-representative 시맨틱과 진단 계측

벤더된(수정 없음) `forward.cu`의 `renderCUDA`를 직접 읽어 확인: 이미 **`median_contributor`**(러닝 transmittance `T`가 0.5를 넘는 순간의 per-tile contributor 위치, `depth_median`/`surf_depth`가 실제로 참조하는 바로 그 surfel)를 계산하고 있지만, 이 값도, 이를 global surfel index로 바꾸는 데 필요한 `point_list`/`ranges`도 Python에 반환되지 않는다(`rasterize_points.cu`의 `RasterizeGaussiansCUDA` 반환 튜플에 없음 — 직접 확인). WL105의 backward-gradient 트릭은 여기서는 "이 뷰에서 최소 1픽셀의 representative였는가"라는 **집계된 boolean**만 주는데, 이번 배치는 픽셀별 이미지-격자 인접성 생성(§5)에 실제 (H,W) ID 맵이 필요해 이 트릭으로는 부족했다.

Directive §3의 지시대로 **진단 전용 CUDA 빌드**를 만들었다(`osn_gs/render/vendor/diff_surfel_rasterization_diag/`, 벤더 원본의 형제 디렉터리 — 원본은 전혀 건드리지 않음). 변경은 정확히 한 곳: `median_contributor`를 설정하는 바로 그 지점(`if (T>0.5)`)에서 이미 레지스터에 있는 `collected_id[j]`(global surfel index)를 함께 캡처해 새 출력 버퍼 `out_representative_id`(H,W, int32, -1=없음)에 쓴다. `forward.cu`/`rasterizer_impl.cu`/`rasterize_points.cu`/`rasterizer.h`/`forward.h`에 각각 한 줄~몇 줄씩만 추가했고(전부 `OSN-GS DIAGNOSTIC ADDITION` 주석으로 표시), `backward.cu`는 전혀 건드리지 않았다(이 진단은 gradient가 필요 없음). 별도 패키지(`diff_surfel_rasterization_diag`)로 pip 빌드해(`scripts/build_surfel_extension_diag.bat`) 설치했고, `osn_gs/render/torch_surfel_representative_diagnostics.py`가 `_C.rasterize_gaussians`를 `torch.no_grad()` 안에서 직접 호출한다(canonical autograd wrapper 우회, gradient 전혀 불필요).

**Rendering 불변성 실측 확인**: 동일 입력에 대해 canonical `OSNSurfelRasterizer.render()`와 이 진단 빌드의 render 출력이 `torch.testing.assert_close`로 완전히 일치함을 확인(`test_diagnostic_rendering_matches_canonical`). 진짜로 가려진 surfel(3개의 완전 불투명 근접 surfel 뒤)이 어떤 픽셀에서도 representative가 되지 않음도 실측 확인(`test_occluded_surfel_never_becomes_the_representative`).

## 5. 이미지-공간 인접성 생성

`accumulate_image_space_pairs`: 매 뷰마다 (H,W) representative 맵에서 우측/아래 4-connectivity 이웃 중 둘 다 유효(≥0)하고 ID가 다른 쌍을 전부 수집, 뷰 전체에 걸쳐 합집합(directive §9) — pixel-radius 파라미터 없음, depth 비교 없음, 두 surfel CENTER를 지나는 가정된 표면 없음(directive Central Intent A).

## 6-7. 3D 국소성 필터와 corridor 폐기

기존 candidate graph를 그대로 재사용해 국소성 제약으로만 썼다(§6) — WL103/106의 center-depth RANGE corridor는 이 배치에 전혀 존재하지 않는다(§7 준수, 코드에 `min/max(center_depth)` 계산 자체가 없음).

## 8. 2차 기하 게이트

WL98/102/103/106의 shape-operator residual + 수정된 signed positional-offset 공식을 코드 그대로 재사용, 모든 뷰의 국소성-통과 쌍 합집합에 단 한 번만 적용.

## 9-10. 다중 뷰 합집합과 occluded-gap 의미론

퍼센트/과반 threshold 전혀 없음. 한 뷰가 못 본 관계는 그냥 증거 부재이지 모순이 아니다 — **`UNRESOLVED_OBSERVATION_CONFLICT` 상태 자체가 이 모듈에 아예 존재하지 않는다**(구조적으로 만들 수 없음, 각 뷰는 자기 자신의 관계만 기여). 진짜로 가려진 gap은 어떤 뷰에서도 이미지-인접 관계가 생성되지 않으므로 자연히 분리 유지(`test_globally_occluded_gap_remains_disconnected`로 검증).

## 11. WL106 대조

WL106을 재실행하지 않고 커밋된 리포트 수치를 그대로 인용했다(동일 체크포인트/카메라, 결정론적, 이미 검증됨).

| | WL106 (pairwise camera approval) | 이번 배치 (camera-induced) |
|---|---|---|
| component 수 | 1,004,080 | **559,989** |
| 최대 component 비율 | 2.91% | **36.77%** |
| singleton 비율 | 83.8% | **45.02%** |
| mean component 크기 | 1.19 | 2.13 |

## 12. Representative coverage

| | 개수 | 비율 |
|---|---|---|
| 전체 | 1,190,469 | 100% |
| renderer-contributing (WL105) | 1,135,884 | 95.4% |
| renderer surface-representative(≥1 뷰) | 785,937 | 66.0% |
| contributing이지만 한 번도 representative 아님 | 385,998 | 34.0% (contributing 대비) |

representative가 아닌 contributing surfel은 이번 배치에서 topology에 강제로 편입되지 않았다 — 여전히 "renderer-contributing primitive"로만 회계 처리된다(§17 준수).

## 13. Camera-induced edge accounting

| | 개수 |
|---|---|
| raw image-space distinct pair (뷰 합산 전, dedup 후) | 9,295,205 |
| 3D 국소성으로 기각 | 7,538,912 (81.1%) |
| 기하 게이트로 기각 (discontinuity 126,351 + positional 268,178) | 394,529 |
| 최종 positive edge | **1,361,764** |
| edge당 지지 뷰 수 (median/mean/max) | 4 / 8.65 / 150 |

## 14. 전역 component 회계

559,989개, 최대 36.77%(437,751개), singleton 45.02%(535,910개), mean 크기 2.13. Coverage identity 유지(1,190,469개 전량).

## 15. WL106 vs 이번 배치 비교 — 핵심 질문

- Singleton 파편화가 실질적으로 감소했는가? **예 (83.8%→45.0%)**
- Scene-wide percolation이 없는가? **최대 36.8%는 WL96-102 시절의 문제적 percolation(70-92%, 테이블·패티오·hedge 뒤섞임)과 질적으로 다르다 — 아래 §16 참조.**
- 테이블이 구조적으로 일관된 하나의 component로 남는가? **예.**
- 테이블이 패티오와 분리되는가? **예.**
- Hedge/배경이 의미 있는 연결성을 회복하는가? **부분적으로 — 극단적 단절(WL106)에서는 벗어났으나 여전히 대부분 파편화.**

## 16. 실제 scene 리뷰

`CAMERA_INDUCED_VISIBLE_COMPONENTS`에서 직접 픽셀 색상을 대조했다: 테이블 중심 RGB `(63,82,200)` vs 패티오 바닥 두 지점 `(154,71,68)`/`(149,67,68)` — **명확히 다른 component**. Hedge/배경 세 지점은 서로도, 테이블·패티오와도 색이 전부 다름(`(126,115,86)`/`(36,116,162)`/`(129,78,108)`) — hedge는 여전히 여러 개의 작은 component로 나뉘어 있다. 즉 **36.8%짜리 최대 component는 패티오 바닥 자체가 이루는 정당한 하나의 넓은 연속 표면으로 보이며, 테이블이나 hedge를 끌어들이지 않는다.** `NON_REPRESENTATIVE_CONTRIBUTOR_VIEW`는 hedge/배경에 집중된 옅은 주황 스페클을 보여준다(대부분 dark) — 대응하는 non-representative contributor가 대부분 hedge의 얇고 복잡한 구조에 몰려 있음을 시각적으로 확인. `GEOMETRIC_REJECTION_VIEW`는 테이블·패티오·hedge 전역에 걸쳐 상당한 마젠타를 보여 2차 기하 게이트가 장면 전반에서 실질적으로 작동하고 있음을 보여준다.

## 17. 곡면 표면 동작

`test_curved_visible_surface_remains_connected`(합성 fixture, 12-point 곡선 밴드, 전체 90도 법선 회전)로 확인 — 법선 회전만으로는 cut되지 않고 1개 component 유지.

## 18. Primitive/topology 분리

전체 1,190,469개 전량 보존. renderer-contributing(95.4%)과 renderer-representative(66.0%)는 서로 다른, 명시적으로 구분된 계약이며, "component size >= 2"를 visibility 증거로 쓰지 않았다(directive §15 — WL104의 그 규칙은 역사적 diagnostic accounting으로만 남기고 이번 배치의 canonical 규칙으로 재사용하지 않음).

## 19. Review export

`output/osn_gs_camera_induced_visible_adjacency/{ORIGINAL_2DGS_SCENE, RENDERER_SURFACE_REPRESENTATIVE_VIEW, CAMERA_INDUCED_PER_VIEW_ADJACENCY, CAMERA_INDUCED_GLOBAL_ADJACENCY, CAMERA_INDUCED_VISIBLE_COMPONENTS, GEOMETRIC_REJECTION_VIEW, NON_REPRESENTATIVE_CONTRIBUTOR_VIEW, WL106_PAIRWISE_BASELINE(WL106 커밋 export에서 복사)}/`, PNG: `preview_png/`, 전체 리포트: `camera_induced_visible_adjacency_report.json`.

## 20. 테스트

- `tests/test_surfel_representative_diagnostics.py` (6 tests, 실제 CUDA+진단 빌드로 실행): canonical 렌더 불변, 단일 surfel 자기 자신의 representative, 미커버 픽셀 -1, 가려진 surfel 오탐 없음, 결정론, primitive tensor 불변.
- `tests/test_camera_induced_visible_adjacency.py` (9 tests): 이미지-인접 representative가 실제로 edge 생성, occlusion이 다른 뷰의 양성을 부정 안 함, 전역 occluded gap 분리 유지, 3D 비국소 쌍 기각, 곡면 연결 유지, 인접 시트 분리 유지, 결정론적 다중뷰 합집합, 뷰 없음 시 무연결, coverage identity.
- 전체 regression: WL106의 1239 + 신규 15 = 1254 passed 1 skipped(CUDA 진단 빌드 필요 테스트 포함, 이 머신에서 실제 실행됨; 실행 결과는 커밋 메시지에 기록).

## 21. 결론

**뚜렷한 개선, 그러나 architecture 최종 채택을 자동으로 선언하지 않는다.** Camera-induced visible adjacency는 WL103/WL106이 실패한 지점(pairwise 3D-edge + per-view corridor의 다중 뷰 모순 폭증)을 회피하면서 singleton을 대폭 줄이고(63.4%/83.8%→45.0%), 테이블의 구조적 일관성과 패티오로부터의 분리를 유지했다. 최대 component(36.8%)는 시각 검토상 문제적 percolation이 아니라 패티오 바닥 자체의 정당한 연속성으로 보인다. 다만: (1) hedge/배경은 여전히 대부분 파편화돼 있고, (2) 이 36.8% 최대 component가 실제로 "순수하게 패티오만"인지 완전히 정량적으로 확정하지는 않았으며(픽셀 색상 대조는 강력한 정황 증거이지 완전한 증명은 아님), (3) 이번 배치는 directive의 명시적 지시대로 이 결과를 "최종 architecture 성공"으로 자동 선언하지 않는다. 다음 단계가 있다면 이 36.8% 컴포넌트의 정체를 더 정밀하게(예: 좌표 clustering) 확인하는 것과 hedge/배경의 잔여 파편화 원인을 규명하는 것이겠으나, 그 판단은 이 배치 범위 밖이다.

## 참고

- 새 모듈: `osn_gs/surface/torch_camera_induced_visible_adjacency.py`, `osn_gs/render/torch_surfel_representative_diagnostics.py`
- 새 벤더 진단 빌드: `osn_gs/render/vendor/diff_surfel_rasterization_diag/` (원본 `diff_surfel_rasterization/`의 형제, 원본 미수정)
- 빌드 스크립트: `scripts/build_surfel_extension_diag.bat`
- 테스트: `tests/test_surfel_representative_diagnostics.py`, `tests/test_camera_induced_visible_adjacency.py`
- Export 스크립트: `scripts/devtools/camera_induced_visible_adjacency_export.py`
- 관련: [[project_renderer_grounded_visible_adjacency]] (WL106, 이번 배치가 반증을 이어받은 baseline), [[project_renderer_contribution_diagnostics]] (WL105)
