# Worklog 91: Curved Annulus Boundary-first Recovery Proof Path

## 상태

isolated `curved_annulus` Boundary-first proof path 구현·검증 완료(국소 범위 100%). 기존 `boundary_first.py` dispatcher 및 production integration은 미착수다.

## 구현 결과

새 경로는 box/trimmed-rectangle fallback을 사용하지 않는다.

```text
scene points
→ 기존 raw component 분석
→ non-face smooth continuation recovery evidence
→ immutable recovered region
→ component-level outer/hole loop 재추출
→ pre-surface boundary pair
→ world-arclength support-curve family
→ explicit cyclic seam multi-patch NURBS
```

- `curved_annulus`의 기존 raw component 수는 2개다.
- 두 component는 normal agreement와 local-spacing 대비 close support witness로 recovery candidate가 된다.
- accepted evidence만 union하여 input component를 mutate하지 않는 recovered region을 만든다.
- recovered region의 기존 boundary extractor는 outer loop 1개와 hole loop 1개를 찾아 `annulus`로 분류한다.
- observed inner/outer loop에서 8개의 radial support curve와 8개의 seam-connected NURBS patch를 생성한다.
- 기존 잘못된 `trimmed_rect_fallback`은 이 path에서 호출하지 않는다.

## 추가 파일

- `osn_gs/surface/torch_boundary_component_recovery.py`
- `osn_gs/surface/torch_boundary_first_visible_builder.py`
- `nurbs_constructor_benchmark/boundary_first_support.py`
- 대응 전용 테스트 2개

## 검증

- 관련 Boundary-first/annulus unittest: 67 tests passed
- `.venv\Scripts\python.exe -B -m pytest`: 공유 작업트리 기준 409 passed, 1 skipped, 1 warning
- warning은 기존 `torch_voxel_hierarchy.py`의 requires-grad tensor scalar conversion이다.

## 명시적 비범위

- 기존 benchmark dispatcher 교체 또는 default 변경
- component builder의 production merge 정책 변경
- disk/non-convex/multi-loop의 interior support-network 구현
- support curve 기반 data/fairness optimizer 정교화
- trainer, renderer, checkpoint integration

다음 integration은 이 proof path의 negative-control sweep과 quality metric을 먼저 확정한 뒤 feature-gated 방식으로만 진행한다.