# 19. 방향성 정정 — NURBS는 관측 Gaussian에서 파생될 뿐, 관측 Gaussian을 움직이지 않는다

날짜: 2026-07-15

> **이 문서는 이전 worklog(01~18)의 전제를 정정한다.** 이전 문서들은 수정하지 않고 그대로 남긴다(그 시점의 기록으로서 유효). 다만 **앞으로의 모든 작업은 이 문서의 방향성을 따른다.** 이전 worklog나 코드에서 이 문서와 충돌하는 서술을 발견하면, 이 문서가 우선한다.

## 무엇이 잘못돼 있었나

`docs/architecture.md`가 프레임워크를 **surface가 geometry의 source of truth이고 Gaussian은 거기서 파생된 렌더링 샘플**이라고 서술하고 있었다:

- "NURBS 기반 parametric surface를 장면의 canonical geometry로 두고 Gaussian을 그 표면에서 파생된 렌더링 샘플로 취급한다"
- "NURBS surface는 OSN-GS의 단일한 geometric source of truth이다"
- "**surface 수정은 Gaussian 위치와 normal을 갱신한다**"
- 학습 루프 의사코드의 `update_surface_bound_gaussian_positions()`

이 전제가 코드로 그대로 흘러갔다. `nurbs_surface_loss`가 **`certain = ~is_uncertain`(= 보이는 Gaussian)** 을 골라 `(gaussian_xyz - patch.evaluate(uv))²`를 최소화했고, `gaussian_xyz`가 grad를 물고 있어 **gradient가 Gaussian 위치로 역류**했다. 즉 NURBS가 보이는 Gaussian을 표면 쪽으로 끌어당기고 있었다. Stage 1에는 uncertain Gaussian이 없어 `uncertain_anchor_loss`는 0만 반환했으므로, **실제로 작동하던 유일한 NURBS 항이 하필 방향이 반대인 그것**이었다.

## 올바른 방향성 (앞으로 이렇게 간다)

```text
보이는 구조(관측 Gaussian)
  -> NURBS 표면 유도
  -> 보이지 않는 표면 유추
  -> 그 표면 위에 uncertain Gaussian 생성
```

- **데이터 흐름은 Gaussian → NURBS 단방향이다.**
- **관측(certain) Gaussian은 NURBS의 영향을 받지 않는다.** baseline 3DGS와 동일하게 image loss만으로 최적화된다. 관측 Gaussian은 표면의 **fitting target**이지 표면의 산출물이 아니다.
- **NURBS가 매 iteration 관측 Gaussian을 따라 갱신되는 것은 유지한다** — 이건 의도된 방향이다. 다만 그 갱신이 Gaussian을 되밀어선 안 된다.
- **surface geometry가 위치를 공급하는 대상은 오직 uncertain Gaussian**(비관측 영역에서 표면으로부터 생성되는 것)이다.
- ADC는 관측 Gaussian에 대해 기존 3DGS 정책을 그대로 유지한다.

NURBS는 **최종 출력이 아니라 중간 표현**이라는 기존 원칙은 그대로다. 존재 이유는 관측 표면 구조를 연속적 parametric form으로 요약해 **비관측 영역으로 연장**하는 것이다.

## 이번에 한 수정

1. **코드** — `osn_gs/losses/torch_losses.py`의 `nurbs_surface_loss`에서 관측 Gaussian 위치를 detach:
   ```python
   xyz = state.model.get_xyz[indices].detach()
   ```
   이제 이 항은 표면만 Gaussian 쪽으로 fitting하고, Gaussian에는 gradient를 주지 않는다. docstring에 단방향 원칙을 명시했다.
2. **문서** — `docs/architecture.md`의 "source of truth" / "surface 수정은 Gaussian 위치를 갱신한다" / `update_surface_bound_gaussian_positions()` / ADC 재해석 서술을 위 방향성대로 정정.

## 검증

`crease` scene(300pts)으로 `nurbs_surface_loss`만 backward한 결과:

```text
surface loss = 0.004132
grad -> Gaussian _xyz      : 0.00000000   (보이는 Gaussian 불변)
grad -> NURBS control_grid : 0.14052598   (표면은 Gaussian을 따라감)
```

단방향 결합이 성립함을 확인했다.

## 파급

- `TODO.md`의 "남은 후보 1 — NURBS surface anchor loss가 certain Gaussian 위치를 구속"은 **설계상 원인이 제거됐다.** baseline 대비 품질 격차 요인 중 이 항목은 해소된 것으로 본다(정량 재측정은 사용자가 직접 수행).
- `lambda_surface`는 이제 **표면이 Gaussian을 얼마나 빠르게 따라가는지**를 정하는 값이지, Gaussian을 얼마나 구속하는지가 아니다. 의미가 바뀌었으므로 기존 ablation 전제도 바뀐다.
- Stage 1은 여전히 visible surface까지만 만든다. **이 프레임워크의 본래 목적(비관측 표면 유추 → 그 위 Gaussian 생성)은 아직 미구현**이며, 그것이 다음 주요 작업이다.
