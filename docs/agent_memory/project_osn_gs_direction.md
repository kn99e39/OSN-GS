---
name: project-osn-gs-direction
description: "OSN-GS's core goal and the one-way Gaussian->NURBS direction rule, plus the current Stage 1 boundary and constraints"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9f58b1e8-0abf-4c0b-a3b4-3b8396c6006c
---

OSN-GS extends 3D Gaussian Splatting to reconstruct surface that was never observed. The intended data flow is **one-way**:

```
visible structure (observed Gaussians)
  -> derive NURBS surface
  -> infer the occluded/unseen surface
  -> generate Gaussians on that inferred surface
```

**The direction rule (stated by the user 2026-07-15, and the single most important constraint): visible/certain Gaussians must NOT be influenced by the NURBS.** They are optimized by the image loss alone, exactly like baseline 3DGS. The NURBS is a *derived intermediate*, not the source of truth — it is fitted to the observed Gaussians (updating every iteration is desirable), and it supplies geometry only to *uncertain* Gaussians created on inferred/occluded regions. Enforced in code by detaching certain-Gaussian positions in `nurbs_surface_loss` (`osn_gs/losses/torch_losses.py`); verified by asserting grad→`model._xyz` is 0 while grad→`control_grid` is > 0. See [[reference-osn-gs-docs]] → `docs/worklogs/19_nurbs_direction_correction.md`.

**Why this matters:** `docs/architecture.md` originally described the opposite ("NURBS is the single geometric source of truth", "surface 수정은 Gaussian 위치와 normal을 갱신한다"), and that framing leaked into the code as an anchor term pulling visible Gaussians onto the surface. The doc has been corrected. If any doc or code again implies NURBS→visible-Gaussian influence, it contradicts the user's intent — flag it rather than following it.

**Current implementation boundary (Stage 1):** visible-surface reconstruction + Gaussian-surface binding only. Occluded-surface generation, the algebraic curve-extension operator, and uncertain-to-certain promotion are explicitly out of scope until the user opens that stage — note this means the *whole point* (occluded inference) is not implemented yet; Stage 1 only builds the NURBS from visible Gaussians.

**Other standing constraints:** NURBS/voxel are not optional scaffolding — never suggest disabling them for throughput (prefer reducing blocking I/O, async streaming, preserving optimizer state across ADC). The voxel bootstrap runs once at init (`state.voxel_regions` is a frozen snapshot); after that NURBS control points are the trainable geometry, and only persistently failing patches get local correction — never a global topology rebuild.
