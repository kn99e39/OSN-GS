# Worklog 150 — Boundary First / Local Surface Decomposition Contract Trace

## Intended architecture

```text
Visible renderer evidence
        ↓
Local Surface Decomposition (`form_surface_regions`)
        ↓
Region / local physical-surface ownership
        ↓
Boundary First: observed boundary candidates → ordered eligibility
        ↓
Region-owned local chart and support
        ↓
PRE-FIT eligible boundary + region-core NURBS fit
        ↓
Supported materialization
```

## Actual WL139–WL149 path

```text
WL145 manual polygon control
        ↓
Independent per-camera renderer median event clouds
        ↓
`clean_points = np.concatenate([cloud.points for cloud in clouds], axis=0)`
        ↓  ← local region/boundary ownership was never created; provenance is flattened
Global PCA (`world_xyz @ axes`) + global min/max rectangle
        ↓  ← event 1527 becomes `v_min`
WL139 fixed physical-UV representative fit
        ↓
WL148 B post-fit support materialization
        ↓
WL149 extrema/influence replay
```

## Result

The first decisive bypass is before representative fitting: WL145's clean-oracle
path does not call the canonical constructor or local decomposition, and its
pooled XYZ union is then used for global PCA. Event 1527 therefore remains in
the representative population and owns the global `v_min` extent. This is
`B. ARCHITECTURE_BYPASS`. It is not a repair and does not show that canonical
Boundary First itself failed.

Required semantic distinction: **WL148 B does NOT by itself restore Boundary First semantics.**

Human review is limited to event 1527:
`HUMAN_REVIEW_PHYSICAL_SHEET_STATUS: CLEAR_NOT_ON_INTENDED_SURFACE`.
No broad rejection rule is inferred for the other events.
