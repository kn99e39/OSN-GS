"""Worklog 127 -- EVIDENCE-BOUNDED PROJECTIVE TSDF for direct Visible Surface
construction.

A deliberately ISOLATED module family. The construction half of this package

    scale.py       canonical spatial scale h from renderer sampling
    field.py       projective signed distance, truncation, authority, fusion
    extraction.py  masked zero level-set extraction
    mesh_ops.py    mesh measurement / raycast / distance (geometry only)
    synthetic.py   deterministic semantic contracts S1-S7

must never import the historical topology / boundary / region / chart / KNN /
NURBS / Trust / occluded-space modules. `tests/test_evidence_bounded_projective_tsdf.py`
asserts that statically over the module source, because that isolation IS the
control experiment (directive section 8).

Only `attribution.py` -- which runs strictly AFTER the mesh exists -- is allowed
to read historical quantities, and only read-only for diagnostic attribution.
"""

CONSTRUCTION_MODULES = ("scale", "field", "extraction", "mesh_ops", "synthetic")
