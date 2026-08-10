from __future__ import annotations

"""Compatibility import for the corrected chart-unit membership-cut topology.

Worklog 89 removed the rejected induced-subgraph PCA rotation and largest-face
construction.  All public calls now resolve to full-region local-frame face
recovery followed by chart-unit face incidence.
"""

from osn_gs.surface.torch_chart_unit_face_incidence_partition_boundary import *  # noqa: F401,F403