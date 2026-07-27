from __future__ import annotations

"""Deterministic closed-contour recovery from component support masks."""

from typing import Any

from osn_gs.surface.torch_boundary_refinement import marching_squares
from osn_gs.utils.torch_ops import require_torch


def ordered_closed_boundary_world_loops(mask: Any, frame: Any) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """Return only degree-two, closed contours in deterministic order.

    Open or branched segment graphs are intentionally omitted: callers must
    retain them as unresolved evidence instead of fabricating a correspondence
    order from unordered boundary cells.
    """
    torch = require_torch()
    segments = marching_squares(torch.as_tensor(mask).float(), 0.5)
    scale = 10**9
    nodes: dict[tuple[int, int], tuple[float, float]] = {}
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}
    remaining: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    def key(point: tuple[float, float]) -> tuple[int, int]:
        return (int(round(point[0] * scale)), int(round(point[1] * scale)))
    for left, right in segments:
        a, b = key(left), key(right)
        if a == b:
            continue
        nodes.setdefault(a, left); nodes.setdefault(b, right)
        adjacency.setdefault(a, set()).add(b); adjacency.setdefault(b, set()).add(a)
        remaining.add(tuple(sorted((a, b))))
    loops = []
    while remaining:
        start = min(min(edge) for edge in remaining)
        if len(adjacency.get(start, ())) != 2:
            remaining = {edge for edge in remaining if start not in edge}
            continue
        path = [start]; previous = None; current = start; closed = False
        while True:
            choices = sorted(value for value in adjacency[current] if value != previous)
            if not choices:
                break
            nxt = choices[0]; edge = tuple(sorted((current, nxt)))
            if edge not in remaining:
                closed = nxt == start
                break
            remaining.remove(edge); previous, current = current, nxt
            if current == start:
                closed = True; break
            path.append(current)
            if len(path) > len(nodes):
                break
        if closed and len(path) >= 3 and all(len(adjacency[node]) == 2 for node in path):
            uv = torch.tensor([nodes[node] for node in path], dtype=frame.origin.dtype, device=frame.origin.device)
            world = frame.to_world(uv).detach().cpu().tolist()
            loops.append(tuple(tuple(float(value) for value in point) for point in world))
        elif not closed:
            touched = set(path) | {current}
            remaining = {edge for edge in remaining if not (edge[0] in touched or edge[1] in touched)}
    return tuple(sorted(loops, key=lambda loop: (len(loop), loop)))