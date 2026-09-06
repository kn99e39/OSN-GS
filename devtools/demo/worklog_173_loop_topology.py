"""Exact integer-lattice diagnostics; no boundary repair or acceptance policy."""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
from scipy import ndimage


def area(poly):
    x, y = np.asarray(poly, dtype=np.int64).T
    return float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))) / 2


def canonical_cycle(poly):
    rows = [tuple(map(int, row)) for row in poly]
    candidates = []
    for sequence in (rows, rows[::-1]):
        smallest = min(sequence)
        for i, vertex in enumerate(sequence):
            if vertex == smallest:
                candidates.append(tuple(sequence[i:] + sequence[:i]))
    return min(candidates)


def stable_ids(loops):
    keys = [canonical_cycle(loop) for loop in loops]
    ordered = {key: f"L{i:03d}" for i, key in enumerate(sorted(set(keys)))}
    return [ordered[key] for key in keys]


def boundary_edges(mask):
    edges = set()
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        for a, b, ny, nx in (((x,y),(x+1,y),y-1,x), ((x+1,y),(x+1,y+1),y,x+1),
                             ((x+1,y+1),(x,y+1),y+1,x), ((x,y+1),(x,y),y,x-1)):
            if ny < 0 or nx < 0 or ny >= height or nx >= width or not mask[ny,nx]:
                edges.add((a,b))
    return edges


def point_relation(points, poly):
    """Even-odd interior and exact on-edge flags; grid/half-grid inputs are exact."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    inside = np.zeros(len(points), bool)
    on = np.zeros(len(points), bool)
    x, y = points.T
    for a,b in zip(poly, np.roll(poly, -1, axis=0)):
        cross = (b[0]-a[0])*(y-a[1]) - (b[1]-a[1])*(x-a[0])
        on |= (cross == 0) & (x >= min(a[0],b[0])) & (x <= max(a[0],b[0])) & (y >= min(a[1],b[1])) & (y <= max(a[1],b[1]))
        if a[1] != b[1]:
            hit_x = a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < hit_x)
    return inside & ~on, on


def intersections(first, second=None):
    """Nonadjacent self contacts or all pair contacts, with strict crossings separate."""
    self_check = second is None
    other = first if self_check else second
    ends = np.roll(other, -1, axis=0)
    contacts = []
    crossings = []
    for i, (a,b) in enumerate(zip(first, np.roll(first, -1, axis=0))):
        low, high = np.minimum(a,b), np.maximum(a,b)
        candidates = np.flatnonzero(np.all(np.maximum(other,ends) >= low, axis=1) & np.all(np.minimum(other,ends) <= high, axis=1))
        for j in candidates:
            if self_check and (j <= i or j == i+1 or (i == 0 and j == len(first)-1)):
                continue
            c,d = other[j], ends[j]
            def cross(u,v):
                return u[0]*v[1]-u[1]*v[0]
            s1,s2,s3,s4 = cross(b-a,c-a), cross(b-a,d-a), cross(d-c,a-c), cross(d-c,b-c)
            if s1*s2 <= 0 and s3*s4 <= 0:
                contacts.append([int(i),int(j)])
                if s1*s2 < 0 and s3*s4 < 0:
                    crossings.append([int(i),int(j)])
    return {"contact_edge_pairs": contacts, "proper_crossing_edge_pairs": crossings}


def loop_record(poly, edges):
    directed = [(tuple(a),tuple(b)) for a,b in zip(poly,np.roll(poly,-1,axis=0))]
    counts = Counter(tuple(sorted(edge)) for edge in directed)
    vertex_counts = Counter(map(tuple, poly))
    crossing = intersections(poly)
    missing = [edge for edge in directed if edge not in edges]
    repeated = [list(p) for p,n in vertex_counts.items() if n > 1]
    duplicate = sum(n-1 for n in counts.values())
    lengths = np.linalg.norm(np.roll(poly,-1,axis=0)-poly,axis=1)
    signed_area = area(poly)
    valid = len(poly) >= 3 and not missing and not crossing["contact_edge_pairs"] and not duplicate and signed_area != 0
    return {"edge_count": len(poly), "vertex_count": len(poly), "unique_vertex_count": len(vertex_counts),
            "support_sample_count": None, "support_sample_count_semantics": "loop vertices are chart-cell corners, not zero-set samples",
            "closed_against_boundary_edges": not missing, "missing_authoritative_edges": len(missing),
            "closing_edge_authoritative": directed[-1] in edges,
            "repeated_vertices": repeated, "repeated_vertex_count": len(poly)-len(vertex_counts),
            "duplicate_undirected_edge_count": duplicate, "self_intersection": crossing,
            "simple_valid_polygon": valid, "signed_area_grid_units_squared": signed_area,
            "orientation": "POSITIVE" if signed_area > 0 else "NEGATIVE" if signed_area < 0 else "ZERO",
            "perimeter_grid_units": float(lengths.sum()), "unit_axis_edges": bool(np.all(lengths == 1)),
            "chart_bbox_grid": [poly.min(axis=0).tolist(),poly.max(axis=0).tolist()],
            "chart_vertex_centroid_grid": poly.mean(axis=0).tolist(),
            "geometry_sha256": hashlib.sha256(np.asarray(canonical_cycle(poly), dtype=np.int64).tobytes()).hexdigest()}


def nesting(loops, records, ids):
    contains = np.zeros((len(loops),len(loops)),bool)
    generalized = np.zeros_like(contains)
    pair_contacts = []
    for i,a in enumerate(loops):
        for j in range(i+1,len(loops)):
            b = loops[j]
            if np.any(a.max(axis=0)<b.min(axis=0)) or np.any(b.max(axis=0)<a.min(axis=0)):
                continue
            contact = intersections(a,b)
            if contact["contact_edge_pairs"]:
                pair_contacts.append({"first":ids[i],"second":ids[j],**contact})
            if not contact["contact_edge_pairs"]:
                generalized[i,j] = bool(point_relation(b[:1],a)[0][0])
                generalized[j,i] = bool(point_relation(a[:1],b)[0][0])
            if not records[i]["simple_valid_polygon"] or not records[j]["simple_valid_polygon"] or contact["contact_edge_pairs"]:
                continue
            # No contact/crossing: one vertex establishes strict containment.
            contains[i,j] = bool(point_relation(b[:1],a)[0][0])
            contains[j,i] = bool(point_relation(a[:1],b)[0][0])
    result = []
    touching = {row[key] for row in pair_contacts for key in ("first","second")}
    for j in range(len(loops)):
        containers = np.flatnonzero(contains[:,j])
        possible = np.flatnonzero(generalized[:,j])
        invalid_containers=[int(i) for i in possible if not records[i]["simple_valid_polygon"]]
        parents = [i for i in containers if not any(contains[i,k] for k in containers if k != i)]
        generalized_parents=[i for i in possible if not any(generalized[i,k] for k in possible if k!=i)]
        invalid = not records[j]["simple_valid_polygon"] or ids[j] in touching or len(parents)>1 or bool(invalid_containers)
        result.append({"loop_id":ids[j], "containers":[ids[i] for i in containers],
                       "parent": ids[parents[0]] if len(parents)==1 else None,
                       "nesting_depth": int(len(containers)) if not invalid else None,
                       "children": [], "ambiguous": invalid,
                       "generalized_evenodd_containers":[ids[i] for i in possible],
                       "generalized_parent":ids[generalized_parents[0]] if len(generalized_parents)==1 else None,
                       "generalized_depth":len(possible),
                       "generalized_children":[],
                       "invalid_containers":[ids[i] for i in invalid_containers],
                       "generalized_relation_is_valid_polygon_nesting":not invalid})
    for row in result:
        if row["parent"] is not None:
            result[ids.index(row["parent"])]["children"].append(row["loop_id"])
        if row["generalized_parent"] is not None:
            result[ids.index(row["generalized_parent"])]["generalized_children"].append(row["loop_id"])
    return result, contains, pair_contacts


def occupancy_diagnostics(mask, loops, records):
    cross = ndimage.generate_binary_structure(2,1)
    padded = np.pad(mask,1,constant_values=False)
    empty_labels, _ = ndimage.label(~padded, structure=cross)
    exterior = int(empty_labels[0,0])
    hole_labels = [int(v) for v in np.unique(empty_labels) if v not in (0,exterior)]
    filled_components = int(ndimage.label(mask,structure=cross)[1])
    records_out = []
    yy,xx = np.indices(mask.shape)
    centers = np.column_stack((xx.ravel()+.5, yy.ravel()+.5))
    center_labels = empty_labels[1:-1,1:-1].ravel()
    for loop,record in zip(loops,records):
        interior,_ = point_relation(centers,loop)
        occupied_inside = int((interior & mask.ravel()).sum())
        holes_inside = sorted(set(map(int,center_labels[interior]))-{0,exterior})
        # Every unit directed edge has occupied left side. Sample both sides
        # at exact quarter-cell offsets; these are diagnostic witnesses only.
        a,b = loop,np.roll(loop,-1,axis=0)
        left = np.column_stack((-(b-a)[:,1],(b-a)[:,0]))
        mid = (a+b)/2
        labels_sides = []
        occupancy_sides = []
        for side in (mid+.25*left,mid-.25*left):
            ij = np.floor(side).astype(int)
            labels_sides.append(empty_labels[ij[:,1]+1,ij[:,0]+1])
            occupancy_sides.append(padded[ij[:,1]+1,ij[:,0]+1])
        empty_adjacent = sorted(set(map(int,labels_sides[1]))-{0})
        if not record["simple_valid_polygon"] or not occupancy_sides[0].all() or occupancy_sides[1].any():
            kind = "AMBIGUOUS_OR_INVALID"
        elif empty_adjacent == [exterior]:
            kind = "EXTERIOR_SUPPORT_BOUNDARY"
        elif len(empty_adjacent)==1 and empty_adjacent[0] in hole_labels:
            kind = "UNSUPPORTED_INTERIOR_HOLE"
        else:
            kind = "AMBIGUOUS_OR_INVALID"
        records_out.append({"class":kind,"left_side_all_occupied":bool(occupancy_sides[0].all()),
            "right_side_all_unsupported":bool((~occupancy_sides[1]).all()),
            "adjacent_empty_component_ids":empty_adjacent,"interior_grid_cell_count":int(interior.sum()),
            "edge_count_by_adjacent_empty_component":{str(i):int((labels_sides[1]==i).sum()) for i in empty_adjacent},
            "occupied_cells_inside":occupied_inside,"unsupported_cells_inside":int(interior.sum())-occupied_inside,
            "enclosed_empty_component_ids":holes_inside,
            "missing_native_cell_cause": "NOT_IDENTIFIABLE_FROM_CHART_OCCUPANCY_ALONE"})
    return records_out, {"occupied_cells":int(mask.sum()),"occupied_face_components":filled_components,
                        "occupied_vertex_connected_components":int(ndimage.label(mask,structure=np.ones((3,3)))[1]),
                        "occupied_face_component_sizes":[int(n) for n in np.bincount(ndimage.label(mask,structure=cross)[0].ravel())[1:]],
                        "unsupported_bounded_components":len(hole_labels),"exterior_empty_component":exterior,
                        "hole_cell_counts":{str(i):int((empty_labels==i).sum()) for i in hole_labels}}, empty_labels[1:-1,1:-1]


def native_cubical_topology(cells):
    """Descriptive closed voxel-union complex, NOT an extracted zero-surface mesh."""
    cells = np.asarray(cells,dtype=np.int64)
    def number(offsets):
        return len(np.unique(np.concatenate([cells+off for off in offsets]),axis=0))
    vertices = number([np.array([x,y,z]) for x in (0,1) for y in (0,1) for z in (0,1)])
    edges=faces=0
    for axis in range(3):
        others=[i for i in range(3) if i!=axis]
        offsets=[]
        for a in (0,1):
            for b in (0,1):
                off=np.zeros(3,dtype=int);off[others]=[a,b];offsets.append(off)
        edges+=number(offsets)
        off=np.zeros(3,dtype=int);off[axis]=1
        faces+=number([np.zeros(3,dtype=int),off])
    shape=cells.max(axis=0)-cells.min(axis=0)+3
    if int(np.prod(shape)) > 20_000_000:
        raise ValueError("native complement diagnostic allocation exceeds fixed memory guard")
    mask=np.zeros(tuple(shape),bool)
    local=cells-cells.min(axis=0)+1
    mask[tuple(local.T)]=True
    solid_components=int(ndimage.label(mask,structure=np.ones((3,3,3)))[1])
    cavities=int(ndimage.label(~mask,structure=ndimage.generate_binary_structure(3,1))[1])-1
    chi=vertices-edges+faces-len(cells)
    return {"vertices":vertices,"edges":edges,"faces":faces,"cubes":len(cells),"euler_characteristic":chi,
            "closed_voxel_union_components":solid_components,"enclosed_voids_beta2":cavities,
            "tunnels_beta1":solid_components+cavities-chi,
            "native_boundary_loop_count":None,
            "interpretation":"Native 3D support has a voxel boundary surface, not canonical ordered 1D loops. Cubical tunnels are not physical zero-set holes."}
