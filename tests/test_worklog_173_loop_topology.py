import json

import numpy as np

from devtools.demo import worklog_173_loop_topology as t
from devtools.demo import worklog_173_tabletop_multi_loop_domain_attribution as w173


def trace(mask):
    loops=w173.native._trace_grid_boundary(mask)
    edges=t.boundary_edges(mask)
    return loops,[t.loop_record(loop,edges) for loop in loops]


def test_ids_do_not_depend_on_loop_start_orientation_or_population_order():
    mask=np.ones((7,7),bool);mask[2:5,2:5]=False
    loops,_=trace(mask)
    original=t.stable_ids(loops)
    variant=[np.roll(loop[::-1],1,axis=0) for loop in loops[::-1]]
    assert t.stable_ids(variant)==original[::-1]


def test_individual_closure_is_independent_of_multi_loop_case_flag():
    mask=np.ones((7,7),bool);mask[2:5,2:5]=False
    loops,records=trace(mask)
    assert len(loops)==2 and all(row["closed_against_boundary_edges"] for row in records)
    edges=t.boundary_edges(mask)
    edges.remove((tuple(loops[0][-1]),tuple(loops[0][0])))
    assert not t.loop_record(loops[0],edges)["closing_edge_authoritative"]


def test_annulus_containment_orientation_and_hole_relation():
    mask=np.ones((7,7),bool);mask[2:5,2:5]=False
    loops,records=trace(mask);ids=t.stable_ids(loops)
    relation,_,contacts=t.nesting(loops,records,ids)
    roles,summary,_=t.occupancy_diagnostics(mask,loops,records)
    assert not contacts and sorted(row["nesting_depth"] for row in relation)==[0,1]
    assert sorted(row["class"] for row in roles)==["EXTERIOR_SUPPORT_BOUNDARY","UNSUPPORTED_INTERIOR_HOLE"]
    assert sorted(row["signed_area_grid_units_squared"] for row in records)==[-9,49]
    assert summary["unsupported_bounded_components"]==1
    assert roles[1]["occupied_cells_inside"]==0


def test_multiple_independent_outers_and_three_level_nesting():
    mask=np.zeros((7,16),bool);mask[:,:7]=True;mask[2:5,2:5]=False;mask[3,3]=True;mask[2:5,12:15]=True
    loops,records=trace(mask);ids=t.stable_ids(loops)
    nesting,_,_=t.nesting(loops,records,ids)
    assert sorted(row["nesting_depth"] for row in nesting)==[0,0,1,2]


def test_self_crossing_and_repeated_vertex_contacts_are_distinct():
    crossing=np.array([[0,0],[2,2],[0,2],[2,0]])
    assert len(t.intersections(crossing)["proper_crossing_edge_pairs"])==1
    touching=np.array([[0,0],[1,0],[1,1],[2,1],[2,2],[1,2],[1,1],[0,1]])
    result=t.intersections(touching)
    assert result["contact_edge_pairs"] and not result["proper_crossing_edge_pairs"]
    edges={(tuple(a),tuple(b)) for a,b in zip(touching,np.roll(touching,-1,axis=0))}
    record=t.loop_record(touching,edges)
    assert record["closed_against_boundary_edges"] and not record["simple_valid_polygon"]
    assert record["repeated_vertex_count"]==1


def test_contact_with_invalid_outer_is_not_an_independent_certified_family():
    outer=np.array([[0,0],[4,0],[4,4],[8,4],[8,8],[4,8],[4,4],[0,4]])
    inner=np.array([[1,1],[1,2],[2,2],[2,1]])
    loops=[outer,inner]
    edges={(tuple(a),tuple(b)) for loop in loops for a,b in zip(loop,np.roll(loop,-1,axis=0))}
    records=[t.loop_record(loop,edges) for loop in loops]
    nesting,_,_=t.nesting(loops,records,t.stable_ids(loops))
    assert any(not row["simple_valid_polygon"] for row in records)
    assert all(row["ambiguous"] for row in nesting)


def test_native_cubical_topology_contract_only():
    single=np.array([[0,0,0]])
    assert t.native_cubical_topology(single)["tunnels_beta1"]==0
    ring=np.array([[x,y,0] for x in range(3) for y in range(3) if (x,y)!=(1,1)])
    result=t.native_cubical_topology(ring)
    assert result["tunnels_beta1"]==1 and result["enclosed_voids_beta2"]==0
    shell=np.array([[x,y,z] for x in range(3) for y in range(3) for z in range(3) if (x,y,z)!=(1,1,1)])
    shell_result=t.native_cubical_topology(shell)
    assert shell_result["tunnels_beta1"]==0 and shell_result["enclosed_voids_beta2"]==1


def test_real_frozen_world_chart_correspondence_and_full_loop_accounting():
    report=json.loads((w173.OUT/"worklog_173_report.json").read_text(encoding="utf-8"))
    assert report["baseline_manifest"]==w173.preservation_manifest()
    complete,samples,indices,boundary,_=w173.load_baseline()
    grid,_,occupancy=w173.chart_mapping(samples,boundary)
    with np.load(w173.OUT/"diagnostic_correspondence.npz") as data:
        assert np.array_equal(grid,data["sample_chart_grid"])
        assert np.array_equal(occupancy,data["chart_occupancy"])
        assert np.array_equal(np.concatenate([loop.numpy() for loop in boundary.loops]),data["loop_world"])
        assert np.array_equal(indices,data["complete_row_indices"])
    assert report["loop_summary"]["loop_count"]==134
    assert report["loop_summary"]["individually_closed"]==134
    assert report["loop_summary"]["unemitted_boundary_edges"]==0
    assert report["loop_summary"]["duplicate_edge_loop_count"]==0
    assert report["no_nurbs_fit"] and not report["production_changed"]


def test_required_visualizations_and_complete_loop_inventory():
    from devtools.demo import worklog_173_review_exports as review
    assert review.validate(w173.OUT)=={"png_count":9,"readme_count":9,"complete":True}
    inventory=(w173.OUT/"loop_inventory.md").read_text(encoding="utf-8")
    assert sum(line.startswith("| ") for line in inventory.splitlines())==135
