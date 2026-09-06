from __future__ import annotations

"""W173 frozen tabletop loop attribution; no fitter call, repair or acceptance change."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from devtools.demo import worklog_172_fragmentation_gate_vs_structural_fit_audit as w172
from devtools.demo import worklog_173_loop_topology as topology
from osn_gs.surface import torch_gaussian_region_owned_tsdf as native

w171=w172.w171
OUT=ROOT/"output/173_tabletop_multi_loop_domain_attribution"
CASE="real_tabletop_coherent"


def stats(values):
    return w171._distribution(np.asarray(values,dtype=float))


def preservation_manifest():
    manifest=w172.frozen_manifest()
    paths=[Path(w172.__file__)]
    for root in (w172.OUT, ROOT/"output/172_render_view_projection_review"):
        paths.extend(sorted(p for p in root.rglob("*") if p.is_file()))
    for p in paths:
        manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return manifest


def load_baseline():
    """Load ONLY the frozen tabletop selection; never resolve another case."""
    case_dir=w172.OUT/"case_artifacts"/CASE
    stored=json.loads((case_dir/"result.json").read_text(encoding="utf-8"))
    spec=stored["baseline"]["selection"]
    source=w171.DEFAULT_W154/"candidate_f_tsdf_surface_samples.npz"
    with np.load(source) as data:
        xyz=data["world_xyz"]
    with np.load(w171.DEFAULT_W154/"candidate_f_region_owned_support.npz") as data:
        owned=data["owned_region_id"]
        mask=data["accepted_mask"].astype(bool)&(owned==spec["prior_region_id"])
    mask&=np.all(xyz>=np.asarray(spec["aabb_min"]),axis=1)&np.all(xyz<=np.asarray(spec["aabb_max"]),axis=1)
    indices=np.flatnonzero(mask)
    regions=owned[indices].copy()
    del xyz,owned,mask
    arrays={}
    for key in w172.ROW_FIELDS[:4]:
        with np.load(source) as data:
            arrays[key]=torch.from_numpy(data[key][indices].copy())
    complete=native.TSDFVisibleSurfaceSamples(**arrays,corner_values=torch.empty((0,8)),corner_support_count=torch.empty((0,8),dtype=torch.int32),
               h=stored["baseline"]["support"]["h"])
    with np.load(case_dir/"selected_support.npz") as data:
        selected=native.TSDFVisibleSurfaceSamples(**{key:torch.from_numpy(data[key].copy()) for key in w172.ROW_FIELDS},h=complete.h)
        row_indices=data["complete_row_indices"].copy()
        selected_regions=data["region_ids"].copy()
    assert len(indices)==17965 and len(row_indices)==15189
    for key in w172.ROW_FIELDS:
        full=getattr(complete,key)
        assert torch.equal(full[row_indices] if len(full) else full,getattr(selected,key))
    assert np.array_equal(regions[row_indices],selected_regions)
    assert w172.row_hash(selected)==stored["selected_rows_hash_before_and_after"]
    assert w172.row_hash(complete)==stored["complete_rows_hash_before_and_after"]
    components,_=native.build_native_tsdf_support_components(complete,w171._all_accepted_support(complete,regions))
    largest=w172.largest_component(complete,components)
    assert len(components)==313 and largest.component_id==stored["diagnostic_component_id"]
    assert np.array_equal(np.sort(largest.sample_indices.numpy()),row_indices)
    single,_=native.build_native_tsdf_support_components(selected,w171._all_accepted_support(selected,selected_regions))
    assert len(single)==1
    boundary=native.derive_native_support_boundary(selected,single[0])
    assert len(boundary.loops)==134 and boundary.reason==stored["diagnostic"]["reason"]
    with np.load(case_dir/"boundary_chart.npz") as data:
        for key in ("chart_origin","tangent_u","tangent_v","chart_occupancy"):
            assert np.array_equal(getattr(boundary,key).numpy(),data[key])
        assert np.array_equal(np.concatenate([loop.numpy() for loop in boundary.loops]),data["loop_world_xyz"])
        assert np.array_equal(np.cumsum([0]+[len(loop) for loop in boundary.loops]),data["loop_offsets"])
    return complete,selected,row_indices,boundary,stored


def chart_mapping(samples,boundary):
    points=samples.world_xyz
    center=points.mean(dim=0)
    projection=torch.stack(((points-center)@boundary.tangent_u,(points-center)@boundary.tangent_v),dim=1)
    grid=torch.floor(projection/samples.h+.5).to(torch.int64)
    low=grid.amin(dim=0)
    shifted=(grid-low).numpy()
    occupancy=np.zeros_like(boundary.chart_occupancy.numpy())
    occupancy[shifted[:,1],shifted[:,0]]=True
    assert np.array_equal(occupancy,boundary.chart_occupancy.numpy())
    return shifted,projection.numpy(),occupancy


def world_diagnostics(samples,boundary,grid,loops_world,records):
    xyz=samples.world_xyz.numpy().astype(float)
    normals=samples.normals.numpy().astype(float)
    normal=boundary.normal.numpy().astype(float)
    normal/=np.linalg.norm(normal)
    distances=(xyz-boundary.chart_origin.numpy())@normal
    norm_lengths=np.linalg.norm(normals,axis=1)
    dot=normals@normal
    cos=dot/np.where(norm_lengths>0,norm_lengths,1)
    keys=samples.source_cell_keys.numpy()
    edge_a,edge_b=[],[]
    for stride in (native.STRIDE_X,native.STRIDE_Y,native.STRIDE_Z):
        locations=np.searchsorted(keys,keys+stride)
        good=locations<len(keys)
        safe=np.minimum(locations,len(keys)-1)
        good&=keys[safe]==keys+stride
        edge_a.extend(np.flatnonzero(good));edge_b.extend(safe[good])
    a,b=np.asarray(edge_a),np.asarray(edge_b)
    delta=xyz[b]-xyz[a]
    local_a=(delta*normals[a]).sum(axis=1)/np.where(norm_lengths[a]>0,norm_lengths[a],1)
    local_b=(-delta*normals[b]).sum(axis=1)/np.where(norm_lengths[b]>0,norm_lengths[b],1)
    neighbor_cos=(normals[a]*normals[b]).sum(axis=1)/np.where(norm_lengths[a]*norm_lengths[b]>0,norm_lengths[a]*norm_lengths[b],1)
    unique,inv,counts=np.unique(grid,axis=0,return_inverse=True,return_counts=True)
    mins=np.full(len(unique),np.inf);maxs=np.full(len(unique),-np.inf)
    np.minimum.at(mins,inv,distances);np.maximum.at(maxs,inv,distances)
    tree=cKDTree(xyz)
    for loop,record in zip(loops_world,records):
        nearest,rows=tree.query(loop,k=1,workers=1)
        record.update({"world_bbox":[loop.min(axis=0).tolist(),loop.max(axis=0).tolist()],
                       "world_extent":np.ptp(loop,axis=0).tolist(),"world_vertex_centroid":loop.mean(axis=0).tolist(),
                       "loop_world_plane_residual":stats((loop-boundary.chart_origin.numpy())@normal),
                       "nearest_support_distance":w171._distribution(nearest,samples.h),
                       "nearest_support_signed_chart_plane_residual":w171._distribution(distances[rows],samples.h)})
    continuous=np.column_stack(((xyz-boundary.chart_origin.numpy())@boundary.tangent_u.numpy(),(xyz-boundary.chart_origin.numpy())@boundary.tangent_v.numpy()))
    continuous_unique=len(np.unique(continuous,axis=0))
    labels,_=ndimage.label(boundary.chart_occupancy.numpy(),structure=ndimage.generate_binary_structure(2,1))
    sample_labels=labels[grid[:,1],grid[:,0]]
    broken=sample_labels[a]!=sample_labels[b]
    bin_worst=int(np.argmax(maxs-mins))
    worst_rows=np.flatnonzero(inv==bin_worst)
    residual_rows=distances[worst_rows]
    min_row=int(worst_rows[np.argmin(residual_rows)]);max_row=int(worst_rows[np.argmax(residual_rows)])
    result={"global_chart_plane_signed_residual":w171._distribution(distances,samples.h),
            "global_chart_plane_absolute_residual":w171._distribution(np.abs(distances),samples.h),
            "local_native_edge_endpoint_plane_absolute_residual":w171._distribution(np.abs(np.concatenate((local_a,local_b))),samples.h),
            "normal_length":stats(norm_lengths),"normal_zero_count":int((norm_lengths==0).sum()),
            "signed_normal_dot_chart_normal":stats(cos),"normal_angle_degrees":stats(np.degrees(np.arccos(np.clip(cos,-1,1)))),
            "unoriented_normal_angle_degrees":stats(np.degrees(np.arccos(np.clip(np.abs(cos),0,1)))),
            "normal_negative_dot_count":int((dot<0).sum()),"normal_zero_dot_count":int((dot==0).sum()),
            "native_neighbor_normal_angle_degrees":stats(np.degrees(np.arccos(np.clip(neighbor_cos,-1,1)))),
            "native_neighbor_normal_opposition_count":int((neighbor_cos<0).sum()),
            "native_face_graph_edges":len(a),"native_face_graph_cycle_rank":len(a)-len(xyz)+1,
            "continuous_projected_duplicate_count":len(xyz)-continuous_unique,
            "native_edges_between_distinct_chart_face_components":int(broken.sum()),
            "chart_connectivity_break_witnesses":[{"sample_rows":[int(i),int(j)],"cell_keys":[int(keys[i]),int(keys[j])],
                "native_cells":[samples.cell_indices[i].tolist(),samples.cell_indices[j].tolist()],
                "chart_bins":[grid[i].tolist(),grid[j].tolist()]} for i,j in zip(a[broken],b[broken])],
            "max_chart_bin_spread_witness":{"chart_bin":unique[bin_worst].tolist(),"sample_rows":[min_row,max_row],
                "world_xyz":xyz[[min_row,max_row]].tolist(),"normals":normals[[min_row,max_row]].tolist(),
                "continuous_chart_separation_world":float(np.linalg.norm(continuous[min_row]-continuous[max_row])),
                "normal_height_separation_world":float(maxs[bin_worst]-mins[bin_worst]),
                "normal_height_separation_h":float((maxs[bin_worst]-mins[bin_worst])/samples.h)},
            "chart_occupied_cells":len(unique),"chart_bin_multiplicity":stats(counts),
            "chart_bins_with_multiple_native_samples":int((counts>1).sum()),
            "multi_sample_bin_height_spread":w171._distribution((maxs-mins)[counts>1],samples.h),
            "native_adjacent_edges_collapsed_to_same_chart_bin":int(np.all(grid[a]==grid[b],axis=1).sum()),
            "native_adjacent_chart_bin_manhattan_distance":stats(np.abs(grid[a]-grid[b]).sum(axis=1)),
            "chart_basis_determinant_relative_normal":float(np.dot(np.cross(boundary.tangent_u.numpy(),boundary.tangent_v.numpy()),normal)),
            "injectivity_interpretation":"Quantized bin collisions do not by themselves prove geometric fold-over. Finite point projection cannot certify a continuous sheet chart; no sample triangle complex exists here."}
    return result,distances,inv,counts


def analyze():
    before=preservation_manifest()
    complete,samples,selected_rows,boundary,baseline=load_baseline()
    print("[W173] exact W172 tabletop identity and all 134 loops verified",flush=True)
    grid,continuous,occupancy=chart_mapping(samples,boundary)
    loops=native._trace_grid_boundary(occupancy)
    world_loops=[loop.numpy().copy() for loop in boundary.loops]
    assert len(loops)==len(world_loops)==134
    # Reconstruct the original float32 embedding exactly, never round the
    # authoritative world loops or use a geometric tolerance for topology.
    for loop,world in zip(loops,world_loops):
        uv=torch.from_numpy(loop).to(boundary.chart_origin.dtype)
        rebuilt=boundary.chart_origin[None,:]+samples.h*(uv[:,0:1]*boundary.tangent_u[None,:]+uv[:,1:2]*boundary.tangent_v[None,:])
        assert np.array_equal(rebuilt.numpy(),world)
    ids=topology.stable_ids(loops)
    edges=topology.boundary_edges(occupancy)
    records=[topology.loop_record(loop,edges) for loop in loops]
    containment,relation,contacts=topology.nesting(loops,records,ids)
    hole_records,occupancy_summary,empty_labels=topology.occupancy_diagnostics(occupancy,loops,records)
    for i,row in enumerate(records):
        row.update({"loop_id":ids[i],"w172_loop_index":i,"w172_size_rank":i+1,
                    "perimeter_world_units":row["perimeter_grid_units"]*samples.h,
                    "signed_area_world_units_squared":row["signed_area_grid_units_squared"]*samples.h**2,
                    "chart_extent_grid":np.ptp(loops[i],axis=0).tolist(),
                    "chart_extent_world_units":(np.ptp(loops[i],axis=0)*samples.h).tolist(),
                    "frozen_world_path_length":float(np.linalg.norm(np.roll(world_loops[i],-1,axis=0)-world_loops[i],axis=1).sum()),
                    "area_interpretation":"simple polygon area" if row["simple_valid_polygon"] else "algebraic winding area only; simple polygon area undefined",
                    "containment":containment[i],"occupancy":hole_records[i]})
    world,plane_distance,bin_ids,bin_counts=world_diagnostics(samples,boundary,grid,world_loops,records)
    native_topology=topology.native_cubical_topology(samples.cell_indices.numpy())
    print(f"[W173] valid={sum(r['simple_valid_polygon'] for r in records)}, contacts={len(contacts)}, chart holes={occupancy_summary['unsupported_bounded_components']}",flush=True)
    print(f"[W173] native cubical topology={native_topology}",flush=True)
    used=Counter((tuple(a),tuple(b)) for loop in loops for a,b in zip(loop,np.roll(loop,-1,axis=0)))
    pair_relations=[{"outer":ids[i],"inner":ids[j]} for i,j in np.argwhere(relation)]
    summary={"loop_count":len(loops),"individually_closed":sum(r["closed_against_boundary_edges"] for r in records),
        "open_loop_count":sum(not r["closed_against_boundary_edges"] for r in records),
        "valid_simple_loop_count":sum(r["simple_valid_polygon"] for r in records),
        "invalid_loop_count":sum(not r["simple_valid_polygon"] for r in records),
        "self_contact_loop_count":sum(bool(r["self_intersection"]["contact_edge_pairs"]) for r in records),
        "self_proper_crossing_loop_count":sum(bool(r["self_intersection"]["proper_crossing_edge_pairs"]) for r in records),
        "repeated_vertex_loop_count":sum(bool(r["repeated_vertex_count"]) for r in records),
        "duplicate_edge_loop_count":sum(bool(r["duplicate_undirected_edge_count"]) for r in records),
        "outer_family_count":sum(r["nesting_depth"]==0 for r in containment),
        "outer_family_count_semantics":"certified simple-polygon roots only; zero means no certified tree, not no exterior boundary",
        "independent_valid_outer_family_count":None if any(r["ambiguous"] for r in containment) else sum(r["nesting_depth"]==0 for r in containment),
        "nested_inner_loop_count":sum(r["nesting_depth"] is not None and r["nesting_depth"]>0 for r in containment),
        "ambiguous_containment_count":sum(r["ambiguous"] for r in containment),
        "generalized_no_contact_free_container_count":sum(r["generalized_depth"]==0 for r in containment),
        "generalized_nested_loops":sum(r["generalized_depth"]>0 for r in containment),
        "nesting_depth":stats([r["nesting_depth"] for r in containment if r["nesting_depth"] is not None]),
        "inter_loop_contact_pair_count":len(contacts),
        "inter_loop_proper_crossing_pair_count":sum(bool(r["proper_crossing_edge_pairs"]) for r in contacts),
        "occupancy_classes":dict(Counter(r["class"] for r in hole_records)),
        "all_boundary_edges":len(edges),"traced_edge_occurrences":sum(used.values()),
        "unemitted_boundary_edges":len(edges-set(used)),"multiply_used_edges":sum(n>1 for n in used.values())}
    distribution={key:stats([r[key] for r in records]) for key in ("edge_count","vertex_count","perimeter_grid_units","perimeter_world_units","signed_area_grid_units_squared")}
    distribution["absolute_area_grid_units_squared_valid"]=stats([abs(r["signed_area_grid_units_squared"]) for r in records if r["simple_valid_polygon"]])
    distribution["absolute_algebraic_area_all_loops"]=stats([abs(r["signed_area_grid_units_squared"]) for r in records])
    distribution["absolute_area_world_squared_valid"]=stats([abs(r["signed_area_world_units_squared"]) for r in records if r["simple_valid_polygon"]])
    distribution["hole_grid_cell_counts"]=stats(list(occupancy_summary["hole_cell_counts"].values()))
    occupancy_summary["all_loop_signed_area_sum"]=sum(r["signed_area_grid_units_squared"] for r in records)
    occupancy_summary["winding_area_equals_occupied_cells"]=occupancy_summary["all_loop_signed_area_sum"]==occupancy_summary["occupied_cells"]
    report={"status":"DIAGNOSTICS_COMPUTED","case":CASE,"baseline_preserved":True,"baseline_manifest":before,
            "baseline_support":{"complete_count":17965,"complete_components":313,"selected_count":15189,"selected_fraction":15189/17965,
                "component_id":baseline["diagnostic_component_id"],"region_id":baseline["diagnostic_region_id"],"h":samples.h,
                "all_selected_arrays_hash":w172.row_hash(samples),"all_134_world_loops_and_row_correspondence_exact":True},
            "loop_summary":summary,"loop_distributions":distribution,"loops":records,
            "containment_relations":pair_relations,"inter_loop_contacts":contacts,"occupancy":occupancy_summary,
            "native_cubical_topology":native_topology,"world_sheet_diagnostics":world,
            "chart_projection_fidelity":{"native_ordered_1d_loops":None,"chart_loops":134,
                "native_to_chart_comparison":"Native 6-face graph and 3D cube union do not supply a canonical 1D boundary-loop population. Direct before/after loop counts or containment changes are undefined.",
                "integer_chart_to_saved_world_embedding_exact":True,"same_loop_edge_correspondence":True,
                "chart_to_embedded_world_closure_changes":0,"chart_to_embedded_world_topology_changes":0,
                "native_face_component_count":1,"chart_face_component_count":occupancy_summary["occupied_face_components"],
                "chart_vertex_connected_component_count":occupancy_summary["occupied_vertex_connected_components"],
                "chart_occupancy_disconnect_witness_count":world["native_edges_between_distinct_chart_face_components"],
                "loop_world_vertices_are_not_native_surface_boundary_vertices":True,
                "chart_lattice_placement":"Quantized sample index floor(projected/h+0.5) is occupied pixel [index,index+1]; traced boundary vertices use this pixel-corner convention, not fitted surface boundary points.",
                "native_to_chart_foldover_certified":False},
            "no_support_or_loop_filtering":True,"no_nurbs_fit":True,"production_changed":False}
    report["w154_loop_semantics"]={
        "native_selection_graph":"same-region occupied source cells joined only by six face-neighbor strides",
        "boundary_input":"canonical mean-normal tangent chart, half-up rounded projected sample occupancy; 2D occupied unit pixel union",
        "edge_definition":"a directed pixel edge whose face-neighbor pixel is unsupported or outside; occupied pixel lies to its left",
        "tracing":"lexicographically minimum remaining edge; greedily take minimum available outgoing endpoint until returning to start or dead end",
        "closure":"retain only chains returning to start with chain length at least four; repeated final start vertex omitted in storage",
        "loop_order":"descending absolute signed area, then first vertex tuple",
        "individual_closed_guaranteed_for_retained_trace":True,"individual_simple_guaranteed":False,
        "case_closed":"bool(loops) and len(loops)==1; this is not an all-individual-loops-closed reduction",
        "geometry":"integer chart pixel corners embedded in canonical chart plane; not direct native zero-surface boundary edges",
        "current_boundary_edges_not_emitted":summary["unemitted_boundary_edges"],
        "half_up_occupancy_index_and_pixel_corner_offset":"sample rounding uses floor(projected/h+0.5), then boundary pixel vertices occupy [index,index+1] with no -0.5 recentering"}
    report["architecture_attribution"]={
        "verdict":"MIXED_ATTRIBUTION",
        "H1_MULTIPLY_CONNECTED_OBSERVED_SUPPORT":{
            "status":"SUPPORTED_FOR_CHART_SUPPORT_HOLES_PHYSICAL_SINGLE_SHEET_NOT_CERTIFIED",
            "evidence":"131 simple loops bound unsupported interior occupancy; two compound traces include three more bounded empty components. All 134 holes remain unsupported, not inferred geometry."},
        "H2_BOUNDARY_CONTRACT_LIMITATION":{
            "status":"EXACT_REJECTION_GATE_CONFIRMED_SOLE_LIMITATION_NOT_ESTABLISHED",
            "evidence":"W154 rejects len(loops)!=1 before LSQ. But the supplied domain also has self/contact singularities, two chart face components and world/chart height conflicts; merely allowing multiple loops is not shown sufficient."},
        "H3_NON_SINGLE_CHART_STRUCTURE":{
            "status":"SUPPORTED_FOR_CURRENT_CHART_DEFECTS_PHYSICAL_MULTI_SHEET_ATTRIBUTION_OPEN",
            "evidence":"Two non-simple closed traces, nine inter-loop contact pairs, native connectivity 1 -> chart 4-connectivity 2, and a 32.88685 h height spread in one chart bin. Normal and height variation are descriptive evidence, not a tuned sheet classifier."},
        "missing_evidence":["A native surface 2-complex with boundary correspondence is absent from this sample/6-face graph contract.",
            "No continuous chart injectivity or physical single-carrier certificate follows from point occupancy; normal sign variations and quantized collisions alone do not identify physical layers.",
            "The causal origin of each empty projected cell (missing observation versus projection sampling versus native zero-set construction) is not recoverable solely from these frozen sample records."],
        "structural_surface_is_not_observed_support_domain":True,
        "unsupported_holes_are_not_occluded_or_latent_evidence":True,
        "stopped_before_domain_redesign_or_nurbs_fit":True}
    assert before==preservation_manifest()
    runtime={"complete":complete,"samples":samples,"selected_rows":selected_rows,"boundary":boundary,"loops":loops,
             "world_loops":world_loops,"ids":ids,"grid":grid,"continuous":continuous,"occupancy":occupancy,
             "empty_labels":empty_labels,"plane_distance":plane_distance,"bin_ids":bin_ids,"bin_counts":bin_counts}
    return report,runtime


def run():
    report,runtime=analyze()
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/"diagnostic_correspondence.npz",sample_chart_grid=runtime["grid"],
                        chart_occupancy=runtime["occupancy"],empty_region_labels=runtime["empty_labels"],
                        loop_offsets=np.cumsum([0]+[len(loop) for loop in runtime["loops"]]),
                        loop_grid=np.concatenate(runtime["loops"]),loop_world=np.concatenate(runtime["world_loops"]),
                        complete_row_indices=runtime["selected_rows"],sample_plane_residual=runtime["plane_distance"])
    from devtools.demo import worklog_173_review_exports as review
    review.write_readmes(report,OUT)
    report["visualization"]=review.export(report,runtime,OUT)
    report["visualization"]["validation"]=review.validate(OUT)
    assert report["baseline_manifest"]==preservation_manifest()
    report["status"]="COMPLETE_TOPOLOGY_ATTRIBUTION_NO_NURBS_FIT"
    w171._write_json(OUT/"worklog_173_report.json",report)
    print("[W173] complete: 9 PNG, 9 README; MIXED_ATTRIBUTION",flush=True)
    return report,runtime


if __name__=="__main__":
    run()
