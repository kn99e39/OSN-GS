"""W175 read-only stored-ID joins. No graph/partition/ownership recomputation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from devtools.demo import worklog_173_tabletop_multi_loop_domain_attribution as w173
from devtools.demo import worklog_155_intrinsic_normal_gaussian_region_viability_audit as w155
from osn_gs.surface.torch_region_coherent_surfel_partition import RegionCoherenceConfig

W171 = w173.w171
W154 = W171.DEFAULT_W154
W155 = ROOT / 'output/confirmed/155_intrinsic_normal_gaussian_region_viability_audit'
W174 = ROOT / 'output/174_reference_surface_complex_attribution'
OUT = ROOT / 'output/175_gaussian_normal_local_surface_lineage_audit'
STATUS = ('core', 'attached', 'ambiguous', 'rejected', 'unassigned')
EXPECTED_DIGEST = '06c9e1cbc730f06581895b32ad683e8822c7626eb3de9017fa8f83aaf0248bce'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return w155._sha256_file(path)


def mapping_digest(mapping):
    order = np.argsort(mapping['stable_gaussian_id'], kind='stable')
    return w155._sha256_bytes(*(mapping[k][order] for k in
        ('stable_gaussian_id', 'region_id', 'membership_status')))


def join_ids(ids, mapping_ids):
    """Exact stable-ID join; never nearest-position or silent missing-owner fallback."""
    order = np.argsort(mapping_ids, kind='stable')
    sorted_ids = mapping_ids[order]
    if len(np.unique(sorted_ids)) != len(sorted_ids):
        raise ValueError('duplicate stable Gaussian ID')
    loc = np.searchsorted(sorted_ids, ids)
    if np.any(loc >= len(sorted_ids)) or not np.array_equal(sorted_ids[loc], ids):
        raise ValueError('missing stable Gaussian ID')
    return order[loc]


def histogram(values):
    ids, count = np.unique(values, return_counts=True)
    return {str(int(i)): int(c) for i, c in zip(ids, count)}


def status_counts(values):
    return {name: int(np.sum(values == i)) for i, name in enumerate(STATUS)}


def preservation_paths():
    """Freeze all W154/W155 and W171-174 artifacts plus audited source contracts."""
    paths = []
    for root in (W154, W155, W171.DEFAULT_OUT, w173.w172.OUT, w173.OUT, W174):
        paths.extend(p for p in root.rglob('*') if p.is_file())
    for rel in (
        'devtools/demo/candidate_f_gaussian_region_owned_tsdf.py',
        'devtools/demo/worklog_155_intrinsic_normal_gaussian_region_viability_audit.py',
        'osn_gs/surface/torch_region_coherent_surfel_partition.py',
        'osn_gs/surface/torch_coverage_first_subset_partition.py',
        'osn_gs/surface/torch_surfel_surface_orientation.py',
        'osn_gs/gaussian/torch_surfel_model.py',
        'osn_gs/surface/torch_gaussian_region_owned_tsdf.py',
        'osn_gs/surface/torch_gaussian_surface_region_formation.py',
        'osn_gs/surface/torch_gaussian_manifold_affinity.py',
        'osn_gs/surface/torch_visible_surface_construction.py',
    ):
        paths.append(ROOT / rel)
    return sorted(set(paths))


def audit():
    torch.set_num_threads(4)
    historical = read_json(W155 / 'worklog_155_report.json')
    with np.load(W155 / 'gaussian_id_region_status_mapping.npz') as data:
        mapping = {k: data[k] for k in data.files}
    digest = mapping_digest(mapping)
    assert digest == EXPECTED_DIGEST == historical['standalone_gaussian_region_replay']['mapping_hash']
    assert digest == (W155 / 'gaussian_id_region_status_mapping.sha256').read_text().strip()
    assert RegionCoherenceConfig().payload() == historical['inputs']['same_current_partition_parameters']
    checkpoint = Path(historical['inputs']['checkpoint'])
    assert sha(checkpoint) == historical['inputs']['checkpoint_sha256']
    model, _ = w155._load_surfel_model_safe(checkpoint, 'cpu')
    orientation, active = w155._active_orientation(model)
    ids = orientation.gaussian_ids.numpy()
    mapped = join_ids(ids, mapping['stable_gaussian_id'])
    assert np.array_equal(mapped, np.arange(len(ids)))  # W154 association-index population
    xyz = orientation.positions.numpy().copy()
    tw = orientation.surface_normal.numpy().copy()
    del model, orientation
    region = mapping['region_id']
    status = mapping['membership_status']
    accepted_gaussian = np.isin(status, (0, 1))
    with np.load(W154 / 'candidate_f_association.npz') as data:
        nearest = data['nearest_gaussian_index']
        stored_ids = data['nearest_gaussian_id']
        assert np.array_equal(ids[nearest], stored_ids)
        distance = data['nearest_distance']
    del stored_ids
    with np.load(W154 / 'candidate_f_region_owned_support.npz') as data:
        nearest_region = data['nearest_region_id']
        owned = data['owned_region_id']
        accepted = data['accepted_mask'].astype(bool)
    # Whole W154 population, not just the positive selected case.
    assert np.array_equal(region[nearest], nearest_region)
    assert np.array_equal(accepted_gaussian[nearest], accepted)
    assert np.array_equal(np.where(accepted, region[nearest], -1), owned)
    global_owned = histogram(owned)
    sample_count = len(owned)
    case_dir = w173.w172.OUT / 'case_artifacts/real_tabletop_coherent'
    baseline = read_json(case_dir / 'result.json')
    spec = baseline['baseline']['selection']
    with np.load(W154 / 'candidate_f_tsdf_surface_samples.npz') as data:
        assert 'region_id' not in data.files
        world = data['world_xyz']
    box = np.all(world >= spec['aabb_min'], axis=1) & np.all(world <= spec['aabb_max'], axis=1)
    complete_rows = np.flatnonzero(box & accepted & (owned == 1))
    box_rows = np.flatnonzero(box)
    complete = world[complete_rows].copy()
    with np.load(case_dir / 'selected_support.npz') as data:
        selected_local = data['complete_row_indices']
        selected = data['world_xyz']
        cells = data['cell_indices']
    assert np.array_equal(complete[selected_local], selected)
    selected_rows = complete_rows[selected_local]
    owner = nearest[selected_rows]
    complete_owner = nearest[complete_rows]
    assert len(complete_rows) == 17965 and len(selected_rows) == 15189
    with np.load(W174 / 'reference_surface_complex.npz') as data:
        vertices, triangles = data['vertices'], data['triangles']
        path_triangles = data['witness_path']
        path_rows = data['triangle_owner_row'][path_triangles]
    path_world = vertices[triangles[path_triangles]].mean(axis=1)
    path_owner = owner[path_rows]
    path_region = region[path_owner]
    witness = []
    for row in (4043, 4051):
        g = int(owner[row])
        witness.append(dict(selected_row=row, complete_row=int(selected_local[row]),
            w154_sample_row=int(selected_rows[row]), cell=cells[row].tolist(),
            world_xyz=selected[row].tolist(), gaussian_active_index=g,
            stable_gaussian_id=int(ids[g]), gaussian_world_xyz=xyz[g].tolist(),
            gaussian_tw=tw[g].tolist(), canonical_w97_subset_id=int(region[g]),
            w154_owned_region_id=int(owned[selected_rows[row]]), membership=STATUS[int(status[g])],
            ambiguous_multi_region=bool(mapping['ambiguous_multi_region'][g]),
            nearest_distance=float(distance[selected_rows[row]]),
            nearest_distance_in_h=float(distance[selected_rows[row]] / baseline['baseline']['support']['h'])))
    same = witness[0]['canonical_w97_subset_id'] == witness[1]['canonical_w97_subset_id']
    assert same and np.all(path_region == 1)
    crop = np.all(xyz >= spec['aabb_min'], axis=1) & np.all(xyz <= spec['aabb_max'], axis=1)
    review_rows = np.union1d(np.flatnonzero(crop), np.unique(complete_owner))
    # Inventory includes every contextual subset, including ambiguous/fallback; no size filter.
    inventory = []
    all_global_counts = histogram(region)
    for sid in np.unique(region[review_rows]):
        sid = int(sid)
        contextual = review_rows[region[review_rows] == sid]
        inventory.append(dict(subset_id=sid, gaussian_count_global=all_global_counts[str(sid)],
            gaussian_count_in_frozen_aabb=int(np.sum(crop & (region == sid))),
            gaussian_count_review=len(contextual), review_membership=status_counts(status[contextual]),
            complete_support_count=int(np.sum(region[complete_owner] == sid)),
            selected_support_count=int(np.sum(region[owner] == sid)),
            complete_unique_owner_gaussians=len(np.unique(complete_owner[region[complete_owner] == sid]))))
    # Read historical cut/conflict edges only. Do not rebuild accepted graph or infer a new path.
    cut_evidence = {}
    with np.load(W155 / 'region_boundary_conflict_edges.npz') as data:
        for key in data.files:
            edges = data[key]
            local = edges[crop[edges[:, 0]] & crop[edges[:, 1]]]
            crossing = region[local[:, 0]] != region[local[:, 1]]
            cut_evidence[key] = dict(global_count=len(edges), within_aabb_count=len(local),
                within_aabb_cross_subset_count=int(crossing.sum()),
                within_aabb_same_subset_count=int((~crossing).sum()))
    r1 = region == 1
    core = r1 & (mapping['partition_role'] == 0)
    concentration = np.linalg.eigvalsh(tw[core].astype(float).T @ tw[core].astype(float))
    concentration = float(concentration[-1] / concentration.sum())
    angle = lambda a, b: float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)) / (np.linalg.norm(a)*np.linalg.norm(b)), 0, 1))))
    report = dict(status='COMPLETE_READ_ONLY_LINEAGE_AUDIT',
        architecture_verdict='CANONICAL_LOCAL_SURFACE_IDENTITY_PRESERVED',
        identity_verdict='CANONICAL_LOCAL_SURFACE_IDENTITY_PRESERVED',
        verdict_scope='W96/W97 intrinsic-normal coverage-first branch, same W154/W155 checkpoint; not equivalence to W150 form_surface_regions',
        richer_contract_status='NOT_THE_ACTIVE_W96_W97_BRANCH; W150 form_surface_regions IDs never enter W154; crosswalk unavailable, not a measured many-to-one merge',
        mapping_digest=digest, checkpoint_sha256=historical['inputs']['checkpoint_sha256'],
        historical_config=historical['inputs']['same_current_partition_parameters'],
        canonical_identity=dict(module='torch_region_coherent_surfel_partition.partition_surfels_region_coherent',
            source_field='subset_ids', artifact_field='region_id', per_gaussian=True,
            population=len(ids), subset_count=len(np.unique(region)), status_counts=status_counts(status),
            raw_partition_unassigned_count=int(np.sum(region < 0)),
            deterministic_scope='fixed checkpoint and row order; size-ranked IDs are not temporal or arbitrary-row-permutation invariant',
            graph_replayed_this_batch=False, historical_two_replays_agreed=True),
        w154_join=dict(sample_count=sample_count, stable_id_index_exact=True, nearest_region_exact=True,
            accepted_mask_exact=True, owned_region_exact=True, many_to_one_subset_merge_count=0,
            sample_npz_has_region_id=False, ownership_source='candidate_f_region_owned_support.npz:owned_region_id',
            global_owned_region1_count=global_owned.get('1', 0)),
        w171_tabletop=dict(selection=spec, complete_support_count=len(complete), selected_support_count=len(selected),
            contributing_subset_ids=histogram(region[complete_owner]),
            selected_subset_ids=histogram(region[owner]), global_region1_gaussian_count=int(r1.sum()),
            global_region1_status=status_counts(status[r1]),
            complete_unique_owner_gaussians=len(np.unique(complete_owner)),
            selected_unique_owner_gaussians=len(np.unique(owner)),
            complete_owner_status=status_counts(status[complete_owner]),
            box_all_support_count=len(box_rows), box_nearest_status=status_counts(status[nearest[box_rows]]),
            region1_core_concentration=concentration),
        witness=dict(classification='SAME_CANONICAL_LOCAL_SURFACE_SUBSET', endpoints=witness,
            endpoint_unsigned_normal_angle_deg=angle(tw[owner[4043]], tw[owner[4051]]),
            reference_path_triangle_count=len(path_rows), reference_path_unique_support_rows=len(np.unique(path_rows)),
            reference_path_subset_histogram=histogram(path_region), reference_path_subset_crossings=int(np.sum(path_region[1:] != path_region[:-1])),
            path_unique_gaussian_owners=len(np.unique(path_owner)),
            path_owner_status=status_counts(status[path_owner]),
            gaussian_graph_path='NOT_PERSISTED; W174 triangle path is not a Gaussian accepted-edge path',
            ownership_meaning='stored Euclidean nearest-center assignment; NOT renderer contributor or TSDF fusion causality'),
        historical_cut_evidence=cut_evidence, all_review_subsets=inventory,
        review=dict(aabb_gaussian_count=int(crop.sum()), review_gaussian_count=len(review_rows),
            review_subset_count=len(inventory), outside_aabb_owners_retained=int(np.sum(~crop[review_rows])),
            normal_arrow_rows=review_rows[np.linspace(0, len(review_rows)-1, min(64,len(review_rows)), dtype=int)].tolist(),
            population='all non-uncertain learned Gaussian centers in frozen AABB plus all stored complete-support owners; not current physical OBSERVED classification'),
        prohibited_changes=dict(partition=False, thresholds=False, graph=False, ownership=False,
            geometry=False, boundary=False, nurbs_fit=False, production=False, replay_cache_copy=False))
    report['contract_audit'] = dict(
        canonical_selection_evidence=['W96 explicitly declares intrinsic-normal coverage-first canonical direction',
            'W97 changes only region union rule on the W96 candidate graph',
            'W154.run explicitly invokes W97 default config', 'W155 stored mapping replays the same W154 checkpoint'],
        same_surface_relation=dict(local_neighborhood='symmetrized/deduplicated 8-NN; median local spacing; distance <= 2*min(spacing_i,spacing_j)',
            intrinsic_normal='TorchGaussianSurfelModel.get_normal -> R[:,:,2] -> derive_surface_orientation_from_surfel.surface_normal',
            sign_independent='abs(dot(t_w_i,t_w_j)) >= 0.85; region scatter sum(n*n.T)',
            tangent_mutual_tangent='NOT_ACTIVE; t_u/t_v/scales carried but unused by W97 membership; require_positional_continuity=False',
            spatial_separation='local center-distance gate only; no active normal-offset/parallel-sheet separation gate',
            multi_edge_consensus='NOT_ACTIVE; region orientation concentration is not shared-neighbor consensus',
            bridge_path_consistency='NOT_ACTIVE as W10/W150 bridge/path mechanism; only concentration-gated union and non-bridging one-hop propagation',
            region_coherence='structural-core scatter concentration >= (1+0.85)/2 = 0.925 at each union',
            ambiguity='ambiguous_multi_region flag retained with one deterministic raw owner; W154 excludes ambiguous/fallback from accepted ownership'),
        transformations=[
            dict(edge='checkpoint -> non-uncertain Gaussian population',type='filtering-only',identity='stable_gaussian_ids retained; all 1190469 active in this checkpoint'),
            dict(edge='intrinsic t_w + local graph -> W97 subset_ids',type='recomputation',identity='Gaussian branch creates identity BEFORE TSDF ownership'),
            dict(edge='W97 subset_ids -> W154 membership.region_ids',type='identity-preserving',identity='direct tensor conversion/copy; no relabel/merge'),
            dict(edge='W97 partition_role/ambiguity -> W154 accepted_mask',type='filtering-only',identity='core/attached accepted; raw ID not coarsened'),
            dict(edge='raw TSDF sample -> nearest_gaussian_index/id',type='projection-only',identity='Euclidean nearest-center association; geometry-to-existing-ID lookup, not observation contributor lineage'),
            dict(edge='nearest Gaussian -> nearest_region_id -> owned_region_id',type='identity-preserving + filtering-only',identity='accepted inherits identical ID, otherwise -1'),
            dict(edge='W154 support -> W171 tabletop',type='filtering-only',identity='frozen AABB and prior region 1; other existing regions excluded, not merged'),
            dict(edge='owned_region_id -> native TSDF components',type='one-to-many split of support, not Gaussian identity',identity='same-region 6-face connected support groups; no new region IDs'),
            dict(edge='W171 complete -> W172 largest diagnostic component',type='filtering-only',identity='stored complete_row_indices; no production filtering rule'),
            dict(edge='W172 rows -> W174 reference triangles/path',type='reference geometry recomputation, ownership identity-preserving',identity='persisted triangle_owner_row composes with W154 nearest-Gaussian owner; never redefines subset')],
        native_connectivity_role='downstream structural-unit decomposition/pre-fit gate in practice; NOT a second Gaussian identity construction',
        unavailable_provenance=['W97 full accepted graph and chronological union path not persisted in W155 (only cut/conflict edges)',
            'W150 form_surface_regions same-checkpoint per-Gaussian ID crosswalk not consumed or persisted by W154',
            'nearest-center assignment is not per-observation TSDF contributor causality'])
    runtime = dict(xyz=xyz, tw=tw, ids=ids, region=region, status=status, review_rows=review_rows,
        complete=complete, selected=selected, complete_owner=complete_owner, owner=owner,
        complete_rows=complete_rows, selected_rows=selected_rows, selected_local=selected_local,
        path_world=path_world, path_rows=path_rows, path_owner=path_owner, path_region=path_region,
        path_triangles=path_triangles, mapping=mapping)
    del world, owned, nearest_region, nearest, distance
    return report, runtime


def run(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    paths = preservation_paths()
    print('[W175] hashing frozen artifacts', flush=True)
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    print('[W175] exact stored lineage joins', flush=True)
    report, rt = audit()
    from devtools.demo import worklog_175_lineage_review_exports as review
    report['visualization'] = review.export(report, rt, out)
    np.savez(out / 'tabletop_lineage.npz', complete_w154_rows=rt['complete_rows'],
        selected_w154_rows=rt['selected_rows'], selected_complete_rows=rt['selected_local'],
        selected_gaussian_active_index=rt['owner'], selected_stable_gaussian_id=rt['ids'][rt['owner']],
        selected_w97_subset_id=rt['region'][rt['owner']], selected_world_xyz=rt['selected'],
        complete_stable_gaussian_id=rt['ids'][rt['complete_owner']],
        path_triangles=rt['path_triangles'], path_selected_rows=rt['path_rows'],
        path_gaussian_active_index=rt['path_owner'], path_w97_subset_id=rt['path_region'],
        path_world_xyz=rt['path_world'], review_gaussian_active_index=rt['review_rows'])
    report['preservation_manifest'] = manifest
    report['frozen_inputs_unchanged'] = all(sha(ROOT / p) == digest for p, digest in manifest.items())
    assert report['frozen_inputs_unchanged']
    (out / 'worklog_175_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    report = run()
    print(json.dumps({k: report[k] for k in ('status', 'identity_verdict', 'w171_tabletop', 'witness')}, indent=2))
