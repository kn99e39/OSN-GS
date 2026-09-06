"""Focused stored-lineage checks; no scene partition or fitter is executed."""
import ast
import inspect
import json

import numpy as np
import pytest

from devtools.demo import worklog_175_gaussian_normal_lineage_audit as audit
from osn_gs.surface import torch_gaussian_region_owned_tsdf as native


@pytest.fixture(scope='module')
def result():
    return audit.audit()


def test_exact_join_is_independent_of_mapping_storage_order():
    assert audit.join_ids(np.array([17,2,100]),np.array([100,17,2])).tolist() == [1,2,0]


def test_join_refuses_missing_or_duplicate_identity():
    with pytest.raises(ValueError,match='missing'):
        audit.join_ids(np.array([3]), np.array([1,2]))
    with pytest.raises(ValueError,match='duplicate'):
        audit.join_ids(np.array([1]), np.array([1,1]))


def test_canonical_mapping_digest_and_roles(result):
    r,rt = result
    assert r['mapping_digest'] == audit.EXPECTED_DIGEST
    assert r['canonical_identity']['population'] == 1190469
    assert r['canonical_identity']['subset_count'] == 104977
    mapping = rt['mapping']
    order = np.arange(len(mapping['region_id']))[::-1]
    assert audit.mapping_digest({k:v[order] for k,v in mapping.items()}) == r['mapping_digest']
    assert r['canonical_identity']['raw_partition_unassigned_count'] == 0
    assert r['canonical_identity']['status_counts']['ambiguous'] == 3532


def test_w154_full_population_lineage(result):
    r,_ = result
    join = r['w154_join']
    assert join['sample_count'] == 21235312
    assert all(join[k] for k in ('stable_id_index_exact','nearest_region_exact','accepted_mask_exact','owned_region_exact'))
    assert join['many_to_one_subset_merge_count'] == 0
    assert not join['sample_npz_has_region_id']


def test_w171_selection_is_one_prior_subset_not_native_identity(result):
    r,rt = result
    assert r['w171_tabletop']['contributing_subset_ids'] == {'1':17965}
    assert r['w171_tabletop']['selected_subset_ids'] == {'1':15189}
    assert r['w171_tabletop']['complete_unique_owner_gaussians'] == 3315
    assert np.array_equal(rt['selected'],rt['complete'][rt['selected_local']])
    assert len(r['all_review_subsets']) == 344  # contextual small subsets retained


def test_w174_exact_witness_and_path_provenance(result):
    r,rt = result
    endpoints = r['witness']['endpoints']
    assert [x['w154_sample_row'] for x in endpoints] == [9008662,9017086]
    assert [x['stable_gaussian_id'] for x in endpoints] == [4937175,3929355]
    assert all(x['canonical_w97_subset_id'] == 1 and x['membership'] == 'core' for x in endpoints)
    with np.load(audit.W174/'reference_surface_complex.npz') as data:
        assert np.array_equal(rt['path_triangles'],data['witness_path'])
        assert np.array_equal(rt['path_rows'],data['triangle_owner_row'][data['witness_path']])
    assert r['witness']['reference_path_subset_crossings'] == 0
    assert r['witness']['path_unique_gaussian_owners'] == 23


def test_source_contract_uses_identity_copy_and_no_new_decomposition():
    source = inspect.getsource(native.gaussian_region_membership_from_partition)
    assert 'partition.subset_ids' in source
    source = inspect.getsource(audit)
    calls = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr
             for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call)
             and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls & {'partition_surfels_region_coherent','form_surface_regions','build_candidate_graph',
                        'associate_tsdf_samples_to_gaussians','derive_native_support_boundary',
                        'fit_boundary_first_region_representative','run_candidate_f'}


def test_visualization_inventory_and_every_directory_readme():
    r = audit.read_json(audit.OUT/'worklog_175_report.json')
    assert r['visualization']['png_count'] == 9
    assert len(list(audit.OUT.rglob('*.png'))) == 9
    assert not list(audit.OUT.rglob('*.ppm'))
    for path in audit.OUT.rglob('*'):
        if path.is_dir():
            text = (path/'README.md').read_text(encoding='utf-8')
            assert len(text)>400 and 'W175' in text
    for name,pair in r['visualization']['mandatory_pair'].items():
        assert audit.sha(audit.OUT/pair['path']) == pair['sha256'] == audit.sha(audit.ROOT/pair['source'])


def test_all_frozen_artifacts_unchanged():
    r = audit.read_json(audit.OUT/'worklog_175_report.json')
    assert r['frozen_inputs_unchanged']
    assert all(audit.sha(audit.ROOT/path) == digest for path,digest in r['preservation_manifest'].items())
