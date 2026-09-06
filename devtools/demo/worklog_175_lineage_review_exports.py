"""Static lineage review: exact saved identities, no new segmentation."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from devtools.demo import worklog_175_gaussian_normal_lineage_audit as audit
from devtools.demo import worklog_172_render_view_projection_review as projection

FAMILIES = ('A_gaussian_local_surface_subsets', 'B_region_vs_subset',
            'C_zeroset_ownership', 'D_height_witness_lineage', 'E_rgb_reference')
CYAN = '#007fa3'


def subset_color(sid):
    """Stable ID-derived RGB. Color collisions are not ID equality evidence."""
    if int(sid) == 1:
        return np.array([0, 127, 163]) / 255
    value = (int(sid) * 2654435761 + 1013904223) & 0xffffffff
    return (np.array([(value >> 16) & 255, (value >> 8) & 255, value & 255]) * .65 + 45) / 255


def export(report, rt, out):
    plt = audit.W171._prepare_matplotlib()
    reviewed = []
    root = out / 'review_views'
    for family in FAMILIES:
        (root / family).mkdir(parents=True, exist_ok=True)
    # Reuse the matched, immutable W154/W155 pair before supplementary diagnostics.
    pair_paths = {}
    pair_root = out / 'mandatory_gaussian_visualization_pair'
    for name in ('Original Scene', 'Observed-Occluded'):
        source = audit.W155 / 'mandatory_gaussian_visualization_pair' / name / 'render.png'
        target = pair_root / name / 'DSC07957.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert audit.sha(source) == audit.sha(target)
        pair_paths[name] = dict(path=str(target.relative_to(out)), source=str(source.relative_to(audit.ROOT)), sha256=audit.sha(source))
    pair_readme = '''# W175 historical Gaussian matched pair

W154에서 생성하고 W155가 보존한 동일 checkpoint iteration 30000의 PNG를 byte-identical 복사했다. 카메라는 DSC07957, 해상도 648×420, black background, OSNSurfelRasterizer, Gaussian 1,190,469행이다. 두 그림의 position/rotation/scale/opacity와 row 수는 동일하다.

Original Scene은 학습된 원래 SH appearance다. Observed-Occluded는 당시 상태의 display color만 바꾼 결과로 OBSERVED=(0.10,0.85,0.35) 초록, OCCLUDED=(0.92,0.18,0.18) 빨강, UNRESOLVED=(0.60,0.60,0.62) 회색이다. 당시 집계는 767,400 / 423,069 / 0이다. 새 Gaussian, shading, geometry는 추가하지 않았다.

이는 historical W154 state의 보존이지 W168–170 이후 physical blocking의 재인증이 아니다. W175는 상태를 재판정하거나 UNRESOLVED를 구현하지 않는다. A–E의 subset 색/normal 화살표와 RGB 점 투영은 별도 진단이며 이 pair를 대체하지 않는다.
'''
    for directory in (pair_root, pair_root/'Original Scene', pair_root/'Observed-Occluded'):
        (directory/'README.md').write_text(pair_readme, encoding='utf-8')

    rr = rt['review_rows']; xyz = rt['xyz'][rr]; labels = rt['region'][rr]
    colors = np.array([subset_color(sid) for sid in labels])
    arrows = np.asarray(report['review']['normal_arrow_rows'])
    # Arrow length is a display glyph in world coordinates, never a decision margin.
    arrow_length = .04 * np.ptp(xyz, axis=0).max()
    normal_color = '#333333'
    def save(fig, family, name='tabletop.png'):
        path = root / family / name
        audit.W171._save_figure(fig, path); plt.close(fig)
        reviewed.append(str(path.relative_to(out)))
    def world_axis(ax, title):
        from matplotlib.ticker import MaxNLocator
        ax.set(xlabel='world X', ylabel='world Y', zlabel='world Z', title=title)
        ax.view_init(elev=24, azim=-65)
        ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), .01))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
    def footer(fig, text):
        fig.text(.5, .025, text, ha='center', fontsize=9)

    fig = plt.figure(figsize=(13,6.5)); ax = fig.add_subplot(121, projection='3d')
    ax.scatter(*xyz.T, c=colors, s=3, alpha=.85, depthshade=False)
    ax.quiver(*rt['xyz'][arrows].T, *rt['tw'][arrows].T, color=normal_color, length=arrow_length, linewidth=.6)
    world_axis(ax, f"W175: {len(rr):,} Gaussian centers / {len(np.unique(labels))} W97 subset IDs")
    ax2 = fig.add_subplot(122)
    ax2.scatter(xyz[:,1], xyz[:,2], c=colors, s=2, alpha=.15)
    ax2.scatter(rt['xyz'][arrows,1], rt['xyz'][arrows,2], c=normal_color, s=9)
    ax2.quiver(rt['xyz'][arrows,1], rt['xyz'][arrows,2], rt['tw'][arrows,1], rt['tw'][arrows,2],
               color=normal_color, angles='xy', scale_units='xy', scale=1/arrow_length, width=.003)
    ax2.set(xlabel='world Y', ylabel='world Z', title='64 deterministic intrinsic t_w directions (Y,Z projection)')
    ax2.set_aspect('equal', adjustable='box'); ax2.margins(.1)
    footer(fig, 'Cyan = subset 1; other colors = stored IDs (CSV). Black arrows = 64 deterministic intrinsic t_w glyphs.')
    fig.subplots_adjust(bottom=.13, top=.9)
    save(fig, FAMILIES[0])

    fig = plt.figure(figsize=(13,6.2))
    for i, title in enumerate(('W97 subset_ids (W155 frozen mapping)', 'W154 Gaussian membership.region_ids')):
        ax = fig.add_subplot(1,2,i+1, projection='3d')
        ax.scatter(*xyz.T, c=colors, s=2.5, alpha=.85, depthshade=False)
        world_axis(ax, title)
    footer(fig, 'Same Gaussian rows / same colors / same world frame. Source assignment is identity-preserving; acceptance is separate.')
    fig.subplots_adjust(bottom=.12, top=.88, wspace=.08)
    save(fig, FAMILIES[1])

    fig = plt.figure(figsize=(13,6.2))
    for i, (points, owners, title) in enumerate(((rt['complete'], rt['complete_owner'], 'W171 complete: 17,965 support rows'),
                                              (rt['selected'], rt['owner'], 'W172 diagnostic selection: 15,189 rows'))):
        ax = fig.add_subplot(1,2,i+1, projection='3d')
        ax.scatter(*points.T, c=[subset_color(sid) for sid in rt['region'][owners]], s=2, alpha=.8, depthshade=False)
        world_axis(ax, title + '\nAll stored Gaussian owners: W97 subset 1')
    footer(fig, 'Colors encode Gaussian subset identity, NOT TSDF component. Every support row is shown; no size filtering.')
    fig.subplots_adjust(bottom=.13, top=.87, wspace=.08)
    save(fig, FAMILIES[2])

    fig, axes = plt.subplots(1,2,figsize=(13,6.5))
    for ax, dims, names in zip(axes, ((1,2),(0,2)), (('Y','Z'),('X','Z'))):
        ax.scatter(*rt['selected'][:,dims].T, c='#cccccc', s=2, alpha=.45)
        ax.plot(*rt['path_world'][:,dims].T, color=CYAN, lw=2, label='W174 path: all owners subset 1')
        for j, ep in enumerate(report['witness']['endpoints']):
            p = np.array(ep['world_xyz']); g = np.array(ep['gaussian_world_xyz']); n = np.array(ep['gaussian_tw'])
            ec = '#b52976' if j == 0 else '#da7615'
            ax.scatter(*p[list(dims)], c=ec, marker='o', s=45, zorder=5)
            ax.scatter(*g[list(dims)], c=ec, marker='*', s=125, zorder=6)
            ax.plot(*np.stack((p,g))[:,dims].T, c=ec, ls='--', lw=1)
            ax.quiver(*g[list(dims)], *n[list(dims)], color=ec, angles='xy', scale_units='xy', scale=1/arrow_length)
            inward = j == 0 and dims == (1,2)
            ax.annotate(f"row {ep['selected_row']}\nG{ep['stable_gaussian_id']} / subset 1 / core",
                        p[list(dims)], xytext=(-12 if inward else 8,20 if j == 0 else -35),
                        ha='right' if inward else 'left', textcoords='offset points', fontsize=8)
        ax.set(xlabel=f'world {names[0]}', ylabel=f'world {names[1]}', title=f'Witness / stored owner / intrinsic normal ({names[0]},{names[1]})')
        ax.margins(.15); ax.set_aspect('equal', adjustable='datalim')
    axes[0].legend(loc='lower left', fontsize=8)
    footer(fig, 'Circle = support; star = stored nearest Gaussian; dashed = ownership link; arrow = t_w. Not a contributor or Gaussian-graph path.')
    fig.subplots_adjust(bottom=.16, top=.9, wspace=.25)
    save(fig, FAMILIES[3])

    cameras, metadata = audit.W171.w155._build_named_cameras(audit.W171.DEFAULT_DATASET, 'images_8','sparse/0',-1,8,'cpu')
    rgb_accounting = {}
    for name, camera in cameras.items():
        reference = audit.W171.DEFAULT_DATASET / 'images_8' / name
        with Image.open(reference) as im:
            rgb = np.asarray(im.convert('RGB'))
        pixels, valid = projection.project(xyz, camera)
        fig, axes = plt.subplots(1,2,figsize=(13,5))
        for ax in axes:
            ax.imshow(rgb); ax.set_axis_off()
            ax.set(xlim=(-.5,camera.image_width-.5),ylim=(camera.image_height-.5,-.5))
        axes[0].set_title(f'{name}: original photograph (not Gaussian render)')
        axes[1].scatter(*pixels[valid].T, c=colors[valid], s=2, alpha=.8, linewidths=0)
        axes[1].set_title(f'Canonical subset centers: {valid.sum():,} projected / {len(rr):,}')
        footer(fig, 'Renderer-canonical pixel projection, no depth culling. RGB overlap is secondary context, never ownership evidence.')
        fig.subplots_adjust(bottom=.12, top=.9, wspace=.04)
        save(fig, FAMILIES[4], Path(name).stem+'.png')
        rgb_accounting[name] = dict(camera=metadata[name], projected_count=int(valid.sum()),
            outside_frustum_or_near=int((~valid).sum()), reference_sha256=audit.sha(reference))

    with (out/'subset_inventory.csv').open('w',encoding='utf-8',newline='') as handle:
        fields = ['subset_id','gaussian_count_global','gaussian_count_in_frozen_aabb','gaussian_count_review',
                  'complete_unique_owner_gaussians','complete_support_count','selected_support_count','rgb_hex']
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for item in report['all_review_subsets']:
            row = {k:item[k] for k in fields[:-1]}
            row['rgb_hex'] = '#' + ''.join(f'{int(x*255):02x}' for x in subset_color(item['subset_id']))
            writer.writerow(row)
    meanings = (
        'A는 frozen tabletop AABB의 learned Gaussian 9,243행과 AABB 밖 stored owner 124행을 포함한 9,367개 center를 W97 subset ID로 표시한다. 344개 subset을 크기로 제거하지 않는다. 검은 화살표 64개는 row-order 균등 index의 deterministic t_w이며 길이는 display glyph다.',
        'B는 동일한 9,367 Gaussian을 왼쪽 W97 subset_ids, 오른쪽 W154 Gaussian membership.region_ids로 비교한다. W154의 source assignment가 그대로 복사한다는 코드와 전체 stored support join이 근거다. 두 패널은 같아야 정상이며 acceptance status는 identity와 별도다.',
        'C는 W171 complete support 17,965점과 W172 diagnostic selected 15,189점을 저장된 nearest Gaussian의 W97 subset ID로 표시한다. 모두 subset 1이므로 청록이다. TSDF component 색칠이 아니며 complete 패널에는 제외 component도 전부 남긴다.',
        'D는 W174의 103-triangle reference path를 그대로 읽어 world YZ/XZ로 표시한다. 회색은 selected support, 청록 path는 Gaussian owner ID가 전부 subset 1임을 나타낸다. 원은 witness 4043(자홍)/4051(주황), 별은 stored owner Gaussian, 점선은 owner 연결, 화살표는 원래 t_w다. Path는 Gaussian graph path가 아니며 소속 boundary crossing 0은 stored ownership에 한정된다.',
        'E는 DSC07960/DSC08003/DSC08043 원본 사진 위에 같은 9,367 Gaussian center의 subset 색을 투영한 secondary context다. 왼쪽은 RGB 사진, 오른쪽은 center overlay다. Renderer canonical half-pixel projection을 사용하고 y-flip은 하지 않는다. Frustum/near 검사만 하며 depth visibility culling은 없다.',
    )
    common = '''
## 입력·상태와 범례

Checkpoint는 W154/W155의 2DGS iteration 30000이다. W155 stable_gaussian_id → region_id mapping을 exact join하며 여기서 canonical subset은 W96/W97 intrinsic-normal coverage-first 계열에 한정한다. W150 form_surface_regions identity와 동치라고 주장하지 않는다. 이 mapping은 TSDF geometry로 재분해한 결과가 아니다.

Subset 1은 청록(#007fa3), 다른 ID는 고정 ID-derived RGB다. 색이 비슷해도 동일 ID의 증거가 아니며 전체 ID·RGB·행 수는 root subset_inventory.csv에 있다. Ambiguous/fallback Gaussian도 raw subset ID 색을 유지하고 status는 report에 별도 회계한다. A–E는 OBSERVED/OCCLUDED 상태 색이 아니다. 입력의 non-uncertain learned population을 역사적 observed branch로 부르되 현재 physical observation 판정으로 승격하지 않는다.

## 공통 조건과 한계

World 좌표와 frozen W171 AABB를 유지한다. Gaussian center는 scientific scatter point이며 splat geometry를 대체하는 새 Gaussian이 아니다. Normal은 canonical getter의 intrinsic t_w이고 부호를 뒤집지 않는다. Support와 subset은 크기별로 버리지 않았고 normal glyph만 가독성을 위해 deterministic 64행을 택한다. Glyph 길이·색은 시각화 전용이며 threshold, ownership, graph에 사용하지 않는다.

Stored nearest-center owner는 renderer contributor 또는 TSDF observation의 causal owner가 아니다. RGB overlap, 그림상 연속성, 같은 subset 색은 physical same-sheet의 증명이 아니다. W174 path는 frozen scalar에서 재추출된 historical reference이며 original persistent triangle 또는 Gaussian accepted-edge path로 부르지 않는다. W154/W171 원본, partition, geometry, boundary, NURBS는 수정하지 않았다.

## 분석 및 평가

W97 ID → W154 owned_region_id → W171 tabletop의 저장 계보는 보존된다. Witness 두 owner는 모두 subset 1의 core이고 경로에서도 subset crossing은 0이다. 그러나 W150의 더 풍부한 tangent/consensus/bridge 계약이 W154에서 실행된 것은 아니다. 그림은 이 계약 차이와 physical provenance의 미해결 부분을 없애지 않는다.
'''
    for family, meaning in zip(FAMILIES, meanings):
        (root/family/'README.md').write_text('# W175 '+family+'\n\n'+meaning+'\n'+common,encoding='utf-8')
    index = '\n'.join(f'- `{f}`: {m}' for f,m in zip(FAMILIES,meanings))
    for directory in (out,root):
        (directory/'README.md').write_text('# W175 Gaussian-normal lineage 검토\n\n'+index+'\n'+common+
            '\n필수 Gaussian pair는 `mandatory_gaussian_visualization_pair`에 historical byte-identical PNG로 별도 보존했다. 새로운 GPU render나 replay cache 복사는 없다.\n',encoding='utf-8')
    all_pngs = list(out.rglob('*.png'))
    for path in all_pngs:
        with Image.open(path) as im:
            im.verify()
    assert len(all_pngs) == 9 and not list(out.rglob('*.ppm'))
    for path in out.rglob('*'):
        if path.is_dir():
            assert (path/'README.md').exists()
    return dict(review_pngs=reviewed, png_count=len(all_pngs), mandatory_pair=pair_paths,
        rgb_cameras=rgb_accounting, gaussian_geometry_changed=False, depth_culling=False,
        support_display_subsampling=False, normal_glyph_count=len(arrows), normal_glyph_length=float(arrow_length),
        skill_effect='static scientific PNG; no inline app or generated imagery')
