"""Static scientific review exports for W173; all support and loops retained."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from devtools.demo import worklog_173_tabletop_multi_loop_domain_attribution as w173
from devtools.demo import worklog_172_render_view_projection_review as projection

w171=w173.w171
FAMILIES=("A_complete_tabletop_support_world","B_selected_component_world","C_all_loops_chart",
          "D_containment_nesting","E_loop_scale_distribution","F_world_vs_chart_correspondence","G_rgb_reference")


def export(report,runtime,out):
    plt=w171._prepare_matplotlib()
    from matplotlib.collections import LineCollection
    xyz=runtime["samples"].world_xyz.numpy()
    full=runtime["complete"].world_xyz.numpy()
    loops=runtime["loops"];world=runtime["world_loops"];ids=runtime["ids"];records=report["loops"]
    colors=["#c026d3" if not row["simple_valid_polygon"] else plt.get_cmap("tab20")(int(row["loop_id"][1:])%20) for row in records]
    reviewed=[]
    for family in FAMILIES:
        (out/"review_views"/family).mkdir(parents=True,exist_ok=True)
    def save(fig,family,name="tabletop.png"):
        target=out/"review_views"/family/name
        w171._save_figure(fig,target);plt.close(fig);reviewed.append(str(target.relative_to(out)))
    def loops_chart(ax,*,labels=False,neutral=False):
        for loop,color,loop_id in zip(loops,colors,ids):
            line=np.vstack((loop,loop[:1]))
            ax.plot(*line.T,color="#858585" if neutral else color,linewidth=.7)
            if labels:
                anchor=loop.mean(axis=0)
                ax.text(*anchor,loop_id,fontsize=6,color="black",ha="center",va="center")
        ax.set(xlabel="canonical chart u / h",ylabel="canonical chart v / h")
        ax.set_aspect("equal",adjustable="box")
    def loops_world(ax):
        for loop,color in zip(world,colors):
            ax.plot(*np.vstack((loop,loop[:1])).T,color=color,linewidth=.7)
    # A: every complete support row, no display sampling.
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection="3d")
    excluded=np.ones(len(full),bool);excluded[runtime["selected_rows"]]=False
    ax.scatter(*full[excluded].T,s=2,c=w171.CONTEXT_RGB,alpha=.8,label="other 312 components / 2,776 rows")
    ax.scatter(*xyz.T,s=1.1,c=w171.SUPPORT_RGB,alpha=.55,label="selected component / 15,189 rows")
    w171._equal_3d_axes(ax,full);w171._label_common_world(ax);ax.legend(fontsize=7)
    ax.set_title("W173 tabletop — complete W171 support, all 17,965 rows")
    save(fig,FAMILIES[0])
    # B: all support and loop world embeddings.
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection="3d")
    ax.scatter(*xyz.T,s=1,c=w171.SUPPORT_RGB,alpha=.4);loops_world(ax)
    for loop_id in ("L000","L009","L023"):
        i=ids.index(loop_id);ax.text(*world[i].mean(axis=0),loop_id,fontsize=8)
    w171._equal_3d_axes(ax,np.concatenate([xyz]+world));w171._label_common_world(ax)
    ax.set_title("Selected native component + all 134 chart-plane loops\nmagenta: self-contact loops L000 / L009; no NURBS")
    save(fig,FAMILIES[1])
    # C: all loops, including the one-cell tail; every loop has an ID.
    fig,ax=plt.subplots(figsize=(14,8))
    ax.imshow(runtime["occupancy"],origin="lower",extent=(0,runtime["occupancy"].shape[1],0,runtime["occupancy"].shape[0]),cmap="Blues",alpha=.28,interpolation="nearest")
    loops_chart(ax,labels=True)
    ax.set(xlim=(-2,runtime["occupancy"].shape[1]+2),ylim=(-2,runtime["occupancy"].shape[0]+2))
    ax.set_title("All 134 authoritative loops — complete ID population, no loop-size filter\nIDs are stable geometry IDs; see loop_inventory.md for the full size-ranked accounting")
    save(fig,FAMILIES[2])
    # D: exact occupancy and singularities, without morphological display edits.
    fig,axes=plt.subplots(1,3,figsize=(15,5),gridspec_kw={"width_ratios":[2.4,1,1]})
    mask=runtime["occupancy"]
    for ax in axes:
        ax.imshow(mask,origin="lower",extent=(0,mask.shape[1],0,mask.shape[0]),cmap="Blues",interpolation="nearest",alpha=.45)
        loops_chart(ax)
    axes[0].set_title("Generalized containment: not a valid polygon tree\n131 simple hole loops; L000/L009 compound; L023 exterior")
    axes[1].set(xlim=(114,121),ylim=(3,11));axes[1].set_title("L000 self-contact at (117,7)")
    axes[2].set(xlim=(33,41),ylim=(61,68));axes[2].set_title("L023: one chart cell\nnative-connected, chart face-disconnected")
    axes[1].scatter([117],[7],marker="x",s=70,c="#c026d3")
    axes[2].plot([36.5,37.5],[64.5,65.5],color="#ef4444",marker="o",linewidth=1)
    fig.suptitle("134 bounded unsupported occupancy regions; 2 occupied face-components (6,608 + 1 cells)")
    save(fig,FAMILIES[3])
    # E: every row in the rank curves, with no threshold line.
    fig,axes=plt.subplots(1,3,figsize=(13,4.4))
    rank=np.arange(1,len(records)+1)
    metrics=(("edge_count","edge count"),("perimeter_world_units","perimeter / world units"),
             ("signed_area_grid_units_squared","absolute algebraic area / h²"))
    for ax,(key,label) in zip(axes,metrics):
        vals=np.abs([row[key] for row in records])
        ax.plot(rank,vals,".",markersize=5,color="#2563eb")
        for i,row in enumerate(records):
            if not row["simple_valid_polygon"]:
                ax.plot(i+1,vals[i],"x",color="#c026d3",markersize=7)
        ax.set(xlabel="frozen W172 size rank (all 134)",ylabel=label,yscale="log")
        ax.grid(alpha=.2)
    fig.suptitle("Loop-scale accounting — every loop retained; magenta = non-simple algebraic area only")
    fig.tight_layout()
    save(fig,FAMILIES[4])
    # F: same points and stable loop IDs on world and chart, plus height conflict.
    fig=plt.figure(figsize=(13,9))
    ax=fig.add_subplot(221,projection="3d");chart=fig.add_subplot(222)
    values=runtime["plane_distance"]/runtime["samples"].h
    scatter=ax.scatter(*xyz.T,c=values,s=1.6,cmap="coolwarm",vmin=values.min(),vmax=values.max())
    loops_world(ax);w171._label_common_world(ax);w171._equal_3d_axes(ax,np.concatenate([xyz]+world))
    chart_uv=np.column_stack(((xyz-runtime["boundary"].chart_origin.numpy())@runtime["boundary"].tangent_u.numpy(),
                              (xyz-runtime["boundary"].chart_origin.numpy())@runtime["boundary"].tangent_v.numpy()))/runtime["samples"].h
    chart.scatter(*chart_uv.T,c=values,s=1.6,cmap="coolwarm",vmin=values.min(),vmax=values.max());loops_chart(chart)
    for loop_id in ("L000","L009","L023"):
        i=ids.index(loop_id)
        ax.text(*world[i].mean(axis=0),loop_id,fontsize=8)
        chart.text(*loops[i].mean(axis=0),loop_id,fontsize=8)
    colorbar_axes=fig.add_axes((.92,.60,.015,.27))
    fig.colorbar(scatter,cax=colorbar_axes,label="signed chart-plane residual / h")
    ax.set_title("Actual support heights + planar loop embeddings")
    chart.set_title("Same support values in canonical chart")
    height=fig.add_subplot(223);height.scatter(chart_uv[:,0],values,s=1,c="#2563eb",alpha=.5)
    height.set(xlabel="canonical chart u / h",ylabel="signed plane residual / h",title="All native support rows; no plane cutoff")
    witness=report["world_sheet_diagnostics"]["max_chart_bin_spread_witness"]
    detail=fig.add_subplot(224)
    rows=np.asarray(witness["sample_rows"])
    detail.scatter(chart_uv[:,1],values,s=1,c="#b0b0b0",alpha=.25)
    detail.plot(chart_uv[rows,1],values[rows],"o-",c="#c026d3",label=f"rows {rows[0]} / {rows[1]}")
    detail.set(xlabel="canonical chart v / h",ylabel="signed plane residual / h",
               title=f"Same quantized bin {witness['chart_bin']}: {witness['normal_height_separation_h']:.4f} h height span")
    detail.legend(fontsize=8)
    fig.suptitle("World / chart correspondence — projection does not certify one physical layer")
    fig.subplots_adjust(hspace=.3,wspace=.25,bottom=.08,top=.92,right=.87)
    save(fig,FAMILIES[5])
    # G: real RGB photographs, not Gaussian recoloring; no state semantics.
    cameras,_=w171.w155._build_named_cameras(w171.DEFAULT_DATASET,"images_8","sparse/0",-1,8,"cpu")
    camera_accounting={}
    for camera_name,camera in cameras.items():
        with Image.open(w171.DEFAULT_DATASET/"images_8"/camera_name) as image:
            rgb=np.asarray(image.convert("RGB"))
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax in axes:
            ax.imshow(rgb);ax.set(xlim=(-.5,camera.image_width-.5),ylim=(camera.image_height-.5,-.5));ax.set_axis_off()
        axes[0].set_title("Original RGB photograph — secondary reference")
        pixels,valid=projection.project(xyz,camera)
        axes[1].scatter(*pixels[valid].T,s=.6,c=w171.SUPPORT_RGB,alpha=.3,linewidths=0)
        segments=[];segment_colors=[]
        for loop,color in zip(world,colors):
            xy,ok=projection.project(loop,camera)
            nxt=np.roll(xy,-1,axis=0);keep=ok&np.roll(ok,-1)
            segments.extend(np.stack((xy[keep],nxt[keep]),axis=1));segment_colors.extend([color]*int(keep.sum()))
        axes[1].add_collection(LineCollection(segments,colors=segment_colors,linewidths=.6,alpha=.8))
        axes[1].set_title("Selected support + all loop projections; no depth culling")
        fig.suptitle(f"W173 {camera_name} — cyan support / stable loop colors / magenta self-contact loops")
        fig.tight_layout()
        save(fig,FAMILIES[6],Path(camera_name).stem+".png")
        camera_accounting[camera_name]={"support_projected":int(valid.sum()),"support_total":len(xyz),"loop_segments_projected":len(segments),"depth_culling":False}
    return {"png_count":len(reviewed),"png_paths":reviewed,"camera_accounting":camera_accounting}


def write_readmes(report,out):
    # Korean prose with technical identifiers left in English.
    shared=("입력은 W171 complete tabletop 17,965개와 W172 selected support 15,189개, 원본 134 loop다. "
        "Geometry, row order, chart, 모든 loop를 유지한다. Cyan은 selected support, gray point는 overview의 나머지 component, "
        "tab20 색은 stable loop ID를 20색으로 반복한 것이며 의미 등급이 아니다. Magenta는 self-contact loop L000/L009 또는 명시된 height witness다. "
        "F의 coolwarm colorbar는 signed chart-plane residual / h다. D의 blue cell은 occupied chart cell, white cell은 unsupported이며 red segment는 native adjacency의 chart-bin 대응이다.\n\n"
        "World view는 기존 fixed orientation과 equal XYZ scale, white figure background, 150-dpi PNG다. Chart 축 단위는 h, area는 h²이며 "
        "모든 support row와 모든 loop를 표시한다. Subsampling, smoothing, gap filling, morphology, loop/component filtering은 없다. "
        "All-loop chart의 작은 ID는 겹칠 수 있으며 full loop_inventory.md와 JSON이 모든 ID/size rank/geometry accounting을 제공한다. "
        "색이 반복되므로 ID로 대응을 확인한다. World loop는 canonical chart plane의 embedding이며 native physical surface edge가 아니다.\n\n"
        "Structural Surface / Parametric Carrier와 Observed Evidence-Supported Domain은 다르다. White hole은 물리적 빈 공간 또는 occluded/latent surface의 정답이 아니다. "
        "Chart hole count, loop count와 physical surface count는 같은 양이 아니다. W154 closed=False는 multi-loop case flag이며 134개 loop는 모두 개별적으로 닫혔다. "
        "Two compound loops와 inter-loop point contacts 때문에 simple polygon containment tree는 인증되지 않았다. Generalized even-odd enclosure는 참고 관계로만 기록한다. "
        "Native 3D cell union에는 1D boundary loop가 정의되지 않아 134를 native loop 수와 직접 비교할 수 없다. NURBS fit, repair, continuation은 실행하지 않았다.\n")
    meanings={
        "":"W173 loop attribution 산출물이다. worklog_173_report.json은 모든134개 loop의 full accounting과 architecture 판단, loop_inventory.md는 size-ranked head와 full tail, diagnostic_correspondence.npz는 world/chart 대응을 보존한다.",
        "review_views":"7개 visualization family로 complete support, selected component, chart loops, containment, loop scale, world/chart 대응, RGB reference를 보여 준다.",
        "review_views/"+FAMILIES[0]:"전체 W171 support와 W172 진단 component를 함께 표시한다. 제외된 312 component/2,776 row도 gray로 전부 남아 있다.",
        "review_views/"+FAMILIES[1]:"선택 component의 모든 world XYZ와 134개 loop의 planar embedding이다. L000/L009/L023은 후속 chart view와 일치하는 stable ID다. 모든 loop가 같은 chart plane에 있다는 것은 모든 support가 그 plane에 있다는 증거가 아니다.",
        "review_views/"+FAMILIES[2]:"기존 canonical chart occupancy와134 loop를 모두 표시한다. 각 loop ID를 표기하며 작은 loop도 제거하지 않는다. Pixel [u,u+1] 경계는 W154 chart-lattice 규약이다.",
        "review_views/"+FAMILIES[3]:"전체 chart containment 문맥과 L000 self-contact, L023의 1-cell exterior family를 확대한다. 확대 범위는 발견된 singularity 위치로 결정하며 acceptance에는 사용하지 않는다. L000은 정상 simple outer가 아니므로 그림을 정상적인 단일-parent polygon tree로 해석하지 않는다.",
        "review_views/"+FAMILIES[4]:"134개 전체 loop의 W172 원래 size rank에 따른 edge count/perimeter/absolute algebraic area 분포다. Log y는 작은 tail을 보여 주기 위한 축 표시이며 threshold가 아니다. Magenta non-simple loop의 algebraic area는 일반 simple-polygon 면적이 아니다.",
        "review_views/"+FAMILIES[5]:"동일 support와 L000/L009/L023을 world/chart에서 대응시킨다. Signed plane residual과 같은 chart bin의 두 sample height witness도 표시한다. 정해진 plane/normal cutoff는 없으며 최대값 witness는 진단용으로만 강조한다. Quantized collision이나 normal 반전만으로 continuous physical fold-over를 확정하지 않는다.",
        "review_views/"+FAMILIES[6]:"기존 고정 camera의 원본 RGB 사진 위에 selected support와134개 loop를 canonical renderer half-pixel 규약으로 투영했다. Gaussian render/state plot은 아니다. 좌우 panel은 동일 full camera extent이고 no depth culling이다. Canonical near/frustum 밖 row와 양 끝점이 유효하지 않은 segment는 표시되지 않을 수 있다. Image-space overlap은 physical topology evidence가 아니며 camera별 PNG를 이 directory에 직접 저장한다."
    }
    for directory,meaning in meanings.items():
        dest=out/directory;dest.mkdir(parents=True,exist_ok=True)
        (dest/"README.md").write_text(f"# W173 {directory or 'loop 진단'}\n\n{meaning}\n\n{shared}",encoding="utf-8")
    lines=["# 전체 loop 목록: head와 full tail\n", "크기는 기술 통계이며 acceptance/selection 규칙이 아니다. 모든134개 row를 W172 원래 size rank 순서로 보존한다. Signed area는 h², perimeter는 h 단위다. Invalid loop는 algebraic winding area만 갖는다. Parent는 generalized even-odd 관계이며 valid polygon tree로 인증되지 않았다.\n",
           "| W172 rank | Stable ID | Edges | Perimeter / h | Signed area / h² | Simple valid | Occupancy role | Generalized parent |",
           "|---:|---|---:|---:|---:|---|---|---|"]
    for row in report["loops"]:
        lines.append(f"| {row['w172_size_rank']} | {row['loop_id']} | {row['edge_count']} | {row['perimeter_grid_units']:g} | {row['signed_area_grid_units_squared']:g} | {row['simple_valid_polygon']} | {row['occupancy']['class']} | {row['containment']['generalized_parent'] or 'none/contact-ambiguous'} |")
    (out/"loop_inventory.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def validate(out):
    paths=[out/"review_views"/family/"tabletop.png" for family in FAMILIES[:-1]]
    paths += [out/"review_views"/FAMILIES[-1]/(Path(name).stem+".png") for name in w171.w155.REVIEW_CAMERAS]
    for path in paths:
        with Image.open(path) as image:
            assert image.format=="PNG";image.verify()
    for directory in [out]+[p for p in out.rglob("*") if p.is_dir()]:
        assert (directory/"README.md").read_text(encoding="utf-8")
    assert not list(out.rglob("*.ppm"))
    return {"png_count":len(paths),"readme_count":len(list(out.rglob("README.md"))),"complete":True}
